//! Stage M3: 24 GiB Sparse GDDR6X VRAM Windows (12 × 2 GiB) for zero-fault,
//! direct 40-bit shader memory addressing (`VICTORY_SPARSE_UMA=1`).
//!
//! Instead of maintaining a 1,023-copy fault-driven directory (`memory.rs`),
//! this module backs a 32 GiB-aligned 24 GiB virtual address aperture
//! (`6 × 4 GiB = 12 × 2 GiB`) with twelve 2 GiB D3D12 `CreateReservedResource`
//! raw UAV buffers (`D3D12_RESOURCE_DIMENSION_BUFFER`,
//! `D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS`, `0x2000_0000` dwords <= D3D12
//! `2^30` raw UAV limit — zero 1 MiB gaps at 4 GiB boundaries).
//!
//! Physical GDDR6X VRAM is committed on demand at 64 KiB tile granularity via
//! `upload_queue.UpdateTileMappings` onto a pool of 64 MiB
//! `D3D12_HEAP_TYPE_DEFAULT` (`D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS`) heaps
//! (never `OpenExistingHeap*` file-mapping heaps). Decommitted tiles are parked
//! LRU up to [`PARK_LIMIT_BYTES`] (256 MiB) before unmapping, empty 64 MiB
//! pools are returned to VRAM, and COM `Release` calls are queued lock-free so
//! releasing a heap can never invert locks against D3D12's internal mutex.
//!
//! Before each submission (`host.rs::submit`), mapped VA ranges inside the
//! aperture undergo scoped `GetWriteWatch(WRITE_WATCH_FLAG_RESET)` + Stage M2
//! 64-byte cacheline diffing (`crate::shadow::diff_cachelines_64`) and upload
//! onto `upload_queue` via the write-only `UploadRing`, ordered into the main
//! graphics queue via GPU-only `direct_queue.Wait(&upload_fence, val)` (zero
//! CPU stall).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicPtr, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
use windows::core::Interface;

/// Number of 2 GiB reserved buffer windows covering the 24 GiB aperture.
pub const WINDOW_COUNT: usize = 12;
/// Size of each reserved buffer window: 2 GiB (`0x2000_0000` dwords <= 2^30).
pub const WINDOW_BYTES: u64 = 2 << 30;
/// Total virtual aperture size: 24 GiB (`12 × 2 GiB = 6 × 4 GiB`).
pub const APERTURE_BYTES: u64 = (WINDOW_COUNT as u64) * WINDOW_BYTES;
/// Required alignment of the 24 GiB aperture (32 GiB, guarantees `lo == 0`
/// and clean high-dword subtraction with no 4 GiB boundary wrap).
pub const APERTURE_ALIGN: u64 = 32 << 30;
/// D3D12 standard tile size: 64 KiB.
pub const TILE_BYTES: u64 = 64 << 10;
/// 64 KiB tiles in one 2 GiB window (`32,768`).
pub const TILES_PER_WINDOW: u32 = (WINDOW_BYTES / TILE_BYTES) as u32;
/// 64 KiB tiles in one 64 MiB `D3D12_HEAP_TYPE_DEFAULT` backing pool heap.
pub const POOL_TILES: u32 = 1024;
/// Maximum decommitted bytes kept parked (mapped) before LRU unmapping: 256 MiB.
pub const PARK_LIMIT_BYTES: u64 = 256 << 20;
/// Maximum parked 64 KiB tiles (`4,096`).
const PARK_LIMIT_TILES: usize = (PARK_LIMIT_BYTES / TILE_BYTES) as usize;
/// Fixed descriptor-heap slots reserved right below `null_slot`:
/// `+0` = null raw UAV, `+1..=12` = 12 × 2 GiB window raw UAVs, `+13..16` spare.
pub(crate) const SPARSE_SLOTS: u32 = 16;

/// Default RAM cap for 64 KiB baseline shadows used by Stage M2 64-byte diffing
/// (`VICTORY_SPARSE_DIFF64_MB`; 512 MiB default keeps RAM lean on 31.1 GB hosts).
const DEFAULT_DIFF_SHADOW_MB: u64 = 512;

/// Statistics for diagnostics / frame status line:
/// `[0]` mapped tiles, `[1]` parked tiles, `[2]` pool bytes alive,
/// `[3]` empty 64 MiB pools released, `[4]` `UpdateTileMappings` map calls,
/// `[5]` `UpdateTileMappings` unmap calls, `[6]` synced bytes, `[7]` synced runs,
/// `[8]` sync microseconds.
pub(crate) static STATS: [AtomicU64; 9] = [const { AtomicU64::new(0) }; 9];

/// Detailed per-kind dirty-scan breakdown for `status()` observability:
/// `[0]` write_watch_tiles (current), `[1]` mapped_view_tiles (current),
/// `[2]` plain_guarded_tiles (current), `[3]` uncommitted_tiles (current),
/// `[4]` fallback_tiles (cum), `[5]` virtual_queries (cum),
/// `[6]` bytes_compared (cum), `[7]` bytes_uploaded (cum),
/// `[8]` submits (cum), `[9]` ww_calls (cum), `[10]` mv_calls (cum),
/// `[11]` reclass_tiles (cum), `[12]` ww_us (cum), `[13]` mv_us (cum),
/// `[14]` pg_us (cum), `[15]` scan_us (cum).
static SCAN_STATS: [AtomicU64; 16] = [const { AtomicU64::new(0) }; 16];

/// Resolves `VICTORY_SPARSE_UMA`: strictly opt-in (`"1"` enables; unset or any
/// other value leaves all legacy paths intact).
fn resolve_sparse_uma(env: Option<&str>) -> bool {
    env.is_some_and(|v| v.trim() == "1")
}

/// Whether Stage M3 sparse 24 GiB GDDR6X VRAM windows are enabled
/// (`VICTORY_SPARSE_UMA=1`).
pub fn active() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let env = std::env::var("VICTORY_SPARSE_UMA").ok();
        resolve_sparse_uma(env.as_deref())
    })
}

/// Resolves `VICTORY_SPARSE_DIFF64_MB` cap in bytes.
fn diff_shadow_cap_bytes() -> u64 {
    static CAP: OnceLock<u64> = OnceLock::new();
    *CAP.get_or_init(|| {
        std::env::var("VICTORY_SPARSE_DIFF64_MB")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(DEFAULT_DIFF_SHADOW_MB)
            << 20
    })
}

fn resolve_diff64_enabled(env: Option<&str>) -> bool {
    env.map(str::trim) == Some("1")
}

/// Whether Stage M2 64-byte cacheline diffing is enabled for sparse uploads
/// (`VICTORY_DIFF64`, opt-in `"1"`; off by default in production to avoid 4x
/// range fragmentation in the 3D world; enabled under `cfg!(test)` when unset
/// so the Stage M2 diff64 integration tests exercise the cacheline path).
fn diff64_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| match std::env::var("VICTORY_DIFF64").ok() {
        Some(v) => resolve_diff64_enabled(Some(&v)),
        None => cfg!(test),
    })
}

/// Probes a 32 GiB-aligned 24 GiB contiguous free VA range strictly inside the
/// 40-bit console address space (`[32 GiB, 1 TiB)`). Immediately releases the
/// probe reservation so `xmem.rs` can issue normal `VirtualAlloc2` calls pinned
/// to `[base, base + 24 GiB - 1]` with native kernel `VirtualFree` tracking.
fn probe_aperture() -> Option<u64> {
    use windows_sys::Win32::System::Memory::{
        MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, VirtualAlloc2, VirtualFree,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    // Prefer 64 GiB (`0x10_0000_0000`) upwards up to 992 GiB (`0xF8_0000_0000`),
    // falling back to 32 GiB (`0x08_0000_0000`) last so low executable/ASLR
    // space stays untouched while staying strictly < 1 TiB (40-bit).
    let candidates = (2u64..=31).chain(std::iter::once(1u64));
    for step in candidates {
        let base = step * APERTURE_ALIGN;
        if base.checked_add(APERTURE_BYTES).is_none_or(|end| end > (1u64 << 40)) {
            continue;
        }
        // SAFETY: probe-reserving a free VA region at `base` with `PAGE_NOACCESS`
        // and immediately releasing it on success.
        let ptr = unsafe {
            VirtualAlloc2(
                GetCurrentProcess(),
                base as *const std::ffi::c_void,
                APERTURE_BYTES as usize,
                MEM_RESERVE,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            )
        };
        if ptr as u64 == base {
            // SAFETY: releasing the exact probe reservation just returned above.
            unsafe { VirtualFree(ptr, 0, MEM_RELEASE) };
            tracing::info!(
                base = format_args!("{base:#x}"),
                end = format_args!("{:#x}", base + APERTURE_BYTES),
                aperture_hi = format_args!("{:#x}", (base >> 32) as u32),
                "sparse UMA 24 GiB VA aperture ready (12 x 2 GiB, 32 GiB-aligned)"
            );
            return Some(base);
        }
        if !ptr.is_null() {
            // SAFETY: unexpected base; release cleanly.
            unsafe { VirtualFree(ptr, 0, MEM_RELEASE) };
        }
    }
    tracing::warn!("sparse UMA: could not locate a 32 GiB-aligned 24 GiB free VA aperture below 1 TiB");
    None
}

/// Ensures the 32 GiB-aligned 24 GiB VA aperture is selected and returns
/// `(aperture_base, APERTURE_BYTES)`.
pub fn ensure_aperture() -> Option<(u64, u64)> {
    static BASE: OnceLock<Option<u64>> = OnceLock::new();
    let base = (*BASE.get_or_init(probe_aperture))?;
    Some((base, APERTURE_BYTES))
}

/// Returns `(aperture_base, APERTURE_BYTES)` when `VICTORY_SPARSE_UMA=1` is
/// active, or `None` when disabled.
pub fn aperture() -> Option<(u64, u64)> {
    if !active() {
        return None;
    }
    ensure_aperture()
}

/// High 32 bits of the 24 GiB aperture base (`USER_MEMORY_APERTURE_HI`), or 0
/// if sparse UMA is inactive.
pub fn aperture_hi() -> u32 {
    aperture().map_or(0, |(base, _)| (base >> 32) as u32)
}

/// Maps a 40-bit title VA `addr` inside `[aperture_base, aperture_base + 24 GiB)`
/// to `(window_index, tile_in_window, byte_offset_in_tile)`.
///
/// Exact host counterpart to Track 4A's shader addressing:
/// `hi_rel = hi - aperture_hi`, `win = (hi_rel << 1) | (lo >> 31)`,
/// `byte = lo & 0x7FFF_FFFF`.
pub fn locate(aperture_base: u64, addr: u64) -> Option<(usize, u32, u32)> {
    let rel = addr.checked_sub(aperture_base)?;
    if rel >= APERTURE_BYTES {
        return None;
    }
    let hi_rel = ((addr >> 32) as u32).wrapping_sub((aperture_base >> 32) as u32);
    if hi_rel >= (WINDOW_COUNT as u32 / 2) {
        return None;
    }
    let lo = addr as u32;
    let win = ((hi_rel << 1) | (lo >> 31)) as usize;
    let byte = lo & 0x7FFF_FFFF;
    let tile = byte / (TILE_BYTES as u32);
    let offset = byte % (TILE_BYTES as u32);
    Some((win, tile, offset))
}

/// Converts `(aperture_base, win, tile)` back to the 64 KiB-aligned title VA.
#[inline]
pub fn tile_va(aperture_base: u64, win: usize, tile: u32) -> u64 {
    aperture_base + (win as u64) * WINDOW_BYTES + u64::from(tile) * TILE_BYTES
}

// ---------------------------------------------------------------------------
// Lock-free deferred COM Release queue (Treiber stack)
// ---------------------------------------------------------------------------

struct ReleaseNode {
    _object: windows::core::IUnknown,
    fence: u64,
    next: *mut ReleaseNode,
}

// SAFETY: Enqueued D3D12 COM objects (`ID3D12Heap`, `ID3D12Resource`) are
// free-threaded and dropped exactly once after their GPU fence completes.
unsafe impl Send for ReleaseNode {}

/// Lock-free deferred COM `Release` queue. Prevents AB-BA lock inversion
/// between our pool/engine mutexes and D3D12's internal runtime lock when
/// releasing empty 64 MiB tile heaps or resources.
pub(crate) struct ReleaseQueue {
    head: AtomicPtr<ReleaseNode>,
}

impl ReleaseQueue {
    pub(crate) const fn new() -> Self {
        Self {
            head: AtomicPtr::new(std::ptr::null_mut()),
        }
    }

    /// Lock-free push of `object` to be released once `completed_fence >= fence`.
    pub(crate) fn push(&self, object: &impl Interface, fence: u64) {
        let Ok(unknown) = object.cast::<windows::core::IUnknown>() else {
            return;
        };
        let node = Box::into_raw(Box::new(ReleaseNode {
            _object: unknown,
            fence,
            next: std::ptr::null_mut(),
        }));
        self.push_raw(node);
    }

    fn push_raw(&self, node: *mut ReleaseNode) {
        let mut head = self.head.load(Ordering::Acquire);
        loop {
            // SAFETY: `node` is exclusively owned until the CAS succeeds.
            unsafe { (*node).next = head };
            match self.head.compare_exchange_weak(
                head,
                node,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => break,
                Err(actual) => head = actual,
            }
        }
    }

    /// Atomically detaches the stack and drops every object whose fence has
    /// completed (`node.fence <= completed_fence`) strictly outside all locks.
    /// Returns the number of COM objects released.
    pub(crate) fn drain(&self, completed_fence: u64) -> usize {
        let mut cur = self.head.swap(std::ptr::null_mut(), Ordering::AcqRel);
        let mut released = 0usize;
        while !cur.is_null() {
            // SAFETY: exclusively detached by `swap` above.
            let node = unsafe { Box::from_raw(cur) };
            let next = node.next;
            if node.fence <= completed_fence {
                drop(node);
                released += 1;
            } else {
                self.push_raw(Box::into_raw(node));
            }
            cur = next;
        }
        released
    }
}

impl Drop for ReleaseQueue {
    fn drop(&mut self) {
        let _ = self.drain(u64::MAX);
    }
}

// ---------------------------------------------------------------------------
// 64 MiB D3D12_HEAP_TYPE_DEFAULT buffer tile pool (mirrors tiles.rs design)
// ---------------------------------------------------------------------------

struct PoolHeap {
    heap: ID3D12Heap,
    /// Sorted, non-adjacent free runs `(first_tile, count)`.
    free: Vec<(u32, u32)>,
}

// SAFETY: `ID3D12Heap` is free-threaded; mutated only under the pool's owner lock.
unsafe impl Send for PoolHeap {}

impl PoolHeap {
    fn free_tiles(&self) -> u32 {
        self.free.iter().map(|r| r.1).sum()
    }
}

struct TilePool {
    heaps: Vec<Option<PoolHeap>>,
    /// Total live 64 MiB pool bytes.
    pool_bytes: u64,
    /// Empty 64 MiB pools handed off to the lock-free release queue.
    pools_released: u64,
}

impl TilePool {
    fn new() -> Self {
        Self {
            heaps: Vec::new(),
            pool_bytes: 0,
            pools_released: 0,
        }
    }

    #[cfg(test)]
    fn alloc_one(&mut self, device: &ID3D12Device) -> Option<(usize, ID3D12Heap, u32)> {
        let runs = self.alloc_runs(device, 1);
        let (pool_id, heap, start, count) = runs.into_iter().next()?;
        debug_assert_eq!(count, 1);
        Some((pool_id, heap, start))
    }

    /// Allocates `count` 64 KiB tiles across one or more 64 MiB `DEFAULT`
    /// (`D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS`) pool heaps, returning
    /// `(pool_id, heap, first_tile, run_count)` segments whose counts sum to
    /// `count` (or fewer if VRAM allocation fails).
    fn alloc_runs(&mut self, device: &ID3D12Device, mut count: u32) -> Vec<(usize, ID3D12Heap, u32, u32)> {
        let mut out = Vec::new();
        while count > 0 {
            let idx = match self
                .heaps
                .iter()
                .position(|h| h.as_ref().is_some_and(|h| !h.free.is_empty()))
            {
                Some(i) => i,
                None => match self.grow(device) {
                    Some(i) => i,
                    None => break,
                },
            };
            let Some(h) = self.heaps[idx].as_mut() else {
                break;
            };
            let heap = h.heap.clone();
            while count > 0 && !h.free.is_empty() {
                let (start, avail) = h.free[0];
                let take = avail.min(count);
                out.push((idx, heap.clone(), start, take));
                if take == avail {
                    h.free.remove(0);
                } else {
                    h.free[0] = (start + take, avail - take);
                }
                count -= take;
            }
        }
        out
    }

    fn grow(&mut self, device: &ID3D12Device) -> Option<usize> {
        let size = u64::from(POOL_TILES) * TILE_BYTES;
        let desc = D3D12_HEAP_DESC {
            SizeInBytes: size,
            Properties: D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            },
            Alignment: 0,
            Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS,
        };
        let mut made: Option<ID3D12Heap> = None;
        // SAFETY: valid D3D12 buffer-only DEFAULT heap creation; never a file-mapping heap.
        if let Err(e) = unsafe { device.CreateHeap::<ID3D12Heap>(&raw const desc, &raw mut made) } {
            tracing::warn!(error = %e, "sparse UMA: failed to create 64 MiB DEFAULT tile pool heap");
            return None;
        }
        let heap = made?;
        crate::host::name(&heap, || format!("sparse UMA 64 MiB tile pool #{}", self.heaps.len()));
        self.pool_bytes += size;
        STATS[2].fetch_add(size, Ordering::Relaxed);
        let fresh = PoolHeap {
            heap,
            free: vec![(0, POOL_TILES)],
        };
        match self.heaps.iter().position(Option::is_none) {
            Some(i) => {
                self.heaps[i] = Some(fresh);
                Some(i)
            }
            None => {
                self.heaps.push(Some(fresh));
                Some(self.heaps.len() - 1)
            }
        }
    }

    /// Returns `(pool_id, pool_tile)` runs to the pool and hands completely
    /// empty 64 MiB heaps off to `releases` (lock-free deferred COM `Release`).
    /// If `keep_one` is true, keeps at least one warm 64 MiB heap in the pool.
    fn give_back(
        &mut self,
        by_pool: &BTreeMap<usize, Vec<(u32, u32)>>,
        releases: &ReleaseQueue,
        fence: u64,
        keep_one: bool,
    ) {
        let mut active_pools = self.heaps.iter().flatten().count();
        for (&pool_id, runs) in by_pool {
            let Some(Some(h)) = self.heaps.get_mut(pool_id) else {
                continue;
            };
            h.free.extend_from_slice(runs);
            h.free.sort_unstable();
            let mut joined: Vec<(u32, u32)> = Vec::with_capacity(h.free.len());
            for &(from, n) in &h.free {
                match joined.last_mut() {
                    Some(last) if last.0 + last.1 == from => last.1 += n,
                    _ => joined.push((from, n)),
                }
            }
            h.free = joined;
            let all_free = h.free_tiles() == POOL_TILES;
            let may_release = if keep_one { active_pools > 1 } else { true };
            if all_free && may_release {
                if let Some(old) = self.heaps[pool_id].take() {
                    let size = u64::from(POOL_TILES) * TILE_BYTES;
                    self.pool_bytes = self.pool_bytes.saturating_sub(size);
                    self.pools_released += 1;
                    STATS[2].fetch_sub(size, Ordering::Relaxed);
                    STATS[3].fetch_add(1, Ordering::Relaxed);
                    active_pools -= 1;
                    releases.push(&old.heap, fence);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Pending notifications from xmem.rs (lock-ordered, never touches D3D12)
// ---------------------------------------------------------------------------

#[derive(Default)]
struct PendingEvents {
    /// Tiles `(win, tile)` newly mapped/committed by `xmem.rs` awaiting
    /// batched `UpdateTileMappings` before the next submission.
    to_map: BTreeSet<(u8, u32)>,
    /// Tiles `(win, tile)` unmapped/decommitted by `xmem.rs` awaiting parking.
    to_unmap: BTreeSet<(u8, u32)>,
}

static PENDING: Mutex<PendingEvents> = Mutex::new(PendingEvents {
    to_map: BTreeSet::new(),
    to_unmap: BTreeSet::new(),
});

/// Notifies `sparse_uma` that title VA `[addr, addr + len)` was mapped or
/// committed. Safe to call while holding `xmem::PHYSICAL` and
/// `title_memory_remapping()`: never calls D3D12.
pub fn on_map(addr: usize, len: usize) {
    if len == 0 || !active() {
        return;
    }
    let Some((base, _)) = aperture() else {
        return;
    };
    let start = (addr as u64).max(base);
    let end = (addr as u64).saturating_add(len as u64).min(base + APERTURE_BYTES);
    if start >= end {
        return;
    }
    let first_tile_va = start & !(TILE_BYTES - 1);
    let last_tile_end = end.next_multiple_of(TILE_BYTES).min(base + APERTURE_BYTES);
    plain_guards_release_range(first_tile_va as usize, last_tile_end as usize);
    let mut pending = PENDING.lock().unwrap_or_else(|e| e.into_inner());
    let mut va = first_tile_va;
    while va < last_tile_end {
        if let Some((win, tile, _)) = locate(base, va) {
            let key = (win as u8, tile);
            pending.to_unmap.remove(&key);
            pending.to_map.insert(key);
        }
        va += TILE_BYTES;
    }
}

/// Notifies `sparse_uma` that title VA `[addr, addr + len)` was unmapped or
/// decommitted. Safe to call while holding `xmem::PHYSICAL` and
/// `title_memory_remapping()`: never calls D3D12.
pub fn on_unmap(addr: usize, len: usize) {
    if len == 0 || !active() {
        return;
    }
    let Some((base, _)) = aperture() else {
        return;
    };
    let start = (addr as u64).max(base);
    let end = (addr as u64).saturating_add(len as u64).min(base + APERTURE_BYTES);
    if start >= end {
        return;
    }
    // Only unmap tiles wholly covered or touched on 64 KiB console page boundaries.
    let first_tile_va = start & !(TILE_BYTES - 1);
    let last_tile_end = end.next_multiple_of(TILE_BYTES).min(base + APERTURE_BYTES);
    plain_guards_release_range(first_tile_va as usize, last_tile_end as usize);
    let mut pending = PENDING.lock().unwrap_or_else(|e| e.into_inner());
    let mut va = first_tile_va;
    while va < last_tile_end {
        if let Some((win, tile, _)) = locate(base, va) {
            let key = (win as u8, tile);
            pending.to_map.remove(&key);
            pending.to_unmap.insert(key);
        }
        va += TILE_BYTES;
    }
}

// ---------------------------------------------------------------------------
// Self-contained read-only write-guard for non-MEM_WRITE_WATCH private tiles
// ---------------------------------------------------------------------------

/// Tracks 64 KiB tiles belonging to ordinary committed `MEM_PRIVATE`
/// allocations that were NOT created with `MEM_WRITE_WATCH`. By protecting
/// untouched tiles read-only and restoring write access on the first CPU write
/// in a vectored exception handler, untouched private tiles cost 0 syscalls
/// and 0 CPU scans on repeat submits while still detecting every CPU write.
///
/// Lock-order leaf: never acquires any other mutex or calls into D3D12.
struct PlainGuards {
    /// 64 KiB tile page number (`va / TILE_BYTES`) -> original writable
    /// protection (`PAGE_READWRITE`, `PAGE_EXECUTE_READWRITE`, etc.) to restore
    /// on fault or release.
    protected: BTreeMap<usize, u32>,
    /// 64 KiB tile page numbers written since the last `sparse_uma` sync.
    dirty: BTreeSet<usize>,
}

static PLAIN_GUARDS: Mutex<PlainGuards> = Mutex::new(PlainGuards {
    protected: BTreeMap::new(),
    dirty: BTreeSet::new(),
});

/// Monotonic epoch bumped whenever any `PLAIN_GUARDS` tile transitions out of
/// the clean protected state (VEH write fault, release, or failed protect).
static PLAIN_GUARD_EPOCH: AtomicU64 = AtomicU64::new(1);

#[inline]
fn plain_guard_epoch() -> u64 {
    PLAIN_GUARD_EPOCH.load(Ordering::Acquire)
}

#[inline]
fn bump_plain_guard_epoch() {
    PLAIN_GUARD_EPOCH.fetch_add(1, Ordering::Release);
}

/// Returns the read-only guard protection matching a writable base protection.
fn guard_prot_for(restore_prot: u32) -> u32 {
    use windows_sys::Win32::System::Memory::{
        PAGE_EXECUTE_READ, PAGE_EXECUTE_READWRITE, PAGE_EXECUTE_WRITECOPY, PAGE_READONLY,
    };
    match restore_prot & 0xFF {
        PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY => PAGE_EXECUTE_READ,
        _ => PAGE_READONLY,
    }
}

/// Registers the vectored exception handler for `PLAIN_GUARDS` once.
/// Strictly handles pages in `PLAIN_GUARDS.protected` and returns
/// `EXCEPTION_CONTINUE_SEARCH` for everything else (fully composable with
/// `watch.rs`'s `MEM_MAPPED` guard handler and the crash reporter).
fn install_plain_guard_handler() {
    use windows_sys::Win32::Foundation::{EXCEPTION_ACCESS_VIOLATION, NTSTATUS};
    use windows_sys::Win32::System::Diagnostics::Debug::{
        AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH,
        EXCEPTION_POINTERS,
    };
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_EXECUTE_READWRITE, PAGE_EXECUTE_WRITECOPY,
        PAGE_READWRITE, PAGE_WRITECOPY, VirtualProtect, VirtualQuery,
    };

    unsafe extern "system" fn on_plain_write(info: *mut EXCEPTION_POINTERS) -> i32 {
        // SAFETY: Windows passes a valid exception record during VEH dispatch.
        let r = unsafe { &*(*info).ExceptionRecord };
        const WRITE: usize = 1;
        if r.ExceptionCode != EXCEPTION_ACCESS_VIOLATION as NTSTATUS
            || r.NumberParameters < 2
            || r.ExceptionInformation[0] != WRITE
        {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        let fault_va = r.ExceptionInformation[1];
        let page_id = fault_va / (TILE_BYTES as usize);
        let mut g = PLAIN_GUARDS.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(restore_prot) = g.protected.remove(&page_id) {
            g.dirty.insert(page_id);
            // Bump epoch BEFORE restoring write access so any concurrent O(1)
            // epoch check observes the invalidation before the faulting store lands.
            bump_plain_guard_epoch();
            let mut old = 0u32;
            // SAFETY: restoring the exact original writable protection on a
            // 64 KiB tile that `sparse_uma` itself protected read-only.
            let _ = unsafe {
                VirtualProtect(
                    (page_id * (TILE_BYTES as usize)) as *const _,
                    TILE_BYTES as usize,
                    restore_prot,
                    &mut old,
                )
            };
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        // Another thread may have faulted on this same tile just before we
        // restored write access; retry cleanly if the page is now writable and
        // recorded in `dirty`.
        if g.dirty.contains(&page_id) {
            let mut mbi: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
            if unsafe { VirtualQuery(fault_va as *const _, &mut mbi, size_of_val(&mbi)) } != 0
                && mbi.State == MEM_COMMIT
                && matches!(
                    mbi.Protect & 0xFF,
                    PAGE_READWRITE | PAGE_WRITECOPY | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY
                )
            {
                return EXCEPTION_CONTINUE_EXECUTION;
            }
        }
        EXCEPTION_CONTINUE_SEARCH
    }

    static ONCE: std::sync::Once = std::sync::Once::new();
    // SAFETY: installs a process-wide VEH callback that touches only `PLAIN_GUARDS`.
    ONCE.call_once(|| unsafe {
        AddVectoredExceptionHandler(1, Some(on_plain_write));
    });
}

/// Arms a read-only write guard on the 64 KiB tile at `va` (`restore_prot` is
/// its original writable protection). Returns `false` if `VirtualProtect` fails
/// (for example if the page was unmapped/freed concurrently).
fn plain_guard_protect(va: usize, restore_prot: u32) -> bool {
    use windows_sys::Win32::System::Memory::VirtualProtect;
    install_plain_guard_handler();
    let page_id = va / (TILE_BYTES as usize);
    let mut g = PLAIN_GUARDS.lock().unwrap_or_else(|e| e.into_inner());
    let mut old = 0u32;
    // SAFETY: `va` is a 64 KiB-aligned committed private tile; `PLAIN_GUARDS`
    // is held across `VirtualProtect` + map insert so a racing write fault
    // observes the `protected` entry immediately.
    let ok = unsafe {
        VirtualProtect(
            va as *const _,
            TILE_BYTES as usize,
            guard_prot_for(restore_prot),
            &mut old,
        )
    } != 0;
    if ok {
        g.protected.insert(page_id, restore_prot);
        g.dirty.remove(&page_id);
    } else {
        g.protected.remove(&page_id);
        g.dirty.remove(&page_id);
        bump_plain_guard_epoch();
    }
    ok
}

/// Releases any `PLAIN_GUARDS` entries covering `[start, end)` and restores
/// their original writable protection.
fn plain_guards_release_range(start: usize, end: usize) {
    use windows_sys::Win32::System::Memory::VirtualProtect;
    if start >= end {
        return;
    }
    let first = start / (TILE_BYTES as usize);
    let last = end.div_ceil(TILE_BYTES as usize);
    let mut g = PLAIN_GUARDS.lock().unwrap_or_else(|e| e.into_inner());
    if g.protected.is_empty() && g.dirty.is_empty() {
        return;
    }
    let to_restore: Vec<(usize, u32)> = g
        .protected
        .range(first..last)
        .map(|(&p, &prot)| (p, prot))
        .collect();
    let dirty_clear: Vec<usize> = g.dirty.range(first..last).copied().collect();
    if !to_restore.is_empty() || !dirty_clear.is_empty() {
        bump_plain_guard_epoch();
    }
    for (page_id, restore_prot) in to_restore {
        g.protected.remove(&page_id);
        let mut old = 0u32;
        // SAFETY: restoring original protection on a tile we guarded; ignores
        // failure if the tile was already freed/unmapped.
        let _ = unsafe {
            VirtualProtect(
                (page_id * (TILE_BYTES as usize)) as *const _,
                TILE_BYTES as usize,
                restore_prot,
                &mut old,
            )
        };
    }
    for page_id in dirty_clear {
        g.dirty.remove(&page_id);
    }
}

/// Releases a single 64 KiB tile from `PLAIN_GUARDS` (restoring its original
/// protection if currently guarded).
fn plain_guard_release_tile(va: usize) {
    plain_guards_release_range(va, va.saturating_add(TILE_BYTES as usize));
}

// ---------------------------------------------------------------------------
// Core SparseEngine (12 x 2 GiB Reserved Buffers + Tile Map + Diff64 Sync)
// ---------------------------------------------------------------------------

/// Cached OS memory-region classification for a 64 KiB sparse tile.
///
/// Populated lazily via `VirtualQuery` (once per contiguous OS region, NOT
/// once per submit!) and invalidated on `on_map`/`on_unmap` or if a write-watch
/// / view-guard query fails (e.g. after lazy `MEM_PRIVATE` -> `MEM_MAPPED`
/// promotion in `xmem`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RegionKind {
    /// Not yet classified (or invalidated after map/unmap/probe failure).
    Unknown,
    /// Committed `MEM_PRIVATE` with `MEM_WRITE_WATCH` under `alloc_base`.
    /// Contiguous tiles sharing the same `alloc_base` are queried in ONE
    /// `watch::writes(..., false)` call and NEVER merged across different
    /// `alloc_base`s.
    PrivateWriteWatch { alloc_base: usize },
    /// Committed `MEM_MAPPED` section view — tracked via `watch::writes(..., true)`
    /// (`PAGE_READONLY` + VEH guard in `watch.rs`).
    MappedView,
    /// Committed `MEM_PRIVATE` without `MEM_WRITE_WATCH` (or read-only private
    /// memory): guarded by `PLAIN_GUARDS` when writable, or skipped after
    /// initial sync when originally read-only.
    PrivateGuarded {
        read_only_originally: bool,
        restore_prot: u32,
    },
    /// Uncommitted (`MEM_RESERVE`, `MEM_FREE`, or `PAGE_NOACCESS`): 0 dirty bytes.
    Uncommitted,
}

/// Diagnostics captured on the most recent `collect_dirty_uploads()` pass.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct SyncMetrics {
    /// `VirtualQuery` calls executed during lazy region classification.
    pub(crate) virtual_queries: u64,
    /// 64 KiB tiles copied & diffed against their baseline (excluding initial sync).
    pub(crate) tiles_scanned: u64,
    /// Full-span unconditional dirty fallbacks triggered (target: 0).
    pub(crate) full_fallbacks: u64,
    /// Tiles scanned via full-span fallback (target: 0).
    pub(crate) fallback_tiles: u64,
    /// Per-`AllocationBase` `watch::writes(..., false)` calls for `MEM_WRITE_WATCH`.
    pub(crate) write_watch_calls: u64,
    /// Coalesced `watch::writes(..., true)` calls for `MEM_MAPPED` views.
    pub(crate) mapped_view_calls: u64,
    /// Tiles invalidated and reclassified during this pass.
    pub(crate) reclass_tiles: u64,
    /// Bytes copied & compared against baseline (`tiles_scanned * TILE_BYTES`).
    pub(crate) bytes_compared: u64,
    /// Bytes emitted in upload chunks during this pass.
    pub(crate) bytes_uploaded: u64,
    /// Microseconds spent in `PrivateWriteWatch` `watch::writes`.
    pub(crate) ww_us: u64,
    /// Microseconds spent in `MappedView` `watch::writes`.
    pub(crate) mv_us: u64,
    /// Microseconds spent in `PrivateGuarded` guard checks.
    pub(crate) pg_us: u64,
    /// Microseconds spent copying & diffing dirty tiles.
    pub(crate) scan_us: u64,
}

struct MappedTile {
    pool_id: usize,
    pool_tile: u32,
    needs_initial_sync: bool,
    region: RegionKind,
    /// Highest `watch::writes` asking sequence synced into VRAM for this tile.
    synced_seq: u64,
    baseline: Option<Box<[u8; TILE_BYTES as usize]>>,
}

struct ParkedTile {
    pool_id: usize,
    pool_tile: u32,
    lru_seq: u64,
}

/// An upload chunk `(window_index, byte_offset_in_window, bytes)` produced by
/// scoped `GetWriteWatch` + Stage M2 64-byte diffing for `upload_queue`.
pub(crate) struct UploadChunk {
    pub(crate) window: usize,
    pub(crate) offset: u64,
    pub(crate) data: Vec<u8>,
}

/// Pre-partitioned contiguous same-`RegionKind` span cached across steady-state
/// submits (rebuilt ONLY on tile map/unmap or region reclassification).
struct CachedSpan {
    kind: RegionKind,
    start: u64,
    end: u64,
    keys: Vec<(u8, u32)>,
    /// True once every tile in `keys` has completed `sync_initial_tile`.
    all_initialized: bool,
    /// Latest `watch::JOURNAL` sequence confirmed synced across this span
    /// (`u64::MAX` transiently before first sync when all tiles need init).
    synced_seq: u64,
}

/// Self-contained 12 × 2 GiB sparse VRAM window engine. Used both by the
/// process-wide singleton (`MANAGER`) and by standalone RTX 3090 D3D12 Debug
/// Layer tests with their own device and queues.
pub(crate) struct SparseEngine {
    aperture_base: u64,
    windows: [ID3D12Resource; WINDOW_COUNT],
    pool: TilePool,
    mapped: BTreeMap<(u8, u32), MappedTile>,
    parked: BTreeMap<(u8, u32), ParkedTile>,
    lru_clock: u64,
    park_limit_tiles: usize,
    diff_shadow_bytes: u64,
    diff_shadow_cap: u64,
    /// Reusable 64 KiB copy buffer (zero heap allocations on clean submits).
    scratch: Box<[u8; TILE_BYTES as usize]>,
    /// True if any mapped tile currently has `RegionKind::Unknown`.
    has_unclassified: bool,
    /// Cached contiguous same-`RegionKind` spans (zero `BTreeMap` scans on steady state).
    cached_spans: Vec<CachedSpan>,
    /// True if `cached_spans` / `kind_counts` need rebuilding from `self.mapped`.
    spans_dirty: bool,
    /// Cached per-kind tile counts: `[ww, mv, pg, unc]`.
    kind_counts: [u64; 4],
    /// Number of writable `PrivateGuarded` spans in `cached_spans`.
    pg_writable_spans: usize,
    /// True if any originally read-only `PrivateGuarded` tile still needs initial sync.
    has_uninit_ro: bool,
    /// Clean `watch::guard_epoch()` when all `MappedView` spans were verified clean.
    mv_clean_epoch: Option<u64>,
    /// Clean `plain_guard_epoch()` when all writable `PrivateGuarded` spans were verified clean.
    pg_clean_epoch: Option<u64>,
    last_sync: SyncMetrics,
    releases: ReleaseQueue,
}

impl Drop for SparseEngine {
    fn drop(&mut self) {
        for (&(win, tile_idx), tile) in &self.mapped {
            if matches!(
                tile.region,
                RegionKind::PrivateGuarded {
                    read_only_originally: false,
                    ..
                }
            ) {
                let va = tile_va(self.aperture_base, win as usize, tile_idx) as usize;
                plain_guard_release_tile(va);
            }
        }
    }
}

// SAFETY: D3D12 resources and heaps are free-threaded; `SparseEngine` is
// accessed exclusively behind a `Mutex` (or owned by a single test thread).
unsafe impl Send for SparseEngine {}

impl SparseEngine {
    /// Creates twelve 2 GiB `CreateReservedResource` (`ALLOW_UNORDERED_ACCESS`,
    /// `D3D12_RESOURCE_STATE_COMMON`) buffers on `device`. Returns `None` if
    /// the device lacks Tiled Resources Tier 1+ or refuses reserved buffers.
    pub(crate) fn new(device: &ID3D12Device, aperture_base: u64) -> Option<Self> {
        let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
        // SAFETY: valid CheckFeatureSupport query.
        let ok = unsafe {
            device.CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS,
                (&raw mut options).cast(),
                size_of_val(&options) as u32,
            )
        };
        if ok.is_err() || options.TiledResourcesTier.0 < 1 {
            tracing::warn!(
                tier = options.TiledResourcesTier.0,
                "sparse UMA requires D3D12 Tiled Resources Tier 1+"
            );
            return None;
        }

        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Alignment: TILE_BYTES,
            Width: WINDOW_BYTES,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };

        let mut windows = Vec::with_capacity(WINDOW_COUNT);
        for w in 0..WINDOW_COUNT {
            let mut res: Option<ID3D12Resource> = None;
            // SAFETY: valid reserved buffer resource creation in COMMON state.
            if let Err(e) = unsafe {
                device.CreateReservedResource(
                    &desc,
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut res,
                )
            } {
                tracing::warn!(window = w, error = %e, "sparse UMA: CreateReservedResource(2 GiB) failed");
                return None;
            }
            let res = res?;
            crate::host::name(&res, || format!("sparse UMA 2 GiB window #{w}"));
            windows.push(res);
        }
        let windows: [ID3D12Resource; WINDOW_COUNT] = windows.try_into().ok()?;
        tracing::info!(
            aperture = format_args!("{aperture_base:#x}"),
            windows = WINDOW_COUNT,
            window_mb = WINDOW_BYTES >> 20,
            tier = options.TiledResourcesTier.0,
            "sparse UMA: created 12 x 2 GiB (24 GiB) reserved GDDR6X windows"
        );
        Some(Self {
            aperture_base,
            windows,
            pool: TilePool::new(),
            mapped: BTreeMap::new(),
            parked: BTreeMap::new(),
            lru_clock: 0,
            park_limit_tiles: PARK_LIMIT_TILES,
            diff_shadow_bytes: 0,
            diff_shadow_cap: diff_shadow_cap_bytes(),
            scratch: Box::new([0u8; TILE_BYTES as usize]),
            has_unclassified: false,
            cached_spans: Vec::new(),
            spans_dirty: false,
            kind_counts: [0; 4],
            pg_writable_spans: 0,
            has_uninit_ro: false,
            mv_clean_epoch: None,
            pg_clean_epoch: None,
            last_sync: SyncMetrics::default(),
            releases: ReleaseQueue::new(),
        })
    }

    /// Processes queued `to_unmap` (moving tiles to `Parked`) and `to_map`
    /// (promoting parked tiles or mapping fresh 64 KiB tiles via
    /// `upload_queue.UpdateTileMappings`), plus LRU eviction if parked tiles
    /// exceed `self.park_limit_tiles`.
    ///
    /// Returns `true` if any `UpdateTileMappings` command was issued on
    /// `upload_queue` (caller MUST signal `upload_fence` and order
    /// `direct_queue.Wait(&upload_fence, val)`).
    pub(crate) fn apply_tile_changes(
        &mut self,
        device: &ID3D12Device,
        upload_queue: &ID3D12CommandQueue,
        to_unmap: &BTreeSet<(u8, u32)>,
        to_map: &BTreeSet<(u8, u32)>,
        release_fence: u64,
        keep_one_pool: bool,
    ) -> bool {
        let mut updated = false;
        let mut mapped_changed = false;

        // 1. Park newly unmapped tiles (zero UpdateTileMappings cost until
        //    parked tiles exceed `park_limit_tiles`!).
        for &key in to_unmap {
            if let Some(m) = self.mapped.remove(&key) {
                mapped_changed = true;
                if matches!(
                    m.region,
                    RegionKind::PrivateGuarded {
                        read_only_originally: false,
                        ..
                    }
                ) {
                    let va = tile_va(self.aperture_base, key.0 as usize, key.1) as usize;
                    plain_guard_release_tile(va);
                }
                if m.baseline.is_some() {
                    self.diff_shadow_bytes = self.diff_shadow_bytes.saturating_sub(TILE_BYTES);
                }
                self.lru_clock += 1;
                self.parked.insert(
                    key,
                    ParkedTile {
                        pool_id: m.pool_id,
                        pool_tile: m.pool_tile,
                        lru_seq: self.lru_clock,
                    },
                );
            }
        }

        // 2. Promote parked tiles or collect brand-new tiles to map.
        let mut fresh: BTreeMap<u8, Vec<u32>> = BTreeMap::new();
        for &key in to_map {
            if let Some(existing) = self.mapped.get_mut(&key) {
                mapped_changed = true;
                self.has_unclassified = true;
                if matches!(
                    existing.region,
                    RegionKind::PrivateGuarded {
                        read_only_originally: false,
                        ..
                    }
                ) {
                    let va = tile_va(self.aperture_base, key.0 as usize, key.1) as usize;
                    plain_guard_release_tile(va);
                }
                existing.needs_initial_sync = true;
                existing.region = RegionKind::Unknown;
                existing.synced_seq = 0;
                continue;
            }
            if let Some(p) = self.parked.remove(&key) {
                mapped_changed = true;
                self.has_unclassified = true;
                // Instant reuse from parked tiles — no UpdateTileMappings needed!
                self.mapped.insert(
                    key,
                    MappedTile {
                        pool_id: p.pool_id,
                        pool_tile: p.pool_tile,
                        needs_initial_sync: true,
                        region: RegionKind::Unknown,
                        synced_seq: 0,
                        baseline: None,
                    },
                );
                continue;
            }
            fresh.entry(key.0).or_default().push(key.1);
        }

        // 3. Evict excess parked tiles if over `park_limit_tiles` (before fresh
        //    allocations so freed pool tiles can immediately be reused!).
        if self.parked.len() > self.park_limit_tiles {
            let evict_n = self.parked.len() - self.park_limit_tiles;
            if self.evict_parked(upload_queue, evict_n, release_fence, keep_one_pool) {
                updated = true;
            }
        }

        // 4. Map fresh tiles onto 64 MiB DEFAULT pools via batched UpdateTileMappings.
        for (win, mut tiles) in fresh {
            tiles.sort_unstable();
            tiles.dedup();
            // Coalesce contiguous tile runs inside `win`.
            let mut runs: Vec<(u32, u32)> = Vec::new();
            for t in tiles {
                match runs.last_mut() {
                    Some(last) if last.0 + last.1 == t => last.1 += 1,
                    _ => runs.push((t, 1)),
                }
            }
            for (mut res_tile, mut remain) in runs {
                let allocs = self.pool.alloc_runs(device, remain);
                for (pool_id, heap, pool_start, count) in allocs {
                    let coord = D3D12_TILED_RESOURCE_COORDINATE {
                        X: res_tile,
                        Y: 0,
                        Z: 0,
                        Subresource: 0,
                    };
                    let region = D3D12_TILE_REGION_SIZE {
                        NumTiles: count,
                        ..Default::default()
                    };
                    let flag = D3D12_TILE_RANGE_FLAG_NONE;
                    // SAFETY: live 2 GiB reserved buffer and live 64 MiB DEFAULT heap;
                    // `res_tile..res_tile+count <= TILES_PER_WINDOW` and
                    // `pool_start..pool_start+count <= POOL_TILES`.
                    unsafe {
                        upload_queue.UpdateTileMappings(
                            &self.windows[win as usize],
                            1,
                            Some(&raw const coord),
                            Some(&raw const region),
                            &heap,
                            1,
                            Some(&raw const flag),
                            Some(&raw const pool_start),
                            Some(&raw const count),
                            D3D12_TILE_MAPPING_FLAG_NONE,
                        );
                    }
                    STATS[4].fetch_add(1, Ordering::Relaxed);
                    updated = true;
                    mapped_changed = true;
                    self.has_unclassified = true;
                    for k in 0..count {
                        self.mapped.insert(
                            (win, res_tile + k),
                            MappedTile {
                                pool_id,
                                pool_tile: pool_start + k,
                                needs_initial_sync: true,
                                region: RegionKind::Unknown,
                                synced_seq: 0,
                                baseline: None,
                            },
                        );
                    }
                    res_tile += count;
                    remain = remain.saturating_sub(count);
                }
                if remain > 0 {
                    tracing::warn!(win, res_tile, remain, "sparse UMA: VRAM pool exhausted for tiles");
                }
            }
        }

        if mapped_changed {
            self.spans_dirty = true;
            self.mv_clean_epoch = None;
            self.pg_clean_epoch = None;
        }

        STATS[0].store(self.mapped.len() as u64, Ordering::Relaxed);
        STATS[1].store(self.parked.len() as u64, Ordering::Relaxed);
        updated
    }

    /// Evicts `count` oldest (lowest `lru_seq`) parked tiles via batched
    /// `UpdateTileMappings(NULL)` and returns them to the 64 MiB pool,
    /// releasing completely empty pools via `self.releases` (lock-free).
    pub(crate) fn evict_parked(
        &mut self,
        upload_queue: &ID3D12CommandQueue,
        count: usize,
        release_fence: u64,
        keep_one_pool: bool,
    ) -> bool {
        if count == 0 || self.parked.is_empty() {
            return false;
        }
        let mut order: Vec<(u64, (u8, u32))> = self
            .parked
            .iter()
            .map(|(&k, p)| (p.lru_seq, k))
            .collect();
        order.sort_unstable_by_key(|&(seq, _)| seq);
        let take = count.min(order.len());

        let mut by_win: BTreeMap<u8, Vec<u32>> = BTreeMap::new();
        let mut by_pool: BTreeMap<usize, Vec<(u32, u32)>> = BTreeMap::new();
        for &(_, key) in &order[..take] {
            if let Some(p) = self.parked.remove(&key) {
                by_win.entry(key.0).or_default().push(key.1);
                by_pool.entry(p.pool_id).or_default().push((p.pool_tile, 1));
            }
        }
        if by_win.is_empty() {
            return false;
        }
        for (win, mut tiles) in by_win {
            tiles.sort_unstable();
            let mut coords = Vec::new();
            let mut regions = Vec::new();
            let mut total = 0u32;
            let mut i = 0;
            while i < tiles.len() {
                let start = tiles[i];
                let mut len = 1u32;
                while i + (len as usize) < tiles.len() && tiles[i + (len as usize)] == start + len {
                    len += 1;
                }
                coords.push(D3D12_TILED_RESOURCE_COORDINATE {
                    X: start,
                    Y: 0,
                    Z: 0,
                    Subresource: 0,
                });
                regions.push(D3D12_TILE_REGION_SIZE {
                    NumTiles: len,
                    ..Default::default()
                });
                total += len;
                i += len as usize;
            }
            let null_flag = D3D12_TILE_RANGE_FLAG_NULL;
            // SAFETY: unmapping `total` tiles of a live reserved buffer with
            // `pHeap = None` and a single NULL range covering `total` tiles.
            unsafe {
                upload_queue.UpdateTileMappings(
                    &self.windows[win as usize],
                    coords.len() as u32,
                    Some(coords.as_ptr()),
                    Some(regions.as_ptr()),
                    None::<&ID3D12Heap>,
                    1,
                    Some(&raw const null_flag),
                    None,
                    Some(&raw const total),
                    D3D12_TILE_MAPPING_FLAG_NONE,
                );
            }
            STATS[5].fetch_add(1, Ordering::Relaxed);
        }
        self.pool
            .give_back(&by_pool, &self.releases, release_fence, keep_one_pool);
        STATS[1].store(self.parked.len() as u64, Ordering::Relaxed);
        true
    }

    /// Lazily classifies mapped tiles whose `region == RegionKind::Unknown`
    /// using one `VirtualQuery` per contiguous OS region (never re-querying
    /// already-classified tiles on steady-state submits).
    fn classify_unknown_tiles(&mut self, filter: Option<&[(u8, u32)]>) {
        use windows_sys::Win32::System::Memory::{
            GetWriteWatch, MEM_COMMIT, MEM_MAPPED, MEM_PRIVATE, MEMORY_BASIC_INFORMATION,
            PAGE_EXECUTE_READ, PAGE_EXECUTE_READWRITE, PAGE_EXECUTE_WRITECOPY, PAGE_NOACCESS,
            PAGE_READONLY, PAGE_READWRITE, PAGE_WRITECOPY, VirtualQuery,
        };
        const KERNEL_PAGE: usize = 4096;

        // Collect contiguous VA runs of unclassified tiles only.
        let mut unknown_runs: Vec<(usize, usize, Vec<(u8, u32)>)> = Vec::new();
        let mut push_key = |key: (u8, u32), region: RegionKind| {
            if region != RegionKind::Unknown {
                return;
            }
            let va = tile_va(self.aperture_base, key.0 as usize, key.1) as usize;
            let end = va + TILE_BYTES as usize;
            match unknown_runs.last_mut() {
                Some(last) if last.1 == va => {
                    last.1 = end;
                    last.2.push(key);
                }
                _ => unknown_runs.push((va, end, vec![key])),
            }
        };
        if let Some(keys) = filter {
            for &key in keys {
                if let Some(tile) = self.mapped.get(&key) {
                    push_key(key, tile.region);
                }
            }
        } else {
            for (&key, tile) in &self.mapped {
                push_key(key, tile.region);
            }
        }
        if unknown_runs.is_empty() {
            return;
        }

        let mut probe_addr = [0usize; 1];
        for (run_start, run_end, run_keys) in unknown_runs {
            let mut at = run_start;
            while at < run_end {
                // SAFETY: plain VirtualQuery into a zeroed struct.
                let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
                // SAFETY: valid out pointer.
                let queried =
                    unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
                self.last_sync.virtual_queries += 1;
                if queried == 0 {
                    // Unqueryable VA: mark remaining tiles in this run uncommitted.
                    let start_idx = (at - run_start) / (TILE_BYTES as usize);
                    for &key in &run_keys[start_idx..] {
                        if let Some(t) = self.mapped.get_mut(&key)
                            && t.region == RegionKind::Unknown
                        {
                            t.region = RegionKind::Uncommitted;
                        }
                    }
                    break;
                }
                let region_end = (info.BaseAddress as usize)
                    .saturating_add(info.RegionSize)
                    .min(run_end);
                if region_end <= at {
                    break;
                }

                let base_prot = info.Protect & 0xFF;
                let kind = if info.State != MEM_COMMIT || base_prot == PAGE_NOACCESS {
                    RegionKind::Uncommitted
                } else if info.Type == MEM_MAPPED {
                    RegionKind::MappedView
                } else if info.Type == MEM_PRIVATE {
                    // Non-destructive probe (`flags = 0`, single 4 KiB page):
                    // checks whether `info.AllocationBase` has `MEM_WRITE_WATCH`
                    // WITHOUT clearing any pending kernel dirty bits.
                    let mut count = 1usize;
                    let mut gran = 0u32;
                    // SAFETY: `probe_addr` holds 1 entry for a 1-page query.
                    let is_ww = unsafe {
                        GetWriteWatch(
                            0,
                            at as *const _,
                            KERNEL_PAGE,
                            probe_addr.as_mut_ptr().cast(),
                            &mut count,
                            &mut gran,
                        )
                    } == 0;
                    if is_ww {
                        RegionKind::PrivateWriteWatch {
                            alloc_base: info.AllocationBase as usize,
                        }
                    } else if matches!(
                        base_prot,
                        PAGE_READWRITE
                            | PAGE_WRITECOPY
                            | PAGE_EXECUTE_READWRITE
                            | PAGE_EXECUTE_WRITECOPY
                    ) {
                        RegionKind::PrivateGuarded {
                            read_only_originally: false,
                            restore_prot: base_prot,
                        }
                    } else if matches!(base_prot, PAGE_READONLY | PAGE_EXECUTE_READ) {
                        // Distinguish a tile we ourselves protected read-only in
                        // `PLAIN_GUARDS` from an originally read-only region.
                        let page_id = at / (TILE_BYTES as usize);
                        let ours = PLAIN_GUARDS
                            .lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .protected
                            .get(&page_id)
                            .copied();
                        match ours {
                            Some(restore_prot) => RegionKind::PrivateGuarded {
                                read_only_originally: false,
                                restore_prot,
                            },
                            None => RegionKind::PrivateGuarded {
                                read_only_originally: true,
                                restore_prot: base_prot,
                            },
                        }
                    } else {
                        RegionKind::Uncommitted
                    }
                } else {
                    RegionKind::PrivateGuarded {
                        read_only_originally: true,
                        restore_prot: base_prot,
                    }
                };

                let first_idx = (at - run_start) / (TILE_BYTES as usize);
                let last_idx = (region_end - run_start)
                    .div_ceil(TILE_BYTES as usize)
                    .min(run_keys.len());
                for &key in &run_keys[first_idx..last_idx] {
                    if let Some(t) = self.mapped.get_mut(&key)
                        && (t.region == RegionKind::Unknown
                            || (t.region == RegionKind::Uncommitted
                                && kind != RegionKind::Uncommitted))
                    {
                        t.region = kind;
                    }
                }
                at = region_end;
            }
        }
    }

    /// Performs the initial full 64 KiB upload for a newly mapped tile and
    /// initializes its Stage M2 diff64 baseline shadow if under budget.
    fn sync_initial_tile(
        &mut self,
        key: (u8, u32),
        va: u64,
        uploads: &mut Vec<UploadChunk>,
        total_bytes: &mut u64,
    ) {
        let win = key.0 as usize;
        let win_tile_off = u64::from(key.1) * TILE_BYTES;
        crate::host::copy_mapped(
            self.scratch.as_mut_ptr(),
            va as *const u8,
            TILE_BYTES as usize,
            false,
        );
        let Some(tile) = self.mapped.get_mut(&key) else {
            return;
        };
        tile.needs_initial_sync = false;
        if tile.baseline.is_none()
            && self.diff_shadow_bytes.saturating_add(TILE_BYTES) <= self.diff_shadow_cap
        {
            let mut b = Box::new([0u8; TILE_BYTES as usize]);
            b.copy_from_slice(&self.scratch[..]);
            tile.baseline = Some(b);
            self.diff_shadow_bytes += TILE_BYTES;
        } else if let Some(b) = tile.baseline.as_mut() {
            b.copy_from_slice(&self.scratch[..]);
        }
        *total_bytes += TILE_BYTES;
        uploads.push(UploadChunk {
            window: win,
            offset: win_tile_off,
            data: self.scratch.to_vec(),
        });
    }

    /// Reads a dirty 64 KiB tile and runs Stage M2 64-byte cacheline diffing
    /// against its baseline shadow, pushing only modified cacheline runs.
    #[allow(clippy::too_many_arguments)]
    fn sync_dirty_tile(
        &mut self,
        key: (u8, u32),
        va: u64,
        diff64: bool,
        diff_runs: &mut Vec<(usize, usize)>,
        uploads: &mut Vec<UploadChunk>,
        total_bytes: &mut u64,
    ) {
        let win = key.0 as usize;
        let win_tile_off = u64::from(key.1) * TILE_BYTES;
        let scan_started = std::time::Instant::now();
        self.last_sync.tiles_scanned += 1;
        self.last_sync.bytes_compared += TILE_BYTES;
        crate::host::copy_mapped(
            self.scratch.as_mut_ptr(),
            va as *const u8,
            TILE_BYTES as usize,
            false,
        );
        let Some(tile) = self.mapped.get_mut(&key) else {
            self.last_sync.scan_us += scan_started.elapsed().as_micros() as u64;
            return;
        };
        if diff64 && let Some(baseline) = tile.baseline.as_mut() {
            diff_runs.clear();
            crate::shadow::diff_cachelines_64(
                &mut baseline[..],
                &self.scratch[..],
                0,
                diff_runs,
            );
            for &(a, b) in diff_runs.iter() {
                *total_bytes += (b - a) as u64;
                uploads.push(UploadChunk {
                    window: win,
                    offset: win_tile_off + a as u64,
                    data: baseline[a..b].to_vec(),
                });
            }
        } else {
            if let Some(baseline) = tile.baseline.as_mut() {
                baseline.copy_from_slice(&self.scratch[..]);
            }
            *total_bytes += TILE_BYTES;
            uploads.push(UploadChunk {
                window: win,
                offset: win_tile_off,
                data: self.scratch.to_vec(),
            });
        }
        self.last_sync.scan_us += scan_started.elapsed().as_micros() as u64;
    }

    /// Rebuilds the cached contiguous same-`RegionKind` spans and per-kind tile
    /// counts from `self.mapped` (called ONLY after tile map/unmap or region
    /// reclassification — NEVER on steady-state submits).
    fn rebuild_cached_spans(&mut self) {
        self.cached_spans.clear();
        let mut counts = [0u64; 4];
        let mut pg_writable_spans = 0usize;
        let mut has_uninit_ro = false;
        for (&key, tile) in &self.mapped {
            match tile.region {
                RegionKind::PrivateWriteWatch { .. } => counts[0] += 1,
                RegionKind::MappedView => counts[1] += 1,
                RegionKind::PrivateGuarded {
                    read_only_originally,
                    ..
                } => {
                    counts[2] += 1;
                    if read_only_originally && tile.needs_initial_sync {
                        has_uninit_ro = true;
                    }
                }
                RegionKind::Uncommitted | RegionKind::Unknown => {
                    counts[3] += 1;
                    continue;
                }
            }
            let va = tile_va(self.aperture_base, key.0 as usize, key.1);
            let end = va + TILE_BYTES;
            match self.cached_spans.last_mut() {
                Some(last) if last.kind == tile.region && last.end == va => {
                    last.end = end;
                    last.keys.push(key);
                    if tile.needs_initial_sync {
                        last.all_initialized = false;
                    } else {
                        last.synced_seq = last.synced_seq.min(tile.synced_seq);
                    }
                }
                _ => {
                    if matches!(
                        tile.region,
                        RegionKind::PrivateGuarded {
                            read_only_originally: false,
                            ..
                        }
                    ) {
                        pg_writable_spans += 1;
                    }
                    self.cached_spans.push(CachedSpan {
                        kind: tile.region,
                        start: va,
                        end,
                        keys: vec![key],
                        all_initialized: !tile.needs_initial_sync,
                        synced_seq: if tile.needs_initial_sync {
                            u64::MAX
                        } else {
                            tile.synced_seq
                        },
                    });
                }
            }
        }
        self.kind_counts = counts;
        self.pg_writable_spans = pg_writable_spans;
        self.has_uninit_ro = has_uninit_ro;
        self.spans_dirty = false;
        self.mv_clean_epoch = None;
        self.pg_clean_epoch = None;
    }

    /// Builds a temporary list of `CachedSpan`s covering only `keys` (used on
    /// rare lazy reclassification passes).
    fn build_spans_for_keys(&self, keys: &[(u8, u32)]) -> Vec<CachedSpan> {
        let mut spans: Vec<CachedSpan> = Vec::new();
        for &key in keys {
            let Some(tile) = self.mapped.get(&key) else {
                continue;
            };
            if matches!(tile.region, RegionKind::Uncommitted | RegionKind::Unknown) {
                continue;
            }
            let va = tile_va(self.aperture_base, key.0 as usize, key.1);
            let end = va + TILE_BYTES;
            match spans.last_mut() {
                Some(last) if last.kind == tile.region && last.end == va => {
                    last.end = end;
                    last.keys.push(key);
                    if tile.needs_initial_sync {
                        last.all_initialized = false;
                    } else {
                        last.synced_seq = last.synced_seq.min(tile.synced_seq);
                    }
                }
                _ => spans.push(CachedSpan {
                    kind: tile.region,
                    start: va,
                    end,
                    keys: vec![key],
                    all_initialized: !tile.needs_initial_sync,
                    synced_seq: if tile.needs_initial_sync {
                        u64::MAX
                    } else {
                        tile.synced_seq
                    },
                }),
            }
        }
        spans
    }

    /// Processes pre-partitioned same-`RegionKind` spans:
    /// - `PrivateWriteWatch`: per-`AllocationBase` `watch::writes_private_since`
    ///   (NEVER merges across `AllocationBase`s; skips per-tile `BTreeMap` loop
    ///   when clean).
    /// - `MappedView`: coalesced `watch::writes_guarded_since` (skips both
    ///   `GetWriteWatch` and `writes_mixed` `VirtualQuery` loops; skipped
    ///   entirely in O(1) when `mv_skip` is true).
    /// - `PrivateGuarded`: `PLAIN_GUARDS` fast range check (skipped entirely in
    ///   O(1) when `pg_skip` is true, or forever after init when originally RO).
    ///
    /// Returns `(reclassify, mv_had_work, mv_all_ok, pg_had_work, pg_all_ok)`.
    #[allow(clippy::too_many_arguments)]
    fn process_span_slice(
        &mut self,
        spans: &mut [CachedSpan],
        mv_skip: bool,
        pg_skip: bool,
        allow_reclassify: bool,
        diff64: bool,
        diff_runs: &mut Vec<(usize, usize)>,
        uploads: &mut Vec<UploadChunk>,
        total_bytes: &mut u64,
    ) -> (Vec<(u8, u32)>, bool, bool, bool, bool) {
        let mut reclassify: Vec<(u8, u32)> = Vec::new();
        let mut mv_had_work = false;
        let mut mv_all_ok = true;
        let mut pg_had_work = false;
        let mut pg_all_ok = true;

        for span in spans.iter_mut() {
            match span.kind {
                RegionKind::PrivateWriteWatch { .. } | RegionKind::MappedView => {
                    let guarded = matches!(span.kind, RegionKind::MappedView);
                    if guarded && mv_skip {
                        continue;
                    }
                    if guarded {
                        self.last_sync.mapped_view_calls += 1;
                    } else {
                        self.last_sync.write_watch_calls += 1;
                        if !span.all_initialized {
                            // Pre-fault any demand-zero 4 KiB pages of newly mapped
                            // `MEM_WRITE_WATCH` tiles BEFORE `GetWriteWatch(RESET)`:
                            // on Windows, the first read of a demand-zero
                            // `MEM_WRITE_WATCH` page faults in a zero frame and sets
                            // its write-watch bit, which would otherwise trigger a
                            // spurious dirty scan on the next submit.
                            for (idx, &key) in span.keys.iter().enumerate() {
                                if self.mapped.get(&key).is_some_and(|t| t.needs_initial_sync) {
                                    let va = span.start + (idx as u64) * TILE_BYTES;
                                    crate::host::copy_mapped(
                                        self.scratch.as_mut_ptr(),
                                        va as *const u8,
                                        TILE_BYTES as usize,
                                        false,
                                    );
                                }
                            }
                        }
                    }
                    // Route through `watch`'s shared `JOURNAL` so `Shadow` copies
                    // and `sparse_uma` NEVER steal each other's dirty state, while
                    // skipping `GetWriteWatch`/`VirtualQuery` walks on known spans
                    // and filtering out unchanged historical journal entries.
                    let q_started = std::time::Instant::now();
                    let watch_res = if guarded {
                        crate::watch::writes_guarded_since(
                            span.start as usize,
                            span.end as usize,
                            span.synced_seq,
                        )
                    } else {
                        crate::watch::writes_private_since(
                            span.start as usize,
                            span.end as usize,
                            span.synced_seq,
                        )
                    };
                    let q_us = q_started.elapsed().as_micros() as u64;
                    if guarded {
                        self.last_sync.mv_us += q_us;
                    } else {
                        self.last_sync.ww_us += q_us;
                    }
                    if let Some((now, pages)) = watch_res {
                        if span.all_initialized {
                            // Steady-state fast path: zero `BTreeMap` lookups when
                            // `pages.is_empty()`, or O(dirty) lookups when sparse.
                            if !pages.is_empty() {
                                let first_page_64k = (span.start / TILE_BYTES) as usize;
                                for &(page_64k, seq) in &pages {
                                    let Some(idx) = page_64k.checked_sub(first_page_64k) else {
                                        continue;
                                    };
                                    let Some(&key) = span.keys.get(idx) else {
                                        continue;
                                    };
                                    let synced_seq = match self.mapped.get(&key) {
                                        Some(t) => t.synced_seq,
                                        None => continue,
                                    };
                                    if seq > synced_seq {
                                        let va = span.start + (idx as u64) * TILE_BYTES;
                                        self.sync_dirty_tile(
                                            key,
                                            va,
                                            diff64,
                                            diff_runs,
                                            uploads,
                                            total_bytes,
                                        );
                                        if guarded {
                                            mv_had_work = true;
                                        }
                                    }
                                    if let Some(t) = self.mapped.get_mut(&key) {
                                        t.synced_seq = now;
                                    }
                                }
                            }
                            span.synced_seq = now;
                        } else {
                            let mut pi = 0usize;
                            for (idx, &key) in span.keys.iter().enumerate() {
                                let va = span.start + (idx as u64) * TILE_BYTES;
                                let page_64k = (va / TILE_BYTES) as usize;
                                while pi < pages.len() && pages[pi].0 < page_64k {
                                    pi += 1;
                                }
                                let (needs_init, synced_seq) = match self.mapped.get(&key) {
                                    Some(t) => (t.needs_initial_sync, t.synced_seq),
                                    None => continue,
                                };
                                let journal_dirty = pi < pages.len()
                                    && pages[pi].0 == page_64k
                                    && pages[pi].1 > synced_seq;
                                if needs_init {
                                    self.sync_initial_tile(key, va, uploads, total_bytes);
                                    if guarded {
                                        mv_had_work = true;
                                    }
                                } else if journal_dirty {
                                    self.sync_dirty_tile(
                                        key,
                                        va,
                                        diff64,
                                        diff_runs,
                                        uploads,
                                        total_bytes,
                                    );
                                    if guarded {
                                        mv_had_work = true;
                                    }
                                }
                                if let Some(t) = self.mapped.get_mut(&key) {
                                    t.synced_seq = now;
                                }
                            }
                            span.all_initialized = true;
                            span.synced_seq = now;
                        }
                    } else if allow_reclassify {
                        if guarded {
                            mv_all_ok = false;
                        }
                        // Cached classification is stale (e.g. `xmem` promoted
                        // a private view to `MEM_MAPPED`, or memory was freed):
                        // invalidate just this span and re-classify once.
                        self.last_sync.reclass_tiles += span.keys.len() as u64;
                        for &key in &span.keys {
                            if let Some(t) = self.mapped.get_mut(&key) {
                                t.region = RegionKind::Unknown;
                            }
                            reclassify.push(key);
                        }
                    } else {
                        if guarded {
                            mv_all_ok = false;
                        }
                        // Last-resort safe fallback only if a span still fails
                        // immediately after fresh `VirtualQuery` classification.
                        self.last_sync.full_fallbacks += 1;
                        self.last_sync.fallback_tiles += span.keys.len() as u64;
                        for (idx, &key) in span.keys.iter().enumerate() {
                            let va = span.start + (idx as u64) * TILE_BYTES;
                            let needs_init =
                                self.mapped.get(&key).is_some_and(|t| t.needs_initial_sync);
                            if needs_init {
                                self.sync_initial_tile(key, va, uploads, total_bytes);
                            } else {
                                self.sync_dirty_tile(
                                    key,
                                    va,
                                    diff64,
                                    diff_runs,
                                    uploads,
                                    total_bytes,
                                );
                            }
                        }
                        span.all_initialized = true;
                    }
                }
                RegionKind::PrivateGuarded {
                    read_only_originally: true,
                    ..
                } => {
                    // Originally read-only private memory cannot be written
                    // without a protection change: sync once on first map, then
                    // skip with 0 syscalls and 0 CPU scans forever.
                    if span.all_initialized {
                        continue;
                    }
                    for (idx, &key) in span.keys.iter().enumerate() {
                        let va = span.start + (idx as u64) * TILE_BYTES;
                        if self.mapped.get(&key).is_some_and(|t| t.needs_initial_sync) {
                            self.sync_initial_tile(key, va, uploads, total_bytes);
                        }
                    }
                    span.all_initialized = true;
                }
                RegionKind::PrivateGuarded {
                    read_only_originally: false,
                    restore_prot,
                } => {
                    if pg_skip {
                        continue;
                    }
                    // Batch-poll `PLAIN_GUARDS` under ONE mutex lock for the
                    // entire span: when all tiles are initialized and clean,
                    // range-count confirms cleanliness with 0 allocations!
                    let pg_started = std::time::Instant::now();
                    let mut work: Vec<(usize, (u8, u32), bool)> = Vec::new();
                    {
                        let mut g = PLAIN_GUARDS.lock().unwrap_or_else(|e| e.into_inner());
                        let first_page = (span.start / TILE_BYTES) as usize;
                        let last_page = (span.end / TILE_BYTES) as usize;
                        let all_clean = span.all_initialized
                            && g.dirty.range(first_page..last_page).next().is_none()
                            && g.protected.range(first_page..last_page).count() == span.keys.len();
                        if !all_clean {
                            for (idx, &key) in span.keys.iter().enumerate() {
                                let needs_init = !span.all_initialized
                                    && self.mapped.get(&key).is_some_and(|t| t.needs_initial_sync);
                                if needs_init {
                                    work.push((idx, key, true));
                                    continue;
                                }
                                let page_id = first_page + idx;
                                let was_dirty = g.dirty.remove(&page_id);
                                let is_protected = g.protected.contains_key(&page_id);
                                if was_dirty || !is_protected {
                                    work.push((idx, key, false));
                                }
                            }
                        }
                    }
                    self.last_sync.pg_us += pg_started.elapsed().as_micros() as u64;
                    if !work.is_empty() {
                        pg_had_work = true;
                        let mut span_ok = true;
                        for (idx, key, is_init) in work {
                            let va = span.start + (idx as u64) * TILE_BYTES;
                            // Arm read-only guard BEFORE copying so any concurrent
                            // CPU write faults and marks the tile dirty for the
                            // next submission (linearizable, zero lost writes).
                            if !plain_guard_protect(va as usize, restore_prot) {
                                span_ok = false;
                                pg_all_ok = false;
                                if allow_reclassify {
                                    self.last_sync.reclass_tiles += 1;
                                    if let Some(t) = self.mapped.get_mut(&key) {
                                        t.region = RegionKind::Unknown;
                                    }
                                    reclassify.push(key);
                                }
                                continue;
                            }
                            if is_init {
                                self.sync_initial_tile(key, va, uploads, total_bytes);
                            } else {
                                self.sync_dirty_tile(
                                    key,
                                    va,
                                    diff64,
                                    diff_runs,
                                    uploads,
                                    total_bytes,
                                );
                            }
                        }
                        if span_ok {
                            span.all_initialized = true;
                        }
                    } else if !span.all_initialized {
                        span.all_initialized = true;
                    }
                }
                RegionKind::Uncommitted | RegionKind::Unknown => {}
            }
        }

        (reclassify, mv_had_work, mv_all_ok, pg_had_work, pg_all_ok)
    }

    /// Scoped pre-submit CPU -> VRAM dirty scanner over all currently mapped
    /// sparse tiles:
    /// 1. Lazily classifies only newly mapped / invalidated tiles via
    ///    `VirtualQuery` (0 `VirtualQuery` calls on steady state).
    /// 2. Caches contiguous `(RegionKind, AllocationBase)` spans across submits
    ///    and uses monotonic `watch::guard_epoch()` / `PLAIN_GUARD_EPOCH`
    ///    tokens so untouched `MappedView` / `PrivateGuarded` tiles take **O(1)**
    ///    time (~1 µs/submit, 0 syscalls, 0 `BTreeMap` walks, 0 allocations).
    /// 3. Runs Stage M2 64-byte cacheline diffing (`diff_cachelines_64`) ONLY
    ///    on genuinely dirty tiles against per-tile baselines.
    pub(crate) fn collect_dirty_uploads(&mut self) -> Vec<UploadChunk> {
        self.last_sync = SyncMetrics::default();
        if self.mapped.is_empty() {
            return Vec::new();
        }
        let started = std::time::Instant::now();

        if self.has_unclassified {
            self.classify_unknown_tiles(None);
            self.has_unclassified = false;
            self.spans_dirty = true;
        }
        if self.spans_dirty {
            self.rebuild_cached_spans();
        }

        // Snapshot epochs BEFORE checking any spans (linearizable: any fault,
        // alias open, `kernel_will_write`, or remap racing with this pass bumps
        // the epoch so we will NOT falsely lock in a stale clean token).
        let mv_ep0 = crate::watch::guard_epoch();
        let pg_ep0 = plain_guard_epoch();
        let mv_skip = self.kind_counts[1] == 0 || self.mv_clean_epoch == Some(mv_ep0);
        let pg_skip = self.pg_writable_spans == 0 || self.pg_clean_epoch == Some(pg_ep0);

        // O(1) steady-state fast exit: zero `PrivateWriteWatch` tiles, zero
        // uninitialized read-only tiles, and both guard epochs unchanged!
        if self.kind_counts[0] == 0 && !self.has_uninit_ro && mv_skip && pg_skip {
            SCAN_STATS[0].store(self.kind_counts[0], Ordering::Relaxed);
            SCAN_STATS[1].store(self.kind_counts[1], Ordering::Relaxed);
            SCAN_STATS[2].store(self.kind_counts[2], Ordering::Relaxed);
            SCAN_STATS[3].store(self.kind_counts[3], Ordering::Relaxed);
            SCAN_STATS[8].fetch_add(1, Ordering::Relaxed);
            STATS[8].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
            return Vec::new();
        }

        let diff64 = diff64_enabled();
        let mut uploads: Vec<UploadChunk> = Vec::new();
        let mut total_bytes = 0u64;
        let mut diff_runs: Vec<(usize, usize)> = Vec::new();

        let mut spans = std::mem::take(&mut self.cached_spans);
        let (reclassify, mv_had_work, mv_all_ok, pg_had_work, pg_all_ok) = self.process_span_slice(
            &mut spans,
            mv_skip,
            pg_skip,
            true,
            diff64,
            &mut diff_runs,
            &mut uploads,
            &mut total_bytes,
        );
        self.cached_spans = spans;

        if !reclassify.is_empty() {
            self.classify_unknown_tiles(Some(&reclassify));
            self.has_unclassified = false;
            let mut reclass_spans = self.build_spans_for_keys(&reclassify);
            let _ = self.process_span_slice(
                &mut reclass_spans,
                false,
                false,
                false,
                diff64,
                &mut diff_runs,
                &mut uploads,
                &mut total_bytes,
            );
            self.rebuild_cached_spans();
        } else {
            self.has_uninit_ro = false;
            if !mv_skip {
                if self.kind_counts[1] > 0
                    && !mv_had_work
                    && mv_all_ok
                    && crate::watch::guard_epoch() == mv_ep0
                {
                    self.mv_clean_epoch = Some(mv_ep0);
                } else {
                    self.mv_clean_epoch = None;
                }
            }
            if !pg_skip {
                if self.pg_writable_spans > 0
                    && !pg_had_work
                    && pg_all_ok
                    && plain_guard_epoch() == pg_ep0
                {
                    self.pg_clean_epoch = Some(pg_ep0);
                } else {
                    self.pg_clean_epoch = None;
                }
            }
        }

        self.last_sync.bytes_uploaded = total_bytes;

        // Publish per-kind residency & cumulative scan telemetry for `status()`.
        SCAN_STATS[0].store(self.kind_counts[0], Ordering::Relaxed);
        SCAN_STATS[1].store(self.kind_counts[1], Ordering::Relaxed);
        SCAN_STATS[2].store(self.kind_counts[2], Ordering::Relaxed);
        SCAN_STATS[3].store(self.kind_counts[3], Ordering::Relaxed);
        SCAN_STATS[4].fetch_add(self.last_sync.fallback_tiles, Ordering::Relaxed);
        SCAN_STATS[5].fetch_add(self.last_sync.virtual_queries, Ordering::Relaxed);
        SCAN_STATS[6].fetch_add(self.last_sync.bytes_compared, Ordering::Relaxed);
        SCAN_STATS[7].fetch_add(total_bytes, Ordering::Relaxed);
        SCAN_STATS[8].fetch_add(1, Ordering::Relaxed);
        SCAN_STATS[9].fetch_add(self.last_sync.write_watch_calls, Ordering::Relaxed);
        SCAN_STATS[10].fetch_add(self.last_sync.mapped_view_calls, Ordering::Relaxed);
        SCAN_STATS[11].fetch_add(self.last_sync.reclass_tiles, Ordering::Relaxed);
        SCAN_STATS[12].fetch_add(self.last_sync.ww_us, Ordering::Relaxed);
        SCAN_STATS[13].fetch_add(self.last_sync.mv_us, Ordering::Relaxed);
        SCAN_STATS[14].fetch_add(self.last_sync.pg_us, Ordering::Relaxed);
        SCAN_STATS[15].fetch_add(self.last_sync.scan_us, Ordering::Relaxed);

        STATS[6].fetch_add(total_bytes, Ordering::Relaxed);
        STATS[7].fetch_add(uploads.len() as u64, Ordering::Relaxed);
        STATS[8].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
        uploads
    }

    #[cfg(test)]
    pub(crate) fn last_sync_metrics(&self) -> SyncMetrics {
        self.last_sync
    }

    #[cfg(test)]
    pub(crate) fn drain_releases(&self, completed_fence: u64) -> usize {
        self.releases.drain(completed_fence)
    }

    #[cfg(test)]
    fn alloc_one_tile_for_test(&mut self, device: &ID3D12Device) -> Option<(usize, ID3D12Heap, u32)> {
        self.pool.alloc_one(device)
    }
}

// ---------------------------------------------------------------------------
// Process-wide singleton manager wired into Host, HostHeap, and memory.rs
// ---------------------------------------------------------------------------

static MANAGER: OnceLock<Option<Mutex<SparseEngine>>> = OnceLock::new();

fn manager() -> Option<&'static Mutex<SparseEngine>> {
    if !active() {
        return None;
    }
    MANAGER
        .get_or_init(|| {
            let host = crate::host::host()?;
            let (base, _) = ensure_aperture()?;
            SparseEngine::new(&host.device, base).map(Mutex::new)
        })
        .as_ref()
}

/// Clones the twelve 2 GiB reserved window resources so `HostHeap` can install
/// raw UAV descriptors at `SPARSE_BASE + 1..=12`.
pub(crate) fn window_resources() -> Option<[ID3D12Resource; WINDOW_COUNT]> {
    let guard = manager()?.lock().unwrap_or_else(|e| e.into_inner());
    Some(guard.windows.clone())
}

/// Pre-submit hook called by `host.rs::Host::submit`:
/// 1. Drains pending `xmem` map/unmap events and issues batched
///    `upload_queue.UpdateTileMappings` (plus LRU eviction past 256 MiB).
/// 2. Scans mapped sparse tiles with scoped `GetWriteWatch` + Stage M2
///    64-byte cacheline diffing.
/// 3. Uploads dirty chunks via Stage M1 `upload_queue` + `UploadRing` and
///    synchronizes the main graphics queue via GPU-only `queue.Wait`.
/// 4. Drains lock-free deferred COM `Release` queue outside all pool locks.
pub(crate) fn sync_before_submit(host: &crate::host::Host, prev_graphics_fence: u64) {
    let Some(mgr) = manager() else {
        return;
    };
    let (to_unmap, to_map) = {
        let mut p = PENDING.lock().unwrap_or_else(|e| e.into_inner());
        (std::mem::take(&mut p.to_unmap), std::mem::take(&mut p.to_map))
    };

    let mut engine = mgr.lock().unwrap_or_else(|e| e.into_inner());
    let has_tile_work = !to_unmap.is_empty()
        || !to_map.is_empty()
        || engine.parked.len() > engine.park_limit_tiles;
    if !has_tile_work && engine.mapped.is_empty() {
        return;
    }

    // Reserve the upcoming upload fence value so any empty 64 MiB pool released
    // during `apply_tile_changes` waits strictly until this copy batch finishes.
    let next_fence = host.next_upload_fence_hint();

    // Perform `UpdateTileMappings` on `upload_queue` (after waiting on
    // `prev_graphics_fence` if needed), then collect dirty uploads (which sees
    // any newly mapped tiles' `needs_initial_sync = true`), and submit copies!
    let windows = engine.windows.clone();
    let submitted = host.async_upload_sparse(
        prev_graphics_fence,
        &windows,
        |device, upload_queue| {
            let tile_ops = engine.apply_tile_changes(
                device,
                upload_queue,
                &to_unmap,
                &to_map,
                next_fence,
                true,
            );
            let chunks = engine.collect_dirty_uploads();
            (tile_ops, chunks)
        },
    );
    let completed = host.upload_completed_value();
    // Drop the engine lock BEFORE draining lock-free COM `Release` nodes!
    let releases_ptr: *const ReleaseQueue = &raw const engine.releases;
    drop(engine);
    // SAFETY: `MANAGER` lives for `'static`, so `engine.releases` is `'static`
    // and its `drain()` is strictly lock-free (`AtomicPtr`).
    let _ = unsafe { (*releases_ptr).drain(completed) };
    let _ = submitted;
}

/// Periodic sweep hook (called from `host.rs::sweep`) draining completed
/// lock-free COM `Release` nodes.
pub(crate) fn sweep() {
    let Some(mgr) = manager() else {
        return;
    };
    let completed = crate::host::host().map_or(0, |h| h.upload_completed_value());
    let engine = mgr.lock().unwrap_or_else(|e| e.into_inner());
    let releases_ptr: *const ReleaseQueue = &raw const engine.releases;
    drop(engine);
    // SAFETY: `MANAGER` lives for `'static`; `ReleaseQueue::drain` is lock-free.
    let _ = unsafe { (*releases_ptr).drain(completed) };
}

/// Periodic status log line for `host.rs::Meter` when `VICTORY_SPARSE_UMA=1`.
pub(crate) fn status() {
    if !active() {
        return;
    }
    let s: [u64; 9] = std::array::from_fn(|i| STATS[i].load(Ordering::Relaxed));
    if s[0] == 0 && s[1] == 0 && s[2] == 0 {
        return;
    }
    let c: [u64; 16] = std::array::from_fn(|i| SCAN_STATS[i].load(Ordering::Relaxed));
    tracing::info!(
        "sparse UMA (24 GiB, 12x2 GiB): {} MiB mapped ({} tiles: ww={} mv={} pg={} unc={}), \
         {} MiB parked ({} tiles), {} MiB DEFAULT pools ({} released), UpdateTileMappings {} map / {} unmap, \
         synced {} MiB in {} diff64 runs ({} us across {} submits; ww_calls={} {} us, mv_calls={} {} us, \
         pg={} us, scan={} us, vq={}, reclass={}, fallback_tiles={}, compared={} MiB, uploaded={} MiB)",
        (s[0] * TILE_BYTES) >> 20,
        s[0],
        c[0],
        c[1],
        c[2],
        c[3],
        (s[1] * TILE_BYTES) >> 20,
        s[1],
        s[2] >> 20,
        s[3],
        s[4],
        s[5],
        s[6] >> 20,
        s[7],
        s[8],
        c[8],
        c[9],
        c[12],
        c[10],
        c[13],
        c[14],
        c[15],
        c[5],
        c[11],
        c[4],
        c[6] >> 20,
        c[7] >> 20,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_uma_env_and_address_math() {
        assert!(!resolve_sparse_uma(None));
        assert!(!resolve_sparse_uma(Some("0")));
        assert!(!resolve_sparse_uma(Some("")));
        assert!(resolve_sparse_uma(Some("1")));
        assert!(resolve_sparse_uma(Some(" 1 ")));

        let base = 0x10_0000_0000u64; // 64 GiB (32 GiB-aligned)
        assert_eq!(base % APERTURE_ALIGN, 0);
        // Below aperture -> None
        assert_eq!(locate(base, base - 4), None);
        // First byte of Window 0
        assert_eq!(locate(base, base), Some((0, 0, 0)));
        // Last 4 bytes of Window 0 (just before 2 GiB boundary)
        assert_eq!(
            locate(base, base + WINDOW_BYTES - 4),
            Some((0, TILES_PER_WINDOW - 1, TILE_BYTES as u32 - 4))
        );
        // Exact 2 GiB boundary -> Window 1, tile 0, offset 0 (zero gap!)
        assert_eq!(locate(base, base + WINDOW_BYTES), Some((1, 0, 0)));
        // Exact 4 GiB boundary -> Window 2, tile 0, offset 0 (zero 1 MiB hole!)
        assert_eq!(locate(base, base + 2 * WINDOW_BYTES), Some((2, 0, 0)));
        // Last byte of Window 11 (24 GiB - 1)
        assert_eq!(
            locate(base, base + APERTURE_BYTES - 1),
            Some((11, TILES_PER_WINDOW - 1, TILE_BYTES as u32 - 1))
        );
        // Past 24 GiB aperture -> None
        assert_eq!(locate(base, base + APERTURE_BYTES), None);
        // Round-trip tile_va
        for w in 0..WINDOW_COUNT {
            for t in [0, 1, 17, TILES_PER_WINDOW - 1] {
                let va = tile_va(base, w, t);
                assert_eq!(locate(base, va), Some((w, t, 0)));
            }
        }
    }

    #[test]
    fn lock_free_release_queue_respects_fence_order() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let b1 = host.uav_buffer(4096).expect("buf 1");
        let b2 = host.uav_buffer(4096).expect("buf 2");
        let q = ReleaseQueue::new();
        q.push(&b1, 10);
        q.push(&b2, 20);
        drop(b1);
        drop(b2);
        assert_eq!(q.drain(5), 0);
        assert_eq!(q.drain(10), 1);
        assert_eq!(q.drain(19), 0);
        assert_eq!(q.drain(20), 1);
        assert_eq!(q.drain(u64::MAX), 0);
    }

    /// Probes a 32 GiB-aligned 24 GiB aperture strictly for unit tests (starting
    /// high in the 40-bit space at 640 GiB..992 GiB so it never collides with
    /// the production `ensure_aperture()` at 64 GiB even under `VICTORY_SPARSE_UMA=1`).
    fn probe_test_aperture() -> Option<u64> {
        use windows_sys::Win32::System::Memory::{
            MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, VirtualAlloc2, VirtualFree,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;

        for step in (20u64..=31).rev() {
            let base = step * APERTURE_ALIGN;
            if base.checked_add(APERTURE_BYTES).is_none_or(|end| end > (1u64 << 40)) {
                continue;
            }
            // SAFETY: probe-reserving and immediately releasing a free VA range.
            let ptr = unsafe {
                VirtualAlloc2(
                    GetCurrentProcess(),
                    base as *const std::ffi::c_void,
                    APERTURE_BYTES as usize,
                    MEM_RESERVE,
                    PAGE_NOACCESS,
                    std::ptr::null_mut(),
                    0,
                )
            };
            if ptr as u64 == base {
                // SAFETY: releases the exact probe reservation above.
                unsafe { VirtualFree(ptr, 0, MEM_RELEASE) };
                return Some(base);
            }
            if !ptr.is_null() {
                // SAFETY: unexpected base; release cleanly.
                unsafe { VirtualFree(ptr, 0, MEM_RELEASE) };
            }
        }
        probe_aperture()
    }

    /// Core live RTX 3090 Stage M3 verification harness:
    /// (a) 64 KiB tile mapping onto 64 MiB DEFAULT (`ALLOW_ONLY_BUFFERS`) pools
    ///     across two 2 GiB reserved windows (Window 0 last tile & Window 1
    ///     first tile, straddling the 2 GiB boundary with zero gap);
    /// (b) CPU write to `MEM_WRITE_WATCH` memory -> scoped `GetWriteWatch` +
    ///     Stage M2 64-byte diff -> dedicated COPY queue upload + GPU-only
    ///     `direct_queue.Wait(&copy_fence, val)` -> GPU shader raw 40-bit read
    ///     (exact Track 4A `hi_rel`/`win`/`byte` math) + `globallycoherent`
    ///     atomic write across the 2 GiB boundary -> readback verification;
    /// (c) Clean tile parking, eviction (`UpdateTileMappings(NULL)`), empty
    ///     64 MiB pool release via lock-free `ReleaseQueue`, and 0 D3D12 Debug
    ///     Layer errors (when `info_queue` is active).
    fn run_live_sparse_uma_test(device: &ID3D12Device, info_queue: Option<ID3D12InfoQueue>) {
        use windows::Win32::Foundation::{HANDLE, WAIT_OBJECT_0};
        use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_TYPELESS;
        use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
        use windows::core::s;
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc2, VirtualFree,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;

        const MEM_WRITE_WATCH: u32 = 0x0020_0000;

        // HLSL compute shader implementing the exact Track 4A 12 x 2 GiB raw
        // 40-bit addressing formula over `globallycoherent` raw UAVs (`win[0]`
        // = null raw UAV, `win[1..=12]` = 12 x 2 GiB reserved windows).
        const SHADER: &str = r#"
#define RS "RootConstants(num32BitConstants=8, b0), DescriptorTable(UAV(u0, numDescriptors=13))"
cbuffer Params : register(b0) {
    uint aperture_hi;
    uint src_lo;
    uint src_hi;
    uint dst_lo;
    uint dst_hi;
    uint oob_lo;
    uint oob_hi;
    uint delta;
};
globallycoherent RWByteAddressBuffer win[13] : register(u0);

uint addr_slot(uint lo, uint hi) {
    uint hi_rel = hi - aperture_hi;
    uint w = (hi_rel << 1) | (lo >> 31);
    return (hi_rel < 6u) ? (1u + w) : 0u;
}
uint addr_byte(uint lo, uint hi) {
    uint hi_rel = hi - aperture_hi;
    return (hi_rel < 6u) ? (lo & 0x7FFFFFFFu) : 0u;
}

[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    uint s_slot = addr_slot(src_lo, src_hi);
    uint s_byte = addr_byte(src_lo, src_hi);
    uint d_slot = addr_slot(dst_lo, dst_hi);
    uint d_byte = addr_byte(dst_lo, dst_hi);
    uint o_slot = addr_slot(oob_lo, oob_hi);
    uint o_byte = addr_byte(oob_lo, oob_hi);

    // Read across Window 0 (last tile) and out-of-range null UAV (must read 0).
    uint v0 = win[NonUniformResourceIndex(s_slot)].Load(s_byte);
    uint v1 = win[NonUniformResourceIndex(s_slot)].Load(s_byte + 4u);
    uint oob_val = win[NonUniformResourceIndex(o_slot)].Load(o_byte);
    // Write to out-of-range null UAV (must safely drop).
    win[NonUniformResourceIndex(o_slot)].Store(o_byte, 0xDEADBEEFu);

    // Globallycoherent atomic add into Window 1 (first tile, across 2 GiB boundary).
    uint prev = 0u;
    win[NonUniformResourceIndex(d_slot)].InterlockedAdd(d_byte, v0 + delta, prev);
    win[NonUniformResourceIndex(d_slot)].Store(d_byte + 4u, prev);
    win[NonUniformResourceIndex(d_slot)].Store(d_byte + 8u, v1 ^ oob_val);
}
"#;

        // Helper asserting 0 D3D12 Debug Layer errors/corruptions and device alive.
        let check_debug_layer = |info: &Option<ID3D12InfoQueue>, step: &str| {
            // SAFETY: plain device status query.
            assert!(
                unsafe { device.GetDeviceRemovedReason() }.is_ok(),
                "D3D12 device removed at {step}"
            );
            let Some(q) = info.as_ref() else { return };
            // SAFETY: reading stored messages from our own debug info queue.
            unsafe {
                let n = q.GetNumStoredMessages();
                let mut errs = Vec::new();
                for i in 0..n {
                    let mut size = 0usize;
                    if q.GetMessage(i, None, &mut size).is_err() || size == 0 {
                        continue;
                    }
                    let mut buf = vec![0u64; size.div_ceil(8)];
                    let msg = buf.as_mut_ptr().cast::<D3D12_MESSAGE>();
                    if q.GetMessage(i, Some(msg), &mut size).is_ok() {
                        let m = &*msg;
                        let text = std::slice::from_raw_parts(m.pDescription, m.DescriptionByteLength);
                        let text = String::from_utf8_lossy(text).trim_end_matches('\0').to_owned();
                        if m.Severity.0 <= D3D12_MESSAGE_SEVERITY_ERROR.0 {
                            errs.push(format!("[id={}] {text}", m.ID.0));
                        }
                    }
                }
                q.ClearStoredMessages();
                assert!(errs.is_empty(), "D3D12 Debug Layer errors at {step}: {errs:#?}");
            }
        };

        // 1. Reserve a 2-tile (128 KiB) MEM_WRITE_WATCH VA block straddling a
        //    real 2 GiB boundary inside a 32 GiB-aligned aperture!
        let Some(aperture_base) = probe_test_aperture() else {
            panic!("could not find a free 32 GiB-aligned 24 GiB aperture for test");
        };
        let boundary_va = aperture_base + WINDOW_BYTES; // exact 2 GiB boundary (Window 0 | Window 1)
        let straddle_base = boundary_va - TILE_BYTES;   // Window 0 last tile + Window 1 first tile
        let va_ptr = unsafe {
            VirtualAlloc2(
                GetCurrentProcess(),
                straddle_base as *const std::ffi::c_void,
                (2 * TILE_BYTES) as usize,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
                std::ptr::null_mut(),
                0,
            )
        };
        assert_eq!(
            va_ptr as u64, straddle_base,
            "failed to allocate 128 KiB MEM_WRITE_WATCH straddling 2 GiB boundary"
        );
        struct VaGuard(*mut std::ffi::c_void);
        impl Drop for VaGuard {
            fn drop(&mut self) {
                // SAFETY: releases our test allocation.
                unsafe { VirtualFree(self.0, 0, MEM_RELEASE) };
            }
        }
        let _va_guard = VaGuard(va_ptr);

        // 2. Create DIRECT + COPY queues, fences, events, upload buffer, readback buffer, and SparseEngine.
        let mut engine = SparseEngine::new(device, aperture_base).expect("SparseEngine::new");
        check_debug_layer(&info_queue, "SparseEngine::new");

        let (direct_queue, copy_queue, direct_fence, copy_fence, event): (
            ID3D12CommandQueue,
            ID3D12CommandQueue,
            ID3D12Fence,
            ID3D12Fence,
            HANDLE,
        ) = unsafe {
            (
                device
                    .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                        Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                        ..Default::default()
                    })
                    .unwrap(),
                device
                    .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                        Type: D3D12_COMMAND_LIST_TYPE_COPY,
                        ..Default::default()
                    })
                    .unwrap(),
                device.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap(),
                device.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap(),
                CreateEventW(None, false, false, None).unwrap(),
            )
        };

        let wait_fence = |fence: &ID3D12Fence, val: u64| unsafe {
            if fence.GetCompletedValue() < val {
                fence.SetEventOnCompletion(val, event).unwrap();
                assert_eq!(WaitForSingleObject(event, 5000), WAIT_OBJECT_0);
            }
        };

        // 3. Write CPU bytes across the 2 GiB boundary:
        //    - Window 0 last tile @ offset 0x0080: [0x11223344, 0xA5A55A5A]
        //    - Window 1 first tile @ offset 0x0040: initial atomic value 0x00001000
        let src_va = straddle_base + 0x80; // Window 0, tile 32767, +0x80
        let dst_va = boundary_va + 0x40;   // Window 1, tile 0, +0x40
        assert_eq!(locate(aperture_base, src_va), Some((0, TILES_PER_WINDOW - 1, 0x80)));
        assert_eq!(locate(aperture_base, dst_va), Some((1, 0, 0x40)));
        unsafe {
            (src_va as *mut u32).write_volatile(0x1122_3344);
            ((src_va + 4) as *mut u32).write_volatile(0xA5A5_5A5A);
            (dst_va as *mut u32).write_volatile(0x0000_1000);
        }

        // 4. (a) Map both 64 KiB tiles across Window 0 & Window 1 onto 64 MiB DEFAULT pool via COPY queue.
        let mut to_map = BTreeSet::new();
        to_map.insert((0u8, TILES_PER_WINDOW - 1));
        to_map.insert((1u8, 0u32));
        let empty = BTreeSet::new();
        assert!(engine.apply_tile_changes(device, &copy_queue, &empty, &to_map, 1, false));
        assert_eq!(engine.mapped.len(), 2);
        assert_eq!(engine.pool.pool_bytes, u64::from(POOL_TILES) * TILE_BYTES);
        check_debug_layer(&info_queue, "apply_tile_changes (map Window 0 & Window 1)");

        // 5. Scoped sync -> initial upload on COPY queue -> GPU-only `direct_queue.Wait(&copy_fence, 1)`.
        let chunks = engine.collect_dirty_uploads();
        assert_eq!(chunks.len(), 2, "initial sync must upload both newly mapped tiles");
        assert!(chunks.iter().all(|c| c.data.len() == TILE_BYTES as usize));

        // Verify zero dirty chunks when CPU hasn't written since initial sync!
        assert!(
            engine.collect_dirty_uploads().is_empty(),
            "unchanged tiles must produce 0 upload chunks"
        );

        // Verify incremental Stage M2 64-byte diffing! Mutate 4 bytes in Window 0 tile.
        unsafe { (src_va as *mut u32).write_volatile(0x2233_4455) };
        let inc_chunks = engine.collect_dirty_uploads();
        assert_eq!(inc_chunks.len(), 1, "only the 1 modified 64-byte cacheline should diff");
        assert_eq!(inc_chunks[0].window, 0);
        assert_eq!(inc_chunks[0].data.len(), 64, "diff64 must isolate exactly 64 bytes");
        assert_eq!(
            inc_chunks[0].offset,
            u64::from(TILES_PER_WINDOW - 1) * TILE_BYTES + 0x80
        );

        // Upload both initial tiles + incremental 64-byte diff on COPY queue via an upload buffer.
        let mut all_chunks = chunks;
        all_chunks.extend(inc_chunks);
        let stage_bytes: u64 = all_chunks.iter().map(|c| c.data.len() as u64).sum();
        let (stage_buf, stage_ptr): (ID3D12Resource, *mut u8) = unsafe {
            let mut b: Option<ID3D12Resource> = None;
            device
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES {
                        Type: D3D12_HEAP_TYPE_UPLOAD,
                        ..Default::default()
                    },
                    D3D12_HEAP_FLAG_NONE,
                    &D3D12_RESOURCE_DESC {
                        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                        Width: stage_bytes.max(256),
                        Height: 1,
                        DepthOrArraySize: 1,
                        MipLevels: 1,
                        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                        ..Default::default()
                    },
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                    None,
                    &mut b,
                )
                .unwrap();
            let b = b.unwrap();
            let mut p = std::ptr::null_mut();
            let read_none = D3D12_RANGE { Begin: 0, End: 0 };
            b.Map(0, Some(&read_none), Some(&mut p)).unwrap();
            (b, p.cast())
        };
        let (copy_alloc, copy_list): (ID3D12CommandAllocator, ID3D12GraphicsCommandList) = unsafe {
            let a: ID3D12CommandAllocator = device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY)
                .unwrap();
            let l: ID3D12GraphicsCommandList = device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, &a, None)
                .unwrap();
            let mut off = 0u64;
            for c in &all_chunks {
                std::ptr::copy_nonoverlapping(c.data.as_ptr(), stage_ptr.add(off as usize), c.data.len());
                l.CopyBufferRegion(
                    &engine.windows[c.window],
                    c.offset,
                    &stage_buf,
                    off,
                    c.data.len() as u64,
                );
                off += c.data.len() as u64;
            }
            l.Close().unwrap();
            let cmd: ID3D12CommandList = l.cast().unwrap();
            copy_queue.ExecuteCommandLists(&[Some(cmd)]);
            copy_queue.Signal(&copy_fence, 1).unwrap();
            // GPU-only wait on DIRECT queue (zero CPU stall on copy_fence!).
            direct_queue.Wait(&copy_fence, 1).unwrap();
            (a, l)
        };
        check_debug_layer(&info_queue, "COPY queue upload + GPU-only direct_queue.Wait");

        // 6. (b) Compile shader, build 13-slot descriptor heap (slot 0 = null raw UAV,
        //    slots 1..=12 = 12 x 2 GiB raw UAVs), dispatch on DIRECT queue, and read back!
        let bytecode = crate::host::compile(SHADER, s!("main"), s!("cs_5_1")).expect("compile cs_5_1");
        let (root_sig, pso, desc_heap, readback, dir_alloc, dir_list) = unsafe {
            let root: ID3D12RootSignature = device.CreateRootSignature(0, &bytecode).unwrap();
            let pso: ID3D12PipelineState = device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: bytecode.as_ptr().cast(),
                        BytecodeLength: bytecode.len(),
                    },
                    ..Default::default()
                })
                .unwrap();
            let heap: ID3D12DescriptorHeap = device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: 13,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    NodeMask: 0,
                })
                .unwrap();
            let step = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            let cpu0 = heap.GetCPUDescriptorHandleForHeapStart();
            // Slot 0: null raw UAV
            let null_desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: DXGI_FORMAT_R32_TYPELESS,
                ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_UAV {
                        NumElements: 1,
                        Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                        ..Default::default()
                    },
                },
            };
            device.CreateUnorderedAccessView(
                None::<&ID3D12Resource>,
                None::<&ID3D12Resource>,
                Some(&null_desc),
                cpu0,
            );
            // Slots 1..=12: 12 x 2 GiB raw UAVs (`0x2000_0000` dwords each)
            let win_desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: DXGI_FORMAT_R32_TYPELESS,
                ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_UAV {
                        FirstElement: 0,
                        NumElements: (WINDOW_BYTES / 4) as u32,
                        StructureByteStride: 0,
                        CounterOffsetInBytes: 0,
                        Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                    },
                },
            };
            for w in 0..WINDOW_COUNT {
                let h = D3D12_CPU_DESCRIPTOR_HANDLE {
                    ptr: cpu0.ptr + ((w + 1) * step as usize),
                };
                device.CreateUnorderedAccessView(
                    &engine.windows[w],
                    None::<&ID3D12Resource>,
                    Some(&win_desc),
                    h,
                );
            }

            let mut rb: Option<ID3D12Resource> = None;
            device
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES {
                        Type: D3D12_HEAP_TYPE_READBACK,
                        ..Default::default()
                    },
                    D3D12_HEAP_FLAG_NONE,
                    &D3D12_RESOURCE_DESC {
                        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                        Width: 256,
                        Height: 1,
                        DepthOrArraySize: 1,
                        MipLevels: 1,
                        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                        ..Default::default()
                    },
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut rb,
                )
                .unwrap();
            let rb = rb.unwrap();

            let alloc: ID3D12CommandAllocator = device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                .unwrap();
            let list: ID3D12GraphicsCommandList = device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, Some(&pso))
                .unwrap();
            list.SetDescriptorHeaps(&[Some(heap.clone())]);
            list.SetComputeRootSignature(&root);
            let oob_va = aperture_base + APERTURE_BYTES + 0x100; // outside 24 GiB -> slot 0 null UAV
            let constants: [u32; 8] = [
                (aperture_base >> 32) as u32,
                src_va as u32,
                (src_va >> 32) as u32,
                dst_va as u32,
                (dst_va >> 32) as u32,
                oob_va as u32,
                (oob_va >> 32) as u32,
                0x0000_0007,
            ];
            list.SetComputeRoot32BitConstants(0, 8, constants.as_ptr().cast(), 0);
            list.SetComputeRootDescriptorTable(1, heap.GetGPUDescriptorHandleForHeapStart());
            list.Dispatch(1, 1, 1);

            // Global UAV barrier (`pResource = NULL`) + transition Window 1 to COPY_SOURCE for readback.
            list.ResourceBarrier(&[
                D3D12_RESOURCE_BARRIER {
                    Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                    Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                    Anonymous: D3D12_RESOURCE_BARRIER_0 {
                        UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                            pResource: std::mem::ManuallyDrop::new(None),
                        }),
                    },
                },
                crate::host::transition(
                    &engine.windows[1],
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                ),
            ]);
            list.CopyBufferRegion(&rb, 0, &engine.windows[1], 0x40, 16);
            list.Close().unwrap();
            let cmd: ID3D12CommandList = list.cast().unwrap();
            direct_queue.ExecuteCommandLists(&[Some(cmd)]);
            direct_queue.Signal(&direct_fence, 1).unwrap();
            wait_fence(&direct_fence, 1);

            let mut mapped = std::ptr::null_mut();
            let range = D3D12_RANGE { Begin: 0, End: 16 };
            rb.Map(0, Some(&range), Some(&mut mapped)).unwrap();
            let words = std::slice::from_raw_parts(mapped.cast::<u32>(), 4);
            // Expected:
            //   words[0] = 0x0000_1000 + (0x2233_4455 + 7) = 0x2233_545C (atomic sum!)
            //   words[1] = 0x0000_1000 (previous atomic value!)
            //   words[2] = 0xA5A5_5A5A ^ 0 = 0xA5A5_5A5A (OOB null UAV read 0!)
            assert_eq!(words[0], 0x2233_545C, "atomic add across 2 GiB boundary mismatch");
            assert_eq!(words[1], 0x0000_1000, "atomic previous value mismatch");
            assert_eq!(words[2], 0xA5A5_5A5A, "second dword / OOB null-UAV read mismatch");
            rb.Unmap(0, None);

            (root, pso, heap, rb, alloc, list)
        };
        check_debug_layer(&info_queue, "DIRECT queue dispatch + readback");

        // 7. (c) Park both tiles -> verify parked count -> evict & unmap (`UpdateTileMappings(NULL)`)
        //    -> release empty 64 MiB pool via lock-free ReleaseQueue -> verify 0 Debug Layer errors!
        assert!(
            !engine.apply_tile_changes(device, &copy_queue, &to_map, &empty, 2, false),
            "parking 2 tiles under 256 MiB limit must not issue UpdateTileMappings"
        );
        assert_eq!(engine.mapped.len(), 0);
        assert_eq!(engine.parked.len(), 2);

        // Order COPY queue after DIRECT queue before unmapping tiles!
        unsafe { copy_queue.Wait(&direct_fence, 1).unwrap() };
        assert!(engine.evict_parked(&copy_queue, 2, 2, false));
        assert_eq!(engine.parked.len(), 0);
        assert_eq!(engine.pool.pool_bytes, 0, "empty 64 MiB pool must be released");
        assert_eq!(engine.pool.pools_released, 1);

        // Signal COPY fence 2 and wait before draining lock-free COM Release queue.
        unsafe { copy_queue.Signal(&copy_fence, 2).unwrap() };
        wait_fence(&copy_fence, 2);
        assert_eq!(
            engine.drain_releases(2),
            1,
            "lock-free ReleaseQueue must release the empty 64 MiB ID3D12Heap"
        );
        check_debug_layer(&info_queue, "tile unmap + empty pool release");

        // Quick sanity check on pool reuse helper (keeps `alloc_one` covered too).
        let _extra = engine.alloc_one_tile_for_test(device);
        drop((root_sig, pso, desc_heap, readback, dir_alloc, dir_list, copy_alloc, copy_list, stage_buf));
        drop(engine);
        check_debug_layer(&info_queue, "final teardown");
    }

    /// Runs live on the shared host GPU inside normal parallel `cargo test`
    /// without calling `EnableDebugLayer` (which removes already-created
    /// devices in the same process).
    #[test]
    fn live_rtx3090_sparse_uma_2gib_boundary_round_trip() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU
        };
        let info = host.device.cast::<ID3D12InfoQueue>().ok();
        run_live_sparse_uma_test(&host.device, info);
    }

    /// Standalone live RTX 3090 test with `ID3D12Debug::EnableDebugLayer()` ON,
    /// asserting 0 D3D12 Debug Layer errors across mapping, 2 GiB boundary
    /// globallycoherent atomic dispatch, readback, parking, unmapping, and
    /// lock-free empty pool release. Run alone (just like `tiles.rs`):
    /// `cargo test -p victory-gpu --lib -- --ignored live_rtx3090_sparse_uma_debug_layer --nocapture`
    #[test]
    #[ignore = "enables D3D12 Debug Layer (must run alone, not alongside shared host::host() tests)"]
    fn live_rtx3090_sparse_uma_debug_layer() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
        use windows::Win32::Graphics::Dxgi::{
            CreateDXGIFactory2, DXGI_ADAPTER_FLAG_SOFTWARE, DXGI_CREATE_FACTORY_FLAGS,
            DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IDXGIAdapter1, IDXGIFactory6,
        };

        // SAFETY: standalone device creation after enabling D3D12 Debug Layer.
        let (device, info_queue) = unsafe {
            let mut layer: Option<ID3D12Debug> = None;
            if D3D12GetDebugInterface(&mut layer).is_ok() {
                if let Some(l) = &layer {
                    l.EnableDebugLayer();
                }
            }
            let mut made: Option<ID3D12Device> = None;
            if let Ok(factory) = CreateDXGIFactory2::<IDXGIFactory6>(DXGI_CREATE_FACTORY_FLAGS(0)) {
                let mut idx = 0u32;
                while let Ok(adapter) = factory.EnumAdapterByGpuPreference::<IDXGIAdapter1>(
                    idx,
                    DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                ) {
                    idx += 1;
                    if let Ok(desc) = adapter.GetDesc1()
                        && (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32) == 0
                        && D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut made).is_ok()
                        && made.is_some()
                    {
                        break;
                    }
                }
            }
            if made.is_none() {
                let _ = D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut made);
            }
            let Some(dev) = made else {
                eprintln!("skipping live_rtx3090_sparse_uma_debug_layer: no D3D12 device");
                return;
            };
            let q = dev.cast::<ID3D12InfoQueue>().ok();
            assert!(
                q.is_some(),
                "D3D12 Debug Layer (Graphics Tools) must be active for live_rtx3090_sparse_uma_debug_layer"
            );
            (dev, q)
        };
        run_live_sparse_uma_test(&device, info_queue);
    }

    /// Creates a `SparseEngine` and a standalone COPY queue on the shared host
    /// device using a free 32 GiB-aligned test aperture.
    fn test_engine_and_copy_queue() -> Option<(ID3D12Device, ID3D12CommandQueue, SparseEngine, u64)> {
        let host = crate::host::host()?;
        let aperture_base = probe_test_aperture()?;
        let engine = SparseEngine::new(&host.device, aperture_base)?;
        // SAFETY: plain COPY command queue creation on the live host device.
        let copy_queue: ID3D12CommandQueue = unsafe {
            host.device
                .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_COPY,
                    ..Default::default()
                })
                .ok()?
        };
        Some((host.device.clone(), copy_queue, engine, aperture_base))
    }

    /// Requirement 4(a): two adjacent `VirtualAlloc2(MEM_WRITE_WATCH)`
    /// allocations (distinct `AllocationBase`s) must be split per
    /// `AllocationBase`, never trigger full-span fallback, perform 0
    /// `VirtualQuery` calls and 0 tile scans on untouched repeat submits,
    /// detect genuine CPU writes (isolating 64 B cachelines), and stay
    /// coherent even if `watch::writes` (`Shadow`) drains kernel dirty bits first.
    #[test]
    fn adjacent_write_watch_allocations_split_per_allocation_base() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_PRESERVE_PLACEHOLDER, MEM_RELEASE, MEM_REPLACE_PLACEHOLDER,
            MEM_RESERVE, MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS, PAGE_READWRITE, VirtualAlloc2,
            VirtualFree,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;

        const MEM_WRITE_WATCH: u32 = 0x0020_0000;

        let Some((device, copy_queue, mut engine, aperture_base)) = test_engine_and_copy_queue()
        else {
            return;
        };

        // Window 2, tiles 0..2: unique VA range that cannot collide with other parallel tests.
        let base_va = tile_va(aperture_base, 2, 0);
        let tile0_va = base_va;
        let tile1_va = base_va + TILE_BYTES;
        let proc = unsafe { GetCurrentProcess() };

        // Reserve a 2-tile placeholder, split into two 64 KiB placeholders, and
        // replace each with its own independent `MEM_WRITE_WATCH` allocation
        // (two distinct `AllocationBase`s sitting back-to-back!).
        let ph = unsafe {
            VirtualAlloc2(
                proc,
                base_va as *const std::ffi::c_void,
                (2 * TILE_BYTES) as usize,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            )
        };
        assert_eq!(ph as u64, base_va, "placeholder reservation failed");
        assert_ne!(
            unsafe {
                VirtualFree(
                    tile0_va as *mut std::ffi::c_void,
                    TILE_BYTES as usize,
                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
                )
            },
            0,
            "placeholder split failed"
        );
        let a0 = unsafe {
            VirtualAlloc2(
                proc,
                tile0_va as *const std::ffi::c_void,
                TILE_BYTES as usize,
                MEM_RESERVE | MEM_COMMIT | MEM_REPLACE_PLACEHOLDER | MEM_WRITE_WATCH,
                PAGE_READWRITE,
                std::ptr::null_mut(),
                0,
            )
        };
        let a1 = unsafe {
            VirtualAlloc2(
                proc,
                tile1_va as *const std::ffi::c_void,
                TILE_BYTES as usize,
                MEM_RESERVE | MEM_COMMIT | MEM_REPLACE_PLACEHOLDER | MEM_WRITE_WATCH,
                PAGE_READWRITE,
                std::ptr::null_mut(),
                0,
            )
        };
        assert_eq!(a0 as u64, tile0_va);
        assert_eq!(a1 as u64, tile1_va);

        struct TwoAllocsGuard(usize, usize);
        impl Drop for TwoAllocsGuard {
            fn drop(&mut self) {
                crate::watch::title_memory_remapped(self.0, (2 * TILE_BYTES) as usize);
                unsafe {
                    VirtualFree(self.0 as *mut std::ffi::c_void, 0, MEM_RELEASE);
                    VirtualFree(self.1 as *mut std::ffi::c_void, 0, MEM_RELEASE);
                }
            }
        }
        let _guard = TwoAllocsGuard(tile0_va as usize, tile1_va as usize);

        unsafe {
            (tile0_va as *mut u32).write_volatile(0x1111_2222);
            (tile1_va as *mut u32).write_volatile(0x3333_4444);
        }

        let mut to_map = BTreeSet::new();
        to_map.insert((2u8, 0u32));
        to_map.insert((2u8, 1u32));
        let empty = BTreeSet::new();
        assert!(engine.apply_tile_changes(&device, &copy_queue, &empty, &to_map, 1, false));

        // 1. Initial sync classifies both tiles into two separate AllocationBases
        //    and uploads both initial 64 KiB tiles with 0 fallbacks.
        let initial = engine.collect_dirty_uploads();
        let m1 = engine.last_sync_metrics();
        assert_eq!(initial.len(), 2);
        assert_eq!(m1.write_watch_calls, 2, "distinct AllocationBases must NOT be merged");
        assert_eq!(m1.full_fallbacks, 0);
        assert!(m1.virtual_queries >= 2);

        // 2. Untouched repeat submits: 0 chunks, 0 tiles scanned, 0 VirtualQueries, 0 fallbacks!
        for _ in 0..3 {
            let chunks = engine.collect_dirty_uploads();
            let m = engine.last_sync_metrics();
            assert!(chunks.is_empty(), "untouched adjacent MEM_WRITE_WATCH tiles must yield 0 uploads");
            assert_eq!(m.virtual_queries, 0, "region classification must stay cached");
            assert_eq!(m.tiles_scanned, 0, "untouched tiles must not be copied/diffed");
            assert_eq!(m.full_fallbacks, 0, "must never fall back to full-span dirty scan");
            assert_eq!(m.write_watch_calls, 2);
        }

        // 3. Mutate 4 bytes in allocation #2 -> ONLY tile 1 scanned & 64-byte cacheline uploaded!
        unsafe { ((tile1_va + 0x140) as *mut u32).write_volatile(0xCAFE_BABE) };
        let diff = engine.collect_dirty_uploads();
        let m3 = engine.last_sync_metrics();
        assert_eq!(diff.len(), 1);
        assert_eq!(diff[0].window, 2);
        assert_eq!(diff[0].offset, TILE_BYTES + 0x140);
        assert_eq!(diff[0].data.len(), 64);
        assert_eq!(u32::from_le_bytes(diff[0].data[0..4].try_into().unwrap()), 0xCAFE_BABE);
        assert_eq!(m3.tiles_scanned, 1);
        assert_eq!(m3.virtual_queries, 0);
        assert_eq!(m3.full_fallbacks, 0);

        // 4. Coherency with `watch::writes` (`Shadow` overlap): mutate allocation #1,
        //    let `watch::writes` consume the kernel dirty bit first, and confirm
        //    `sparse_uma` STILL detects and uploads the 64-byte diff via the journal!
        unsafe { ((tile0_va + 0x80) as *mut u32).write_volatile(0xDEAD_C0DE) };
        let shadow_seen = crate::watch::writes(
            tile0_va as usize,
            (tile0_va + TILE_BYTES) as usize,
            false,
        );
        assert!(shadow_seen.is_some());
        let diff_after_shadow = engine.collect_dirty_uploads();
        let m4 = engine.last_sync_metrics();
        assert_eq!(diff_after_shadow.len(), 1, "must not miss writes already drained into watch::JOURNAL");
        assert_eq!(diff_after_shadow[0].offset, 0x80);
        assert_eq!(diff_after_shadow[0].data.len(), 64);
        assert_eq!(
            u32::from_le_bytes(diff_after_shadow[0].data[0..4].try_into().unwrap()),
            0xDEAD_C0DE
        );
        assert_eq!(m4.tiles_scanned, 1);
        assert_eq!(m4.full_fallbacks, 0);
    }

    /// Requirement 4(b): untouched non-`MEM_WRITE_WATCH` `MEM_PRIVATE` regions
    /// and `MEM_MAPPED` file-mapping views must NOT burn full-span CPU scans on
    /// repeat submits (`tiles_scanned == 0`, `full_fallbacks == 0`), while still
    /// detecting genuine CPU writes via guard faults and uploading only the
    /// modified 64-byte cacheline.
    #[test]
    fn untouched_plain_private_and_mapped_views_avoid_full_scans_and_detect_writes() {
        use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_COMMIT, MEM_PRESERVE_PLACEHOLDER, MEM_RELEASE,
            MEM_REPLACE_PLACEHOLDER, MEM_RESERVE, MEM_RESERVE_PLACEHOLDER,
            MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile3, PAGE_NOACCESS, PAGE_READWRITE, SEC_COMMIT,
            UnmapViewOfFile2, VirtualAlloc2, VirtualFree,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;

        let Some((device, copy_queue, mut engine, aperture_base)) = test_engine_and_copy_queue()
        else {
            return;
        };

        // Window 3, tiles 0..4 (256 KiB):
        //   tiles 0..2 = 128 KiB plain `MEM_PRIVATE` (NO `MEM_WRITE_WATCH`)
        //   tiles 2..4 = 128 KiB `MEM_MAPPED` section view
        let base_va = tile_va(aperture_base, 3, 0);
        let plain_va = base_va;
        let mapped_va = base_va + 2 * TILE_BYTES;
        let span_len = (2 * TILE_BYTES) as usize;
        let proc = unsafe { GetCurrentProcess() };

        let ph = unsafe {
            VirtualAlloc2(
                proc,
                base_va as *const std::ffi::c_void,
                (4 * TILE_BYTES) as usize,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            )
        };
        assert_eq!(ph as u64, base_va);
        assert_ne!(
            unsafe {
                VirtualFree(
                    plain_va as *mut std::ffi::c_void,
                    span_len,
                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
                )
            },
            0
        );

        // Plain `MEM_PRIVATE` WITHOUT `MEM_WRITE_WATCH`.
        let priv_ptr = unsafe {
            VirtualAlloc2(
                proc,
                plain_va as *const std::ffi::c_void,
                span_len,
                MEM_RESERVE | MEM_COMMIT | MEM_REPLACE_PLACEHOLDER,
                PAGE_READWRITE,
                std::ptr::null_mut(),
                0,
            )
        };
        assert_eq!(priv_ptr as u64, plain_va);

        // `MEM_MAPPED` section view.
        let section = unsafe {
            CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE | SEC_COMMIT,
                0,
                span_len as u32,
                std::ptr::null(),
            )
        };
        assert!(!section.is_null());
        let view = unsafe {
            MapViewOfFile3(
                section,
                proc,
                mapped_va as *const std::ffi::c_void,
                0,
                span_len,
                MEM_REPLACE_PLACEHOLDER,
                PAGE_READWRITE,
                std::ptr::null_mut(),
                0,
            )
        };
        assert_eq!(view.Value as u64, mapped_va);

        struct MixedGuard {
            plain_va: usize,
            mapped_va: usize,
            span_len: usize,
            section: HANDLE,
        }
        impl Drop for MixedGuard {
            fn drop(&mut self) {
                plain_guards_release_range(self.plain_va, self.plain_va + self.span_len);
                // Notify `watch` BEFORE unmapping `MEM_MAPPED` view so no stale
                // `GUARDS` block lingers across parallel tests.
                crate::watch::title_memory_remapped(self.mapped_va, self.span_len);
                unsafe {
                    UnmapViewOfFile2(
                        GetCurrentProcess(),
                        MEMORY_MAPPED_VIEW_ADDRESS {
                            Value: self.mapped_va as *mut std::ffi::c_void,
                        },
                        0,
                    );
                    VirtualFree(self.plain_va as *mut std::ffi::c_void, 0, MEM_RELEASE);
                    CloseHandle(self.section);
                }
            }
        }
        let _guard = MixedGuard {
            plain_va: plain_va as usize,
            mapped_va: mapped_va as usize,
            span_len,
            section,
        };

        unsafe {
            (plain_va as *mut u32).write_volatile(0x1010_2020);
            ((plain_va + TILE_BYTES) as *mut u32).write_volatile(0x3030_4040);
            (mapped_va as *mut u32).write_volatile(0x5050_6060);
            ((mapped_va + TILE_BYTES) as *mut u32).write_volatile(0x7070_8080);
        }

        let mut to_map = BTreeSet::new();
        for t in 0u32..4 {
            to_map.insert((3u8, t));
        }
        let empty = BTreeSet::new();
        assert!(engine.apply_tile_changes(&device, &copy_queue, &empty, &to_map, 1, false));

        // 1. Initial sync uploads all 4 tiles and arms both PLAIN_GUARDS and VIEW_GUARD.
        let init = engine.collect_dirty_uploads();
        let m1 = engine.last_sync_metrics();
        assert_eq!(init.len(), 4);
        assert_eq!(m1.full_fallbacks, 0);
        assert_eq!(m1.mapped_view_calls, 1, "contiguous MEM_MAPPED tiles must coalesce");

        // 2. Untouched repeat submits must scan ZERO tiles and issue ZERO VirtualQueries!
        for _ in 0..4 {
            let chunks = engine.collect_dirty_uploads();
            let m = engine.last_sync_metrics();
            assert!(chunks.is_empty(), "untouched plain private & MEM_MAPPED tiles must yield 0 uploads");
            assert_eq!(m.tiles_scanned, 0, "untouched tiles must NEVER trigger full-span scans");
            assert_eq!(m.full_fallbacks, 0);
            assert_eq!(m.virtual_queries, 0, "region classifications must remain cached");
        }

        // 3. Write 4 bytes into plain private tile 1 AND 4 bytes into MEM_MAPPED tile 3.
        unsafe {
            ((plain_va + TILE_BYTES + 0x200) as *mut u32).write_volatile(0xA1B2_C3D4);
            ((mapped_va + TILE_BYTES + 0x300) as *mut u32).write_volatile(0xE5F6_0718);
        }

        let inc = engine.collect_dirty_uploads();
        let m3 = engine.last_sync_metrics();
        assert_eq!(inc.len(), 2, "both the plain-private write and MEM_MAPPED write must be detected");
        assert_eq!(m3.tiles_scanned, 2, "only the 2 modified tiles should be scanned");
        assert_eq!(m3.full_fallbacks, 0);
        assert_eq!(m3.virtual_queries, 0);
        assert_eq!(inc[0].offset, TILE_BYTES + 0x200);
        assert_eq!(inc[0].data.len(), 64);
        assert_eq!(u32::from_le_bytes(inc[0].data[0..4].try_into().unwrap()), 0xA1B2_C3D4);
        assert_eq!(inc[1].offset, 3 * TILE_BYTES + 0x300);
        assert_eq!(inc[1].data.len(), 64);
        assert_eq!(u32::from_le_bytes(inc[1].data[0..4].try_into().unwrap()), 0xE5F6_0718);

        // 4. Subsequent untouched submit returns immediately to 0 tiles scanned!
        let after = engine.collect_dirty_uploads();
        let m4 = engine.last_sync_metrics();
        assert!(after.is_empty());
        assert_eq!(m4.tiles_scanned, 0);
        assert_eq!(m4.full_fallbacks, 0);
        assert_eq!(m4.virtual_queries, 0);

        // Drop engine BEFORE MixedGuard so `SparseEngine::drop` exercises guard teardown.
        drop(engine);
    }

    /// Simulates `xmem::promote_view_containing`: a single-view `MEM_WRITE_WATCH`
    /// private tile is lazily replaced in place with a `MEM_MAPPED` section view
    /// without calling `sparse_uma::on_map`. Verifies `SparseEngine` detects the
    /// stale classification, reclassifies cleanly to `RegionKind::MappedView`,
    /// syncs the new bytes, and returns to 0 scans on untouched submits.
    #[test]
    fn lazy_promotion_from_private_write_watch_to_mapped_view_reclassifies_cleanly() {
        use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_COMMIT, MEM_PRESERVE_PLACEHOLDER, MEM_RELEASE,
            MEM_REPLACE_PLACEHOLDER, MEM_RESERVE, MEM_RESERVE_PLACEHOLDER,
            MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile3, PAGE_NOACCESS, PAGE_READWRITE, SEC_COMMIT,
            UnmapViewOfFile2, VirtualAlloc2, VirtualFree,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;

        const MEM_WRITE_WATCH: u32 = 0x0020_0000;

        let Some((device, copy_queue, mut engine, aperture_base)) = test_engine_and_copy_queue()
        else {
            return;
        };

        // Window 4, tile 0: unique VA range for parallel test safety.
        let va = tile_va(aperture_base, 4, 0);
        let len = TILE_BYTES as usize;
        let proc = unsafe { GetCurrentProcess() };

        let ph = unsafe {
            VirtualAlloc2(
                proc,
                va as *const std::ffi::c_void,
                len,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            )
        };
        assert_eq!(ph as u64, va);
        let priv_ptr = unsafe {
            VirtualAlloc2(
                proc,
                va as *const std::ffi::c_void,
                len,
                MEM_RESERVE | MEM_COMMIT | MEM_REPLACE_PLACEHOLDER | MEM_WRITE_WATCH,
                PAGE_READWRITE,
                std::ptr::null_mut(),
                0,
            )
        };
        assert_eq!(priv_ptr as u64, va);

        let section = unsafe {
            CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE | SEC_COMMIT,
                0,
                len as u32,
                std::ptr::null(),
            )
        };
        assert!(!section.is_null());

        struct PromoGuard {
            va: usize,
            len: usize,
            promoted: bool,
            section: HANDLE,
        }
        impl Drop for PromoGuard {
            fn drop(&mut self) {
                crate::watch::title_memory_remapped(self.va, self.len);
                unsafe {
                    if self.promoted {
                        UnmapViewOfFile2(
                            GetCurrentProcess(),
                            MEMORY_MAPPED_VIEW_ADDRESS {
                                Value: self.va as *mut std::ffi::c_void,
                            },
                            0,
                        );
                    } else {
                        VirtualFree(self.va as *mut std::ffi::c_void, 0, MEM_RELEASE);
                    }
                    CloseHandle(self.section);
                }
            }
        }
        let mut guard = PromoGuard {
            va: va as usize,
            len,
            promoted: false,
            section,
        };

        unsafe { (va as *mut u32).write_volatile(0x1234_5678) };

        let mut to_map = BTreeSet::new();
        to_map.insert((4u8, 0u32));
        let empty = BTreeSet::new();
        assert!(engine.apply_tile_changes(&device, &copy_queue, &empty, &to_map, 1, false));

        // Initial + untouched steady state as `PrivateWriteWatch`.
        assert_eq!(engine.collect_dirty_uploads().len(), 1);
        assert!(engine.collect_dirty_uploads().is_empty());
        assert_eq!(engine.last_sync_metrics().write_watch_calls, 1);
        assert_eq!(engine.last_sync_metrics().tiles_scanned, 0);

        // Lazily promote in place to `MEM_MAPPED` WITHOUT notifying `sparse_uma::on_map`.
        crate::watch::title_memory_remapped(va as usize, len);
        assert_ne!(
            unsafe {
                VirtualFree(
                    va as *mut std::ffi::c_void,
                    len,
                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
                )
            },
            0
        );
        let view = unsafe {
            MapViewOfFile3(
                section,
                proc,
                va as *const std::ffi::c_void,
                0,
                len,
                MEM_REPLACE_PLACEHOLDER,
                PAGE_READWRITE,
                std::ptr::null_mut(),
                0,
            )
        };
        assert_eq!(view.Value as u64, va);
        guard.promoted = true;
        unsafe { (va as *mut u32).write_volatile(0x9ABC_DEF0) };
        crate::watch::title_memory_remapped(va as usize, len);

        // Next submit detects stale `PrivateWriteWatch` -> reclassifies to `MappedView` -> syncs new value!
        let promo_chunks = engine.collect_dirty_uploads();
        let mp = engine.last_sync_metrics();
        assert_eq!(promo_chunks.len(), 1);
        assert_eq!(promo_chunks[0].data.len(), 64);
        assert_eq!(
            u32::from_le_bytes(promo_chunks[0].data[0..4].try_into().unwrap()),
            0x9ABC_DEF0
        );
        assert_eq!(mp.full_fallbacks, 0);
        assert_eq!(mp.mapped_view_calls, 1);
        assert!(mp.virtual_queries >= 1);

        // Following untouched submit uses cached `MappedView`: 0 scans, 0 VirtualQueries!
        let clean = engine.collect_dirty_uploads();
        let mc = engine.last_sync_metrics();
        assert!(clean.is_empty());
        assert_eq!(mc.tiles_scanned, 0);
        assert_eq!(mc.virtual_queries, 0);
        assert_eq!(mc.full_fallbacks, 0);
        assert_eq!(mc.fallback_tiles, 0);
        assert_eq!(mc.bytes_compared, 0);
        assert_eq!(mc.mapped_view_calls, 1);

        // Subsequent untouched submit hits the O(1) `GUARD_EPOCH` fast exit (`mapped_view_calls == 0`).
        // Retry across concurrent parallel tests that may bump process-wide `GUARD_EPOCH`.
        let mut hit_o1_fast_exit = false;
        for _ in 0..256 {
            let steady = engine.collect_dirty_uploads();
            let ms = engine.last_sync_metrics();
            assert!(steady.is_empty());
            assert_eq!(ms.tiles_scanned, 0);
            assert_eq!(ms.virtual_queries, 0);
            assert_eq!(ms.full_fallbacks, 0);
            assert_eq!(ms.bytes_compared, 0);
            if ms.mapped_view_calls == 0 {
                hit_o1_fast_exit = true;
                break;
            }
            std::thread::yield_now();
        }
        assert!(hit_o1_fast_exit, "steady-state MappedView must hit O(1) GUARD_EPOCH fast exit");
    }

    /// Comprehensive Phase 6B verification across EVERY `RegionKind` variant
    /// simultaneously (`PrivateWriteWatch`, multi-view `MappedView` + external
    /// alias, writable `PrivateGuarded`, originally read-only `PrivateGuarded`,
    /// and `Uncommitted`):
    /// - Zero redundant scans (`tiles_scanned == 0`, `bytes_compared == 0`),
    ///   zero `VirtualQuery` syscalls (both `sparse_uma` and `watch::writes_mixed`),
    ///   zero fallbacks (`fallback_tiles == 0`), and O(1) `GUARD_EPOCH` fast exit
    ///   on untouched steady-state submits;
    /// - 100% coherency across direct CPU writes, alias view writes,
    ///   `kernel_will_write` (`ReadFile`), and `Shadow` (`watch::writes`) overlap.
    #[test]
    fn all_region_kinds_zero_redundant_scans_and_o1_epoch_coherency() {
        use std::io::{Read, Seek, Write};
        use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEM_COMMIT, MEM_PRESERVE_PLACEHOLDER, MEM_RELEASE,
            MEM_REPLACE_PLACEHOLDER, MEM_RESERVE, MEM_RESERVE_PLACEHOLDER,
            MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile, MapViewOfFile3, PAGE_NOACCESS,
            PAGE_READONLY, PAGE_READWRITE, SEC_COMMIT, UnmapViewOfFile, UnmapViewOfFile2,
            VirtualAlloc2, VirtualFree, VirtualProtect,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;

        const MEM_WRITE_WATCH: u32 = 0x0020_0000;

        let Some((device, copy_queue, mut engine, aperture_base)) = test_engine_and_copy_queue()
        else {
            return;
        };

        // Window 5, tiles 0..8 (512 KiB, unique VA range):
        //   tile 0      : `PrivateWriteWatch` (`MEM_WRITE_WATCH`)
        //   tiles 1..5  : 4 separate 64 KiB `MappedView`s back-to-back (exact `xmem` layout)
        //   tile 5      : writable `PrivateGuarded` (plain `MEM_PRIVATE` `PAGE_READWRITE`)
        //   tile 6      : originally read-only `PrivateGuarded` (`MEM_PRIVATE` `PAGE_READONLY`)
        //   tile 7      : `Uncommitted` placeholder (`MEM_RESERVE`, `PAGE_NOACCESS`)
        let base_va = tile_va(aperture_base, 5, 0);
        let tile = |i: u64| base_va + i * TILE_BYTES;
        let proc = unsafe { GetCurrentProcess() };

        let ph = unsafe {
            VirtualAlloc2(
                proc,
                base_va as *const std::ffi::c_void,
                (8 * TILE_BYTES) as usize,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            )
        };
        assert_eq!(ph as u64, base_va);

        // Split tiles 0..7 off individually (leaving tile 7 as an uncommitted placeholder).
        for i in 0u64..7 {
            assert_ne!(
                unsafe {
                    VirtualFree(
                        tile(i) as *mut std::ffi::c_void,
                        TILE_BYTES as usize,
                        MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
                    )
                },
                0
            );
        }

        // Tile 0: `PrivateWriteWatch`.
        let p_ww = unsafe {
            VirtualAlloc2(
                proc,
                tile(0) as *const std::ffi::c_void,
                TILE_BYTES as usize,
                MEM_RESERVE | MEM_COMMIT | MEM_REPLACE_PLACEHOLDER | MEM_WRITE_WATCH,
                PAGE_READWRITE,
                std::ptr::null_mut(),
                0,
            )
        };
        assert_eq!(p_ww as u64, tile(0));

        // Tiles 1..5: 4 separate 64 KiB `MapViewOfFile3` views + external alias of tile 2.
        let section = unsafe {
            CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE | SEC_COMMIT,
                0,
                (4 * TILE_BYTES) as u32,
                std::ptr::null(),
            )
        };
        assert!(!section.is_null());
        for i in 0u64..4 {
            let va = tile(1 + i);
            let v = unsafe {
                MapViewOfFile3(
                    section,
                    proc,
                    va as *const std::ffi::c_void,
                    i * TILE_BYTES,
                    TILE_BYTES as usize,
                    MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                )
            };
            assert_eq!(v.Value as u64, va);
            crate::inplace::title_memory_view(
                va as usize,
                TILE_BYTES as usize,
                section as usize,
                i * TILE_BYTES,
            );
        }
        let alias_va = unsafe {
            MapViewOfFile(section, FILE_MAP_WRITE, 0, TILE_BYTES as u32, TILE_BYTES as usize).Value
                as usize
        };
        assert_ne!(alias_va, 0);
        crate::inplace::title_memory_view(
            alias_va,
            TILE_BYTES as usize,
            section as usize,
            TILE_BYTES,
        );

        // Tile 5: plain writable `MEM_PRIVATE`.
        let p_pg_rw = unsafe {
            VirtualAlloc2(
                proc,
                tile(5) as *const std::ffi::c_void,
                TILE_BYTES as usize,
                MEM_RESERVE | MEM_COMMIT | MEM_REPLACE_PLACEHOLDER,
                PAGE_READWRITE,
                std::ptr::null_mut(),
                0,
            )
        };
        assert_eq!(p_pg_rw as u64, tile(5));

        // Tile 6: originally read-only `MEM_PRIVATE` (seed bytes then make `PAGE_READONLY`).
        let p_pg_ro = unsafe {
            VirtualAlloc2(
                proc,
                tile(6) as *const std::ffi::c_void,
                TILE_BYTES as usize,
                MEM_RESERVE | MEM_COMMIT | MEM_REPLACE_PLACEHOLDER,
                PAGE_READWRITE,
                std::ptr::null_mut(),
                0,
            )
        };
        assert_eq!(p_pg_ro as u64, tile(6));
        unsafe { (tile(6) as *mut u32).write_volatile(0x524F_524F) };
        let mut old_prot = 0u32;
        assert_ne!(
            unsafe {
                VirtualProtect(
                    tile(6) as *const std::ffi::c_void,
                    TILE_BYTES as usize,
                    PAGE_READONLY,
                    &mut old_prot,
                )
            },
            0
        );

        struct AllKindsGuard {
            base_va: u64,
            alias_va: usize,
            section: HANDLE,
        }
        impl Drop for AllKindsGuard {
            fn drop(&mut self) {
                let tile = |i: u64| (self.base_va + i * TILE_BYTES) as usize;
                plain_guards_release_range(tile(5), tile(6));
                crate::watch::title_memory_remapped(tile(0), (5 * TILE_BYTES) as usize);
                crate::watch::title_memory_remapped(self.alias_va, TILE_BYTES as usize);
                unsafe {
                    let proc = GetCurrentProcess();
                    VirtualFree(tile(0) as *mut std::ffi::c_void, 0, MEM_RELEASE);
                    for i in 1u64..5 {
                        UnmapViewOfFile2(
                            proc,
                            MEMORY_MAPPED_VIEW_ADDRESS {
                                Value: tile(i) as *mut std::ffi::c_void,
                            },
                            0,
                        );
                    }
                    UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                        Value: self.alias_va as *mut std::ffi::c_void,
                    });
                    VirtualFree(tile(5) as *mut std::ffi::c_void, 0, MEM_RELEASE);
                    VirtualFree(tile(6) as *mut std::ffi::c_void, 0, MEM_RELEASE);
                    VirtualFree(tile(7) as *mut std::ffi::c_void, 0, MEM_RELEASE);
                    CloseHandle(self.section);
                }
            }
        }
        let _guard = AllKindsGuard {
            base_va,
            alias_va,
            section,
        };

        // Seed initial bytes in writable tiles.
        unsafe {
            (tile(0) as *mut u32).write_volatile(0x0001_0001);
            (tile(1) as *mut u32).write_volatile(0x0002_0002);
            (tile(2) as *mut u32).write_volatile(0x0003_0003);
            (tile(3) as *mut u32).write_volatile(0x0004_0004);
            (tile(4) as *mut u32).write_volatile(0x0005_0005);
            (tile(5) as *mut u32).write_volatile(0x0006_0006);
        }

        let mut to_map = BTreeSet::new();
        for t in 0u32..8 {
            to_map.insert((5u8, t));
        }
        let empty = BTreeSet::new();
        assert!(engine.apply_tile_changes(&device, &copy_queue, &empty, &to_map, 1, false));

        // 1. Initial sync: uploads the 7 committed tiles (0 for `Uncommitted` tile 7).
        let init = engine.collect_dirty_uploads();
        let m1 = engine.last_sync_metrics();
        assert_eq!(init.len(), 7);
        assert_eq!(m1.full_fallbacks, 0);
        assert_eq!(m1.fallback_tiles, 0);
        assert_eq!(m1.write_watch_calls, 1);
        assert_eq!(m1.mapped_view_calls, 1, "all 4 separate 64 KiB MEM_MAPPED views coalesce");

        // 2. Steady state: 0 scans, 0 bytes_compared, 0 VirtualQueries (sparse_uma & watch),
        //    0 fallbacks, and O(1) `GUARD_EPOCH` skip (`mapped_view_calls == 0`)!
        //    Retry across parallel tests that may bump process-wide `GUARD_EPOCH`.
        let mut hit_o1_mv = false;
        for _ in 0..256 {
            crate::watch::reset_mixed_virtual_queries();
            let chunks = engine.collect_dirty_uploads();
            let m = engine.last_sync_metrics();
            assert!(chunks.is_empty());
            assert_eq!(m.tiles_scanned, 0);
            assert_eq!(m.bytes_compared, 0);
            assert_eq!(m.virtual_queries, 0);
            assert_eq!(m.full_fallbacks, 0);
            assert_eq!(m.fallback_tiles, 0);
            if m.mapped_view_calls == 0 && crate::watch::mixed_virtual_queries() == 0 {
                hit_o1_mv = true;
                break;
            }
            std::thread::yield_now();
        }
        assert!(hit_o1_mv, "untouched MappedView span must hit O(1) GUARD_EPOCH skip");

        // 3a. Direct CPU write to `MappedView` tile 1 -> caught immediately despite O(1) epoch cache!
        unsafe { ((tile(1) + 0x100) as *mut u32).write_volatile(0x1111_AAAA) };
        let c_mv = engine.collect_dirty_uploads();
        let m_mv = engine.last_sync_metrics();
        assert_eq!(c_mv.len(), 1);
        assert_eq!(c_mv[0].offset, TILE_BYTES + 0x100);
        assert_eq!(c_mv[0].data.len(), 64);
        assert_eq!(u32::from_le_bytes(c_mv[0].data[0..4].try_into().unwrap()), 0x1111_AAAA);
        assert_eq!(m_mv.tiles_scanned, 1);
        assert_eq!(m_mv.bytes_compared, TILE_BYTES);

        // Re-lock O(1) epoch cache before next coherency test.
        for _ in 0..2 {
            assert!(engine.collect_dirty_uploads().is_empty());
        }

        // 3b. External alias write into tile 2 (`alias_va`) -> caught despite O(1) epoch cache!
        unsafe { ((alias_va + 0x140) as *mut u32).write_volatile(0x2222_BBBB) };
        let c_alias = engine.collect_dirty_uploads();
        let m_alias = engine.last_sync_metrics();
        assert_eq!(c_alias.len(), 1);
        assert_eq!(c_alias[0].offset, 2 * TILE_BYTES + 0x140);
        assert_eq!(c_alias[0].data.len(), 64);
        assert_eq!(u32::from_le_bytes(c_alias[0].data[0..4].try_into().unwrap()), 0x2222_BBBB);
        assert_eq!(m_alias.tiles_scanned, 1);

        for _ in 0..2 {
            assert!(engine.collect_dirty_uploads().is_empty());
        }

        // 3c. `kernel_will_write` (`watch::writing`) + `ReadFile` into `MappedView` tile 3!
        let mut f = tempfile::tempfile().unwrap();
        f.write_all(b"SPARSE_UMA_KW64!").unwrap();
        f.rewind().unwrap();
        crate::watch::writing((tile(3) + 0x180) as usize, 16);
        let dst = unsafe { std::slice::from_raw_parts_mut((tile(3) + 0x180) as *mut u8, 16) };
        assert_eq!(f.read(dst).unwrap(), 16);
        let c_kw = engine.collect_dirty_uploads();
        let m_kw = engine.last_sync_metrics();
        assert_eq!(c_kw.len(), 1);
        assert_eq!(c_kw[0].offset, 3 * TILE_BYTES + 0x180);
        assert_eq!(c_kw[0].data.len(), 64);
        assert_eq!(&c_kw[0].data[0..16], b"SPARSE_UMA_KW64!");
        assert_eq!(m_kw.tiles_scanned, 1);

        for _ in 0..2 {
            assert!(engine.collect_dirty_uploads().is_empty());
        }

        // 3d. `Shadow` overlap (`watch::writes` drains `MappedView` tile 4 fault first)!
        unsafe { ((tile(4) + 0x1C0) as *mut u32).write_volatile(0x4444_DDDD) };
        assert!(crate::watch::writes(tile(4) as usize, (tile(4) + TILE_BYTES) as usize, true).is_some());
        let c_shadow = engine.collect_dirty_uploads();
        let m_shadow = engine.last_sync_metrics();
        assert_eq!(c_shadow.len(), 1, "must stay 100% coherent when Shadow drains guard fault first");
        assert_eq!(c_shadow[0].offset, 4 * TILE_BYTES + 0x1C0);
        assert_eq!(u32::from_le_bytes(c_shadow[0].data[0..4].try_into().unwrap()), 0x4444_DDDD);
        assert_eq!(m_shadow.tiles_scanned, 1);

        for _ in 0..2 {
            assert!(engine.collect_dirty_uploads().is_empty());
        }

        // 3e. Writable `PrivateGuarded` tile 5 write -> caught despite `PLAIN_GUARD_EPOCH` cache!
        unsafe { ((tile(5) + 0x200) as *mut u32).write_volatile(0x5555_EEEE) };
        let c_pg = engine.collect_dirty_uploads();
        let m_pg = engine.last_sync_metrics();
        assert_eq!(c_pg.len(), 1);
        assert_eq!(c_pg[0].offset, 5 * TILE_BYTES + 0x200);
        assert_eq!(u32::from_le_bytes(c_pg[0].data[0..4].try_into().unwrap()), 0x5555_EEEE);
        assert_eq!(m_pg.tiles_scanned, 1);

        // 3f. `PrivateWriteWatch` tile 0 write -> caught immediately!
        unsafe { ((tile(0) + 0x240) as *mut u32).write_volatile(0x0000_FFFF) };
        let c_ww = engine.collect_dirty_uploads();
        assert_eq!(c_ww.len(), 1);
        assert_eq!(c_ww[0].offset, 0x240);
        assert_eq!(u32::from_le_bytes(c_ww[0].data[0..4].try_into().unwrap()), 0x0000_FFFF);

        // 4. Unmap `PrivateWriteWatch` tile 0 so only `MappedView` + `PrivateGuarded` + `Uncommitted`
        //    remain, proving the complete O(1) ultra-fast exit (0 ww calls, 0 mv calls, 0 scans, 0 VQs)!
        let mut to_unmap = BTreeSet::new();
        to_unmap.insert((5u8, 0u32));
        engine.apply_tile_changes(&device, &copy_queue, &to_unmap, &empty, 2, false);
        let mut hit_full_o1 = false;
        for _ in 0..256 {
            crate::watch::reset_mixed_virtual_queries();
            let chunks = engine.collect_dirty_uploads();
            let m = engine.last_sync_metrics();
            assert!(chunks.is_empty());
            assert_eq!(m.write_watch_calls, 0);
            assert_eq!(m.tiles_scanned, 0);
            assert_eq!(m.bytes_compared, 0);
            assert_eq!(m.virtual_queries, 0);
            assert_eq!(m.fallback_tiles, 0);
            if m.mapped_view_calls == 0 && crate::watch::mixed_virtual_queries() == 0 {
                hit_full_o1 = true;
                break;
            }
            std::thread::yield_now();
        }
        assert!(hit_full_o1, "steady-state MappedView+PrivateGuarded+Uncommitted must hit full O(1) exit");

        drop(engine);
    }
}
