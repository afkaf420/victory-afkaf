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

/// Whether Stage M2 64-byte cacheline diffing is enabled for sparse uploads
/// (`VICTORY_DIFF64`, on by default; `"0"` disables).
fn diff64_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_DIFF64").is_none_or(|v| v != "0"))
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
// Core SparseEngine (12 x 2 GiB Reserved Buffers + Tile Map + Diff64 Sync)
// ---------------------------------------------------------------------------

struct MappedTile {
    pool_id: usize,
    pool_tile: u32,
    needs_initial_sync: bool,
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
    watch_buf: Vec<usize>,
    releases: ReleaseQueue,
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
            watch_buf: Vec::new(),
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

        // 1. Park newly unmapped tiles (zero UpdateTileMappings cost until
        //    parked tiles exceed `park_limit_tiles`!).
        for &key in to_unmap {
            if let Some(m) = self.mapped.remove(&key) {
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
                existing.needs_initial_sync = true;
                continue;
            }
            if let Some(p) = self.parked.remove(&key) {
                // Instant reuse from parked tiles — no UpdateTileMappings needed!
                self.mapped.insert(
                    key,
                    MappedTile {
                        pool_id: p.pool_id,
                        pool_tile: p.pool_tile,
                        needs_initial_sync: true,
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
                    for k in 0..count {
                        self.mapped.insert(
                            (win, res_tile + k),
                            MappedTile {
                                pool_id,
                                pool_tile: pool_start + k,
                                needs_initial_sync: true,
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

    /// Scoped pre-submit CPU -> VRAM dirty scanner over all currently mapped
    /// sparse tiles: runs `GetWriteWatch(WRITE_WATCH_FLAG_RESET)` (falling back
    /// per `VirtualQuery` sub-region to `watch::writes` for promoted aliased
    /// `MEM_MAPPED` views), then Stage M2 64-byte cacheline diffing against
    /// per-tile baselines.
    pub(crate) fn collect_dirty_uploads(&mut self) -> Vec<UploadChunk> {
        if self.mapped.is_empty() {
            return Vec::new();
        }
        let started = std::time::Instant::now();
        let diff64 = diff64_enabled();

        // Group currently mapped tiles into contiguous VA runs.
        let keys: Vec<(u8, u32)> = self.mapped.keys().copied().collect();
        let mut runs: Vec<(u64, u64, Vec<(u8, u32)>)> = Vec::new();
        for key in keys {
            let va = tile_va(self.aperture_base, key.0 as usize, key.1);
            match runs.last_mut() {
                Some(last) if last.1 == va => {
                    last.1 = va + TILE_BYTES;
                    last.2.push(key);
                }
                _ => runs.push((va, va + TILE_BYTES, vec![key])),
            }
        }

        let mut uploads: Vec<UploadChunk> = Vec::new();
        let mut total_bytes = 0u64;
        let mut scratch = vec![0u8; TILE_BYTES as usize];
        let mut diff_runs: Vec<(usize, usize)> = Vec::new();

        for (run_start, run_end, run_keys) in runs {
            // Query dirty 4 KiB sub-pages across `[run_start, run_end)`.
            let dirty_masks = query_dirty_4k_masks(
                &mut self.watch_buf,
                run_start as usize,
                run_end as usize,
            );
            for (idx, key) in run_keys.into_iter().enumerate() {
                let va = run_start + (idx as u64) * TILE_BYTES;
                let Some(tile) = self.mapped.get_mut(&key) else {
                    continue;
                };
                let win = key.0 as usize;
                let win_tile_off = u64::from(key.1) * TILE_BYTES;

                if tile.needs_initial_sync {
                    tile.needs_initial_sync = false;
                    crate::host::copy_mapped(
                        scratch.as_mut_ptr(),
                        va as *const u8,
                        TILE_BYTES as usize,
                        false,
                    );
                    if tile.baseline.is_none()
                        && self.diff_shadow_bytes.saturating_add(TILE_BYTES) <= self.diff_shadow_cap
                    {
                        let mut b = Box::new([0u8; TILE_BYTES as usize]);
                        b.copy_from_slice(&scratch);
                        tile.baseline = Some(b);
                        self.diff_shadow_bytes += TILE_BYTES;
                    } else if let Some(b) = tile.baseline.as_mut() {
                        b.copy_from_slice(&scratch);
                    }
                    total_bytes += TILE_BYTES;
                    uploads.push(UploadChunk {
                        window: win,
                        offset: win_tile_off,
                        data: scratch.clone(),
                    });
                    continue;
                }

                let mask = dirty_masks[idx];
                if mask == 0 {
                    continue;
                }

                // Coalesce adjacent dirty 4 KiB sub-pages within this 64 KiB tile.
                let mut sub = 0usize;
                while sub < 16 {
                    if (mask & (1u16 << sub)) == 0 {
                        sub += 1;
                        continue;
                    }
                    let sub_start = sub;
                    while sub < 16 && (mask & (1u16 << sub)) != 0 {
                        sub += 1;
                    }
                    let lo = sub_start * 4096;
                    let hi = sub * 4096;
                    let len = hi - lo;
                    crate::host::copy_mapped(
                        scratch[lo..hi].as_mut_ptr(),
                        (va + lo as u64) as *const u8,
                        len,
                        false,
                    );
                    if diff64 && let Some(baseline) = tile.baseline.as_mut() {
                        diff_runs.clear();
                        crate::shadow::diff_cachelines_64(
                            &mut baseline[lo..hi],
                            &scratch[lo..hi],
                            lo,
                            &mut diff_runs,
                        );
                        for &(a, b) in &diff_runs {
                            total_bytes += (b - a) as u64;
                            uploads.push(UploadChunk {
                                window: win,
                                offset: win_tile_off + a as u64,
                                data: baseline[a..b].to_vec(),
                            });
                        }
                    } else {
                        if let Some(baseline) = tile.baseline.as_mut() {
                            baseline[lo..hi].copy_from_slice(&scratch[lo..hi]);
                        }
                        total_bytes += len as u64;
                        uploads.push(UploadChunk {
                            window: win,
                            offset: win_tile_off + lo as u64,
                            data: scratch[lo..hi].to_vec(),
                        });
                    }
                }
            }
        }

        STATS[6].fetch_add(total_bytes, Ordering::Relaxed);
        STATS[7].fetch_add(uploads.len() as u64, Ordering::Relaxed);
        STATS[8].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
        uploads
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

/// Queries dirty 4 KiB OS pages for the 64 KiB-aligned range `[start, end)`,
/// returning one `u16` bitmask per 64 KiB tile (bit `k` = 4 KiB page `k` dirty).
///
/// Fast path: one `GetWriteWatch(WRITE_WATCH_FLAG_RESET)` over the entire run.
/// Fallback: walks `VirtualQuery` sub-regions so mixed private write-watched
/// allocations, promoted `MEM_MAPPED` views, and placeholder holes each use
/// their optimal tracking mechanism.
fn query_dirty_4k_masks(buf: &mut Vec<usize>, start: usize, end: usize) -> Vec<u16> {
    use windows_sys::Win32::System::Memory::{
        GetWriteWatch, MEM_COMMIT, MEM_MAPPED, MEM_PRIVATE, MEMORY_BASIC_INFORMATION,
        PAGE_NOACCESS, VirtualQuery,
    };
    const WRITE_WATCH_FLAG_RESET: u32 = 1;
    const KERNEL_PAGE: usize = 4096;

    let n_tiles = (end - start) / (TILE_BYTES as usize);
    let mut masks = vec![0u16; n_tiles];
    if n_tiles == 0 {
        return masks;
    }
    let max_pages = (end - start) / KERNEL_PAGE;
    if buf.len() < max_pages {
        buf.resize(max_pages, 0);
    }

    let mut count = max_pages;
    let mut granularity = 0u32;
    // SAFETY: `buf` has room for `count` page addresses.
    let ok = unsafe {
        GetWriteWatch(
            WRITE_WATCH_FLAG_RESET,
            start as *const _,
            end - start,
            buf.as_mut_ptr().cast(),
            &mut count,
            &mut granularity,
        )
    } == 0;
    if ok {
        for &addr in &buf[..count] {
            if addr >= start && addr < end {
                let rel = addr - start;
                let tile_idx = rel / (TILE_BYTES as usize);
                let sub = (rel % (TILE_BYTES as usize)) / KERNEL_PAGE;
                masks[tile_idx] |= 1u16 << sub;
            }
        }
        return masks;
    }

    // Mixed sub-region fallback via VirtualQuery.
    let mut at = start;
    while at < end {
        // SAFETY: plain VirtualQuery into a zeroed struct.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: valid out pointer.
        if unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) } == 0 {
            break;
        }
        let region_end = (info.BaseAddress as usize)
            .saturating_add(info.RegionSize)
            .min(end);
        if region_end <= at {
            break;
        }
        if info.State == MEM_COMMIT && info.Protect != PAGE_NOACCESS {
            let sub_len = region_end - at;
            if info.Type == MEM_PRIVATE {
                let mut sub_count = sub_len.div_ceil(KERNEL_PAGE);
                if buf.len() < sub_count {
                    buf.resize(sub_count, 0);
                }
                let mut gran = 0u32;
                // SAFETY: `buf` holds `sub_count` pointers.
                let sub_ok = unsafe {
                    GetWriteWatch(
                        WRITE_WATCH_FLAG_RESET,
                        at as *const _,
                        sub_len,
                        buf.as_mut_ptr().cast(),
                        &mut sub_count,
                        &mut gran,
                    )
                } == 0;
                if sub_ok {
                    for &addr in &buf[..sub_count] {
                        if addr >= start && addr < end {
                            let rel = addr - start;
                            masks[rel / (TILE_BYTES as usize)] |=
                                1u16 << ((rel % (TILE_BYTES as usize)) / KERNEL_PAGE);
                        }
                    }
                } else {
                    // Non-write-watched private pages: mark covered 4 KiB pages
                    // dirty so diff64 still filters unchanged cachelines.
                    let mut p = at & !(KERNEL_PAGE - 1);
                    while p < region_end {
                        if p >= start && p < end {
                            let rel = p - start;
                            masks[rel / (TILE_BYTES as usize)] |=
                                1u16 << ((rel % (TILE_BYTES as usize)) / KERNEL_PAGE);
                        }
                        p += KERNEL_PAGE;
                    }
                }
            } else if info.Type == MEM_MAPPED {
                // Promoted aliased section view: use existing VIEW_GUARD tracker.
                if let Some((now, pages)) = crate::watch::writes(at, region_end, true) {
                    for (page_64k, asked) in pages {
                        if asked == now {
                            let p_va = page_64k * (TILE_BYTES as usize);
                            if p_va >= start && p_va < end {
                                masks[(p_va - start) / (TILE_BYTES as usize)] = u16::MAX;
                            }
                        }
                    }
                } else {
                    let mut p = at & !(KERNEL_PAGE - 1);
                    while p < region_end {
                        if p >= start && p < end {
                            let rel = p - start;
                            masks[rel / (TILE_BYTES as usize)] |=
                                1u16 << ((rel % (TILE_BYTES as usize)) / KERNEL_PAGE);
                        }
                        p += KERNEL_PAGE;
                    }
                }
            }
        }
        at = region_end;
    }
    masks
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
    tracing::info!(
        "sparse UMA (24 GiB, 12x2 GiB): {} MiB mapped ({} tiles), {} MiB parked ({} tiles), \
         {} MiB DEFAULT pools ({} released), UpdateTileMappings {} map / {} unmap, \
         synced {} MiB in {} diff64 runs ({} us)",
        (s[0] * TILE_BYTES) >> 20,
        s[0],
        (s[1] * TILE_BYTES) >> 20,
        s[1],
        s[2] >> 20,
        s[3],
        s[4],
        s[5],
        s[6] >> 20,
        s[7],
        s[8],
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
}
