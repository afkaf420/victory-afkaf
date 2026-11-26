//! PC-Native VRAM Promotion — Move 2 (`VICTORY_LAZY_READBACK`, default mode `1`).
//!
//! Eliminates eager PCIe readbacks (`Card::before`, `CARD_COMPARE`,
//! `relay_card_back`) for GPU-written `WriteCombinedGpuRw` (`0x2404`) buffers
//! (and optionally `CachedGpuRw` `0x2004` in mode `2`) by leaving their latest
//! bytes resident in VRAM and marking the corresponding 64 KiB title-memory
//! blocks `PAGE_NOACCESS` ONCE on first promotion:
//!
//! - Subsequent GPU writes to an already-`GpuResident` block perform an O(1)
//!   fence update only (`0 VirtualProtect`, `0 Card::before`, `0 PCIe readback`).
//! - Any CPU read or write fault (`watch::on_write` VEH) or Victory internal
//!   CPU read (`ensure_cpu_resident`) pulls that single 64 KiB block on demand
//!   over a dedicated `victory-lazy-fetch` `D3D12_COMMAND_LIST_TYPE_COPY` queue
//!   (`rb_queue.Wait(&main_fence, last_fence)` — zero lock contention with
//!   `Host::submitted`, `LINE`, `GUARDS`, `MAPPING`, or `xmem::PHYSICAL`).
//! - Adaptive demotion: any 64 KiB block that faults `>= 2` times is
//!   permanently demoted to `CpuHot` until unmapped/remapped (`evict_range`),
//!   reverting that block to standard `Card::before` / `relay_card_back` with
//!   zero future faults.

use std::cell::Cell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicPtr, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock, Weak, mpsc};
use std::time::{Duration, Instant};

use windows::Win32::Foundation::{CloseHandle, HANDLE};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
use windows::core::Interface;

use crate::shadow::Shadow;
use crate::watch::ConsoleMemClass;

/// Granularity of lazy residency tracking (64 KiB, matching `watch::PAGE` and
/// D3D12/console allocation granularity).
pub(crate) const BLOCK: usize = 64 << 10;

/// Number of CPU faults/materializations after which a 64 KiB block is
/// permanently demoted to `CpuHot` until unmapped/remapped.
const DEMOTE_THRESHOLD: u8 = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Off,
    WriteCombined,
    WriteCombinedAndCached,
}

impl Mode {
    #[inline]
    fn code(self) -> u8 {
        match self {
            Mode::Off => 0,
            Mode::WriteCombined => 1,
            Mode::WriteCombinedAndCached => 2,
        }
    }
}

fn resolve_mode(env: Option<&str>) -> Mode {
    match env.map(|v| v.trim().to_ascii_lowercase()).as_deref() {
        Some("0" | "off" | "false") => Mode::Off,
        Some("2" | "all" | "cached") => Mode::WriteCombinedAndCached,
        _ => Mode::WriteCombined,
    }
}

fn active_mode() -> Mode {
    static MODE: OnceLock<Mode> = OnceLock::new();
    *MODE.get_or_init(|| {
        let env = std::env::var("VICTORY_LAZY_READBACK").ok();
        let m = resolve_mode(env.as_deref());
        tracing::info!(
            mode = m.code(),
            "PC-native VRAM lazy readback (VICTORY_LAZY_READBACK={}): {}",
            m.code(),
            match m {
                Mode::Off => "OFF (eager Card::before / relay_card_back for all writable classes)",
                Mode::WriteCombined =>
                    "ON for WriteCombinedGpuRw (0x2404) with VEH PAGE_NOACCESS demand fetch and 2-fault CpuHot demotion",
                Mode::WriteCombinedAndCached =>
                    "ON for WriteCombinedGpuRw (0x2404) + CachedGpuRw (0x2004) with VEH PAGE_NOACCESS demand fetch",
            }
        );
        m
    })
}

/// Active `VICTORY_LAZY_READBACK` mode (`0` = off, `1` = WC default, `2` = WC + Cached).
#[inline]
pub(crate) fn mode() -> u8 {
    active_mode().code()
}

/// True when `VICTORY_LAZY_READBACK` is enabled (`mode() != 0`).
#[inline]
pub(crate) fn on() -> bool {
    active_mode() != Mode::Off
}

/// True if `class` is eligible for `GpuResident` promotion under the active
/// `VICTORY_LAZY_READBACK` mode.
#[inline]
pub(crate) fn class_eligible(class: ConsoleMemClass) -> bool {
    class_eligible_in(class, active_mode())
}

#[inline]
fn class_eligible_in(class: ConsoleMemClass, m: Mode) -> bool {
    match m {
        Mode::Off => false,
        Mode::WriteCombined => class == ConsoleMemClass::WriteCombinedGpuRw,
        Mode::WriteCombinedAndCached => matches!(
            class,
            ConsoleMemClass::WriteCombinedGpuRw | ConsoleMemClass::CachedGpuRw
        ),
    }
}

/// Lock-free atomic bitmap of 64 KiB blocks covering the 128 TiB user address
/// space (`4 GiB` pieces allocated lazily and never freed).
struct Bits([AtomicPtr<BitsPiece>; Bits::PIECES]);
type BitsPiece = [AtomicU64; Bits::WORDS];

impl Bits {
    const WORDS: usize = 1 << 10;
    const PIECES: usize = 1 << 15;

    const fn new() -> Self {
        Self([const { AtomicPtr::new(std::ptr::null_mut()) }; Self::PIECES])
    }

    #[inline]
    fn piece(&self, word: usize) -> Option<&BitsPiece> {
        let ptr = self.0.get(word / Self::WORDS)?.load(Ordering::Acquire);
        // SAFETY: null, or a leaked `BitsPiece` allocated in `piece_or_init`.
        unsafe { ptr.as_ref() }
    }

    fn piece_or_init(&self, word: usize) -> Option<&BitsPiece> {
        let slot = self.0.get(word / Self::WORDS)?;
        let mut ptr = slot.load(Ordering::Acquire);
        if ptr.is_null() {
            let made: Box<BitsPiece> = Box::new([const { AtomicU64::new(0) }; Self::WORDS]);
            let raw = Box::into_raw(made);
            ptr = match slot.compare_exchange(
                std::ptr::null_mut(),
                raw,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => raw,
                Err(existing) => {
                    // SAFETY: allocated above; another thread installed its own piece.
                    drop(unsafe { Box::from_raw(raw) });
                    existing
                }
            };
        }
        // SAFETY: non-null leaked `BitsPiece`.
        unsafe { ptr.as_ref() }
    }

    #[inline]
    fn contains(&self, block: usize) -> bool {
        let word = block / 64;
        self.piece(word)
            .is_some_and(|p| (p[word % Self::WORDS].load(Ordering::Acquire) >> (block % 64)) & 1 != 0)
    }

    fn set(&self, block: usize) {
        let word = block / 64;
        if let Some(piece) = self.piece_or_init(word) {
            piece[word % Self::WORDS].fetch_or(1u64 << (block % 64), Ordering::SeqCst);
        }
    }

    fn clear(&self, block: usize) {
        let word = block / 64;
        if let Some(piece) = self.piece(word) {
            piece[word % Self::WORDS].fetch_and(!(1u64 << (block % 64)), Ordering::SeqCst);
        }
    }

    fn any(&self, first: usize, last: usize) -> bool {
        let mut block = first;
        while block < last {
            let word = block / 64;
            let until = ((word + 1) * 64).min(last);
            let mask = (u64::MAX >> (64 - (until - block))) << (block % 64);
            match self.piece(word) {
                Some(piece) => {
                    if piece[word % Self::WORDS].load(Ordering::Acquire) & mask != 0 {
                        return true;
                    }
                    block = until;
                }
                None => {
                    let next = (word / Self::WORDS + 1)
                        .saturating_mul(Self::WORDS)
                        .saturating_mul(64);
                    if next <= block {
                        break;
                    }
                    block = next;
                }
            }
        }
        false
    }
}

/// Lock-free bitmap: bit `block = va >> 16` is `1` while `block` is
/// `GpuResident` or `Fetching`. Checked in ~1 ns by `watch::on_write`,
/// `Shadow::update`, `ask_once`, `ensure_cpu_resident`, and `prepare_cpu_overwrite`.
static RESIDENT: Bits = Bits::new();

/// Number of tracked entries (`blocks.len() + fault_counts.len()`), enabling
/// O(1) lock-free no-ops in `evict_range` / `flush_shadow` when idle.
static TRACKED_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Highest fence value confirmed signaled on `Host::queue` (`Host::fence`).
/// Defense-in-depth guard ensuring `victory-lazy-fetch` can never issue
/// `rb_queue.Wait(&main_fence, F)` for an unsignaled future fence `F`.
static SIGNALED_FENCE: AtomicU64 = AtomicU64::new(0);

/// Records that `val` has been signaled on `Host::queue` (`Host::fence`).
#[inline]
pub(crate) fn note_signaled_fence(val: u64) {
    SIGNALED_FENCE.fetch_max(val, Ordering::Release);
}

/// Fast O(1) lock-free check: true if 64 KiB block `block = va >> 16` is
/// currently `GpuResident` (or mid-fetch transitioning back to CPU).
#[inline]
pub(crate) fn is_gpu_resident(block: usize) -> bool {
    RESIDENT.contains(block)
}

/// Fast O(1) lock-free check: true if any 64 KiB block overlapping `[at, at + len)`
/// is currently `GpuResident` (or mid-fetch).
#[inline]
pub(crate) fn any_gpu_resident(at: usize, len: usize) -> bool {
    len != 0 && RESIDENT.any(at / BLOCK, at.saturating_add(len).div_ceil(BLOCK))
}

/// Optional `xmem` physical-page alias writer callback (fallback after
/// `crate::inplace::beside` for section-backed views).
type AliasWriter = fn(usize, &[u8; BLOCK]) -> bool;
static ALIAS_WRITER: AtomicPtr<()> = AtomicPtr::new(std::ptr::null_mut());

/// Registers an optional `xmem` physical-page alias writer that copies a 64 KiB
/// block into the section backing `block_va` while `block_va` remains
/// `PAGE_NOACCESS`. Must be non-blocking with respect to `xmem::PHYSICAL`
/// (`try_lock`) so it can never invert lock order with `evict_range`.
pub fn use_alias_writer(writer: AliasWriter) {
    ALIAS_WRITER.store(writer as *mut (), Ordering::Release);
}

#[inline]
fn try_alias_writer(block_va: usize, data: &[u8; BLOCK]) -> bool {
    let raw = ALIAS_WRITER.load(Ordering::Acquire);
    if raw.is_null() {
        return false;
    }
    // SAFETY: stored from an `AliasWriter` function pointer in `use_alias_writer`.
    let f: AliasWriter = unsafe { std::mem::transmute(raw) };
    f(block_va, data)
}

/// Per-interval and cumulative counters:
/// `[0]` promotions (`CpuResident -> GpuResident`),
/// `[1]` repeat GPU fence updates (0 `VirtualProtect`),
/// `[2]` VEH read faults serviced,
/// `[3]` VEH write faults serviced,
/// `[4]` internal `ensure_cpu_resident` / partial-edge fetches,
/// `[5]` coalesced fault waits,
/// `[6]` full-block `prepare_cpu_overwrite` drops (0 readback),
/// `[7]` `evict_range` drops (0 readback),
/// `[8]` `flush_shadow` flushes,
/// `[9]` blocks demoted to `CpuHot`,
/// `[10]` fetch microseconds total.
static STATS: [AtomicU64; 11] = [const { AtomicU64::new(0) }; 11];
/// Per-interval `plan_gpu_resident` rejection counters:
/// `[0]` out-of-shadow-span, `[1]` class ineligible (e.g. `0x2004` in mode 1),
/// `[2]` multi-view aliased (`inplace::pages_of.len() >= 2`),
/// `[3]` `CpuHot` / `Fetching`, `[4]` `!is_committable_block`.
static REJ: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];
/// Currently active `GpuResident` block count.
static LIVE_BLOCKS: AtomicU64 = AtomicU64::new(0);
/// Cumulative blocks demoted to `CpuHot`.
static TOTAL_DEMOTED: AtomicU64 = AtomicU64::new(0);

#[inline]
fn stat(idx: usize, delta: u64) {
    STATS[idx].fetch_add(delta, Ordering::Relaxed);
}

#[inline]
fn rej(idx: usize) {
    REJ[idx].fetch_add(1, Ordering::Relaxed);
}

struct ResidentInfo {
    shadow_id: u64,
    shadow: Option<Weak<Shadow>>,
    gpu_offset: u64,
    last_fence: u64,
}

enum BlockState {
    GpuResident(ResidentInfo),
    Fetching {
        shadow_id: u64,
        any_write: bool,
        epoch: u64,
    },
    CpuHot,
}

struct State {
    blocks: HashMap<usize, BlockState>,
    fault_counts: HashMap<usize, u8>,
    next_gen: u64,
}

impl State {
    fn sync_tracked(&self) {
        TRACKED_COUNT.store(self.blocks.len() + self.fault_counts.len(), Ordering::Relaxed);
    }
}

static STATE: OnceLock<(Mutex<State>, Condvar)> = OnceLock::new();

fn state() -> &'static (Mutex<State>, Condvar) {
    STATE.get_or_init(|| {
        (
            Mutex::new(State {
                blocks: HashMap::new(),
                fault_counts: HashMap::new(),
                next_gen: 1,
            }),
            Condvar::new(),
        )
    })
}

thread_local! {
    /// Prevents VEH self-reentrancy while this thread is inside `handle_fault`
    /// or writing fetched bytes back to title RAM.
    static IN_LAZY: Cell<bool> = const { Cell::new(false) };
}

struct LazyGuard(bool);
impl LazyGuard {
    fn enter() -> Option<Self> {
        IN_LAZY.with(|c| {
            if c.get() {
                None
            } else {
                c.set(true);
                Some(Self(true))
            }
        })
    }
}
impl Drop for LazyGuard {
    fn drop(&mut self) {
        if self.0 {
            IN_LAZY.with(|c| c.set(false));
        }
    }
}

/// Request sent to the dedicated `victory-lazy-fetch` D3D12 COPY queue worker.
struct FetchReq {
    buffer: ID3D12Resource,
    gpu_offset: u64,
    last_fence: u64,
    reply: mpsc::SyncSender<Option<Box<[u8; BLOCK]>>>,
}

// SAFETY: `ID3D12Resource` COM interfaces are free-threaded and owned until the
// COPY command list completes and drops the request.
unsafe impl Send for FetchReq {}

struct FetcherHandle {
    tx: mpsc::Sender<FetchReq>,
}

static FETCHER: OnceLock<Option<FetcherHandle>> = OnceLock::new();

#[cfg(test)]
type TestFetcherFn = Arc<dyn Fn(usize, u64, u64) -> Option<Box<[u8; BLOCK]>> + Send + Sync>;
#[cfg(test)]
static TEST_FETCHERS: OnceLock<Mutex<HashMap<usize, TestFetcherFn>>> = OnceLock::new();

#[cfg(test)]
fn test_fetchers() -> &'static Mutex<HashMap<usize, TestFetcherFn>> {
    TEST_FETCHERS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn has_live_fetcher() -> bool {
    FETCHER.get().is_some_and(Option::is_some)
}

/// Initializes the dedicated `victory-lazy-fetch` `D3D12_COMMAND_LIST_TYPE_COPY`
/// queue, allocator, command list, 64 KiB persistently mapped `READBACK` buffer,
/// fence, and worker thread using `Host`'s device and main graphics fence.
/// Idempotent; no-op when `VICTORY_LAZY_READBACK=0`.
pub(crate) fn init(device: &ID3D12Device, main_fence: ID3D12Fence) {
    if !on() {
        return;
    }
    FETCHER.get_or_init(|| match spawn_worker(device, main_fence) {
        Ok(handle) => Some(handle),
        Err(e) => {
            tracing::warn!(error = %e, "victory-lazy-fetch COPY queue init failed; lazy readback inactive");
            None
        }
    });
}

struct SendCom<T>(T);
impl<T> SendCom<T> {
    fn into_inner(self) -> T {
        self.0
    }
}
// SAFETY: D3D12 device/queue/allocator/list/resource/fence objects are
// apartment-neutral free-threaded COM interfaces transferred once to the
// dedicated `victory-lazy-fetch` thread.
unsafe impl<T> Send for SendCom<T> {}

struct SendHandle(HANDLE);
impl SendHandle {
    fn into_inner(self) -> HANDLE {
        self.0
    }
}
// SAFETY: Win32 event handle owned exclusively by the worker thread.
unsafe impl Send for SendHandle {}

fn spawn_worker(device: &ID3D12Device, main_fence: ID3D12Fence) -> windows::core::Result<FetcherHandle> {
    // SAFETY: standard D3D12 COPY queue, allocator, closed list, 64 KiB readback
    // buffer, fence, and Win32 auto-reset event creation.
    let (rb_queue, rb_alloc, rb_list, rb_buffer, rb_ptr, rb_fence, rb_event) = unsafe {
        let rb_queue: ID3D12CommandQueue = device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
            Type: D3D12_COMMAND_LIST_TYPE_COPY,
            ..Default::default()
        })?;
        let rb_alloc: ID3D12CommandAllocator =
            device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY)?;
        let rb_list: ID3D12GraphicsCommandList = device.CreateCommandList(
            0,
            D3D12_COMMAND_LIST_TYPE_COPY,
            &rb_alloc,
            None,
        )?;
        rb_list.Close()?;
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_READBACK,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: BLOCK as u64,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut rb_buffer: Option<ID3D12Resource> = None;
        device.CreateCommittedResource(
            &heap,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_COPY_DEST,
            None,
            &mut rb_buffer,
        )?;
        let rb_buffer = rb_buffer.ok_or_else(windows::core::Error::empty)?;
        let mut ptr = std::ptr::null_mut();
        rb_buffer.Map(0, None, Some(&mut ptr))?;
        let rb_fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
        let rb_event = CreateEventW(None, false, false, None)?;
        (
            SendCom(rb_queue),
            SendCom(rb_alloc),
            SendCom(rb_list),
            SendCom(rb_buffer),
            ptr as usize,
            SendCom(rb_fence),
            SendHandle(rb_event),
        )
    };
    let main_fence = SendCom(main_fence);
    let (tx, rx) = mpsc::channel::<FetchReq>();
    let spawned = std::thread::Builder::new()
        .name("victory-lazy-fetch".to_owned())
        .spawn(move || {
            let rb_queue = rb_queue.into_inner();
            let rb_alloc = rb_alloc.into_inner();
            let rb_list = rb_list.into_inner();
            let rb_buffer = rb_buffer.into_inner();
            let rb_fence = rb_fence.into_inner();
            let rb_event = rb_event.into_inner();
            let main_fence = main_fence.into_inner();
            let rb_ptr = rb_ptr as *const u8;
            let mut next_val = 0u64;

            while let Ok(req) = rx.recv() {
                let res = (|| -> windows::core::Result<Box<[u8; BLOCK]>> {
                    // SAFETY: `rb_queue` is privately owned by this thread; `req.buffer`
                    // is kept alive by `req` until after `WaitForSingleObject` completes.
                    unsafe {
                        let max_sig = SIGNALED_FENCE.load(Ordering::Acquire);
                        let wait_fence = if req.last_fence > max_sig {
                            tracing::error!(
                                requested = req.last_fence,
                                signaled = max_sig,
                                offset = req.gpu_offset,
                                "victory-lazy-fetch clamped unsignaled future fence to prevent self-deadlock"
                            );
                            debug_assert!(
                                req.last_fence <= max_sig,
                                "lazy_readback requested wait on unsignaled fence {} > {}",
                                req.last_fence,
                                max_sig
                            );
                            max_sig
                        } else {
                            req.last_fence
                        };
                        if wait_fence > 0 && main_fence.GetCompletedValue() < wait_fence {
                            rb_queue.Wait(&main_fence, wait_fence)?;
                        }
                        rb_alloc.Reset()?;
                        rb_list.Reset(&rb_alloc, None)?;
                        rb_list.CopyBufferRegion(
                            &rb_buffer,
                            0,
                            &req.buffer,
                            req.gpu_offset,
                            BLOCK as u64,
                        );
                        rb_list.Close()?;
                        let lists = [Some(rb_list.cast::<ID3D12CommandList>()?)];
                        rb_queue.ExecuteCommandLists(&lists);
                        next_val += 1;
                        rb_queue.Signal(&rb_fence, next_val)?;
                        rb_fence.SetEventOnCompletion(next_val, rb_event)?;
                        WaitForSingleObject(rb_event, INFINITE);
                        let mut out = Box::new([0u8; BLOCK]);
                        std::ptr::copy_nonoverlapping(rb_ptr, out.as_mut_ptr(), BLOCK);
                        Ok(out)
                    }
                })();
                match res {
                    Ok(bytes) => {
                        let _ = req.reply.send(Some(bytes));
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, offset = req.gpu_offset, "victory-lazy-fetch GPU copy failed");
                        let _ = req.reply.send(None);
                    }
                }
            }
            // SAFETY: worker owns `rb_buffer` mapping and `rb_event`.
            unsafe {
                rb_buffer.Unmap(0, None);
                let _ = CloseHandle(rb_event);
            }
        });
    if let Err(e) = spawned {
        tracing::warn!(error = %e, "could not spawn victory-lazy-fetch thread");
        return Err(windows::core::Error::empty());
    }
    Ok(FetcherHandle { tx })
}

fn fetch_from_gpu(
    block: usize,
    buffer: Option<ID3D12Resource>,
    gpu_offset: u64,
    last_fence: u64,
) -> Option<Box<[u8; BLOCK]>> {
    #[cfg(test)]
    if let Some(f) = test_fetchers()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(&block)
        .cloned()
    {
        return f(block, gpu_offset, last_fence);
    }
    let _ = block;
    let buffer = buffer?;
    let handle = FETCHER.get().and_then(|o| o.as_ref())?;
    let (reply_tx, reply_rx) = mpsc::sync_channel(1);
    handle
        .tx
        .send(FetchReq {
            buffer,
            gpu_offset,
            last_fence,
            reply: reply_tx,
        })
        .ok()?;
    reply_rx.recv().ok().flatten()
}

/// Checks whether `[block_va, block_va + BLOCK)` is uniformly classified under
/// an eligible `ConsoleMemClass` for the active `VICTORY_LAZY_READBACK` mode.
fn block_class_eligible(block_va: usize, m: Mode) -> bool {
    let parts = crate::watch::console_mem_class_ranges(block_va, block_va.saturating_add(BLOCK));
    !parts.is_empty() && parts.iter().all(|&(_, _, c)| class_eligible_in(c, m))
}

/// Checks (read-only) whether `[block_va, block_va + BLOCK)` is a committed
/// read-write or read-only 64 KiB title block eligible for `PAGE_NOACCESS`.
fn is_committable_block(block_va: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_READONLY, PAGE_READWRITE, VirtualQuery,
    };
    // SAFETY: plain out struct for `VirtualQuery`.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: valid pointer and struct size.
    let n = unsafe { VirtualQuery(block_va as *const _, &mut info, size_of_val(&info)) };
    if n == 0 || info.State != MEM_COMMIT {
        return false;
    }
    let region_end = (info.BaseAddress as usize).saturating_add(info.RegionSize);
    region_end >= block_va.saturating_add(BLOCK)
        && matches!(info.Protect, PAGE_READWRITE | PAGE_READONLY)
}

/// Checks whether `[block_va, block_va + BLOCK)` is a committed read-write or
/// read-only title page (not uncommitted/free/noaccess) and flips it to
/// `PAGE_NOACCESS`. Returns `true` on success.
fn protect_noaccess(block_va: usize) -> bool {
    use windows_sys::Win32::System::Memory::{PAGE_NOACCESS, VirtualProtect};
    if !is_committable_block(block_va) {
        return false;
    }
    let mut old = 0u32;
    // SAFETY: verified committed 64 KiB title block.
    unsafe { VirtualProtect(block_va as *const _, BLOCK, PAGE_NOACCESS, &mut old) != 0 }
}

fn protect_readwrite(block_va: usize) -> bool {
    use windows_sys::Win32::System::Memory::{PAGE_READWRITE, VirtualProtect};
    let mut old = 0u32;
    // SAFETY: restoring read-write access to a 64 KiB title block we previously set `PAGE_NOACCESS`.
    unsafe { VirtualProtect(block_va as *const _, BLOCK, PAGE_READWRITE, &mut old) != 0 }
}

/// A 64 KiB block validated in Phase 1 ([`plan_gpu_resident`], inside
/// `Host::card_before`) and ready to be armed/updated in Phase 2
/// ([`commit_gpu_resident`]) once `Host::queue.Signal(&Host::fence, F)` has
/// succeeded.
pub(crate) struct PendingClaim {
    pub(crate) block: usize,
    pub(crate) shadow: Arc<Shadow>,
    pub(crate) gpu_offset: u64,
}

/// **Phase 1 (plan)** of 2-phase lazy readback claiming, called inside
/// `Host::card_before()` BEFORE `card_dirty()` / `copy_dirty_to_ring()` and
/// BEFORE `Host::queue.Signal(&Host::fence, F)`.
///
/// Validates every 64 KiB block in `ranges` against all safety filters and
/// returns `(pending_claims, remaining_ranges)` WITHOUT mutating `RESIDENT`,
/// WITHOUT calling `VirtualProtect(PAGE_NOACCESS)`, and WITHOUT storing an
/// unsignaled future fence in `last_fence` — eliminating any possibility of
/// self-deadlock when `card_dirty` uploads pending CPU bytes of the same block
/// in the same submission.
pub(crate) fn plan_gpu_resident(
    shadow: &Arc<Shadow>,
    ranges: &[(usize, usize)],
) -> (Vec<PendingClaim>, Vec<(usize, usize)>) {
    let m = active_mode();
    if m == Mode::Off
        || ranges.is_empty()
        || !shadow.carded()
        || shadow.in_place()
        || !has_live_fetcher()
    {
        return (Vec::new(), ranges.to_vec());
    }
    let (span_start_u64, span_len_u64) = shadow.span();
    let span_start = span_start_u64 as usize;
    let span_end = span_start.saturating_add(span_len_u64 as usize);

    let (lock, _cv) = state();
    let mut pending: Vec<PendingClaim> = Vec::new();
    let mut remaining: Vec<(usize, usize)> = Vec::new();
    let push_rem = |out: &mut Vec<(usize, usize)>, lo: usize, hi: usize| {
        if lo >= hi {
            return;
        }
        if let Some(last) = out.last_mut()
            && last.1 == lo
        {
            last.1 = hi;
        } else {
            out.push((lo, hi));
        }
    };

    for &(from, to) in ranges {
        if from >= to {
            continue;
        }
        let first_block = from / BLOCK;
        let last_block = to.div_ceil(BLOCK);
        for block in first_block..last_block {
            let b_lo = block * BLOCK;
            let b_hi = b_lo.saturating_add(BLOCK);
            let sub_lo = from.max(b_lo);
            let sub_hi = to.min(b_hi);
            if b_lo < span_start || b_hi > span_end {
                rej(0);
                push_rem(&mut remaining, sub_lo, sub_hi);
                continue;
            }
            if !block_class_eligible(b_lo, m) {
                rej(1);
                push_rem(&mut remaining, sub_lo, sub_hi);
                continue;
            }
            // Skip multi-view aliased blocks so CPU access via another alias
            // VA can never bypass `PAGE_NOACCESS`.
            if crate::inplace::pages_of(b_lo).len() >= 2 {
                rej(2);
                push_rem(&mut remaining, sub_lo, sub_hi);
                continue;
            }
            let gpu_offset = (b_lo - span_start) as u64;
            let eligible = {
                let mut s = lock.lock().unwrap_or_else(|e| e.into_inner());
                match s.blocks.get(&block) {
                    Some(BlockState::GpuResident(_)) => Some(true),
                    Some(BlockState::CpuHot) | Some(BlockState::Fetching { .. }) => {
                        rej(3);
                        Some(false)
                    }
                    None => {
                        if s.fault_counts.get(&block).copied().unwrap_or(0) >= DEMOTE_THRESHOLD {
                            s.blocks.insert(block, BlockState::CpuHot);
                            s.sync_tracked();
                            rej(3);
                            Some(false)
                        } else {
                            None
                        }
                    }
                }
            };
            let take = match eligible {
                Some(v) => v,
                None => {
                    let ok = is_committable_block(b_lo);
                    if !ok {
                        rej(4);
                    }
                    ok
                }
            };
            if take {
                if pending.last().is_none_or(|c| c.block != block) {
                    pending.push(PendingClaim {
                        block,
                        shadow: shadow.clone(),
                        gpu_offset,
                    });
                }
            } else {
                push_rem(&mut remaining, sub_lo, sub_hi);
            }
        }
    }
    (pending, remaining)
}

/// **Phase 2 (commit)** of 2-phase lazy readback claiming, called IMMEDIATELY
/// AFTER `Host::queue.Signal(&Host::fence, signaled_fence)` succeeds (and after
/// [`note_signaled_fence`]). Arms `PAGE_NOACCESS` / `RESIDENT` on newly
/// promoted blocks and advances `last_fence` in O(1) on already-`GpuResident`
/// blocks. Returns `(block, shadow_id)` for every committed block.
pub(crate) fn commit_gpu_resident(
    claims: Vec<PendingClaim>,
    signaled_fence: u64,
) -> Vec<(usize, u64)> {
    if claims.is_empty() {
        return Vec::new();
    }
    // Ensure `watch::on_write` VEH is installed ahead of any `PAGE_NOACCESS`.
    crate::watch::install();

    let (lock, _cv) = state();
    let mut committed = Vec::with_capacity(claims.len());
    for claim in claims {
        let PendingClaim {
            block,
            shadow,
            gpu_offset,
        } = claim;
        let shadow_id = shadow.id();
        let b_lo = block * BLOCK;
        let mut s = lock.lock().unwrap_or_else(|e| e.into_inner());
        match s.blocks.get_mut(&block) {
            Some(BlockState::GpuResident(info)) => {
                // O(1) steady-state fast path: 0 `VirtualProtect` syscalls!
                info.shadow_id = shadow_id;
                info.shadow = Some(Arc::downgrade(&shadow));
                info.gpu_offset = gpu_offset;
                info.last_fence = info.last_fence.max(signaled_fence);
                stat(1, 1);
                committed.push((block, shadow_id));
            }
            Some(BlockState::CpuHot) | Some(BlockState::Fetching { .. }) => {
                // Racing thread faulted/demoted between plan and commit; leave intact.
            }
            None => {
                if s.fault_counts.get(&block).copied().unwrap_or(0) >= DEMOTE_THRESHOLD {
                    s.blocks.insert(block, BlockState::CpuHot);
                    s.sync_tracked();
                    continue;
                }
                // Release any `watch` read-only guard on `block` before
                // setting `PAGE_NOACCESS` (strict lock order: `STATE -> GUARDS`
                // never reverses because `GUARDS` never locks `STATE`).
                crate::watch::release_guard_for_lazy(block);
                // Mark `RESIDENT` BEFORE `VirtualProtect(PAGE_NOACCESS)` so any
                // racing VEH fault on another thread sees `RESIDENT == 1`.
                RESIDENT.set(block);
                if protect_noaccess(b_lo) {
                    s.blocks.insert(
                        block,
                        BlockState::GpuResident(ResidentInfo {
                            shadow_id,
                            shadow: Some(Arc::downgrade(&shadow)),
                            gpu_offset,
                            last_fence: signaled_fence,
                        }),
                    );
                    s.sync_tracked();
                    LIVE_BLOCKS.fetch_add(1, Ordering::Relaxed);
                    stat(0, 1);
                    committed.push((block, shadow_id));
                } else {
                    RESIDENT.clear(block);
                }
            }
        }
    }
    committed
}

/// Convenience wrapper for unit tests that have already signaled `fence`.
#[cfg(test)]
pub(crate) fn claim_gpu_resident(
    shadow: &Arc<Shadow>,
    ranges: &[(usize, usize)],
    fence: u64,
) -> Vec<(usize, usize)> {
    let (claims, remaining) = plan_gpu_resident(shadow, ranges);
    let _ = commit_gpu_resident(claims, fence);
    remaining
}

/// Writes 64 KiB `payload` back into title memory `[block_va, block_va + BLOCK)`,
/// restoring `PAGE_READWRITE`, and resets `GetWriteWatch` on `MEM_PRIVATE` pure
/// read faults so clean GPU bytes are not spuriously re-uploaded.
fn restore_title_block(block_va: usize, payload: &[u8; BLOCK], any_write: bool) {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_MAPPED, MEM_PRIVATE, MEMORY_BASIC_INFORMATION, ResetWriteWatch,
        VirtualQuery,
    };
    // SAFETY: plain out struct.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: valid pointer and struct size.
    let n = unsafe { VirtualQuery(block_va as *const _, &mut info, size_of_val(&info)) };
    if n == 0 || info.State != MEM_COMMIT {
        return;
    }

    if info.Type == MEM_MAPPED {
        // Section-backed view: populate physical RAM while title VA is STILL
        // `PAGE_NOACCESS` (zero torn-read window), then flip to `PAGE_READWRITE`.
        let mut wrote_beside = false;
        if let Some(_reading) = crate::watch::title_memory_reading_now() {
            if let Some(alias_ptr) = crate::inplace::beside(block_va, BLOCK) {
                if crate::safecopy::on() {
                    // SAFETY: `payload` is valid 64 KiB; `safecopy` returns false
                    // cleanly if the section page is uncommitted/stale.
                    wrote_beside = unsafe {
                        crate::safecopy::copy(alias_ptr as *mut u8, payload.as_ptr(), BLOCK)
                    };
                } else {
                    // SAFETY: `alias_ptr` is a writable persistent view of the same
                    // 64 KiB section pages held stable by `_reading`.
                    unsafe {
                        std::ptr::copy_nonoverlapping(payload.as_ptr(), alias_ptr as *mut u8, BLOCK);
                    }
                    wrote_beside = true;
                }
            }
        }
        if !wrote_beside {
            wrote_beside = try_alias_writer(block_va, payload);
        }
        if wrote_beside {
            protect_readwrite(block_va);
            return;
        }
        // Fallback if section cannot be aliased right now.
        if protect_readwrite(block_va) {
            if crate::safecopy::on() {
                // SAFETY: `payload` is valid 64 KiB; `safecopy` fails cleanly if unmapped.
                unsafe {
                    let _ = crate::safecopy::copy(block_va as *mut u8, payload.as_ptr(), BLOCK);
                }
            } else {
                // SAFETY: `block_va` verified committed and switched to `PAGE_READWRITE`.
                unsafe {
                    std::ptr::copy_nonoverlapping(payload.as_ptr(), block_va as *mut u8, BLOCK);
                }
            }
        }
        return;
    }

    // `MEM_PRIVATE` (including `MEM_WRITE_WATCH`): flip `PAGE_READWRITE` and
    // copy all 64 KiB.
    if protect_readwrite(block_va) {
        if crate::safecopy::on() {
            // SAFETY: `payload` is valid 64 KiB; `safecopy` fails cleanly if unmapped.
            unsafe {
                let _ = crate::safecopy::copy(block_va as *mut u8, payload.as_ptr(), BLOCK);
            }
        } else {
            // SAFETY: `block_va` verified committed and switched to `PAGE_READWRITE`.
            unsafe {
                std::ptr::copy_nonoverlapping(payload.as_ptr(), block_va as *mut u8, BLOCK);
            }
        }
        if info.Type == MEM_PRIVATE && !any_write {
            // Clear dirty bits produced by our own restore copy on pure read
            // faults so `Shadow::update` does not re-upload unchanged GPU bytes.
            // On write faults (or any coalesced writer), skip `ResetWriteWatch`
            // so the retried store is guaranteed dirty.
            // SAFETY: valid committed `MEM_PRIVATE` range (harmless failure if not `MEM_WRITE_WATCH`).
            unsafe {
                let _ = ResetWriteWatch(block_va as *mut _, BLOCK);
            }
        }
    }
}

/// Core state-machine materialization of one 64 KiB block (`GpuResident -> CpuResident`
/// or `CpuHot`). Returns `true` if `block` belonged to lazy readback (was
/// `GpuResident` or `Fetching`) and is now restored to `PAGE_READWRITE`.
fn materialize_block(
    block: usize,
    is_write: bool,
    count_fault: bool,
    from_veh: bool,
    direct_buffer: Option<(u64, ID3D12Resource)>,
) -> bool {
    let Some(_guard) = LazyGuard::enter() else {
        return false;
    };
    let (lock, cv) = state();
    let mut dead_shadow_waits = 0u32;

    loop {
        let work = {
            let mut s = lock.lock().unwrap_or_else(|e| e.into_inner());
            match s.blocks.get_mut(&block) {
                Some(BlockState::Fetching { any_write, .. }) => {
                    if is_write {
                        *any_write = true;
                    }
                    stat(5, 1);
                    while matches!(s.blocks.get(&block), Some(BlockState::Fetching { .. })) {
                        s = cv.wait(s).unwrap_or_else(|e| e.into_inner());
                    }
                    return true;
                }
                Some(BlockState::GpuResident(info)) => {
                    let (shadow_arc, buffer_opt) = if let Some((sid, ref buf)) = direct_buffer {
                        if info.shadow_id != sid {
                            return false;
                        }
                        (None, Some(buf.clone()))
                    } else if let Some(ref weak) = info.shadow {
                        match weak.upgrade() {
                            Some(arc) => {
                                let buf = arc.buffer.clone();
                                (Some(arc), Some(buf))
                            }
                            None => {
                                // `Arc<Shadow>` hit 0; `Shadow::drop` is about to call
                                // `flush_shadow(self)` which will fetch & restore this block.
                                if dead_shadow_waits < 10 {
                                    dead_shadow_waits += 1;
                                    let _ = cv
                                        .wait_timeout(s, Duration::from_millis(5))
                                        .unwrap_or_else(|e| e.into_inner());
                                    continue;
                                }
                                // Safety fallback if shadow was dropped without flushing.
                                s.blocks.remove(&block);
                                s.sync_tracked();
                                LIVE_BLOCKS.fetch_sub(1, Ordering::Relaxed);
                                protect_readwrite(block * BLOCK);
                                RESIDENT.clear(block);
                                cv.notify_all();
                                return true;
                            }
                        }
                    } else {
                        // Test-mode entry without real D3D12 `Shadow`.
                        (None, None)
                    };
                    let gpu_offset = info.gpu_offset;
                    let last_fence = info.last_fence;
                    let shadow_id = info.shadow_id;
                    let epoch = s.next_gen;
                    s.next_gen = s.next_gen.wrapping_add(1).max(1);
                    s.blocks.insert(
                        block,
                        BlockState::Fetching {
                            shadow_id,
                            any_write: is_write,
                            epoch,
                        },
                    );
                    LIVE_BLOCKS.fetch_sub(1, Ordering::Relaxed);
                    (shadow_arc, buffer_opt, gpu_offset, last_fence, epoch)
                }
                Some(BlockState::CpuHot) | None => return false,
            }
        };

        let (shadow_arc, buffer_opt, gpu_offset, last_fence, my_epoch) = work;
        let started = Instant::now();
        let fetched = fetch_from_gpu(block, buffer_opt, gpu_offset, last_fence);
        let us = started.elapsed().as_micros() as u64;
        stat(10, us);
        if from_veh {
            stat(if is_write { 3 } else { 2 }, 1);
        } else if direct_buffer.is_some() {
            stat(8, 1);
        } else {
            stat(4, 1);
        }

        // Keep `Card::staging` coherent when still allocated (Move 3 Stage 3A
        // may leave `cpu_data()` null on stagingless shadows).
        if let (Some(arc), Some(payload)) = (&shadow_arc, &fetched) {
            let dst = arc.cpu_data() as *mut u8;
            if !dst.is_null() {
                let (_, span_len) = arc.span();
                if gpu_offset.saturating_add(BLOCK as u64) <= span_len {
                    // SAFETY: `arc` keeps `Card::staging` mapped and alive for `span_len` bytes.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            payload.as_ptr(),
                            dst.add(gpu_offset as usize),
                            BLOCK,
                        );
                    }
                }
            }
        }

        // Verify this `Fetching` epoch was NOT evicted by `evict_range`
        // while the GPU readback was in flight, and read final `any_write`
        // (which covers any coalesced writer currently blocked on `Fetching`).
        let still_ours = {
            let s = lock.lock().unwrap_or_else(|e| e.into_inner());
            match s.blocks.get(&block) {
                Some(&BlockState::Fetching { any_write, epoch, .. }) if epoch == my_epoch => {
                    Some(any_write)
                }
                _ => None,
            }
        };
        let Some(final_any_write) = still_ours else {
            // `evict_range` already unmapped/remapped this VA: do NOT touch it.
            return true;
        };

        let block_va = block * BLOCK;
        if let Some(payload) = fetched {
            restore_title_block(block_va, &payload, final_any_write);
        } else {
            protect_readwrite(block_va);
        }

        // Complete transition under `STATE` lock and wake all coalesced waiters.
        {
            let mut s = lock.lock().unwrap_or_else(|e| e.into_inner());
            let still_current = matches!(
                s.blocks.get(&block),
                Some(BlockState::Fetching { epoch, .. }) if *epoch == my_epoch
            );
            if still_current {
                if count_fault {
                    let count = s.fault_counts.entry(block).or_insert(0);
                    *count = count.saturating_add(1);
                    if *count >= DEMOTE_THRESHOLD {
                        s.blocks.insert(block, BlockState::CpuHot);
                        stat(9, 1);
                        TOTAL_DEMOTED.fetch_add(1, Ordering::Relaxed);
                    } else {
                        s.blocks.remove(&block);
                    }
                } else {
                    s.blocks.remove(&block);
                }
                s.sync_tracked();
            }
            RESIDENT.clear(block);
            cv.notify_all();
        }
        return true;
    }
}

/// VEH entry point called from `watch::on_write` **BEFORE**
/// `safecopy::failing_exit` for BOTH read (`info[0] == 0`) and write
/// (`info[0] == 1`) access violations.
///
/// Returns `true` if `addr` hit a `GpuResident` / `Fetching` block and has now
/// been fetched and restored to `PAGE_READWRITE` (`EXCEPTION_CONTINUE_EXECUTION`).
pub(crate) fn handle_fault(addr: usize, is_write: bool) -> bool {
    let block = addr / BLOCK;
    // Lock-free 1-ns fast reject for all non-`GpuResident` faults.
    if !RESIDENT.contains(block) {
        return false;
    }
    materialize_block(block, is_write, true, true, None)
}

/// Synchronously materializes any `GpuResident` blocks overlapping `[at, at + len)`
/// back to CPU RAM (`PAGE_READWRITE`). Lock-free O(1) no-op if none overlap.
pub(crate) fn ensure_cpu_resident(at: usize, len: usize) {
    if !any_gpu_resident(at, len) {
        return;
    }
    let first = at / BLOCK;
    let last = at.saturating_add(len).div_ceil(BLOCK);
    for block in first..last {
        if RESIDENT.contains(block) {
            materialize_block(block, false, true, false, None);
        }
    }
}

/// Prepares `[at, at + len)` for a CPU or kernel overwrite (`ReadFile`,
/// DirectStorage, `xmem::write_physical`, `watch::writing`):
/// - Full 64 KiB aligned blocks completely covered by `[at, at + len)` drop
///   `GpuResident` and flip to `PAGE_READWRITE` with **zero GPU readback**.
/// - Partial leading/trailing edge blocks fetch 64 KiB from GPU first so the
///   un-overwritten bytes of the block remain intact.
///
/// Lock-free O(1) no-op if no `GpuResident` blocks overlap `[at, at + len)`.
pub fn prepare_cpu_overwrite(at: usize, len: usize) {
    if !any_gpu_resident(at, len) {
        return;
    }
    let end = at.saturating_add(len);
    let first = at / BLOCK;
    let last = end.div_ceil(BLOCK);
    let (lock, cv) = state();

    for block in first..last {
        if !RESIDENT.contains(block) {
            continue;
        }
        let b_lo = block * BLOCK;
        let b_hi = b_lo.saturating_add(BLOCK);
        let full_block = at <= b_lo && end >= b_hi;
        if !full_block {
            // Partial overwrite: fetch 64 KiB first (marked `is_write = true`
            // so `ResetWriteWatch` is NOT called and the overwrite stays dirty).
            materialize_block(block, true, false, false, None);
            continue;
        }
        // Full 64 KiB block overwrite: drop `GpuResident` with ZERO GPU readback!
        let mut s = lock.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            match s.blocks.get(&block) {
                Some(BlockState::Fetching { .. }) => {
                    s = cv.wait(s).unwrap_or_else(|e| e.into_inner());
                }
                Some(BlockState::GpuResident(_)) => {
                    s.blocks.remove(&block);
                    s.sync_tracked();
                    LIVE_BLOCKS.fetch_sub(1, Ordering::Relaxed);
                    protect_readwrite(b_lo);
                    RESIDENT.clear(block);
                    stat(6, 1);
                    cv.notify_all();
                    break;
                }
                Some(BlockState::CpuHot) | None => {
                    RESIDENT.clear(block);
                    break;
                }
            }
        }
    }
}

/// Evicts all `GpuResident`, `Fetching`, `CpuHot`, and fault-count state in
/// `[at, at + len)` with **zero GPU readback** and **zero blocking** (safe to
/// call from `watch::title_memory_remapped` under `PHYSICAL -> MAPPING`).
pub(crate) fn evict_range(at: usize, len: usize) {
    if len == 0 || TRACKED_COUNT.load(Ordering::Relaxed) == 0 {
        return;
    }
    let first = at / BLOCK;
    let last = at.saturating_add(len).div_ceil(BLOCK);
    let (lock, cv) = state();
    let mut s = lock.lock().unwrap_or_else(|e| e.into_inner());
    let mut evicted = 0u64;
    let mut woke = false;
    for block in first..last {
        s.fault_counts.remove(&block);
        if let Some(prev) = s.blocks.remove(&block) {
            match prev {
                BlockState::GpuResident(_) => {
                    LIVE_BLOCKS.fetch_sub(1, Ordering::Relaxed);
                    RESIDENT.clear(block);
                    evicted += 1;
                }
                BlockState::Fetching { .. } => {
                    RESIDENT.clear(block);
                    evicted += 1;
                    woke = true;
                }
                BlockState::CpuHot => {}
            }
        }
    }
    s.sync_tracked();
    if evicted > 0 {
        stat(7, evicted);
    }
    if woke {
        cv.notify_all();
    }
}

/// Flushes any still-live `GpuResident` blocks owned by `shadow` back to CPU RAM
/// (and waits for any in-flight `Fetching` referencing `shadow`) before
/// `Shadow::drop` releases `shadow.buffer`.
pub(crate) fn flush_shadow(shadow: &Shadow) {
    if TRACKED_COUNT.load(Ordering::Relaxed) == 0 {
        return;
    }
    let shadow_id = shadow.id();
    let (lock, cv) = state();
    loop {
        let mut to_flush: Vec<usize> = Vec::new();
        {
            let s = lock.lock().unwrap_or_else(|e| e.into_inner());
            let mut has_fetching = false;
            for (&block, st) in &s.blocks {
                match st {
                    BlockState::GpuResident(info) if info.shadow_id == shadow_id => {
                        to_flush.push(block);
                    }
                    BlockState::Fetching { shadow_id: sid, .. } if *sid == shadow_id => {
                        has_fetching = true;
                    }
                    _ => {}
                }
            }
            if to_flush.is_empty() {
                if has_fetching {
                    drop(cv.wait(s).unwrap_or_else(|e| e.into_inner()));
                    continue;
                }
                return;
            }
        }
        to_flush.sort_unstable();
        for block in to_flush {
            materialize_block(
                block,
                false,
                false,
                false,
                Some((shadow_id, shadow.buffer.clone())),
            );
        }
    }
}

/// Periodic 5-second telemetry log (called from `Meter`).
pub(crate) fn log_periodic() {
    if !on() {
        return;
    }
    let p: [u64; 11] = std::array::from_fn(|i| STATS[i].swap(0, Ordering::Relaxed));
    let r: [u64; 5] = std::array::from_fn(|i| REJ[i].swap(0, Ordering::Relaxed));
    let live = LIVE_BLOCKS.load(Ordering::Relaxed);
    let total_demoted = TOTAL_DEMOTED.load(Ordering::Relaxed);
    if live == 0 && p.iter().all(|&v| v == 0) && r.iter().all(|&v| v == 0) && total_demoted == 0 {
        return;
    }
    let faults = p[2] + p[3];
    tracing::info!(
        "PC-native VRAM lazy readback (VICTORY_LAZY_READBACK={}): live {} blocks ({:.1} MiB), \
         promoted {}, repeat-fence-updates {} (0 VirtualProtect), VEH faults {} (read {}, write {}, coalesced {}), \
         internal fetches {}, overwrite-drops {} (0 readback), evicts {}, shadow-flushes {}, \
         demoted-CpuHot {} (cumulative {}), fetch {:.2} ms; \
         rejected [span={}, class={}, alias={}, cpuhot={}, uncommittable={}]",
        mode(),
        live,
        (live * BLOCK as u64) as f64 / (1024.0 * 1024.0),
        p[0],
        p[1],
        faults,
        p[2],
        p[3],
        p[5],
        p[4],
        p[6],
        p[7],
        p[8],
        p[9],
        total_demoted,
        p[10] as f64 / 1000.0,
        r[0],
        r[1],
        r[2],
        r[3],
        r[4],
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Barrier;
    use windows_sys::Win32::System::Memory::{
        GetWriteWatch, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc,
        VirtualFree,
    };

    const MEM_WRITE_WATCH: u32 = 0x20_0000;

    struct WatchPages {
        ptr: *mut u8,
        len: usize,
    }

    impl WatchPages {
        fn new(blocks: usize) -> Self {
            let len = blocks * BLOCK;
            // SAFETY: fresh `MEM_WRITE_WATCH` allocation aligned to 64 KiB, freed on Drop.
            let ptr = unsafe {
                VirtualAlloc(
                    std::ptr::null(),
                    len,
                    MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                    PAGE_READWRITE,
                )
            };
            assert!(!ptr.is_null());
            assert_eq!((ptr as usize) % BLOCK, 0);
            Self {
                ptr: ptr.cast(),
                len,
            }
        }

        fn va(&self) -> usize {
            self.ptr as usize
        }

        fn dirty_pages(&self) -> Vec<usize> {
            let mut addrs = vec![0usize; self.len / 4096];
            let mut count = addrs.len();
            let mut gran = 0u32;
            // SAFETY: `addrs` holds `count` entries for our `MEM_WRITE_WATCH` allocation.
            let rc = unsafe {
                GetWriteWatch(
                    1, // WRITE_WATCH_FLAG_RESET
                    self.ptr.cast(),
                    self.len,
                    addrs.as_mut_ptr().cast(),
                    &mut count,
                    &mut gran,
                )
            };
            assert_eq!(rc, 0);
            addrs.truncate(count);
            addrs
        }
    }

    impl Drop for WatchPages {
        fn drop(&mut self) {
            let base = self.va();
            evict_range(base, self.len);
            let mut fetchers = test_fetchers().lock().unwrap_or_else(|e| e.into_inner());
            for block in base / BLOCK..(base + self.len) / BLOCK {
                fetchers.remove(&block);
            }
            drop(fetchers);
            // SAFETY: releasing our test allocation.
            unsafe {
                VirtualFree(self.ptr.cast(), 0, MEM_RELEASE);
            }
        }
    }

    /// Promotes a test block to `GpuResident` (`PAGE_NOACCESS`) with an injected
    /// mock fetcher (returns `false` if block is `CpuHot`).
    fn promote_test_block(
        block_va: usize,
        shadow_id: u64,
        gpu_offset: u64,
        fence: u64,
        fetcher: TestFetcherFn,
    ) -> bool {
        crate::watch::install();
        let block = block_va / BLOCK;
        test_fetchers()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(block, fetcher);
        let (lock, _) = state();
        let mut s = lock.lock().unwrap_or_else(|e| e.into_inner());
        match s.blocks.get(&block) {
            Some(BlockState::CpuHot) => return false,
            Some(BlockState::Fetching { .. }) => return false,
            Some(BlockState::GpuResident(_)) => return true,
            None => {}
        }
        if s.fault_counts.get(&block).copied().unwrap_or(0) >= DEMOTE_THRESHOLD {
            s.blocks.insert(block, BlockState::CpuHot);
            s.sync_tracked();
            return false;
        }
        RESIDENT.set(block);
        assert!(protect_noaccess(block_va));
        s.blocks.insert(
            block,
            BlockState::GpuResident(ResidentInfo {
                shadow_id,
                shadow: None,
                gpu_offset,
                last_fence: fence,
            }),
        );
        s.sync_tracked();
        LIVE_BLOCKS.fetch_add(1, Ordering::Relaxed);
        true
    }

    fn is_cpu_hot(block: usize) -> bool {
        let (lock, _) = state();
        let s = lock.lock().unwrap_or_else(|e| e.into_inner());
        matches!(s.blocks.get(&block), Some(BlockState::CpuHot))
    }

    #[test]
    fn mode_parsing_defaults_and_overrides() {
        assert_eq!(resolve_mode(None), Mode::WriteCombined);
        assert_eq!(resolve_mode(Some("")), Mode::WriteCombined);
        assert_eq!(resolve_mode(Some("1")), Mode::WriteCombined);
        assert_eq!(resolve_mode(Some("wc")), Mode::WriteCombined);
        assert_eq!(resolve_mode(Some("2")), Mode::WriteCombinedAndCached);
        assert_eq!(resolve_mode(Some("cached")), Mode::WriteCombinedAndCached);
        assert_eq!(resolve_mode(Some("0")), Mode::Off);
        assert_eq!(resolve_mode(Some("off")), Mode::Off);
        assert_eq!(resolve_mode(Some("false")), Mode::Off);

        assert!(class_eligible_in(ConsoleMemClass::WriteCombinedGpuRw, Mode::WriteCombined));
        assert!(!class_eligible_in(ConsoleMemClass::CachedGpuRw, Mode::WriteCombined));
        assert!(!class_eligible_in(ConsoleMemClass::GpuReadOnly, Mode::WriteCombined));
        assert!(!class_eligible_in(ConsoleMemClass::CoherentGpuRw, Mode::WriteCombinedAndCached));
        assert!(class_eligible_in(ConsoleMemClass::CachedGpuRw, Mode::WriteCombinedAndCached));
    }

    #[test]
    fn cpu_read_fault_recovers_gpu_bytes_and_resets_write_watch() {
        let pages = WatchPages::new(1);
        let base = pages.va();
        // Reset initial commit dirty bits.
        let _ = pages.dirty_pages();

        let fetches = Arc::new(AtomicU64::new(0));
        let fetches_c = fetches.clone();
        let promoted = promote_test_block(
            base,
            101,
            0,
            7,
            Arc::new(move |_blk, _off, _fence| {
                fetches_c.fetch_add(1, Ordering::SeqCst);
                let mut data = Box::new([0u8; BLOCK]);
                for (i, b) in data.iter_mut().enumerate() {
                    *b = ((i * 17 + 3) & 0xff) as u8;
                }
                Some(data)
            }),
        );
        assert!(promoted);
        assert!(is_gpu_resident(base / BLOCK));

        // Real CPU read hits `PAGE_NOACCESS` -> VEH -> fetches 64 KiB -> resumes!
        // SAFETY: `base` is our 64 KiB `WatchPages` allocation.
        let sample = unsafe {
            let slice = std::slice::from_raw_parts(pages.ptr, BLOCK);
            (slice[0], slice[1234], slice[BLOCK - 1])
        };
        assert_eq!(sample.0, 3);
        assert_eq!(sample.1, ((1234 * 17 + 3) & 0xff) as u8);
        assert_eq!(sample.2, (((BLOCK - 1) * 17 + 3) & 0xff) as u8);
        assert_eq!(fetches.load(Ordering::SeqCst), 1);
        assert!(!is_gpu_resident(base / BLOCK));
        // Read fault must call `ResetWriteWatch` so restored GPU bytes are NOT dirty.
        assert!(
            pages.dirty_pages().is_empty(),
            "read fault restore must not leave GetWriteWatch dirty bits set"
        );
    }

    #[test]
    fn partial_4k_cpu_write_fault_preserves_other_60k_gpu_bytes() {
        let pages = WatchPages::new(1);
        let base = pages.va();
        let _ = pages.dirty_pages();

        let promoted = promote_test_block(
            base,
            102,
            0,
            9,
            Arc::new(move |_blk, _off, _fence| Some(Box::new([0xA5u8; BLOCK]))),
        );
        assert!(promoted);

        // Real CPU partial 4-byte write at offset 0x2000 hits `PAGE_NOACCESS`.
        // VEH must fetch all 64 KiB GPU bytes BEFORE retrying the 4-byte write!
        // SAFETY: offset 0x2000 is inside our 64 KiB allocation.
        unsafe {
            (pages.ptr.add(0x2000).cast::<u32>()).write_volatile(0x1122_3344);
        }
        // SAFETY: reading back our restored 64 KiB allocation.
        let buf = unsafe { std::slice::from_raw_parts(pages.ptr, BLOCK) };
        assert_eq!(&buf[0x2000..0x2004], &0x1122_3344u32.to_le_bytes());
        assert!(buf[..0x2000].iter().all(|&b| b == 0xA5));
        assert!(buf[0x2004..].iter().all(|&b| b == 0xA5));
        // Write fault must NOT call `ResetWriteWatch`, so the CPU write is visible to `Shadow::update`.
        let dirty = pages.dirty_pages();
        assert!(
            dirty.contains(&(base + 0x2000)),
            "write fault must leave written 4 KiB page dirty in GetWriteWatch"
        );
    }

    #[test]
    fn concurrent_faults_on_same_block_coalesce_to_single_fetch() {
        let pages = WatchPages::new(1);
        let base = pages.va();
        let fetches = Arc::new(AtomicU64::new(0));
        let fetches_c = fetches.clone();
        assert!(promote_test_block(
            base,
            103,
            0,
            11,
            Arc::new(move |_blk, _off, _fence| {
                fetches_c.fetch_add(1, Ordering::SeqCst);
                std::thread::sleep(Duration::from_millis(25));
                Some(Box::new([0x5Cu8; BLOCK]))
            }),
        ));

        let barrier = Arc::new(Barrier::new(4));
        std::thread::scope(|s| {
            for t in 0..4usize {
                let b = barrier.clone();
                s.spawn(move || {
                    b.wait();
                    // SAFETY: each thread reads inside the shared 64 KiB block.
                    let val = unsafe { *((base + t * 4096 + 128) as *const u8) };
                    assert_eq!(val, 0x5C);
                });
            }
        });
        assert_eq!(
            fetches.load(Ordering::SeqCst),
            1,
            "4 concurrent faults on one 64 KiB block must coalesce to 1 GPU fetch"
        );
    }

    #[test]
    fn adaptive_cpu_hot_demotion_after_two_faults_and_evict_resets() {
        let pages = WatchPages::new(1);
        let base = pages.va();
        let block = base / BLOCK;
        let fetcher: TestFetcherFn = Arc::new(|_, _, _| Some(Box::new([0x33u8; BLOCK])));

        // Fault #1 -> returns to CpuResident (re-promotable once).
        assert!(promote_test_block(base, 104, 0, 1, fetcher.clone()));
        // SAFETY: inside our allocation.
        assert_eq!(unsafe { *pages.ptr }, 0x33);
        assert!(!is_gpu_resident(block));
        assert!(!is_cpu_hot(block));

        // Fault #2 -> permanently demoted to CpuHot!
        assert!(promote_test_block(base, 104, 0, 2, fetcher.clone()));
        // SAFETY: inside our allocation.
        assert_eq!(unsafe { *pages.ptr }, 0x33);
        assert!(!is_gpu_resident(block));
        assert!(is_cpu_hot(block));

        // 3rd promotion attempt MUST refuse while CpuHot.
        assert!(!promote_test_block(base, 104, 0, 3, fetcher.clone()));
        assert!(!is_gpu_resident(block));

        // `evict_range` (unmap/remap VA reuse) clears `CpuHot` and fault counts.
        evict_range(base, BLOCK);
        assert!(!is_cpu_hot(block));
        assert!(promote_test_block(base, 105, 0, 4, fetcher));
        assert!(is_gpu_resident(block));
    }

    #[test]
    fn prepare_cpu_overwrite_drops_full_block_and_fetches_partial_edges() {
        let pages = WatchPages::new(3);
        let base = pages.va();
        let fetches = Arc::new(AtomicU64::new(0));
        for i in 0..3usize {
            let fetches_c = fetches.clone();
            let fill = (0x10 + i as u8) as u8;
            assert!(promote_test_block(
                base + i * BLOCK,
                106,
                (i * BLOCK) as u64,
                5,
                Arc::new(move |_, _, _| {
                    fetches_c.fetch_add(1, Ordering::SeqCst);
                    Some(Box::new([fill; BLOCK]))
                }),
            ));
        }

        // Overwrite `[base + 0x4000, base + 2*BLOCK + 0x4000)`:
        // - block 0: partial tail -> fetched (1)
        // - block 1: fully covered -> dropped with ZERO GPU readback!
        // - block 2: partial head -> fetched (2)
        prepare_cpu_overwrite(base + 0x4000, 2 * BLOCK);
        assert_eq!(
            fetches.load(Ordering::SeqCst),
            2,
            "middle fully-overwritten block must drop with 0 GPU readback"
        );
        assert!(!any_gpu_resident(base, 3 * BLOCK));

        // SAFETY: all 3 blocks are now `PAGE_READWRITE`.
        unsafe {
            assert_eq!(*pages.ptr, 0x10, "block 0 untouched head preserved");
            assert_eq!(*pages.ptr.add(2 * BLOCK + 0x8000), 0x12, "block 2 untouched tail preserved");
            // Middle block is writable without faulting.
            pages.ptr.add(BLOCK + 123).write_volatile(0xEE);
            assert_eq!(pages.ptr.add(BLOCK + 123).read_volatile(), 0xEE);
        }
    }

    #[test]
    fn evict_range_drops_without_readback_and_ensure_cpu_resident_fetches() {
        let pages = WatchPages::new(2);
        let base = pages.va();
        let fetches = Arc::new(AtomicU64::new(0));
        let f0 = fetches.clone();
        let f1 = fetches.clone();
        assert!(promote_test_block(
            base,
            107,
            0,
            1,
            Arc::new(move |_, _, _| {
                f0.fetch_add(1, Ordering::SeqCst);
                Some(Box::new([0x77u8; BLOCK]))
            }),
        ));
        assert!(promote_test_block(
            base + BLOCK,
            107,
            BLOCK as u64,
            1,
            Arc::new(move |_, _, _| {
                f1.fetch_add(1, Ordering::SeqCst);
                Some(Box::new([0x88u8; BLOCK]))
            }),
        ));

        // `evict_range` on block 0 must drop with 0 fetches.
        evict_range(base, BLOCK);
        assert_eq!(fetches.load(Ordering::SeqCst), 0);
        assert!(!is_gpu_resident(base / BLOCK));
        assert!(is_gpu_resident((base + BLOCK) / BLOCK));
        // Restore protection on block 0 so test drop doesn't leave it noaccess.
        protect_readwrite(base);

        // `ensure_cpu_resident` on block 1 pulls its GPU bytes cleanly.
        ensure_cpu_resident(base + BLOCK + 512, 128);
        assert_eq!(fetches.load(Ordering::SeqCst), 1);
        assert!(!is_gpu_resident((base + BLOCK) / BLOCK));
        // SAFETY: block 1 is restored to `PAGE_READWRITE`.
        assert_eq!(unsafe { *pages.ptr.add(BLOCK + 512) }, 0x88);
    }

    #[test]
    fn mem_mapped_section_read_and_write_faults_recover_via_beside_alias() {
        use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEM_MAPPED, MEMORY_BASIC_INFORMATION,
            MapViewOfFile, PAGE_NOACCESS, SEC_COMMIT, UnmapViewOfFile, VirtualQuery,
        };

        // Real Windows section-backed `MEM_MAPPED` 64 KiB view (Func9 layout).
        // SAFETY: anonymous pagefile section mapped and cleaned up inside test.
        let sec = unsafe {
            CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE | SEC_COMMIT,
                0,
                BLOCK as u32,
                std::ptr::null(),
            )
        };
        assert!(!sec.is_null());
        // SAFETY: mapping 1 full 64 KiB block read-write.
        let view = unsafe { MapViewOfFile(sec, FILE_MAP_WRITE, 0, 0, BLOCK) };
        assert!(!view.Value.is_null());
        let base = view.Value as usize;
        assert_eq!(base % BLOCK, 0);

        // Register with `inplace` so `restore_title_block` takes the zero-torn-read
        // `MEM_MAPPED` + `inplace::beside` path while `base` stays `PAGE_NOACCESS`.
        crate::inplace::title_memory_section(sec as usize, BLOCK as u64, true);
        crate::inplace::title_memory_view(base, BLOCK, sec as usize, 0);

        assert!(is_committable_block(base));
        assert_eq!(crate::inplace::pages_of(base).len(), 1);
        assert!(crate::inplace::beside(base, BLOCK).is_some());

        // 1. CPU read fault on `MEM_MAPPED` `PAGE_NOACCESS` recovers via `beside` alias.
        let fetches = Arc::new(AtomicU64::new(0));
        let f1 = fetches.clone();
        assert!(promote_test_block(
            base,
            201,
            0,
            1,
            Arc::new(move |_, _, _| {
                f1.fetch_add(1, Ordering::SeqCst);
                Some(Box::new([0xD4u8; BLOCK]))
            }),
        ));
        let mut mbi: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        assert_ne!(unsafe { VirtualQuery(base as *const _, &mut mbi, size_of_val(&mbi)) }, 0);
        assert_eq!(mbi.Type, MEM_MAPPED);
        assert_eq!(mbi.Protect, PAGE_NOACCESS);

        // SAFETY: `base` is our 64 KiB `MEM_MAPPED` view; VEH recovers `PAGE_NOACCESS`.
        let sample = unsafe {
            let s = std::slice::from_raw_parts(base as *const u8, BLOCK);
            (s[0], s[0x1234], s[BLOCK - 1])
        };
        assert_eq!(sample, (0xD4, 0xD4, 0xD4));
        assert_eq!(fetches.load(Ordering::SeqCst), 1);
        assert!(!is_gpu_resident(base / BLOCK));
        assert_ne!(unsafe { VirtualQuery(base as *const _, &mut mbi, size_of_val(&mbi)) }, 0);
        assert_eq!(mbi.Protect, PAGE_READWRITE);

        // 2. Partial 4-byte CPU write fault on `MEM_MAPPED` preserves all other 65,532 bytes.
        let f2 = fetches.clone();
        assert!(promote_test_block(
            base,
            201,
            0,
            2,
            Arc::new(move |_, _, _| {
                f2.fetch_add(1, Ordering::SeqCst);
                Some(Box::new([0xB2u8; BLOCK]))
            }),
        ));
        // SAFETY: partial write at offset 0x3000 hits `PAGE_NOACCESS`, recovers via `beside`,
        // then retries the 4-byte store cleanly.
        unsafe {
            ((base + 0x3000) as *mut u32).write_volatile(0xCAFE_BABE);
            let s = std::slice::from_raw_parts(base as *const u8, BLOCK);
            assert_eq!(&s[0x3000..0x3004], &0xCAFE_BABEu32.to_le_bytes());
            assert!(s[..0x3000].iter().all(|&b| b == 0xB2));
            assert!(s[0x3004..].iter().all(|&b| b == 0xB2));
        }
        assert_eq!(fetches.load(Ordering::SeqCst), 2);
        // 2nd fault also triggered `CpuHot` demotion on this `MEM_MAPPED` block.
        assert!(is_cpu_hot(base / BLOCK));

        // Clean up `lazy_readback`, `inplace`, and Win32 handles.
        evict_range(base, BLOCK);
        test_fetchers()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&(base / BLOCK));
        crate::inplace::views_gone(base, BLOCK);
        crate::inplace::title_memory_section_gone(sec as usize);
        // SAFETY: unmapping and closing our test section.
        unsafe {
            UnmapViewOfFile(view);
            CloseHandle(sec);
        }
    }
}
