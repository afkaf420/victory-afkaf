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

/// Maximum number of virtual alias views (`inplace::pages_of`) supported for a
/// single `GpuResident` 64 KiB physical block (`1..=4`).
const MAX_ALIASES: usize = 4;

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
/// `[10]` fetch microseconds total,
/// `[11]` multi-view aliased (`2..=4` VAs) promotions.
static STATS: [AtomicU64; 12] = [const { AtomicU64::new(0) }; 12];
/// Per-interval `plan_gpu_resident` rejection counters:
/// `[0]` out-of-shadow-span, `[1]` class ineligible (e.g. `0x2004` in mode 1),
/// `[2]` multi-view aliased (`> 4` VAs or intra-shadow self-alias),
/// `[3]` `CpuHot` / `Fetching`, `[4]` `!is_committable_block`.
static REJ: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];
/// Currently active `GpuResident` physical 64 KiB block count.
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

/// Sorted, deduplicated set of `1..=MAX_ALIASES` 64 KiB block indices
/// (`va / BLOCK`) that all map to the same physical 64 KiB page.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AliasSet {
    blocks: [usize; MAX_ALIASES],
    len: u8,
}

impl AliasSet {
    /// Builds a sorted, deduplicated `AliasSet` from `primary_block` and the
    /// raw VA list returned by `crate::inplace::pages_of(primary_block * BLOCK)`.
    /// Returns `None` if there are `> MAX_ALIASES` distinct alias VAs or any
    /// VA is not 64 KiB aligned.
    fn from_pages(primary_block: usize, page_vas: &[usize]) -> Option<Self> {
        let mut blocks = [0usize; MAX_ALIASES];
        let mut len = 0usize;
        let mut push_unique = |blk: usize| -> bool {
            if blocks[..len].contains(&blk) {
                return true;
            }
            if len >= MAX_ALIASES {
                return false;
            }
            blocks[len] = blk;
            len += 1;
            true
        };
        if !push_unique(primary_block) {
            return None;
        }
        for &va in page_vas {
            if !va.is_multiple_of(BLOCK) {
                return None;
            }
            if !push_unique(va / BLOCK) {
                return None;
            }
        }
        blocks[..len].sort_unstable();
        Some(Self {
            blocks,
            len: len as u8,
        })
    }

    #[inline]
    fn as_slice(&self) -> &[usize] {
        &self.blocks[..self.len as usize]
    }

    /// Deterministic canonical primary key in `State::blocks` (minimum block index).
    #[inline]
    fn canonical(&self) -> usize {
        self.blocks[0]
    }

    #[inline]
    fn contains(&self, block: usize) -> bool {
        self.as_slice().contains(&block)
    }
}

struct ResidentInfo {
    /// Block index (`va / BLOCK`) inside `shadow`'s span matching `gpu_offset`.
    owner_block: usize,
    shadow_id: u64,
    shadow: Option<Weak<Shadow>>,
    gpu_offset: u64,
    last_fence: u64,
    aliases: AliasSet,
}

enum BlockState {
    GpuResident(ResidentInfo),
    Fetching {
        shadow_id: u64,
        any_write: bool,
        epoch: u64,
        aliases: AliasSet,
    },
    /// Secondary alias block pointing to the canonical primary key in
    /// `State::blocks` (which holds `GpuResident` or `Fetching`).
    AliasOf(usize),
    CpuHot(AliasSet),
}

struct State {
    blocks: HashMap<usize, BlockState>,
    fault_counts: HashMap<usize, u8>,
    next_gen: u64,
}

impl State {
    #[inline]
    fn resolve_canonical(&self, block: usize) -> usize {
        match self.blocks.get(&block) {
            Some(&BlockState::AliasOf(primary)) => primary,
            _ => block,
        }
    }

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

/// Checks (read-only) whether `[block_va, block_va + BLOCK)` is currently
/// committed `PAGE_READWRITE` (used to resolve a tight VEH race where another
/// thread finished restoring the page right before we acquired `STATE`).
fn is_readwrite_block(block_va: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_READWRITE, VirtualQuery,
    };
    // SAFETY: plain out struct for `VirtualQuery`.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: valid pointer and struct size.
    let n = unsafe { VirtualQuery(block_va as *const _, &mut info, size_of_val(&info)) };
    if n == 0 || info.State != MEM_COMMIT {
        return false;
    }
    let region_end = (info.BaseAddress as usize).saturating_add(info.RegionSize);
    region_end >= block_va.saturating_add(BLOCK) && info.Protect == PAGE_READWRITE
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

/// Releases `watch` guards, sets `RESIDENT`, and applies `PAGE_NOACCESS` to
/// every block in `aliases` not already covered by `already_armed`. Rolls back
/// all newly armed blocks to `PAGE_READWRITE` + `RESIDENT.clear` if any
/// `VirtualProtect` fails, guaranteeing an atomic all-or-nothing transition.
fn arm_aliases_noaccess(aliases: &AliasSet, already_armed: Option<&AliasSet>) -> bool {
    let mut armed = [0usize; MAX_ALIASES];
    let mut n_armed = 0usize;
    for &a in aliases.as_slice() {
        if already_armed.is_some_and(|prev| prev.contains(a)) {
            continue;
        }
        // Release any `watch` read-only guard before setting `PAGE_NOACCESS`
        // (strict lock order: `STATE -> GUARDS` never reverses).
        crate::watch::release_guard_for_lazy(a);
        // Mark `RESIDENT` BEFORE `VirtualProtect(PAGE_NOACCESS)` so any racing
        // VEH fault on another thread sees `RESIDENT == 1`.
        RESIDENT.set(a);
        if protect_noaccess(a * BLOCK) {
            armed[n_armed] = a;
            n_armed += 1;
        } else {
            RESIDENT.clear(a);
            for &prev in &armed[..n_armed] {
                protect_readwrite(prev * BLOCK);
                RESIDENT.clear(prev);
            }
            return false;
        }
    }
    true
}

/// A 64 KiB block validated in Phase 1 ([`plan_gpu_resident`], inside
/// `Host::card_before`) and ready to be armed/updated in Phase 2
/// ([`commit_gpu_resident`]) once `Host::queue.Signal(&Host::fence, F)` has
/// succeeded.
pub(crate) struct PendingClaim {
    pub(crate) block: usize,
    pub(crate) shadow: Arc<Shadow>,
    pub(crate) gpu_offset: u64,
    aliases: AliasSet,
}

/// **Phase 1 (plan)** of 2-phase lazy readback claiming, called inside
/// `Host::card_before()` BEFORE `card_dirty()` / `copy_dirty_to_ring()` and
/// BEFORE `Host::queue.Signal(&Host::fence, F)`.
///
/// Validates every 64 KiB block in `ranges` (including multi-view aliases with
/// `1..=4` virtual addresses) against all safety filters and returns
/// `(pending_claims, remaining_ranges)` WITHOUT arming `PAGE_NOACCESS` and
/// WITHOUT storing an unsignaled future fence in `last_fence`.
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
            // Support 1..=4 virtual alias views of the same 64 KiB physical
            // page by arming `PAGE_NOACCESS` across all aliases together.
            let page_vas = crate::inplace::pages_of(b_lo);
            let Some(aliases) = AliasSet::from_pages(block, &page_vas) else {
                if RESIDENT.contains(block) {
                    materialize_block(block, false, false, false, None);
                }
                rej(2);
                push_rem(&mut remaining, sub_lo, sub_hi);
                continue;
            };
            // Reject intra-shadow self-aliasing where two distinct alias VAs
            // lie inside the same `Shadow::span()` (writes at one `gpu_offset`
            // would not be mirrored to the second offset inside that buffer).
            if aliases.len > 1
                && aliases.as_slice().iter().any(|&a| {
                    if a == block {
                        return false;
                    }
                    let a_lo = a * BLOCK;
                    a_lo >= span_start && a_lo < span_end
                })
            {
                if RESIDENT.contains(block) {
                    materialize_block(block, false, false, false, None);
                }
                rej(2);
                push_rem(&mut remaining, sub_lo, sub_hi);
                continue;
            }

            let gpu_offset = (b_lo - span_start) as u64;
            let mut need_commit_check = [0usize; MAX_ALIASES];
            let mut n_check = 0usize;
            let state_ok = {
                let mut s = lock.lock().unwrap_or_else(|e| e.into_inner());
                let mut existing_primary: Option<usize> = None;
                let mut ok = true;
                for &a in aliases.as_slice() {
                    let p = s.resolve_canonical(a);
                    match s.blocks.get(&p) {
                        Some(BlockState::GpuResident(_)) => {
                            if let Some(ep) = existing_primary {
                                if ep != p {
                                    rej(2);
                                    ok = false;
                                    break;
                                }
                            } else {
                                existing_primary = Some(p);
                            }
                        }
                        Some(BlockState::CpuHot(_)) | Some(BlockState::Fetching { .. }) => {
                            rej(3);
                            ok = false;
                            break;
                        }
                        Some(BlockState::AliasOf(_)) => {
                            rej(2);
                            ok = false;
                            break;
                        }
                        None => {
                            if s.fault_counts.get(&a).copied().unwrap_or(0) >= DEMOTE_THRESHOLD {
                                s.blocks.insert(a, BlockState::CpuHot(aliases));
                                s.sync_tracked();
                                rej(3);
                                ok = false;
                                break;
                            }
                            need_commit_check[n_check] = a;
                            n_check += 1;
                        }
                    }
                }
                ok
            };
            let take = if !state_ok {
                false
            } else if n_check == 0 {
                // Fast path: all aliases are already armed in one `GpuResident` group.
                true
            } else {
                let all_ok = need_commit_check[..n_check]
                    .iter()
                    .all(|&a| is_committable_block(a * BLOCK));
                if !all_ok {
                    rej(4);
                }
                all_ok
            };
            if take {
                if pending.last().is_none_or(|c| c.block != block) {
                    pending.push(PendingClaim {
                        block,
                        shadow: shadow.clone(),
                        gpu_offset,
                        aliases,
                    });
                }
            } else {
                if RESIDENT.contains(block) {
                    materialize_block(block, false, false, false, None);
                }
                push_rem(&mut remaining, sub_lo, sub_hi);
            }
        }
    }
    (pending, remaining)
}

/// Core Phase-2 commit routine for one physical 64 KiB page and its `1..=4`
/// virtual alias blocks. Shared by [`commit_gpu_resident`] and unit tests.
fn commit_claim_entry(
    owner_block: usize,
    aliases: AliasSet,
    shadow_id: u64,
    shadow: Option<Weak<Shadow>>,
    gpu_offset: u64,
    signaled_fence: u64,
    committed: &mut Vec<(usize, u64)>,
) -> bool {
    let (lock, _cv) = state();
    let mut s = lock.lock().unwrap_or_else(|e| e.into_inner());

    // Validate every alias VA against concurrent faults, demotions, or collisions.
    let mut existing_primary: Option<usize> = None;
    for &a in aliases.as_slice() {
        let p = s.resolve_canonical(a);
        match s.blocks.get(&p) {
            Some(BlockState::CpuHot(_)) | Some(BlockState::Fetching { .. }) => {
                return false;
            }
            Some(BlockState::GpuResident(_)) => {
                if let Some(ep) = existing_primary {
                    if ep != p {
                        return false;
                    }
                } else {
                    existing_primary = Some(p);
                }
            }
            Some(BlockState::AliasOf(_)) => return false,
            None => {
                if s.fault_counts.get(&a).copied().unwrap_or(0) >= DEMOTE_THRESHOLD {
                    s.blocks.insert(a, BlockState::CpuHot(aliases));
                    s.sync_tracked();
                    return false;
                }
            }
        }
    }

    if let Some(old_primary) = existing_primary {
        let old_aliases = match s.blocks.get(&old_primary) {
            Some(BlockState::GpuResident(info)) => info.aliases,
            _ => return false,
        };
        if old_aliases == aliases {
            // O(1) steady-state fast path: 0 `VirtualProtect` syscalls!
            if let Some(BlockState::GpuResident(info)) = s.blocks.get_mut(&old_primary) {
                info.owner_block = owner_block;
                info.shadow_id = shadow_id;
                info.shadow = shadow;
                info.gpu_offset = gpu_offset;
                info.last_fence = info.last_fence.max(signaled_fence);
            }
            stat(1, 1);
            for &a in aliases.as_slice() {
                committed.push((a, shadow_id));
            }
            return true;
        }
        // Alias set grew or changed while already `GpuResident`: arm any newly
        // added alias VAs atomically before rewriting `State::blocks`.
        if !arm_aliases_noaccess(&aliases, Some(&old_aliases)) {
            return false;
        }
        let prev_fence = match s.blocks.remove(&old_primary) {
            Some(BlockState::GpuResident(info)) => info.last_fence,
            _ => 0,
        };
        for &oa in old_aliases.as_slice() {
            s.blocks.remove(&oa);
            if !aliases.contains(oa) {
                protect_readwrite(oa * BLOCK);
                RESIDENT.clear(oa);
            }
        }
        let new_primary = aliases.canonical();
        s.blocks.insert(
            new_primary,
            BlockState::GpuResident(ResidentInfo {
                owner_block,
                shadow_id,
                shadow,
                gpu_offset,
                last_fence: prev_fence.max(signaled_fence),
                aliases,
            }),
        );
        for &a in aliases.as_slice() {
            if a != new_primary {
                s.blocks.insert(a, BlockState::AliasOf(new_primary));
            }
        }
        s.sync_tracked();
        stat(1, 1);
        for &a in aliases.as_slice() {
            committed.push((a, shadow_id));
        }
        return true;
    }

    // Fresh promotion (`CpuResident -> GpuResident`): atomically arm all 1..=4 aliases.
    if !arm_aliases_noaccess(&aliases, None) {
        return false;
    }
    let primary = aliases.canonical();
    s.blocks.insert(
        primary,
        BlockState::GpuResident(ResidentInfo {
            owner_block,
            shadow_id,
            shadow,
            gpu_offset,
            last_fence: signaled_fence,
            aliases,
        }),
    );
    for &a in aliases.as_slice() {
        if a != primary {
            s.blocks.insert(a, BlockState::AliasOf(primary));
        }
    }
    s.sync_tracked();
    LIVE_BLOCKS.fetch_add(1, Ordering::Relaxed);
    stat(0, 1);
    if aliases.len > 1 {
        stat(11, 1);
    }
    for &a in aliases.as_slice() {
        committed.push((a, shadow_id));
    }
    true
}

/// **Phase 2 (commit)** of 2-phase lazy readback claiming, called IMMEDIATELY
/// AFTER `Host::queue.Signal(&Host::fence, signaled_fence)` succeeds (and after
/// [`note_signaled_fence`]). Arms `PAGE_NOACCESS` / `RESIDENT` across all
/// `1..=4` alias VAs on newly promoted blocks and advances `last_fence` in O(1)
/// on already-`GpuResident` blocks. Returns `(block, shadow_id)` for every
/// committed alias block.
pub(crate) fn commit_gpu_resident(
    claims: Vec<PendingClaim>,
    signaled_fence: u64,
) -> Vec<(usize, u64)> {
    if claims.is_empty() {
        return Vec::new();
    }
    // Ensure `watch::on_write` VEH is installed ahead of any `PAGE_NOACCESS`.
    crate::watch::install();

    let mut committed = Vec::with_capacity(claims.len());
    for claim in claims {
        let PendingClaim {
            block,
            shadow,
            gpu_offset,
            aliases: planned_aliases,
        } = claim;
        let b_lo = block * BLOCK;
        // Re-check `inplace::pages_of` so any alias mapped between Phase 1 and
        // Phase 2 is also armed (or skipped if the physical page grew > 4 aliases
        // or gained an intra-shadow self-alias).
        let page_vas = crate::inplace::pages_of(b_lo);
        let aliases = if page_vas.is_empty() {
            planned_aliases
        } else {
            let Some(fresh) = AliasSet::from_pages(block, &page_vas) else {
                continue;
            };
            fresh
        };
        if aliases.len > 1 {
            let (span_start_u64, span_len_u64) = shadow.span();
            let span_start = span_start_u64 as usize;
            let span_end = span_start.saturating_add(span_len_u64 as usize);
            if aliases.as_slice().iter().any(|&a| {
                a != block && {
                    let a_lo = a * BLOCK;
                    a_lo >= span_start && a_lo < span_end
                }
            }) {
                continue;
            }
        }
        let shadow_id = shadow.id();
        commit_claim_entry(
            block,
            aliases,
            shadow_id,
            Some(Arc::downgrade(&shadow)),
            gpu_offset,
            signaled_fence,
            &mut committed,
        );
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

/// Writes 64 KiB `payload` back into the physical page backing `aliases`,
/// restoring `PAGE_READWRITE` across all alias VAs, and resets `GetWriteWatch`
/// on single-view `MEM_PRIVATE` pure read faults so clean GPU bytes are not
/// spuriously re-uploaded.
fn restore_title_block(aliases: &AliasSet, payload: &[u8; BLOCK], any_write: bool) {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_MAPPED, MEM_PRIVATE, MEMORY_BASIC_INFORMATION, ResetWriteWatch,
        VirtualQuery,
    };
    let slice = aliases.as_slice();
    let mut first_type = 0u32;
    let mut first_committed_va = 0usize;
    for &blk in slice {
        let va = blk * BLOCK;
        // SAFETY: plain out struct.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: valid pointer and struct size.
        let n = unsafe { VirtualQuery(va as *const _, &mut info, size_of_val(&info)) };
        if n != 0 && info.State == MEM_COMMIT {
            first_type = info.Type;
            first_committed_va = va;
            if info.Type == MEM_MAPPED {
                break;
            }
        }
    }
    if first_committed_va == 0 {
        return;
    }

    if first_type == MEM_MAPPED || slice.len() > 1 {
        // Section-backed view (1..=4 aliases): populate the shared physical
        // page ONCE while ALL alias VAs are STILL `PAGE_NOACCESS` (zero
        // torn-read window across every view), then flip every alias VA to
        // `PAGE_READWRITE`.
        let mut wrote_beside = false;
        if let Some(_reading) = crate::watch::title_memory_reading_now() {
            for &blk in slice {
                if let Some(alias_ptr) = crate::inplace::beside(blk * BLOCK, BLOCK) {
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
                            std::ptr::copy_nonoverlapping(
                                payload.as_ptr(),
                                alias_ptr as *mut u8,
                                BLOCK,
                            );
                        }
                        wrote_beside = true;
                    }
                    if wrote_beside {
                        break;
                    }
                }
            }
        }
        if !wrote_beside {
            for &blk in slice {
                if try_alias_writer(blk * BLOCK, payload) {
                    wrote_beside = true;
                    break;
                }
            }
        }
        if wrote_beside {
            for &blk in slice {
                protect_readwrite(blk * BLOCK);
            }
            return;
        }
        // Fallback if section cannot be aliased right now: unprotect the first
        // committed alias VA, write the shared physical page, then unprotect
        // all remaining alias VAs.
        if protect_readwrite(first_committed_va) {
            if crate::safecopy::on() {
                // SAFETY: `payload` is valid 64 KiB; `safecopy` fails cleanly if unmapped.
                unsafe {
                    let _ = crate::safecopy::copy(
                        first_committed_va as *mut u8,
                        payload.as_ptr(),
                        BLOCK,
                    );
                }
            } else {
                // SAFETY: `first_committed_va` verified committed and switched to `PAGE_READWRITE`.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        payload.as_ptr(),
                        first_committed_va as *mut u8,
                        BLOCK,
                    );
                }
            }
        }
        for &blk in slice {
            let va = blk * BLOCK;
            if va != first_committed_va {
                protect_readwrite(va);
            }
        }
        return;
    }

    // Single-view `MEM_PRIVATE` (including `MEM_WRITE_WATCH`): flip
    // `PAGE_READWRITE` and copy all 64 KiB.
    if protect_readwrite(first_committed_va) {
        if crate::safecopy::on() {
            // SAFETY: `payload` is valid 64 KiB; `safecopy` fails cleanly if unmapped.
            unsafe {
                let _ = crate::safecopy::copy(
                    first_committed_va as *mut u8,
                    payload.as_ptr(),
                    BLOCK,
                );
            }
        } else {
            // SAFETY: `first_committed_va` verified committed and switched to `PAGE_READWRITE`.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    payload.as_ptr(),
                    first_committed_va as *mut u8,
                    BLOCK,
                );
            }
        }
        if first_type == MEM_PRIVATE && !any_write {
            // Clear dirty bits produced by our own restore copy on pure read
            // faults so `Shadow::update` does not re-upload unchanged GPU bytes.
            // On write faults (or any coalesced writer), skip `ResetWriteWatch`
            // so the retried store is guaranteed dirty.
            // SAFETY: valid committed `MEM_PRIVATE` range (harmless failure if not `MEM_WRITE_WATCH`).
            unsafe {
                let _ = ResetWriteWatch(first_committed_va as *mut _, BLOCK);
            }
        }
    }
}

/// Core state-machine materialization of one 64 KiB physical page (resolving
/// any alias VA in `1..=4` to its shared canonical entry: `GpuResident ->
/// CpuResident` or `CpuHot`). Returns `true` if `block` belonged to lazy
/// readback (was `GpuResident` or `Fetching`) and all alias VAs are now
/// restored to `PAGE_READWRITE`.
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
            let primary = s.resolve_canonical(block);
            match s.blocks.get_mut(&primary) {
                Some(BlockState::Fetching { any_write, .. }) => {
                    if is_write {
                        *any_write = true;
                    }
                    stat(5, 1);
                    while matches!(s.blocks.get(&primary), Some(BlockState::Fetching { .. })) {
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
                                let aliases = info.aliases;
                                for &a in aliases.as_slice() {
                                    s.blocks.remove(&a);
                                }
                                s.sync_tracked();
                                LIVE_BLOCKS.fetch_sub(1, Ordering::Relaxed);
                                for &a in aliases.as_slice() {
                                    protect_readwrite(a * BLOCK);
                                    RESIDENT.clear(a);
                                }
                                cv.notify_all();
                                return true;
                            }
                        }
                    } else {
                        // Test-mode entry without real D3D12 `Shadow`.
                        (None, None)
                    };
                    let owner_block = info.owner_block;
                    let gpu_offset = info.gpu_offset;
                    let last_fence = info.last_fence;
                    let shadow_id = info.shadow_id;
                    let aliases = info.aliases;
                    let epoch = s.next_gen;
                    s.next_gen = s.next_gen.wrapping_add(1).max(1);
                    s.blocks.insert(
                        primary,
                        BlockState::Fetching {
                            shadow_id,
                            any_write: is_write,
                            epoch,
                            aliases,
                        },
                    );
                    LIVE_BLOCKS.fetch_sub(1, Ordering::Relaxed);
                    (
                        primary,
                        owner_block,
                        aliases,
                        shadow_arc,
                        buffer_opt,
                        gpu_offset,
                        last_fence,
                        epoch,
                    )
                }
                Some(BlockState::CpuHot(_)) | Some(BlockState::AliasOf(_)) | None => {
                    // If a VEH thread saw `RESIDENT == 1` just before another
                    // thread finished restoring `block` to `PAGE_READWRITE`,
                    // retry the faulting instruction cleanly.
                    return from_veh && is_readwrite_block(block * BLOCK);
                }
            }
        };

        let (primary, owner_block, aliases, shadow_arc, buffer_opt, gpu_offset, last_fence, my_epoch) =
            work;
        let started = Instant::now();
        let fetched = fetch_from_gpu(owner_block, buffer_opt, gpu_offset, last_fence);
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
        // (which covers any coalesced writer across any alias VA).
        let still_ours = {
            let s = lock.lock().unwrap_or_else(|e| e.into_inner());
            match s.blocks.get(&primary) {
                Some(&BlockState::Fetching {
                    any_write, epoch, ..
                }) if epoch == my_epoch => Some(any_write),
                _ => None,
            }
        };
        let Some(final_any_write) = still_ours else {
            // `evict_range` already unmapped/remapped this group: do NOT touch it.
            return true;
        };

        if let Some(payload) = fetched {
            restore_title_block(&aliases, &payload, final_any_write);
        } else {
            for &a in aliases.as_slice() {
                protect_readwrite(a * BLOCK);
            }
        }

        // Complete transition under `STATE` lock and wake all coalesced waiters.
        {
            let mut s = lock.lock().unwrap_or_else(|e| e.into_inner());
            let still_current = matches!(
                s.blocks.get(&primary),
                Some(BlockState::Fetching { epoch, .. }) if *epoch == my_epoch
            );
            if still_current {
                if count_fault {
                    let mut max_count = 0u8;
                    for &a in aliases.as_slice() {
                        let count = s.fault_counts.entry(a).or_insert(0);
                        *count = count.saturating_add(1);
                        max_count = max_count.max(*count);
                    }
                    if max_count >= DEMOTE_THRESHOLD {
                        for &a in aliases.as_slice() {
                            s.fault_counts.insert(a, max_count);
                            s.blocks.insert(a, BlockState::CpuHot(aliases));
                        }
                        stat(9, 1);
                        TOTAL_DEMOTED.fetch_add(1, Ordering::Relaxed);
                    } else {
                        for &a in aliases.as_slice() {
                            s.blocks.remove(&a);
                        }
                    }
                } else {
                    for &a in aliases.as_slice() {
                        s.blocks.remove(&a);
                    }
                }
                s.sync_tracked();
                for &a in aliases.as_slice() {
                    RESIDENT.clear(a);
                }
            }
            cv.notify_all();
        }
        return true;
    }
}

/// VEH entry point called from `watch::on_write` **BEFORE**
/// `safecopy::failing_exit` for BOTH read (`info[0] == 0`) and write
/// (`info[0] == 1`) access violations.
///
/// Returns `true` if `addr` hit a `GpuResident` / `Fetching` block (or any of
/// its alias VAs) and has now been fetched and restored to `PAGE_READWRITE`
/// (`EXCEPTION_CONTINUE_EXECUTION`).
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
///   `GpuResident` across ALL alias VAs and flip them to `PAGE_READWRITE` with
///   **zero GPU readback**.
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
        // Full 64 KiB block overwrite: drop `GpuResident` across all alias VAs
        // with ZERO GPU readback!
        let mut s = lock.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            let primary = s.resolve_canonical(block);
            match s.blocks.get(&primary) {
                Some(BlockState::Fetching { .. }) => {
                    s = cv.wait(s).unwrap_or_else(|e| e.into_inner());
                }
                Some(BlockState::GpuResident(info)) => {
                    let aliases = info.aliases;
                    for &a in aliases.as_slice() {
                        s.blocks.remove(&a);
                    }
                    s.sync_tracked();
                    LIVE_BLOCKS.fetch_sub(1, Ordering::Relaxed);
                    for &a in aliases.as_slice() {
                        protect_readwrite(a * BLOCK);
                        RESIDENT.clear(a);
                    }
                    stat(6, 1);
                    cv.notify_all();
                    break;
                }
                Some(BlockState::CpuHot(_)) | Some(BlockState::AliasOf(_)) | None => {
                    RESIDENT.clear(block);
                    break;
                }
            }
        }
    }
}

/// Evicts all `GpuResident`, `Fetching`, `CpuHot`, and fault-count state in
/// `[at, at + len)` (plus any linked multi-view aliases of those physical
/// pages) with **zero GPU readback** and **zero blocking** (safe to call from
/// `watch::title_memory_remapped` under `PHYSICAL -> MAPPING`). Restores
/// `PAGE_READWRITE` on any still-mapped sibling alias outside `[at, at + len)`
/// so no alias VA is ever left stranded at `PAGE_NOACCESS`.
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
        let primary = s.resolve_canonical(block);
        if let Some(prev) = s.blocks.remove(&primary) {
            match prev {
                BlockState::GpuResident(info) => {
                    for &a in info.aliases.as_slice() {
                        s.blocks.remove(&a);
                        s.fault_counts.remove(&a);
                        if a < first || a >= last {
                            protect_readwrite(a * BLOCK);
                        }
                        RESIDENT.clear(a);
                    }
                    LIVE_BLOCKS.fetch_sub(1, Ordering::Relaxed);
                    evicted += 1;
                }
                BlockState::Fetching { aliases, .. } => {
                    for &a in aliases.as_slice() {
                        s.blocks.remove(&a);
                        s.fault_counts.remove(&a);
                        if a < first || a >= last {
                            protect_readwrite(a * BLOCK);
                        }
                        RESIDENT.clear(a);
                    }
                    evicted += 1;
                    woke = true;
                }
                BlockState::CpuHot(aliases) => {
                    for &a in aliases.as_slice() {
                        s.blocks.remove(&a);
                        s.fault_counts.remove(&a);
                    }
                }
                BlockState::AliasOf(_) => {
                    s.blocks.remove(&block);
                    RESIDENT.clear(block);
                }
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
    let p: [u64; 12] = std::array::from_fn(|i| STATS[i].swap(0, Ordering::Relaxed));
    let r: [u64; 5] = std::array::from_fn(|i| REJ[i].swap(0, Ordering::Relaxed));
    let live = LIVE_BLOCKS.load(Ordering::Relaxed);
    let total_demoted = TOTAL_DEMOTED.load(Ordering::Relaxed);
    if live == 0 && p.iter().all(|&v| v == 0) && r.iter().all(|&v| v == 0) && total_demoted == 0 {
        return;
    }
    let faults = p[2] + p[3];
    tracing::info!(
        "PC-native VRAM lazy readback (VICTORY_LAZY_READBACK={}): live {} blocks ({:.1} MiB), \
         promoted {} (aliased {}), repeat-fence-updates {} (0 VirtualProtect), \
         VEH faults {} (read {}, write {}, coalesced {}), \
         internal fetches {}, overwrite-drops {} (0 readback), evicts {}, shadow-flushes {}, \
         demoted-CpuHot {} (cumulative {}), fetch {:.2} ms; \
         rejected [span={}, class={}, alias={}, cpuhot={}, uncommittable={}]",
        mode(),
        live,
        (live * BLOCK as u64) as f64 / (1024.0 * 1024.0),
        p[0],
        p[11],
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

    /// Promotes a test block (plus any `inplace::pages_of` aliases, `1..=4`) to
    /// `GpuResident` (`PAGE_NOACCESS`) through the real [`commit_claim_entry`]
    /// state machine with an injected mock fetcher (returns `false` if `CpuHot`
    /// or `> 4` aliases).
    fn promote_test_block(
        block_va: usize,
        shadow_id: u64,
        gpu_offset: u64,
        fence: u64,
        fetcher: TestFetcherFn,
    ) -> bool {
        crate::watch::install();
        let block = block_va / BLOCK;
        let Some(aliases) = AliasSet::from_pages(block, &crate::inplace::pages_of(block_va)) else {
            return false;
        };
        test_fetchers()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(block, fetcher);
        let mut committed = Vec::new();
        commit_claim_entry(
            block,
            aliases,
            shadow_id,
            None,
            gpu_offset,
            fence,
            &mut committed,
        )
    }

    fn is_cpu_hot(block: usize) -> bool {
        let (lock, _) = state();
        let s = lock.lock().unwrap_or_else(|e| e.into_inner());
        matches!(s.blocks.get(&block), Some(BlockState::CpuHot(_)))
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

    #[test]
    fn multi_alias_two_views_protect_fault_coalesce_and_evict_together() {
        use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEM_MAPPED, MEMORY_BASIC_INFORMATION,
            MapViewOfFile, PAGE_NOACCESS, SEC_COMMIT, UnmapViewOfFile, VirtualQuery,
        };

        // 0. Verify `AliasSet::from_pages` bounds (1..=4 accepted, >4 or unaligned rejected).
        let a1 = AliasSet::from_pages(10, &[]).expect("empty pages_of -> single block");
        assert_eq!(a1.as_slice(), &[10]);
        let a4 = AliasSet::from_pages(20, &[40 * BLOCK, 10 * BLOCK, 30 * BLOCK, 20 * BLOCK])
            .expect("4 unique aligned aliases accepted & sorted");
        assert_eq!(a4.as_slice(), &[10, 20, 30, 40]);
        assert_eq!(a4.canonical(), 10);
        assert!(
            AliasSet::from_pages(1, &[2 * BLOCK, 3 * BLOCK, 4 * BLOCK, 5 * BLOCK]).is_none(),
            "5 distinct alias VAs must be rejected (> MAX_ALIASES)"
        );
        assert!(
            AliasSet::from_pages(1, &[2 * BLOCK + 4096]).is_none(),
            "unaligned alias VA must be rejected"
        );

        let query_prot = |va: usize| -> (u32, u32) {
            // SAFETY: plain out struct for `VirtualQuery`.
            let mut mbi: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
            // SAFETY: valid mapped VA.
            assert_ne!(
                unsafe { VirtualQuery(va as *const _, &mut mbi, size_of_val(&mbi)) },
                0
            );
            (mbi.Type, mbi.Protect)
        };

        // 1. Real Windows 2-alias section (`CreateFileMappingW` + `MapViewOfFile` x 2).
        // SAFETY: anonymous 64 KiB pagefile section cleaned up at end of test.
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
        // SAFETY: mapping two distinct read-write views of the same 64 KiB section.
        let view1 = unsafe { MapViewOfFile(sec, FILE_MAP_WRITE, 0, 0, BLOCK) };
        let view2 = unsafe { MapViewOfFile(sec, FILE_MAP_WRITE, 0, 0, BLOCK) };
        assert!(!view1.Value.is_null() && !view2.Value.is_null());
        let va1 = view1.Value as usize;
        let va2 = view2.Value as usize;
        assert_ne!(va1, va2);
        assert_eq!(va1 % BLOCK, 0);
        assert_eq!(va2 % BLOCK, 0);
        let blk1 = va1 / BLOCK;
        let blk2 = va2 / BLOCK;

        crate::inplace::title_memory_section(sec as usize, BLOCK as u64, true);
        crate::inplace::title_memory_view(va1, BLOCK, sec as usize, 0);
        crate::inplace::title_memory_view(va2, BLOCK, sec as usize, 0);
        assert_eq!(crate::inplace::pages_of(va1).len(), 2);
        assert!(crate::inplace::beside(va1, BLOCK).is_some());

        // 2. Promoting `va1` arms `PAGE_NOACCESS` and `RESIDENT` on BOTH `va1` & `va2`.
        let fetches = Arc::new(AtomicU64::new(0));
        let f1 = fetches.clone();
        let aliases = AliasSet::from_pages(blk1, &crate::inplace::pages_of(va1)).unwrap();
        assert_eq!(aliases.len, 2);
        test_fetchers()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(
                blk1,
                Arc::new(move |_, _, _| {
                    f1.fetch_add(1, Ordering::SeqCst);
                    Some(Box::new([0xE7u8; BLOCK]))
                }),
            );
        crate::watch::install();
        let mut committed = Vec::new();
        assert!(commit_claim_entry(
            blk1,
            aliases,
            301,
            None,
            0,
            1,
            &mut committed
        ));
        assert!(committed.contains(&(blk1, 301)) && committed.contains(&(blk2, 301)));
        assert!(is_gpu_resident(blk1) && is_gpu_resident(blk2));
        assert_eq!(query_prot(va1), (MEM_MAPPED, PAGE_NOACCESS));
        assert_eq!(query_prot(va2), (MEM_MAPPED, PAGE_NOACCESS));

        // 3. Real CPU read fault on `va2` restores the physical page via `beside`
        //    and unprotects BOTH `va1` & `va2`; subsequent read on `va1` sees
        //    the exact GPU bytes with 0 second fault!
        // SAFETY: `va2` is a committed 64 KiB `MEM_MAPPED` view armed by lazy_readback.
        let got_va2 = unsafe {
            let s = std::slice::from_raw_parts(va2 as *const u8, BLOCK);
            (s[0], s[0x4321], s[BLOCK - 1])
        };
        assert_eq!(got_va2, (0xE7, 0xE7, 0xE7));
        assert_eq!(fetches.load(Ordering::SeqCst), 1);
        assert!(!is_gpu_resident(blk1) && !is_gpu_resident(blk2));
        assert_eq!(query_prot(va1).1, PAGE_READWRITE);
        assert_eq!(query_prot(va2).1, PAGE_READWRITE);
        // SAFETY: `va1` is already `PAGE_READWRITE` and shares the restored physical page.
        let got_va1 = unsafe {
            let s = std::slice::from_raw_parts(va1 as *const u8, BLOCK);
            (s[0], s[0x4321], s[BLOCK - 1])
        };
        assert_eq!(got_va1, (0xE7, 0xE7, 0xE7));
        assert_eq!(
            fetches.load(Ordering::SeqCst),
            1,
            "reading va1 after va2 fault must incur 0 second fault"
        );

        // 4. Re-promote & fault concurrently across `va1` (read) and `va2`
        //    (4-byte partial write): must coalesce to 1 fetch and demote BOTH
        //    `va1` and `va2` to `CpuHot`!
        let f2 = fetches.clone();
        assert!(promote_test_block(
            va1,
            301,
            0,
            2,
            Arc::new(move |_, _, _| {
                f2.fetch_add(1, Ordering::SeqCst);
                std::thread::sleep(Duration::from_millis(20));
                Some(Box::new([0x9Au8; BLOCK]))
            }),
        ));
        assert_eq!(query_prot(va1).1, PAGE_NOACCESS);
        assert_eq!(query_prot(va2).1, PAGE_NOACCESS);

        let barrier = Arc::new(Barrier::new(2));
        std::thread::scope(|sc| {
            let b1 = barrier.clone();
            sc.spawn(move || {
                b1.wait();
                // SAFETY: reading inside `va1`.
                let v = unsafe { *((va1 + 0x100) as *const u8) };
                assert_eq!(v, 0x9A);
            });
            let b2 = barrier.clone();
            sc.spawn(move || {
                b2.wait();
                // SAFETY: partial 4-byte write inside `va2`.
                unsafe { ((va2 + 0x2000) as *mut u32).write_volatile(0xDEAD_BEEF) };
            });
        });
        assert_eq!(
            fetches.load(Ordering::SeqCst),
            2,
            "concurrent faults on va1 and va2 must coalesce to 1 fetch"
        );
        // SAFETY: both views are now `PAGE_READWRITE` and coherent.
        unsafe {
            let s1 = std::slice::from_raw_parts(va1 as *const u8, BLOCK);
            assert_eq!(&s1[0x2000..0x2004], &0xDEAD_BEEFu32.to_le_bytes());
            assert!(s1[..0x2000].iter().all(|&b| b == 0x9A));
            assert!(s1[0x2004..].iter().all(|&b| b == 0x9A));
        }
        assert!(
            is_cpu_hot(blk1) && is_cpu_hot(blk2),
            "2nd fault must demote BOTH alias VAs to CpuHot"
        );
        let noop_fetcher: TestFetcherFn = Arc::new(|_, _, _| Some(Box::new([0x11u8; BLOCK])));
        assert!(!promote_test_block(va1, 301, 0, 3, noop_fetcher.clone()));
        assert!(!promote_test_block(va2, 301, 0, 3, noop_fetcher.clone()));

        // 5. `evict_range(va1)` clears `CpuHot` on BOTH `va1` and `va2`, and
        //    when `GpuResident`, evicts & clears BOTH with 0 fetches while
        //    restoring still-mapped sibling `va2` to `PAGE_READWRITE`.
        evict_range(va1, BLOCK);
        assert!(!is_cpu_hot(blk1) && !is_cpu_hot(blk2));

        assert!(promote_test_block(va1, 302, 0, 4, noop_fetcher.clone()));
        assert!(is_gpu_resident(blk1) && is_gpu_resident(blk2));
        assert_eq!(query_prot(va1).1, PAGE_NOACCESS);
        assert_eq!(query_prot(va2).1, PAGE_NOACCESS);

        evict_range(va1, BLOCK);
        assert_eq!(fetches.load(Ordering::SeqCst), 2, "evict_range must perform 0 GPU fetches");
        assert!(!is_gpu_resident(blk1) && !is_gpu_resident(blk2));
        assert_eq!(
            query_prot(va2).1,
            PAGE_READWRITE,
            "evict_range(va1) must restore still-mapped sibling va2 to PAGE_READWRITE"
        );
        protect_readwrite(va1);

        // 6. Full-block `prepare_cpu_overwrite(va2, BLOCK)` drops BOTH `va1`
        //    and `va2` with 0 fetches and restores BOTH to `PAGE_READWRITE`.
        assert!(promote_test_block(va1, 303, 0, 5, noop_fetcher));
        assert_eq!(query_prot(va1).1, PAGE_NOACCESS);
        assert_eq!(query_prot(va2).1, PAGE_NOACCESS);
        prepare_cpu_overwrite(va2, BLOCK);
        assert_eq!(fetches.load(Ordering::SeqCst), 2);
        assert!(!is_gpu_resident(blk1) && !is_gpu_resident(blk2));
        assert_eq!(query_prot(va1).1, PAGE_READWRITE);
        assert_eq!(query_prot(va2).1, PAGE_READWRITE);

        // Clean up `lazy_readback`, `inplace`, and Win32 handles.
        evict_range(va1, BLOCK);
        evict_range(va2, BLOCK);
        let mut tf = test_fetchers().lock().unwrap_or_else(|e| e.into_inner());
        tf.remove(&blk1);
        tf.remove(&blk2);
        drop(tf);
        crate::inplace::views_gone(va1, BLOCK);
        crate::inplace::views_gone(va2, BLOCK);
        crate::inplace::title_memory_section_gone(sec as usize);
        // SAFETY: unmapping both views and closing section handle.
        unsafe {
            UnmapViewOfFile(view1);
            UnmapViewOfFile(view2);
            CloseHandle(sec);
        }
    }
}
