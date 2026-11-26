//! Which title memory the title wrote since a host copy took it (decision
//! 0007, "Copies of title memory"). The runtime reserves title GPU memory
//! with write watching (`victory-hle` xmem), so the kernel keeps a dirty bit
//! per 4 KiB page; asking for a range's written pages also clears them. Host
//! copies of the same memory can overlap (nested buffers, textures in
//! buffers), so what one asks is kept here for the others: a journal of
//! each page's last write, numbered by the asking.
//!
//! Physical-page views (xmem's sections) cannot be write-watched. There a
//! copy leaves each 64 KiB block it took read-only, and our vectored handler
//! takes the first write to it: it notes the block as written and gives the
//! write access back. The runtime tells us when it maps or unmaps views
//! ([`title_memory_remapped`]), which resets their protection. Blocks found
//! wholly uncommitted are skipped while they remain so; one query can
//! confirm a whole region, including ordinary reservations between views.
//!
//! Writes into physical pages through another view are seen too: the
//! runtime tells us the views of the pages it wrote itself ([`writing`],
//! xmem's `write_physical` and DirectStorage's page destinations), and a
//! write by the title through any view of a guarded page faults like one
//! at the guarded address (the guard makes every view of the page
//! read-only, [`open`] gives them all back; not a page of more than
//! [`MANY_VIEWS`] views). A
//! system call writing into a guarded block fails (none is known to).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Mutex, RwLock};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// PC-native memory class derived from Xbox console `XMemVirtualAlloc` flags
/// and `flProtect` bits before host sanitization strips them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConsoleMemClass {
    Unknown,
    CpuOnly,
    GpuReadOnly,
    WriteCombinedGpuRw,
    CachedGpuRw,
    CoherentGpuRw,
}

fn classify_console_memory(flags: u32, protect: u32) -> ConsoleMemClass {
    const MEM_GRAPHICS: u32 = 0x2;
    const PAGE_WRITECOMBINE: u32 = 0x400;
    const PAGE_GRAPHICS_NOACCESS: u32 = 0x800;
    const PAGE_GRAPHICS_READONLY: u32 = 0x1000;
    const PAGE_GRAPHICS_READWRITE: u32 = 0x2000;
    const PAGE_GRAPHICS_EXECUTE_READ: u32 = 0x8000;
    const PAGE_GRAPHICS_COHERENT: u32 = 0x20000;
    const PAGE_GRAPHICS_NOCACHE: u32 = 0x40000;

    if (flags & MEM_GRAPHICS) == 0 || (protect & PAGE_GRAPHICS_NOACCESS) != 0 {
        ConsoleMemClass::CpuOnly
    } else if (protect & PAGE_GRAPHICS_COHERENT) != 0 || (protect & PAGE_GRAPHICS_NOCACHE) != 0 {
        ConsoleMemClass::CoherentGpuRw
    } else if (protect & PAGE_GRAPHICS_READWRITE) == 0
        && ((protect & PAGE_GRAPHICS_READONLY) != 0 || (protect & PAGE_GRAPHICS_EXECUTE_READ) != 0)
    {
        ConsoleMemClass::GpuReadOnly
    } else if (protect & PAGE_GRAPHICS_READWRITE) != 0 && (protect & PAGE_WRITECOMBINE) != 0 {
        ConsoleMemClass::WriteCombinedGpuRw
    } else if (protect & PAGE_GRAPHICS_READWRITE) != 0 {
        ConsoleMemClass::CachedGpuRw
    } else {
        ConsoleMemClass::Unknown
    }
}

/// Disjoint sorted intervals `start -> (end, ConsoleMemClass)` with adjacent
/// identical classes merged. Protected by its own `RwLock` (never nests with
/// `JOURNAL` or `GUARDS`).
static CONSOLE_MEM: RwLock<BTreeMap<usize, (usize, ConsoleMemClass)>> =
    RwLock::new(BTreeMap::new());

fn console_mem_insert(
    map: &mut BTreeMap<usize, (usize, ConsoleMemClass)>,
    lo: usize,
    hi: usize,
    class: ConsoleMemClass,
) {
    if lo >= hi {
        return;
    }
    // Remove/split any intervals overlapping `[lo, hi)`.
    let mut overlapping: Vec<(usize, usize, ConsoleMemClass)> = Vec::new();
    if let Some((&s, &(e, c))) = map.range(..lo).next_back()
        && e > lo
    {
        overlapping.push((s, e, c));
    }
    for (&s, &(e, c)) in map.range(lo..hi) {
        overlapping.push((s, e, c));
    }
    for &(s, _, _) in &overlapping {
        map.remove(&s);
    }
    for (s, e, c) in overlapping {
        if s < lo {
            map.insert(s, (lo, c));
        }
        if e > hi {
            map.insert(hi, (e, c));
        }
    }
    // Merge with touching left/right neighbors of the same class.
    let mut new_lo = lo;
    let mut new_hi = hi;
    if let Some((&s, &(e, c))) = map.range(..lo).next_back()
        && e == lo
        && c == class
    {
        map.remove(&s);
        new_lo = s;
    }
    if let Some(&(e, c)) = map.get(&hi)
        && c == class
    {
        map.remove(&hi);
        new_hi = e;
    }
    map.insert(new_lo, (new_hi, class));
}

fn console_mem_clear(map: &mut BTreeMap<usize, (usize, ConsoleMemClass)>, lo: usize, hi: usize) {
    if lo >= hi {
        return;
    }
    let mut overlapping: Vec<(usize, usize, ConsoleMemClass)> = Vec::new();
    if let Some((&s, &(e, c))) = map.range(..lo).next_back()
        && e > lo
    {
        overlapping.push((s, e, c));
    }
    for (&s, &(e, c)) in map.range(lo..hi) {
        overlapping.push((s, e, c));
    }
    for &(s, _, _) in &overlapping {
        map.remove(&s);
    }
    for (s, e, c) in overlapping {
        if s < lo {
            map.insert(s, (lo, c));
        }
        if e > hi {
            map.insert(hi, (e, c));
        }
    }
}

/// Records the console memory classification of `[base, base + len)` from raw
/// Xbox `XMemVirtualAlloc` / placeholder split `flags` and `protect` bits.
pub fn note_console_memory(base: usize, len: usize, flags: u32, protect: u32) {
    if base == 0 || len == 0 {
        return;
    }
    let end = base.saturating_add(len);
    if end <= base {
        return;
    }
    let class = classify_console_memory(flags, protect);
    let mut map = CONSOLE_MEM.write().unwrap_or_else(|e| e.into_inner());
    console_mem_insert(&mut map, base, end, class);
}

/// Clears any recorded console memory classification over `[base, base + len)`
/// (called when virtual memory is released/unmapped).
pub fn clear_console_memory(base: usize, len: usize) {
    if base == 0 || len == 0 {
        return;
    }
    let end = base.saturating_add(len);
    if end <= base {
        return;
    }
    let mut map = CONSOLE_MEM.write().unwrap_or_else(|e| e.into_inner());
    console_mem_clear(&mut map, base, end);
}

/// Returns a complete, sorted partition of `[start, end)` into contiguous
/// `(lo, hi, ConsoleMemClass)` sub-ranges (unregistered gaps report `Unknown`).
pub(crate) fn console_mem_class_ranges(
    start: usize,
    end: usize,
) -> Vec<(usize, usize, ConsoleMemClass)> {
    let mut out = Vec::new();
    if start >= end {
        return out;
    }
    let map = CONSOLE_MEM.read().unwrap_or_else(|e| e.into_inner());
    let mut cur = start;
    let push_merged = |out: &mut Vec<(usize, usize, ConsoleMemClass)>,
                       lo: usize,
                       hi: usize,
                       c: ConsoleMemClass| {
        if lo >= hi {
            return;
        }
        if let Some(last) = out.last_mut()
            && last.1 == lo
            && last.2 == c
        {
            last.1 = hi;
        } else {
            out.push((lo, hi, c));
        }
    };
    if let Some((_, &(e, c))) = map.range(..start).next_back()
        && e > start
    {
        let hi = e.min(end);
        push_merged(&mut out, start, hi, c);
        cur = hi;
    }
    if cur >= end {
        return out;
    }
    for (&s, &(e, c)) in map.range(cur..end) {
        if s > cur {
            push_merged(&mut out, cur, s.min(end), ConsoleMemClass::Unknown);
        }
        let lo = s.max(cur);
        let hi = e.min(end);
        push_merged(&mut out, lo, hi, c);
        cur = cur.max(hi);
        if cur >= end {
            break;
        }
    }
    if cur < end {
        push_merged(&mut out, cur, end, ConsoleMemClass::Unknown);
    }
    out
}

/// Monotonic epoch incremented whenever any guarded `MEM_MAPPED` block leaves
/// the clean state (write fault, alias open, `writing`, `views_mapped`,
/// `title_memory_remapped`, `wrote`, or `guard` reporting newly dirty blocks).
/// Enables O(1) clean steady-state skipping in `sparse_uma`.
static GUARD_EPOCH: AtomicU64 = AtomicU64::new(1);

/// Current monotonic guard epoch (`Acquire` pairs with `Release` bumps).
pub(crate) fn guard_epoch() -> u64 {
    GUARD_EPOCH.load(Ordering::Acquire)
}

#[inline]
fn bump_guard_epoch() {
    GUARD_EPOCH.fetch_add(1, Ordering::Release);
}

#[cfg(test)]
thread_local! {
    static THREAD_MIXED_VQS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    static THREAD_GUARD_GAP_BLOCKS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    static THREAD_UNCOMMITTED_VQS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    static KNOWN_HOLES_IN_TEST: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
    static REST_IN_TEST: std::cell::Cell<Duration> = const { std::cell::Cell::new(Duration::ZERO) };
}

#[cfg(test)]
pub(crate) fn mixed_virtual_queries() -> u64 {
    THREAD_MIXED_VQS.with(std::cell::Cell::get)
}

#[cfg(test)]
pub(crate) fn reset_mixed_virtual_queries() {
    THREAD_MIXED_VQS.with(|c| c.set(0));
}

#[cfg(test)]
fn reset_guard_gap_blocks() {
    THREAD_GUARD_GAP_BLOCKS.with(|c| c.set(0));
}

#[cfg(test)]
fn guard_gap_blocks() -> u64 {
    THREAD_GUARD_GAP_BLOCKS.with(std::cell::Cell::get)
}

#[cfg(test)]
fn reset_uncommitted_vqs() {
    THREAD_UNCOMMITTED_VQS.with(|c| c.set(0));
}

#[cfg(test)]
fn uncommitted_vqs() -> u64 {
    THREAD_UNCOMMITTED_VQS.with(std::cell::Cell::get)
}

/// Whether placeholder holes are trusted via `title_memory_remapped` instead of
/// issuing `VirtualQuery` on every submission (`VICTORY_KNOWN_HOLES`, default ON,
/// `=0` disables).
fn known_holes() -> bool {
    #[cfg(test)]
    if let Some(v) = KNOWN_HOLES_IN_TEST.with(std::cell::Cell::get) {
        return v;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_KNOWN_HOLES").is_none_or(|v| v != "0"))
}

/// How long a block opened by a CPU write fault rests before being re-guarded
/// (`VICTORY_GUARD_REST=<microseconds>`, default `3000` µs = 3 ms, `=0` disables).
/// Resting blocks still report as dirty at each check so no writes are missed,
/// avoiding back-to-back `VirtualProtect(PAGE_READONLY)` TLB shootdown storms.
fn guard_rest() -> Duration {
    #[cfg(test)]
    {
        REST_IN_TEST.with(std::cell::Cell::get)
    }
    #[cfg(not(test))]
    {
        static US: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
        let us = *US.get_or_init(|| match std::env::var("VICTORY_GUARD_REST") {
            Ok(v) => v.trim().parse::<u64>().unwrap_or(3000),
            Err(_) => 3000,
        });
        Duration::from_micros(us)
    }
}

/// True if `block` lies in a Windows `MEM_RESERVE_PLACEHOLDER` reservation,
/// where pages can only appear via `MapViewOfFile3`/`VirtualAlloc2(MEM_REPLACE_PLACEHOLDER)`
/// (always routed through `xmem` -> `title_memory_remapped`), never via plain
/// `VirtualAlloc(MEM_COMMIT)`.
fn placeholder(block: usize) -> bool {
    #[repr(C)]
    #[derive(Default)]
    struct Region {
        allocation_base: usize,
        allocation_protect: u32,
        region_type: u32,
        region_size: usize,
        commit_size: usize,
        partition_id: usize,
        node_preference: usize,
    }
    #[link(name = "ntdll", kind = "raw-dylib")]
    unsafe extern "system" {
        fn NtQueryVirtualMemory(
            process: *mut std::ffi::c_void,
            at: *const std::ffi::c_void,
            class: u32,
            out: *mut std::ffi::c_void,
            len: usize,
            returned: *mut usize,
        ) -> i32;
    }
    const MEMORY_REGION_INFORMATION_EX: u32 = 7;
    const PLACEHOLDER_RESERVATION: u32 = 1 << 8;
    let mut region = Region::default();
    // SAFETY: valid output buffer and size for current process query.
    let status = unsafe {
        NtQueryVirtualMemory(
            windows_sys::Win32::System::Threading::GetCurrentProcess(),
            (block * BLOCK) as *const _,
            MEMORY_REGION_INFORMATION_EX,
            (&raw mut region).cast(),
            size_of::<Region>(),
            std::ptr::null_mut(),
        )
    };
    status >= 0 && (region.region_type & PLACEHOLDER_RESERVATION) != 0
}

/// Whether `VICTORY_GUARD_ASK=1` forces Windows `VirtualQuery` checks on
/// every guarded range query instead of trusting `GUARDS`.
fn guard_ask() -> bool {
    static ASK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ASK.get_or_init(|| std::env::var_os("VICTORY_GUARD_ASK").is_some_and(|v| v == "1"))
}

/// The unit tracked and copied: 64 KiB, the console's page (the kernel's
/// 4 KiB pages fold into it). Finer units made the checks of the world's
/// big buffers cost more than the copies they saved.
pub(crate) const PAGE: usize = 64 << 10;
/// The kernel's page, the unit of its answers.
const KERNEL_PAGE: usize = 4096;

struct Journal {
    /// Numbers the askings; 0 is "never".
    asked: u64,
    /// Page number → the asking that last found it written.
    written: BTreeMap<usize, u64>,
    /// Room for the kernel's answer.
    found: Vec<usize>,
    /// Page ranges asked ahead for a submission's refreshes ([`ask_ahead`])
    /// and the asking that did: `writes` answers for a range inside one
    /// from the journal, without another kernel call.
    ahead: Vec<(usize, usize)>,
    ahead_asked: u64,
    /// The ahead asking stands for a whole submission ([`ask_once`]): no
    /// other ahead asking is made, and `ahead` stays, until
    /// [`ask_once_end`].
    kept: bool,
}

static JOURNAL: Mutex<Journal> = Mutex::new(Journal {
    asked: 0,
    written: BTreeMap::new(),
    found: Vec::new(),
    ahead: Vec::new(),
    ahead_asked: 0,
    kept: false,
});

impl Journal {
    #[cfg(test)]
    fn new_for_test() -> Self {
        Self {
            asked: 0,
            written: BTreeMap::new(),
            found: Vec::new(),
            ahead: Vec::new(),
            ahead_asked: 0,
            kept: false,
        }
    }
}

/// Ranges of one submission closer than this are asked about together:
/// walking the pages between them costs less than another kernel call (a
/// call that resets written pages costs 30-70 microseconds whatever the
/// range).
const AHEAD_GAP: usize = (8 << 20) / PAGE;
/// The most one asking covers (its answer needs 8 bytes per 4 KiB page).
const AHEAD_MAX: usize = (1 << 30) / PAGE;

/// Asks the kernel about many ranges `(start, end)` at once, before the
/// [`writes`] of each: ranges close together are one asking, and `writes`
/// then answers from the journal until [`ask_ahead_done`]. The world's
/// submissions refresh thousands of host copies (6,664 askings a frame,
/// 456 ms, in run 14). Ranges that are not write-watched, or lie alone, are
/// left to `writes`.
pub(crate) fn ask_ahead(ranges: &[(usize, usize)]) {
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    ask_ahead_on(&mut j, ranges);
}

fn ask_ahead_on(j: &mut Journal, ranges: &[(usize, usize)]) {
    // The submission's one asking stands ([`ask_once`]): what it does not
    // hold is left to `writes`.
    if j.kept {
        return;
    }
    // Skip ranges already known to be guarded `MEM_MAPPED` views so `ask_run`
    // does not waste failing `GetWriteWatch` syscalls binary-splitting them.
    let mut pieces: Vec<(usize, usize)> = if guard_ask() {
        ranges
            .iter()
            .filter(|r| r.0 < r.1)
            .map(|&(start, end)| (start / PAGE, end.div_ceil(PAGE)))
            .collect()
    } else {
        let g = guards();
        ranges
            .iter()
            .filter(|r| r.0 < r.1)
            .map(|&(start, end)| (start / PAGE, end.div_ceil(PAGE)))
            .filter(|&(first, last)| !g.known_views.covers(first, last))
            .collect()
    };
    if pieces.len() < 2 {
        return;
    }
    pieces.sort_unstable();
    j.asked += 1;
    j.ahead_asked = j.asked;
    j.ahead.clear();
    let mut i = 0;
    while i < pieces.len() {
        let first = pieces[i].0;
        let mut last = pieces[i].1;
        let mut k = i + 1;
        while k < pieces.len()
            && pieces[k].0 <= last + AHEAD_GAP
            && last.max(pieces[k].1) - first <= AHEAD_MAX
        {
            last = last.max(pieces[k].1);
            k += 1;
        }
        ask_run(j, &pieces[i..k]);
        i = k;
    }
}

/// Since the last frame line: the kernel askings that took a millisecond
/// or more, their microseconds, and the longest asking's. An asking walks
/// page tables, some ten microseconds for most ranges; in run 75 the 2,644
/// askings of a frame took 219 ms in the lines where the pass over the
/// heap's views ran long, 25-50 ms in the others. These numbers say
/// whether a few askings wait for something or all of them are slow.
#[cfg(feature = "diagnostics")]
static SLOW: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

#[cfg(feature = "diagnostics")]
fn slow(us: u64) {
    use std::sync::atomic::Ordering::Relaxed;
    if us >= 1000 {
        SLOW[0].fetch_add(1, Relaxed);
        SLOW[1].fetch_add(us, Relaxed);
    }
    SLOW[2].fetch_max(us, Relaxed);
}

/// [`SLOW`], taken.
#[cfg(feature = "diagnostics")]
pub(crate) fn slow_asks() -> [u64; 3] {
    [0, 1, 2].map(|i| SLOW[i].swap(0, std::sync::atomic::Ordering::Relaxed))
}

/// The submission's refreshes are done: `writes` asks the kernel again.
pub(crate) fn ask_ahead_done() {
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    if !j.kept {
        j.ahead.clear();
    }
}

/// Whether a submission asks the kernel once, before its first refresh,
/// about every range its lists refresh (`VICTORY_ASK_ONCE=1`; off by
/// default).
///
/// A submission's refreshes are done in groups, between the title's own
/// buffer copies (each copy flushes the refreshes recorded before it), and
/// every group asked the kernel again: 2,644 askings a frame, 70 a
/// submission. With the switch the journal answers all of them from one
/// asking made when the submission starts.
///
/// What changes: memory the title's other threads write while the
/// submission is being prepared is seen by the next submission's asking,
/// not by this one's later groups. What this submission's lists read was
/// written before they were submitted, so before the asking. Our own
/// writes into title memory while preparing are told to the journal
/// ([`wrote`]) or end the asking ([`ask_once_end`]).
pub(crate) fn once() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // On by default since run 81 (with card buffers: 375-450 ms a frame
    // against 573-647, the picture unchanged over four minutes);
    // `VICTORY_ASK_ONCE=0` asks per range, as before.
    *ON.get_or_init(|| std::env::var_os("VICTORY_ASK_ONCE").is_none_or(|v| v != "0"))
}

/// With [`once`]: asks about `ranges` (everything a submission's lists
/// refresh) as [`ask_ahead`] does, and lets that asking stand until
/// [`ask_once_end`]. Covers both `MEM_PRIVATE` write-watched ranges AND
/// `MEM_MAPPED` guarded views (`guard_if_known_view` / `writes_mixed`) so
/// repeated `flush_one`/`flush` refreshes of large mapped pools (e.g. 182 MiB
/// buffers refreshed 20x/frame) hit the submission journal in O(log N + K)
/// with zero `GetWriteWatch` and zero `VirtualQuery` syscalls.
pub(crate) fn ask_once(ranges: &[(usize, usize)]) {
    if !once() {
        return;
    }
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    ask_once_on(&mut j, ranges);
}

fn ask_once_on(j: &mut Journal, ranges: &[(usize, usize)]) {
    ask_once_end_on(j);
    ask_ahead_on(j, ranges);
    if j.ahead.is_empty() {
        // `ask_ahead` asked nothing (fewer than two ranges, or all mapped).
        j.asked += 1;
        j.ahead_asked = j.asked;
    }
    // `ask_ahead` leaves a range that lies alone (or is `MEM_MAPPED`/mixed) to
    // `writes`, which asks once per flush: ask each remaining range ONCE here
    // and record it in `j.ahead` so every subsequent refresh in this submission
    // hits the journal directly.
    for &(start, end) in ranges {
        if start >= end {
            continue;
        }
        let (first, last) = (start / PAGE, end.div_ceil(PAGE));
        if j.ahead.iter().any(|&(a, b)| a <= first && last <= b) {
            continue;
        }
        // Fast path for known `MEM_MAPPED` views (clean OR sparsely dirty):
        // handles `[first, last)` in O(log N + K) under ONE `GUARDS` lock with
        // 0 `GetWriteWatch` and 0 `VirtualQuery` syscalls!
        if let Some(guarded_pages) = guard_if_known_view(first, last) {
            if !guarded_pages.is_empty() {
                let now = j.ahead_asked;
                for page in guarded_pages {
                    j.written.insert(page, now);
                }
                bump_guard_epoch();
            }
            j.ahead.push((first, last));
            continue;
        }
        if ask_kernel(j, first, last) {
            j.ahead.push((first, last));
            continue;
        }
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        if let Some(guarded_pages) = writes_mixed(j, first * PAGE, last * PAGE, true) {
            #[cfg(feature = "diagnostics")]
            {
                use crate::copystat::{C, add};
                add(C::Watches, 1);
                add(C::WatchUs, started.elapsed().as_micros() as u64);
                slow(started.elapsed().as_micros() as u64);
            }
            let now = j.ahead_asked;
            let any_guarded = !guarded_pages.is_empty();
            for page in guarded_pages {
                j.written.insert(page, now);
            }
            if any_guarded {
                bump_guard_epoch();
            }
            j.ahead.push((first, last));
        }
    }
    j.kept = !j.ahead.is_empty();
}

/// The submission's one asking stands no longer: its refreshes are done,
/// or title memory was written in a way the journal was not told.
pub(crate) fn ask_once_end() {
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    ask_once_end_on(&mut j);
}

fn ask_once_end_on(j: &mut Journal) {
    if j.kept {
        j.kept = false;
        j.ahead.clear();
    }
}

/// We wrote title memory `[at, at + len)` while a submission's one asking
/// stands (the title's buffer copies, done at submission): its pages count
/// as written after that asking, so the refreshes that follow copy them.
pub(crate) fn wrote(at: usize, len: usize) {
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    wrote_on(&mut j, at, len);
}

fn wrote_on(j: &mut Journal, at: usize, len: usize) {
    if !j.kept || len == 0 {
        return;
    }
    j.asked += 1;
    let now = j.asked;
    // The asking that stands moves with it: `asked` and `writes` give it as
    // "now", and a page noted written after "now" stays stale for every
    // copy that takes it (copied at `now`, written at `now + 1`): copied
    // again at every refresh, and a reason to settle flights
    // (fxceless/victory-fork#410).
    j.ahead_asked = now;
    for page in at / PAGE..at.saturating_add(len).div_ceil(PAGE) {
        j.written.insert(page, now);
    }
    bump_guard_epoch();
}

/// One kernel asking for `pieces` (page ranges in order); by halves if it
/// fails, as a run across two reservations or into memory that is not
/// write-watched does. A piece alone is left to [`writes`].
fn ask_run(j: &mut Journal, pieces: &[(usize, usize)]) {
    if pieces.len() < 2 {
        return;
    }
    let first = pieces[0].0;
    let last = pieces.iter().map(|p| p.1).max().unwrap_or(first);
    if ask_kernel(j, first, last) {
        j.ahead.push((first, last));
        return;
    }
    let (a, b) = pieces.split_at(pieces.len() / 2);
    ask_run(j, a);
    ask_run(j, b);
}

/// Asks the kernel for the written pages of pages `[first, last)` and notes
/// them in the journal under the ahead asking; false (and nothing changed)
/// if the range is not write-watched memory of one reservation.
fn ask_kernel(j: &mut Journal, first: usize, last: usize) -> bool {
    use windows_sys::Win32::System::Memory::GetWriteWatch;
    const WRITE_WATCH_FLAG_RESET: u32 = 1;
    let mut count = (last - first) * PAGE / KERNEL_PAGE;
    j.found.resize(count.max(j.found.len()), 0);
    let mut granularity = 0;
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    // SAFETY: the answer buffer holds `count` entries.
    let failed = unsafe {
        GetWriteWatch(
            WRITE_WATCH_FLAG_RESET,
            (first * PAGE) as *const _,
            (last - first) * PAGE,
            j.found.as_mut_ptr().cast(),
            &mut count,
            &mut granularity,
        )
    } != 0;
    if failed {
        return false;
    }
    #[cfg(feature = "diagnostics")]
    {
        use crate::copystat::{C, add};
        add(C::Watches, 1);
        add(C::WatchUs, started.elapsed().as_micros() as u64);
        slow(started.elapsed().as_micros() as u64);
    }
    let now = j.ahead_asked;
    for k in 0..count {
        let page = j.found[k] / PAGE;
        j.written.insert(page, now);
    }
    true
}

/// What [`writes`] would say of `[start, end)` from the journal alone, if
/// the range lies in one asked ahead ([`ask_ahead`], [`ask_once`]): the
/// pages written since tracking began and the asking that last found each
/// written, with the number of the asking that stands (as [`writes`] gives
/// it). `None` if the kernel would have to be asked.
pub(crate) fn asked(start: usize, end: usize) -> Option<(u64, Vec<(usize, u64)>)> {
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    let j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    j.ahead
        .iter()
        .any(|&(a, b)| a <= first && last <= b)
        .then(|| (j.ahead_asked, j.written.range(first..last).map(|(&p, &w)| (p, w)).collect()))
}

/// The pages of `[start, end)` (whole pages) as last written, for pages
/// written since tracking began: `(page number, asking)` in page order, and
/// this asking's number, which is higher than any in the list but those the
/// kernel just reported. `None` if the memory is not write-watched (nor,
/// with `guarded`, view memory, which is then guarded).
pub(crate) fn writes(start: usize, end: usize, guarded: bool) -> Option<(u64, Vec<(usize, u64)>)> {
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    writes_on(&mut j, start, end, guarded)
}

fn writes_on(
    j: &mut Journal,
    start: usize,
    end: usize,
    guarded: bool,
) -> Option<(u64, Vec<(usize, u64)>)> {
    use windows_sys::Win32::System::Memory::GetWriteWatch;
    const WRITE_WATCH_FLAG_RESET: u32 = 1;
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    // Asked ahead with the submission's other ranges: the journal has it.
    if j.ahead.iter().any(|&(a, b)| a <= first && last <= b) {
        return Some((
            j.ahead_asked,
            j.written
                .range(first..last)
                .map(|(&p, &w)| (p, w))
                .collect(),
        ));
    }
    // Fast path for known `MEM_MAPPED` views (clean OR sparsely dirty):
    // skips guaranteed-failing `GetWriteWatch` AND `writes_mixed` `VirtualQuery`!
    if guarded && let Some(guarded_pages) = guard_if_known_view(first, last) {
        j.asked += 1;
        let now = j.asked;
        if !guarded_pages.is_empty() {
            for page in guarded_pages {
                j.written.insert(page, now);
            }
            bump_guard_epoch();
        }
        return Some((
            now,
            j.written
                .range(first..last)
                .map(|(&p, &w)| (p, w))
                .collect(),
        ));
    }
    let mut count = (last - first) * PAGE / KERNEL_PAGE;
    j.found.resize(count.max(j.found.len()), 0);
    let mut granularity = 0;
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    // SAFETY: the answer buffer holds `count` entries.
    let failed = unsafe {
        GetWriteWatch(
            WRITE_WATCH_FLAG_RESET,
            (first * PAGE) as *const _,
            (last - first) * PAGE,
            j.found.as_mut_ptr().cast(),
            &mut count,
            &mut granularity,
        )
    } != 0;
    let guarded = if failed {
        count = 0;
        writes_mixed(j, first * PAGE, last * PAGE, guarded)?
    } else {
        Vec::new()
    };
    #[cfg(feature = "diagnostics")]
    {
        use crate::copystat::{C, add};
        add(C::Watches, 1);
        add(C::WatchUs, started.elapsed().as_micros() as u64);
        slow(started.elapsed().as_micros() as u64);
    }
    j.asked += 1;
    let now = j.asked;
    let any_guarded = !guarded.is_empty();
    let pages = j.found[..count].iter().map(|&at| at / PAGE).chain(guarded);
    for page in pages.collect::<Vec<_>>() {
        j.written.insert(page, now);
    }
    if any_guarded {
        bump_guard_epoch();
    }
    Some((
        now,
        j.written
            .range(first..last)
            .map(|(&p, &w)| (p, w))
            .collect(),
    ))
}

/// Fast-path variant of [`writes`] for spans already known to be committed
/// `MEM_MAPPED` views (`sparse_uma::RegionKind::MappedView`):
/// - Skips the guaranteed-failing `GetWriteWatch` syscall and `writes_mixed`'s
///   per-view `VirtualQuery` walk, calling [`guard`] directly.
/// - Shares `JOURNAL` with [`writes`] (`Shadow` / `texture`) and returns only
///   journal entries with `asking > min_seq`, avoiding copying thousands of
///   unchanged historical entries on clean or sparsely dirty spans.
/// - Returns `None` if `guard` rejects the range (signaling the caller to
///   re-classify).
pub(crate) fn writes_guarded_since(
    start: usize,
    end: usize,
    min_seq: u64,
) -> Option<(u64, Vec<(usize, u64)>)> {
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    if first >= last {
        return Some((min_seq, Vec::new()));
    }
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    let j = &mut *j;
    if j.ahead.iter().any(|&(a, b)| a <= first && last <= b) {
        let now = j.ahead_asked;
        if now <= min_seq {
            return Some((now, Vec::new()));
        }
        return Some((
            now,
            j.written
                .range(first..last)
                .filter(|&(_, &w)| w > min_seq)
                .map(|(&p, &w)| (p, w))
                .collect(),
        ));
    }
    let guarded = guard(first * PAGE, last * PAGE)?;
    if !guarded.is_empty() {
        j.asked += 1;
        let now = j.asked;
        for page in guarded {
            j.written.insert(page, now);
        }
        bump_guard_epoch();
    }
    let now = j.asked;
    if now <= min_seq {
        return Some((now, Vec::new()));
    }
    Some((
        now,
        j.written
            .range(first..last)
            .filter(|&(_, &w)| w > min_seq)
            .map(|(&p, &w)| (p, w))
            .collect(),
    ))
}

/// Fast-path variant of [`writes`] for spans already known to be a single
/// `AllocationBase` of `MEM_WRITE_WATCH` private memory
/// (`sparse_uma::RegionKind::PrivateWriteWatch`):
/// - Runs one `GetWriteWatch(WRITE_WATCH_FLAG_RESET)` syscall; returns `None`
///   immediately on failure (e.g. after lazy `MEM_PRIVATE` -> `MEM_MAPPED`
///   promotion) so the caller reclassifies without falling into `writes_mixed`.
/// - Shares `JOURNAL` with [`writes`] (`Shadow`) and returns only journal
///   entries with `asking > min_seq` (zero allocations when clean).
pub(crate) fn writes_private_since(
    start: usize,
    end: usize,
    min_seq: u64,
) -> Option<(u64, Vec<(usize, u64)>)> {
    use windows_sys::Win32::System::Memory::GetWriteWatch;
    const WRITE_WATCH_FLAG_RESET: u32 = 1;
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    if first >= last {
        return Some((min_seq, Vec::new()));
    }
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    let j = &mut *j;
    if j.ahead.iter().any(|&(a, b)| a <= first && last <= b) {
        let now = j.ahead_asked;
        if now <= min_seq {
            return Some((now, Vec::new()));
        }
        return Some((
            now,
            j.written
                .range(first..last)
                .filter(|&(_, &w)| w > min_seq)
                .map(|(&p, &w)| (p, w))
                .collect(),
        ));
    }
    let mut count = (last - first) * PAGE / KERNEL_PAGE;
    j.found.resize(count.max(j.found.len()), 0);
    let mut granularity = 0;
    // SAFETY: `j.found` holds `count` entries for `[first * PAGE, last * PAGE)`.
    let failed = unsafe {
        GetWriteWatch(
            WRITE_WATCH_FLAG_RESET,
            (first * PAGE) as *const _,
            (last - first) * PAGE,
            j.found.as_mut_ptr().cast(),
            &mut count,
            &mut granularity,
        )
    } != 0;
    if failed {
        return None;
    }
    if count > 0 {
        j.asked += 1;
        let now = j.asked;
        for k in 0..count {
            let page = j.found[k] / PAGE;
            j.written.insert(page, now);
        }
    }
    let now = j.asked;
    if now <= min_seq {
        return Some((now, Vec::new()));
    }
    Some((
        now,
        j.written
            .range(first..last)
            .filter(|&(_, &w)| w > min_seq)
            .map(|(&p, &w)| (p, w))
            .collect(),
    ))
}

/// Fallback when a single `GetWriteWatch` call over `[start, end)` fails:
/// walks VirtualQuery sub-regions so ranges spanning multiple adjacent
/// single-view `MEM_WRITE_WATCH` private mappings, unmapped placeholder holes,
/// and promoted `MEM_MAPPED` aliased views each use their optimal tracking
/// mechanism (`GetWriteWatch` for `MEM_WRITE_WATCH` private regions with zero
/// `PAGE_READONLY` faults; `guard()` only for `MEM_MAPPED` views and holes).
/// Returns `None` if any committed private sub-region is not write-watched.
fn writes_mixed(j: &mut Journal, start: usize, end: usize, guarded: bool) -> Option<Vec<usize>> {
    use windows_sys::Win32::System::Memory::{
        GetWriteWatch, MEM_COMMIT, MEM_MAPPED, MEM_PRIVATE, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };
    const WRITE_WATCH_FLAG_RESET: u32 = 1;
    // Fast path: if `guarded` and `[first, last)` is already known view memory
    // (clean OR sparsely dirty), process in O(log N + K) with 0 `VirtualQuery`!
    if guarded {
        let (first, last) = (start / BLOCK, end.div_ceil(BLOCK));
        if let Some(pages) = guard_if_known_view(first, last) {
            return Some(pages);
        }
    }
    let mut at = start;
    let mut dirty = Vec::new();
    let mut any = false;
    while at < end {
        // SAFETY: plain VirtualQuery into a zeroed struct.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        #[cfg(test)]
        THREAD_MIXED_VQS.with(|c| c.set(c.get() + 1));
        // SAFETY: valid out pointer.
        let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
        if n == 0 {
            return None;
        }
        let region_end = (info.BaseAddress as usize)
            .saturating_add(info.RegionSize)
            .min(end);
        if region_end <= at {
            return None;
        }
        if info.State == MEM_COMMIT && info.Type == MEM_PRIVATE {
            let sub_first = at / PAGE;
            let sub_last = region_end.div_ceil(PAGE);
            let mut count = (sub_last - sub_first) * PAGE / KERNEL_PAGE;
            j.found.resize(count.max(j.found.len()), 0);
            let mut granularity = 0;
            // SAFETY: `j.found` holds `count` entries; querying committed private region.
            let sub_failed = unsafe {
                GetWriteWatch(
                    WRITE_WATCH_FLAG_RESET,
                    (sub_first * PAGE) as *const _,
                    (sub_last - sub_first) * PAGE,
                    j.found.as_mut_ptr().cast(),
                    &mut count,
                    &mut granularity,
                )
            } != 0;
            if sub_failed {
                // Ordinary committed private memory (not `MEM_WRITE_WATCH`).
                return None;
            }
            dirty.extend(j.found[..count].iter().map(|&addr| addr / PAGE));
            any = true;
        } else {
            if !guarded {
                return None;
            }
            // Coalesce consecutive non-private (`MEM_MAPPED` / placeholder hole)
            // sub-regions into one `guard()` call.
            let mut guard_end = region_end;
            while guard_end < end {
                // SAFETY: plain VirtualQuery probe.
                let mut next_info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
                #[cfg(test)]
                THREAD_MIXED_VQS.with(|c| c.set(c.get() + 1));
                // SAFETY: valid out pointer.
                let m = unsafe {
                    VirtualQuery(guard_end as *const _, &mut next_info, size_of_val(&next_info))
                };
                if m == 0 || (next_info.State == MEM_COMMIT && next_info.Type == MEM_PRIVATE) {
                    break;
                }
                let next_end = (next_info.BaseAddress as usize)
                    .saturating_add(next_info.RegionSize)
                    .min(end);
                if next_end <= guard_end {
                    break;
                }
                // Stop if we hit free memory that isn't a placeholder/view.
                if next_info.Type != MEM_MAPPED && next_info.State == MEM_COMMIT {
                    break;
                }
                guard_end = next_end;
            }
            #[cfg(feature = "diagnostics")]
            let guarding = std::time::Instant::now();
            let sub_written = guard(at, guard_end)?;
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::GuardUs, guarding.elapsed().as_micros() as u64);
            dirty.extend(sub_written);
            any = true;
            at = guard_end;
            continue;
        }
        at = region_end;
    }
    any.then_some(dirty)
}

/// Guarded blocks of view memory.
const BLOCK: usize = PAGE;
const _: () = assert!(BLOCK == PAGE);

/// Coalesced set of disjoint, maximal half-open block intervals `[start, end)`
/// (`start < end`, with `prev_end < next_start`). Represents an entire 182 MiB
/// (`2,848`-block) `MEM_MAPPED` pool in ONE `BTreeMap` entry so membership,
/// full-range coverage, gap extraction (`O(log N + K)`), and contiguous range
/// removal never walk clean blocks individually.
struct IntervalSet(BTreeMap<usize, usize>);

impl IntervalSet {
    const fn new() -> Self {
        Self(BTreeMap::new())
    }

    #[inline]
    fn contains(&self, x: usize) -> bool {
        self.0.range(..=x).next_back().is_some_and(|(_, &e)| x < e)
    }

    #[inline]
    fn covering_end(&self, x: usize) -> Option<usize> {
        self.0
            .range(..=x)
            .next_back()
            .and_then(|(_, &e)| (x < e).then_some(e))
    }

    /// True if every block in `[lo, hi)` is covered (O(log N): 1 lookup because
    /// adjacent intervals are always merged).
    #[inline]
    fn covers(&self, lo: usize, hi: usize) -> bool {
        lo >= hi || self.0.range(..=lo).next_back().is_some_and(|(_, &e)| hi <= e)
    }

    #[inline]
    fn insert(&mut self, x: usize) {
        self.insert_range(x, x.saturating_add(1));
    }

    fn insert_range(&mut self, lo: usize, hi: usize) {
        if lo >= hi {
            return;
        }
        let mut new_lo = lo;
        let mut new_hi = hi;
        // Include a preceding interval if it overlaps or touches `lo` (`prev_e >= lo`).
        if let Some((&s, &e)) = self.0.range(..=lo).next_back()
            && e >= lo
        {
            new_lo = s;
            new_hi = new_hi.max(e);
        }
        let to_remove: Vec<usize> = self
            .0
            .range(new_lo..=hi)
            .map(|(&s, &e)| {
                new_hi = new_hi.max(e);
                s
            })
            .collect();
        for s in to_remove {
            self.0.remove(&s);
        }
        self.0.insert(new_lo, new_hi);
    }

    #[inline]
    fn remove(&mut self, x: usize) -> bool {
        !self.remove_range(x, x.saturating_add(1)).is_empty()
    }

    /// Removes `[lo, hi)` and returns the exact removed sub-intervals clipped to
    /// `[lo, hi)` in ascending, disjoint order.
    fn remove_range(&mut self, lo: usize, hi: usize) -> Vec<(usize, usize)> {
        let mut removed = Vec::new();
        if lo >= hi {
            return removed;
        }
        let mut overlapping: Vec<(usize, usize)> = Vec::new();
        if let Some((&s, &e)) = self.0.range(..lo).next_back()
            && e > lo
        {
            overlapping.push((s, e));
        }
        for (&s, &e) in self.0.range(lo..hi) {
            overlapping.push((s, e));
        }
        if overlapping.is_empty() {
            return removed;
        }
        for &(s, _) in &overlapping {
            self.0.remove(&s);
        }
        for (s, e) in overlapping {
            let cut_lo = s.max(lo);
            let cut_hi = e.min(hi);
            if cut_lo < cut_hi {
                removed.push((cut_lo, cut_hi));
            }
            if s < lo {
                self.0.insert(s, lo);
            }
            if e > hi {
                self.0.insert(hi, e);
            }
        }
        removed
    }

    /// Returns all uncovered sub-intervals `[a, b) ⊆ [lo, hi)` in ascending order
    /// in `O(log N + K)` where `K` is the number of gaps.
    fn gaps(&self, lo: usize, hi: usize) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        if lo >= hi {
            return out;
        }
        let mut cur = lo;
        if let Some((_, &e)) = self.0.range(..=lo).next_back()
            && e > cur
        {
            cur = e.min(hi);
        }
        if cur >= hi {
            return out;
        }
        for (&s, &e) in self.0.range(cur..hi) {
            if s > cur {
                out.push((cur, s.min(hi)));
            }
            cur = cur.max(e).min(hi);
            if cur >= hi {
                break;
            }
        }
        if cur < hi {
            out.push((cur, hi));
        }
        out
    }

    /// Number of covered blocks inside `[lo, hi)` in `O(log N + M)`.
    fn count_in(&self, lo: usize, hi: usize) -> usize {
        if lo >= hi {
            return 0;
        }
        let mut total = 0;
        if let Some((_, &e)) = self.0.range(..lo).next_back()
            && e > lo
        {
            total += e.min(hi) - lo;
        }
        for (&s, &e) in self.0.range(lo..hi) {
            total += e.min(hi) - s;
        }
        total
    }
}

struct Guards {
    /// Blocks we made read-only (coalesced intervals).
    protected: IntervalSet,
    /// Of those, the ones not written since a copy took them (coalesced intervals).
    clean: IntervalSet,
    /// Blocks already copied as zeros and confirmed uncommitted (coalesced intervals).
    absent: IntervalSet,
    /// Subset of `absent` confirmed to be `MEM_RESERVE_PLACEHOLDER` reservations
    /// (`VICTORY_KNOWN_HOLES`): skipped with 0 `VirtualQuery` until `title_memory_remapped`.
    held: IntervalSet,
    /// Intervals confirmed to be protected `MEM_MAPPED` views or `MEM_RESERVE_PLACEHOLDER`
    /// holes (populated only after `VirtualProtect` succeeds or `placeholder()` confirms;
    /// cleared by `title_memory_remapped()`).
    known_views: IntervalSet,
    /// Blocks whose physical page was guarded together with its other views
    /// (those views' blocks are in it too): a write to one opens them all.
    /// A write range can be huge (`fread`), so only these are looked up.
    aliased: BTreeSet<usize>,
    /// Blocks recently opened by a CPU write fault and when (`VICTORY_GUARD_REST`).
    opened: BTreeMap<usize, Instant>,
}

impl Guards {
    fn note_opened(&mut self, block: usize) {
        const MANY: usize = 8192;
        if self.opened.len() >= MANY {
            let cutoff = Duration::from_millis(50);
            self.opened.retain(|_, at| at.elapsed() < cutoff);
            if self.opened.len() >= MANY {
                self.opened.clear();
            }
        }
        self.opened.insert(block, Instant::now());
    }

    #[inline]
    fn is_resting(&self, block: usize, rest: Duration) -> bool {
        !rest.is_zero() && self.opened.get(&block).is_some_and(|at| at.elapsed() < rest)
    }
}

static GUARDS: Mutex<Guards> = Mutex::new(Guards {
    protected: IntervalSet::new(),
    clean: IntervalSet::new(),
    absent: IntervalSet::new(),
    held: IntervalSet::new(),
    known_views: IntervalSet::new(),
    aliased: BTreeSet::new(),
    opened: BTreeMap::new(),
});

fn guards() -> std::sync::MutexGuard<'static, Guards> {
    GUARDS.lock().unwrap_or_else(|e| e.into_inner())
}

/// Processes ONLY the non-clean gaps of `[first, last)` inside `g` in
/// `O(log N + K)` (skipping contiguous clean intervals, confirmed placeholder
/// `held` holes with 0 `VirtualQuery`, and confirmed-uncommitted `absent` runs;
/// honoring `VICTORY_GUARD_REST` on recently faulted blocks).
fn guard_gaps(g: &mut Guards, first: usize, last: usize) -> Vec<usize> {
    let mut written = Vec::new();
    if g.clean.covers(first, last) {
        return written;
    }
    let holes_known = known_holes();
    let rest = guard_rest();
    let gaps = g.clean.gaps(first, last);
    let mut uncommitted_until = first;
    for (gap_start, gap_end) in gaps {
        let mut block = gap_start;
        let mut run: Option<usize> = None;
        while block < gap_end {
            if holes_known && let Some(held_end) = g.held.covering_end(block) {
                if let Some(from) = run.take() {
                    protect(g, from, block);
                }
                block = held_end.min(gap_end);
                continue;
            }
            if let Some(abs_end) = g.absent.covering_end(block) {
                let abs_hi = abs_end.min(gap_end);
                if block >= uncommitted_until {
                    uncommitted_until = uncommitted_end(block);
                }
                if block < uncommitted_until {
                    if let Some(from) = run.take() {
                        protect(g, from, block);
                    }
                    block = abs_hi.min(uncommitted_until);
                    continue;
                }
                g.absent.remove(block);
                g.held.remove(block);
            }
            #[cfg(test)]
            THREAD_GUARD_GAP_BLOCKS.with(|c| c.set(c.get() + 1));
            if let Some(prot_end) = g.protected.covering_end(block) {
                if let Some(from) = run.take() {
                    protect(g, from, block);
                }
                // Written through the runtime or alias-protected, still read-only.
                let prot_hi = prot_end.min(gap_end);
                written.extend(block..prot_hi);
                g.clean.insert_range(block, prot_hi);
                block = prot_hi;
            } else if g.is_resting(block, rest) {
                if let Some(from) = run.take() {
                    protect(g, from, block);
                }
                // Still resting after a recent write fault: report dirty so
                // nothing is missed, without thrashing `VirtualProtect`.
                written.push(block);
                block += 1;
            } else {
                g.opened.remove(&block);
                written.push(block);
                run.get_or_insert(block);
                block += 1;
            }
        }
        if let Some(from) = run.take() {
            protect(g, from, gap_end);
        }
    }
    written
}

/// Fast single-lock path for `[first, last)` when already recorded in
/// `GUARDS.known_views`: returns `Some(written)` in `O(log N + K)` with
/// ZERO `GetWriteWatch` and ZERO `VirtualQuery` syscalls, or `None` if any
/// part of `[first, last)` is not yet a known view (or `VICTORY_GUARD_ASK=1`).
fn guard_if_known_view(first: usize, last: usize) -> Option<Vec<usize>> {
    if guard_ask() || first >= last {
        return None;
    }
    let mut g = guards();
    if !g.known_views.covers(first, last) {
        return None;
    }
    if g.clean.covers(first, last) {
        return Some(Vec::new());
    }
    let written = guard_gaps(&mut g, first, last);
    Some(written)
}

/// For view memory: the pages of `[start, end)`'s blocks written since
/// they were last guarded (all of a block's pages), guarding them again
/// before the caller copies them. `None` if the memory is not a view.
fn guard(start: usize, end: usize) -> Option<Vec<usize>> {
    let (first, last) = (start / BLOCK, end.div_ceil(BLOCK));
    if first >= last {
        return Some(Vec::new());
    }
    // Single-lock fast path when `[first, last)` is already known view memory.
    if let Some(written) = guard_if_known_view(first, last) {
        return Some(written);
    }
    let known = !guard_ask() && {
        let g = guards();
        g.protected.contains(first) || g.absent.contains(first)
    };
    if !known && !is_view(start, end) {
        return None;
    }
    install();
    let mut g = guards();
    let written = guard_gaps(&mut g, first, last);
    Some(written)
}

/// Whether `[start, end)` is view memory. The title's views lie in
/// placeholders, mapped and unmapped block by block as it streams, so a
/// range can begin in a block that is not mapped right now: the first
/// pages of the range that are there decide. A range with nothing there
/// counts as a view (its blocks are noted absent, and looked at again when
/// the runtime maps them).
///
/// Looking at the first address only took such a range for "not a view":
/// its copy was marked not watched for good and copied whole at every
/// refresh, around its unmapped pages (run 54: 300 such ranges, 570 MB
/// copied a frame for them, 255 copies a frame reading around holes).
/// `VICTORY_GUARD_HOLES=0` looks at the first address only, as before.
fn is_view(start: usize, end: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_MAPPED, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };
    static HOLES: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let holes = *HOLES.get_or_init(|| std::env::var_os("VICTORY_GUARD_HOLES").is_none_or(|v| v != "0"));
    let mut at = start;
    loop {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
        if n == 0 {
            return false;
        }
        if info.Type == MEM_MAPPED {
            return true;
        }
        // Committed memory that is no view (the title's heaps, ours), or
        // the answer as it was before.
        if info.State == MEM_COMMIT || !holes {
            return false;
        }
        // A hole (a placeholder, or free): on to what follows it.
        let next = (info.BaseAddress as usize).saturating_add(info.RegionSize);
        if next <= at || next >= end {
            return true;
        }
        at = next;
    }
}

/// Makes blocks `[from, to)` read-only and notes them guarded and clean;
/// block by block if they span views, leaving out any that fail, and
/// noting those not committed (unmapped) absent. The same physical pages
/// at every other view of them are made read-only too: a write through
/// one of those is this memory written, and only a fault can say so.
/// `VICTORY_VIEW_GUARD=0` leaves the other views as they are, as before.
fn protect(g: &mut Guards, from: usize, to: usize) {
    protect_here(g, from, to);
    if !view_guard() {
        return;
    }
    for (base, bytes) in crate::inplace::views_between(from.saturating_mul(BLOCK), to.saturating_mul(BLOCK)) {
        let first = (base / BLOCK).max(from);
        let last = (base.saturating_add(bytes).div_ceil(BLOCK)).min(to);
        for block in first..last {
            let shown = crate::inplace::pages_of(block * BLOCK);
            // One view (nothing else to guard), or the page the title shows
            // wherever it has nothing (thousands of views): each view of
            // that one is guarded by itself.
            if shown.len() < 2 || shown.len() > MANY_VIEWS {
                continue;
            }
            #[cfg(feature = "diagnostics")]
            ALIASES[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            for alias in shown {
                let alias = alias / BLOCK;
                g.aliased.insert(alias);
                if alias != block {
                    protect_alias(g, alias);
                }
            }
        }
    }
}

/// Views of one page past which its views are not guarded together.
pub(crate) const MANY_VIEWS: usize = 64;

/// Whether all the views of a guarded page are guarded together
/// (`VICTORY_VIEW_GUARD`).
fn view_guard() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_VIEW_GUARD").is_none_or(|v| v != "0"))
}

/// Diagnostics: blocks guarded together with their other views, blocks
/// opened because another view of theirs was written, and blocks opened
/// because a view of theirs was mapped; since the start.
#[cfg(feature = "diagnostics")]
pub(crate) static ALIASES: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// Makes `block` read-only: another view of pages a copy has just taken
/// through its own view, so that a write through this one faults too. It is
/// not noted clean. The copies of title memory at this view have not taken
/// the pages: noted clean, the block would not be reported written at their
/// next check and they would keep the bytes from before the write (`guard`
/// reports a block that is read-only and not clean as written, then notes it
/// clean). A block that is clean stays so: had the pages been written since
/// its copies took them, `open` would have opened it.
fn protect_alias(g: &mut Guards, block: usize) {
    use windows_sys::Win32::System::Memory::{PAGE_READONLY, VirtualProtect};
    if g.protected.contains(block) {
        return;
    }
    let mut old = 0;
    // SAFETY: title view memory; only its protection changes.
    if unsafe { VirtualProtect((block * BLOCK) as *const _, BLOCK, PAGE_READONLY, &mut old) } != 0 {
        g.protected.insert(block);
        g.known_views.insert(block);
    }
}

/// [`protect`] for one range, without the other views of its pages.
fn protect_here(g: &mut Guards, from: usize, to: usize) {
    use windows_sys::Win32::System::Memory::{PAGE_READONLY, VirtualProtect};
    let set = |from: usize, to: usize| {
        let mut old = 0;
        // SAFETY: title view memory; only its protection changes.
        unsafe {
            VirtualProtect(
                (from * BLOCK) as *const _,
                (to - from) * BLOCK,
                PAGE_READONLY,
                &mut old,
            ) != 0
        }
    };
    let note = |g: &mut Guards, from: usize, to: usize| {
        g.protected.insert_range(from, to);
        g.clean.insert_range(from, to);
        g.known_views.insert_range(from, to);
    };
    if set(from, to) {
        note(g, from, to);
        return;
    }
    let mut block = from;
    while block < to {
        if set(block, block + 1) {
            note(g, block, block + 1);
            block += 1;
        } else {
            let uncomm = uncommitted_end(block).min(to);
            if block < uncomm {
                g.absent.insert_range(block, uncomm);
                // Only `MEM_RESERVE_PLACEHOLDER` holes are safe to cache in
                // `held`/`known_views`: ordinary `MEM_RESERVE` private regions
                // can be committed via `VirtualAlloc(MEM_COMMIT)` without
                // `title_memory_remapped()`.
                if known_holes() && placeholder(block) {
                    g.held.insert_range(block, uncomm);
                    g.known_views.insert_range(block, uncomm);
                }
                block = uncomm;
            } else {
                block += 1;
            }
        }
    }
}

/// The end (exclusive) of the whole uncommitted blocks starting at `block`.
fn uncommitted_end(block: usize) -> usize {
    use windows_sys::Win32::System::Memory::{
        MEM_FREE, MEM_RESERVE, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };
    #[cfg(test)]
    THREAD_UNCOMMITTED_VQS.with(|c| c.set(c.get() + 1));
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery((block * BLOCK) as *const _, &mut info, size_of_val(&info)) };
    if n != 0 && matches!(info.State, MEM_RESERVE | MEM_FREE) {
        (info.BaseAddress as usize).saturating_add(info.RegionSize) / BLOCK
    } else {
        block
    }
}

/// Gives write access back to the guarded blocks of `[start, end)` and
/// notes them written; true if any was guarded. The same physical pages at
/// every other view of them are opened too: the write that faulted (or is
/// ours, `crate::host::copy_mapped`) goes through one view, and every
/// guard over the pages must copy them again.
fn open(g: &mut Guards, start: usize, end: usize) -> bool {
    let any = open_here(g, start, end);
    if start >= end || g.aliased.is_empty() {
        return any;
    }
    let aliased: Vec<usize> = g.aliased.range(start / BLOCK..end.div_ceil(BLOCK)).copied().collect();
    for block in aliased {
        g.aliased.remove(&block);
        for alias in crate::inplace::pages_of(block * BLOCK) {
            if alias / BLOCK != block && open_here(g, alias, alias.saturating_add(BLOCK)) {
                #[cfg(feature = "diagnostics")]
                ALIASES[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            g.aliased.remove(&(alias / BLOCK));
        }
    }
    any
}

/// [`open`] for one range, without the other views of its pages.
/// Coalesces contiguous protected block runs into a single `VirtualProtect`
/// call per run (falling back per-block only when a run crosses OS views).
fn open_here(g: &mut Guards, start: usize, end: usize) -> bool {
    use windows_sys::Win32::System::Memory::{PAGE_READWRITE, VirtualProtect};
    if start >= end {
        return false;
    }
    let (first, last) = (start / BLOCK, end.div_ceil(BLOCK));
    let runs = g.protected.remove_range(first, last);
    if runs.is_empty() {
        return false;
    }
    g.clean.remove_range(first, last);
    // Bump epoch BEFORE making pages writable so any concurrent reader's
    // O(1) epoch check observes the invalidation before a store can land.
    bump_guard_epoch();
    for (run_from, run_to) in runs {
        let mut old = 0;
        // SAFETY: contiguous run of blocks we previously made read-only.
        let ok = unsafe {
            VirtualProtect(
                (run_from * BLOCK) as *const _,
                (run_to - run_from) * BLOCK,
                PAGE_READWRITE,
                &mut old,
            ) != 0
        };
        if !ok {
            for block in run_from..run_to {
                let mut old_b = 0;
                // SAFETY: individual block we previously made read-only.
                unsafe {
                    VirtualProtect((block * BLOCK) as *const _, BLOCK, PAGE_READWRITE, &mut old_b)
                };
            }
        }
    }
    true
}

/// Evicts `[at, at + len)` from `JOURNAL.written` and invalidates overlapping
/// `ahead` cache entries when a VA range is unmapped or remapped. Removing
/// freed VAs prevents stale journal entries from accumulating at runtime or
/// poisoning subsequent `VirtualAlloc(MEM_WRITE_WATCH)` allocations that reuse
/// the same VA; live `Shadow` copies are invalidated separately via
/// `crate::shadow::remapped`. Must never be called while holding `GUARDS`.
fn journal_clear(at: usize, len: usize) {
    if len == 0 {
        return;
    }
    let first = at / PAGE;
    let last = at.saturating_add(len).div_ceil(PAGE);
    if first >= last {
        return;
    }
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    j.ahead.retain(|&(a, b)| b <= first || a >= last);
    let stale: Vec<usize> = j.written.range(first..last).map(|(&p, _)| p).collect();
    for page in stale {
        j.written.remove(&page);
    }
}

/// Before our own write into title memory `[at, at + len)` through the
/// kernel (which does not fault into our handler): opens its guarded blocks.
/// The runtime calls it (as `kernel_will_write`) before the title's kernel
/// reads into its own memory.
pub fn writing(at: usize, len: usize) {
    wrote(at, len);
    // The length is the title's (`fread`'s size times count, saturated).
    open(&mut guards(), at, at.saturating_add(len));
    if len > 0 {
        bump_guard_epoch();
    }
    // A placed texture there changes with no descriptor written.
    crate::descriptor::host::remapped(at, len);
}

/// What `VICTORY_PAGE_WRITES` asks to be done with a write the guards do
/// not see once its bytes are there ([`title_memory_written`]), as bits:
/// 1, the guarded blocks under it are opened and count as written (host
/// copies of buffers and textures read through tables take them again);
/// 2, the placed textures picked by index that lie under it are due again;
/// 4, both also for reads to plain addresses when they have finished.
/// Unset or `0`: nothing, counted only. `1`: bits 1 and 2. `guards`: bit 1
/// alone. `slots`: bit 2 alone. `all`: everything.
pub(crate) fn page_writes() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("VICTORY_PAGE_WRITES").as_deref().map(str::trim) {
        Ok("1") => 3,
        Ok("guards") => 1,
        Ok("slots") => 2,
        Ok("all") => 7,
        _ => 0,
    })
}

/// Since the start: writes into physical pages through the runtime's own
/// views that landed at addresses the title has mapped (one for each run
/// of addresses); their bytes; guarded blocks that lay under them (host
/// copies that would otherwise keep the bytes from before); reads to
/// plain addresses that finished.
pub(crate) static WRITTEN: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// The bytes of a write the guards do not see are there: title memory
/// `[at, at + len)`. `pages`: DirectStorage wrote them into physical pages
/// through a view of the runtime's own, and this is an address where the
/// title has those pages mapped (no title address faulted, nothing was
/// mapped or unmapped; the shortcut this file's header names). Otherwise a
/// read to a plain address has finished, of which [`writing`] was told
/// before it began.
///
/// Upstream's c75fdba does the same for its trackers (its `Writing` marks
/// the pages once more when dropped; page destinations are reported at
/// each address showing them). Here it is counted always and acted on as
/// [`page_writes`] says. Called without the runtime's page-table lock;
/// takes the guards' lock, then, alone, the lock of the ranges kept for
/// the texture pass.
pub fn title_memory_written(at: usize, len: usize, pages: bool) {
    use std::sync::atomic::Ordering::Relaxed;
    if len == 0 {
        return;
    }
    let mode = page_writes();
    let end = at.saturating_add(len);
    if pages {
        WRITTEN[0].fetch_add(1, Relaxed);
        WRITTEN[1].fetch_add(len as u64, Relaxed);
    } else {
        WRITTEN[3].fetch_add(1, Relaxed);
        if mode & 4 == 0 {
            return;
        }
    }
    {
        let mut g = guards();
        let guarded = g.protected.count_in(at / BLOCK, end.div_ceil(BLOCK)) as u64;
        if pages {
            WRITTEN[2].fetch_add(guarded, Relaxed);
        }
        if mode & 1 != 0 && guarded != 0 {
            open(&mut g, at, end);
        }
    }
    // A placed texture there changed with no descriptor written.
    crate::descriptor::host::page_written(at, len);
}

/// The runtime mapped a view of the physical bytes `[offset, offset + len)`
/// of `section` (`crate::inplace::title_memory_view`): the view is
/// writable, so the same bytes guarded at another view must be checked
/// again — and are, the pages noted written, which also has the next check
/// guard the new view (`protect`).
pub(crate) fn views_mapped(section: usize, offset: u64, len: usize) {
    if !view_guard() {
        return;
    }
    let ranges = crate::inplace::views_of_section(section, offset, len);
    if ranges.is_empty() {
        return;
    }
    let mut g = guards();
    for &(at, n) in &ranges {
        if open_here(&mut g, at, at.saturating_add(n)) {
            #[cfg(feature = "diagnostics")]
            ALIASES[2].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

/// Held shared by a read of title memory in place, and alone by the runtime
/// while it changes what title memory shows.
///
/// The runtime does not change a mapping in one step. To unmap one page of
/// a view it unmaps the whole view and maps what stays again
/// (`xmem.rs` `unmap`), and a view it maps is a reservation until the next
/// call commits it (`map`, then `commit`). For that moment pages the title
/// never gave up are "reserved, never committed", and a thread reading
/// them in place takes an access violation: run 71 in `Layout::detile`,
/// run 74 in the hash before it, both under `Staging::refresh_now`, which
/// had asked whether the texture was all mapped and been told yes. Asking
/// cannot help; not changing the mapping while such a read is under way
/// does.
static MAPPING: std::sync::RwLock<()> = std::sync::RwLock::new(());

/// What the runtime holds while it changes what title memory shows
/// ([`title_memory_remapping`]).
pub struct Remapping(#[allow(dead_code)] std::sync::RwLockWriteGuard<'static, ()>);

/// The runtime is about to map, unmap or move physical pages of title
/// memory: until what it returns is dropped, no read of title memory in
/// place is under way and none begins ([`title_memory_reading`]). Taken
/// around the whole change, the unmapping and the mapping again.
pub fn title_memory_remapping() -> Remapping {
    Remapping(MAPPING.write().unwrap_or_else(|e| e.into_inner()))
}

/// What a read of title memory in place holds ([`title_memory_reading`]).
pub struct Reading(#[allow(dead_code)] std::sync::RwLockReadGuard<'static, ()>);

/// For reading title memory in place: until what it returns is dropped,
/// the runtime changes no mapping, so memory found mapped stays mapped.
/// Not to be taken twice by one thread, nor held across anything that
/// maps title memory.
pub fn title_memory_reading() -> Reading {
    Reading(MAPPING.read().unwrap_or_else(|e| e.into_inner()))
}

/// [`title_memory_reading`] if it can be had at once: `None` while the
/// runtime changes a mapping (or waits to), for a caller that has another
/// way to read and would rather not wait.
pub fn title_memory_reading_now() -> Option<Reading> {
    match MAPPING.try_read() {
        Ok(guard) => Some(Reading(guard)),
        Err(std::sync::TryLockError::Poisoned(e)) => Some(Reading(e.into_inner())),
        Err(std::sync::TryLockError::WouldBlock) => None,
    }
}

/// The runtime mapped or unmapped title memory `[at, at + len)`: what it
/// shows is new and read-write, so it is no longer guarded (nor known
/// unmapped).
pub fn title_memory_remapped(at: usize, len: usize) {
    // Evict stale write-watch journal entries BEFORE taking `GUARDS` (lock
    // order: `JOURNAL` -> `GUARDS`) so freed/reused VAs start clean, and reset
    // `Shadow::copied` on live shadows so remapped single-view `MEM_WRITE_WATCH`
    // pages are re-copied even when their kernel dirty bits start clear.
    journal_clear(at, len);
    crate::shadow::remapped(at, len);
    // A placed texture there arrives, goes or changes with no descriptor
    // written.
    crate::descriptor::host::remapped(at, len);
    // The views the runtime told us of there are no longer what it shows.
    crate::inplace::views_gone(at, len);
    // And nothing stays in place over it.
    crate::inplace::title_memory_freeing(at, len);
    let mut g = guards();
    let first = at / BLOCK;
    let last = at.saturating_add(len).div_ceil(BLOCK);
    g.absent.remove_range(first, last);
    g.held.remove_range(first, last);
    g.protected.remove_range(first, last);
    g.clean.remove_range(first, last);
    g.known_views.remove_range(first, last);
    let aliased: Vec<usize> = g.aliased.range(first..last).copied().collect();
    for block in aliased {
        g.aliased.remove(&block);
    }
    let opened: Vec<usize> = g.opened.range(first..last).map(|(&b, _)| b).collect();
    for block in opened {
        g.opened.remove(&block);
    }
    if len > 0 {
        bump_guard_epoch();
    }
}

/// Registers the handler that takes writes to guarded blocks, once, ahead
/// of the crash reporter's.
fn install() {
    use windows_sys::Win32::Foundation::{EXCEPTION_ACCESS_VIOLATION, NTSTATUS};
    use windows_sys::Win32::System::Diagnostics::Debug::{
        AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH,
        EXCEPTION_POINTERS,
    };
    unsafe extern "system" fn on_write(info: *mut EXCEPTION_POINTERS) -> i32 {
        // SAFETY: the system passes the exception being dispatched.
        let r = unsafe { &*(*info).ExceptionRecord };
        const WRITE: usize = 1;
        if r.ExceptionCode != EXCEPTION_ACCESS_VIOLATION as NTSTATUS
            || r.NumberParameters < 2
            || r.ExceptionInformation[0] != WRITE
        {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        let at = r.ExceptionInformation[1];
        // Held across both checks: another thread may have opened the block
        // since this one faulted, and a copy may have guarded it again since
        // (then it is ours again, not read-only for some other reason).
        let mut g = guards();
        if open(&mut g, at, at + 1) {
            g.note_opened(at / BLOCK);
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::GuardFaults, 1);
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        if writable(at) {
            EXCEPTION_CONTINUE_EXECUTION
        } else {
            EXCEPTION_CONTINUE_SEARCH
        }
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    // SAFETY: the handler only touches our guard state and protections.
    ONCE.call_once(|| unsafe {
        AddVectoredExceptionHandler(1, Some(on_write));
    });
}

/// Whether `at` is committed read-write memory now.
fn writable(at: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_READWRITE, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    n != 0 && info.State == MEM_COMMIT && info.Protect == PAGE_READWRITE
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
    };
    const MEM_WRITE_WATCH: u32 = 0x20_0000;

    #[test]
    fn written_pages_are_reported_to_every_asker() {
        // SAFETY: a fresh write-watched allocation, freed at the end.
        let p = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                4 * PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(p, 0);
        let (start, end) = (p, p + 4 * PAGE);
        let (_, before) = writes(start, end, false).expect("watched");
        assert!(before.is_empty());
        // SAFETY: committed pages of ours.
        unsafe { ((p + PAGE + 8) as *mut u8).write(1) };
        let (now, after) = writes(start, end, false).expect("watched");
        assert_eq!(after, [(p / PAGE + 1, now)]);
        // Another asker after the kernel's bit was cleared still sees it.
        let (later, again) = writes(p + PAGE, p + 2 * PAGE, false).expect("watched");
        assert!(later > now);
        assert_eq!(again, [(p / PAGE + 1, now)]);
        let mut plain = vec![0u8; PAGE];
        assert!(
            writes(
                plain.as_mut_ptr() as usize,
                plain.as_ptr() as usize + PAGE,
                true
            )
            .is_none()
        );
        title_memory_remapped(p, 4 * PAGE);
        // SAFETY: our allocation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    /// A kernel read into a guarded view block fails; opened first, it
    /// lands and the block counts as written.
    #[test]
    fn kernel_reads_into_guarded_blocks_after_opening() {
        use std::io::{Read, Seek, Write};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, UnmapViewOfFile,
        };
        // SAFETY: a pagefile section of two blocks, mapped read-write.
        let view = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * BLOCK).Value as usize
        };
        assert_ne!(view, 0);
        assert!(guard(view, view + 2 * BLOCK).is_some()); // now read-only
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(b"from the disk").unwrap();
        // `ReadFile` inside: the kernel writes the view.
        let read = |file: &mut std::fs::File| {
            file.rewind().unwrap();
            // SAFETY: 13 bytes of our view, nothing else refers to them.
            let into = unsafe { std::slice::from_raw_parts_mut((view + 100) as *mut u8, 13) };
            file.read(into).is_ok_and(|n| n == 13)
        };
        assert!(
            !read(&mut file),
            "the kernel does not fault into our handler"
        );
        writing(view + 100, 13);
        assert!(read(&mut file));
        // SAFETY: the bytes just read.
        let landed = unsafe { std::slice::from_raw_parts((view + 100) as *const u8, 13) };
        assert_eq!(landed, b"from the disk");
        assert_eq!(
            guard(view, view + 2 * BLOCK),
            Some((view / PAGE..view / PAGE + 1).collect())
        );
        title_memory_remapped(view, 2 * BLOCK);
        // SAFETY: our view.
        unsafe {
            UnmapViewOfFile(
                windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: view as *mut _,
                },
            )
        };
    }

    #[test]
    fn writes_in_partly_committed_view_blocks_are_reported() {
        use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile,
            SEC_RESERVE, UnmapViewOfFile,
        };

        const SIZE: usize = 3 * BLOCK;
        struct View(MEMORY_MAPPED_VIEW_ADDRESS);
        impl Drop for View {
            fn drop(&mut self) {
                title_memory_remapped(self.0.Value as usize, SIZE);
                // SAFETY: our view is still mapped.
                unsafe { UnmapViewOfFile(self.0) };
            }
        }
        // SAFETY: a new reserved pagefile section. The view holds its own
        // reference after we close the handle, including on assertion failure.
        let view = unsafe {
            let section = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE | SEC_RESERVE,
                0,
                SIZE as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let view = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, SIZE);
            CloseHandle(section);
            assert!(!view.Value.is_null());
            View(view)
        };
        let start = view.0.Value as usize;
        let commit = |at: usize, len| {
            // SAFETY: these pages belong to our reserved view.
            let committed = unsafe { VirtualAlloc(at as _, len, MEM_COMMIT, PAGE_READWRITE) };
            assert_eq!(committed as usize, at);
        };
        let early = start + BLOCK + 3 * KERNEL_PAGE;
        let late = start + 2 * BLOCK + 3 * KERNEL_PAGE;
        commit(start, BLOCK);
        commit(early, KERNEL_PAGE);
        let refresh = || guard(start, start + SIZE).unwrap();
        assert_eq!(refresh(), [start / PAGE, early / PAGE, late / PAGE]);
        // The last block is still entirely reserved.
        assert!(!refresh().contains(&(late / PAGE)));

        commit(late, KERNEL_PAGE); // Ordinary commit, without a runtime mapping notification.
        for at in [early, late] {
            assert!(writable(at));
            // SAFETY: each address is inside its committed read-write page.
            unsafe { (at as *mut u8).write_volatile(1) };
        }
        assert_eq!(refresh(), [early / PAGE, late / PAGE]);
    }

    /// A range of view memory whose first block is not mapped is a view
    /// all the same: its blocks are reported once, then the ones written
    /// or mapped since.
    #[test]
    fn a_range_that_begins_in_an_unmapped_block_is_guarded() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER,
            MEM_RESERVE_PLACEHOLDER, MapViewOfFile3, PAGE_NOACCESS, UnmapViewOfFile2,
            VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        // SAFETY: a fresh placeholder of three blocks and a pagefile
        // section of three; the middle block is mapped first.
        let (p, section) = unsafe {
            let p = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                3 * BLOCK,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(p, 0);
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (3 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            (p, section)
        };
        let map = |block: usize| {
            // SAFETY: splits the placeholder at the block and maps the
            // section's block there.
            let view = unsafe {
                VirtualFree(
                    (p + block * BLOCK) as *mut _,
                    BLOCK,
                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
                );
                MapViewOfFile3(
                    section,
                    GetCurrentProcess(),
                    (p + block * BLOCK) as *const _,
                    (block * BLOCK) as u64,
                    BLOCK,
                    MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                )
            };
            assert_eq!(view.Value as usize, p + block * BLOCK);
            view
        };
        let pages = |block: usize| (p + block * BLOCK) / PAGE..(p + (block + 1) * BLOCK) / PAGE;
        let all: Vec<usize> = (0..3).flat_map(pages).collect();
        // Nothing mapped at all: a view with every block absent.
        assert!(is_view(p, p + 3 * BLOCK));
        let middle = map(1);
        title_memory_remapped(p + BLOCK, BLOCK);
        // The range begins in the placeholder, its middle block is a view.
        assert!(is_view(p, p + 3 * BLOCK));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(all));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(Vec::new()), "nothing written since");
        // SAFETY: inside the mapped middle block.
        unsafe { ((p + BLOCK + 8) as *mut u8).write_volatile(1) };
        assert_eq!(guard(p, p + 3 * BLOCK), Some(pages(1).collect()));
        let first = map(0);
        title_memory_remapped(p, BLOCK);
        assert_eq!(guard(p, p + 3 * BLOCK), Some(pages(0).collect()));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(Vec::new()));
        // Memory of ours that is no view stays none.
        let mut local = [0u8; 16];
        let at = local.as_mut_ptr() as usize;
        assert!(!is_view(at, at + local.len()));
        title_memory_remapped(p, 3 * BLOCK);
        // SAFETY: our views and placeholder.
        unsafe {
            UnmapViewOfFile2(GetCurrentProcess(), first, MEM_PRESERVE_PLACEHOLDER);
            UnmapViewOfFile2(GetCurrentProcess(), middle, MEM_PRESERVE_PLACEHOLDER);
            for block in 0..3 {
                VirtualFree((p + block * BLOCK) as *mut _, 0, MEM_RELEASE);
            }
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
    }

    /// Blocks of a placeholder between views are reported once, then not
    /// again (they cannot be guarded) until the runtime maps them.
    #[test]
    fn unmapped_blocks_are_reported_once_until_mapped() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER,
            MEM_RESERVE_PLACEHOLDER, MapViewOfFile3, PAGE_NOACCESS, UnmapViewOfFile2,
            VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        // SAFETY: a fresh placeholder of three blocks and a pagefile
        // section of two; block 0 shows the section's first block, as the
        // runtime maps physical pages.
        let (p, section) = unsafe {
            let p = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                3 * BLOCK,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(p, 0);
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            (p, section)
        };
        let map = |block: usize| {
            // SAFETY: splits the placeholder at the block and maps the
            // section's block there.
            let view = unsafe {
                VirtualFree(
                    (p + block * BLOCK) as *mut _,
                    BLOCK,
                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
                );
                MapViewOfFile3(
                    section,
                    GetCurrentProcess(),
                    (p + block * BLOCK) as *const _,
                    (block * BLOCK) as u64,
                    BLOCK,
                    MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                )
            };
            assert_eq!(view.Value as usize, p + block * BLOCK);
            view
        };
        let first = map(0);
        let pages = |block: usize| (p + block * BLOCK) / PAGE..(p + (block + 1) * BLOCK) / PAGE;
        let all: Vec<usize> = (0..3).flat_map(pages).collect();
        assert_eq!(guard(p, p + 3 * BLOCK), Some(all));
        assert_eq!(
            guard(p, p + 3 * BLOCK),
            Some(Vec::new()),
            "nothing written since"
        );
        let second = map(1);
        title_memory_remapped(p + BLOCK, BLOCK);
        assert_eq!(guard(p, p + 3 * BLOCK), Some(pages(1).collect()));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(Vec::new()));
        title_memory_remapped(p, 3 * BLOCK);
        // SAFETY: our views and placeholder.
        unsafe {
            UnmapViewOfFile2(GetCurrentProcess(), first, MEM_PRESERVE_PLACEHOLDER);
            UnmapViewOfFile2(GetCurrentProcess(), second, MEM_PRESERVE_PLACEHOLDER);
            for block in 0..3 {
                VirtualFree((p + block * BLOCK) as *mut _, 0, MEM_RELEASE);
            }
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
    }

    /// A pagefile section mapped twice: two views of the same physical
    /// pages, the second left for the test; the section handle, the two
    /// addresses.
    fn two_views(bytes: usize) -> (usize, usize, usize) {
        use windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile,
        };
        // SAFETY: a fresh pagefile section, mapped twice; the views keep
        // the section alive and the handle is closed by the caller.
        unsafe {
            let section = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                bytes as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let one = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, bytes).Value as usize;
            let two = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, bytes).Value as usize;
            assert_ne!(one, 0);
            assert_ne!(two, 0);
            crate::inplace::title_memory_view(one, bytes, section as usize, 0);
            crate::inplace::title_memory_view(two, bytes, section as usize, 0);
            (section as usize, one, two)
        }
    }

    /// The section of [`two_views`] is closed and its views unmapped; the
    /// guard is told they are gone first.
    fn drop_views(section: usize, views: &[usize], bytes: usize) {
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Memory::{MEMORY_MAPPED_VIEW_ADDRESS, UnmapViewOfFile};
        for &view in views {
            title_memory_remapped(view, bytes);
            // SAFETY: a view of the section mapped by `two_views`.
            unsafe {
                UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: view as *mut _,
                })
            };
        }
        // SAFETY: the section's last reference.
        unsafe { CloseHandle(section as *mut std::ffi::c_void) };
    }

    /// The title wrote the pages of a guarded texture through another view
    /// of them: the fault at that view is the guard's to see, and the
    /// guarded view's next check reports the pages written.
    #[test]
    fn a_write_through_another_view_is_reported_at_the_guarded_one() {
        const SIZE: usize = 2 * BLOCK;
        let (section, guarded, other) = two_views(SIZE);
        let pages = |block: usize| (guarded + block * BLOCK) / PAGE;
        // The first check takes it in (and every view of its pages).
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()), "nothing written yet");
        // Writing through the other view faults into the handler, which
        // opens that view and the guarded one: the next check sees it.
        // SAFETY: the second view, read-write once the handler has opened it.
        unsafe { ((other + BLOCK + 8) as *mut u8).write(1) };
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()));
        drop_views(section, &[guarded, other], SIZE);
    }

    /// A view of guarded pages is mapped anew: it is writable, so the
    /// guarded view checks its pages again (and guards the new view).
    #[test]
    fn a_view_mapped_beside_a_guarded_one_makes_it_check_again() {
        const SIZE: usize = 2 * BLOCK;
        let (section, guarded, other) = two_views(SIZE);
        let pages = |block: usize| (guarded + block * BLOCK) / PAGE;
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()), "nothing written yet");
        title_memory_remapped(other, SIZE);
        crate::inplace::title_memory_view(other, SIZE, section, 0);
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()));
        drop_views(section, &[guarded, other], SIZE);
    }

    /// A write of the runtime's own, told at one view of the pages
    /// (`writing`): the other views' next checks report the pages written.
    #[test]
    fn a_write_told_at_one_view_is_reported_at_the_others() {
        const SIZE: usize = 2 * BLOCK;
        let (section, guarded, other) = two_views(SIZE);
        let pages = |block: usize| (guarded + block * BLOCK) / PAGE;
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()), "nothing written yet");
        writing(other + BLOCK, BLOCK);
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()));
        // Half a page and the next: both blocks, not just the first's.
        writing(other + BLOCK / 2, BLOCK);
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        drop_views(section, &[guarded, other], SIZE);
    }

    /// Both views of the pages are checked, each for a copy of its own
    /// (the title shows one physical page at several addresses, and each
    /// address has its host copy): a write through either is reported at
    /// both, whichever is checked first. Guarding the pages for the first
    /// view's check must not note the second view clean, or its copy would
    /// keep the bytes from before the write.
    #[test]
    fn a_write_is_reported_at_every_checked_view() {
        const SIZE: usize = 2 * BLOCK;
        let (section, one, two) = two_views(SIZE);
        let pages = |view: usize, block: usize| (view + block * BLOCK) / PAGE;
        assert_eq!(guard(one, one + SIZE), Some(vec![pages(one, 0), pages(one, 1)]));
        assert_eq!(guard(two, two + SIZE), Some(vec![pages(two, 0), pages(two, 1)]), "the second view's first check");
        assert_eq!(guard(one, one + SIZE), Some(Vec::new()), "nothing written yet");
        assert_eq!(guard(two, two + SIZE), Some(Vec::new()), "nothing written yet");
        // SAFETY: the second view, read-write once the handler has opened it.
        unsafe { ((two + BLOCK + 8) as *mut u8).write(1) };
        assert_eq!(guard(one, one + SIZE), Some(vec![pages(one, 1)]));
        assert_eq!(guard(two, two + SIZE), Some(vec![pages(two, 1)]), "written through this view, checked second");
        // SAFETY: the first view, as above.
        unsafe { ((one + 8) as *mut u8).write(1) };
        assert_eq!(guard(two, two + SIZE), Some(vec![pages(two, 0)]));
        assert_eq!(guard(one, one + SIZE), Some(vec![pages(one, 0)]), "written through this view, checked second");
        assert_eq!(guard(one, one + SIZE), Some(Vec::new()));
        assert_eq!(guard(two, two + SIZE), Some(Vec::new()));
        drop_views(section, &[one, two], SIZE);
    }

    /// A page shown in more than `MANY_VIEWS` views is guarded at each by
    /// itself: the others stay writable, and nothing looks through them all.
    #[test]
    fn a_page_of_many_views_is_guarded_one_view_at_a_time() {
        let (section, one, two) = two_views(BLOCK);
        let filler = one + (1 << 45);
        for n in 0..MANY_VIEWS {
            crate::inplace::title_memory_view(filler + n * BLOCK, BLOCK, section, 0);
        }
        assert_eq!(guard(one, one + BLOCK), Some(vec![one / PAGE]));
        assert!(writable(two), "another view of it is left alone");
        // SAFETY: the guarded view, opened by the handler.
        unsafe { ((one + 8) as *mut u8).write(1) };
        assert_eq!(guard(one, one + BLOCK), Some(vec![one / PAGE]));
        title_memory_remapped(filler, MANY_VIEWS * BLOCK);
        drop_views(section, &[one, two], BLOCK);
    }

    /// A write or mapping that runs past the end of the address space (a
    /// title's `fread` with a huge count) covers up to the end; the sum used
    /// to overflow. Up there no test has guarded blocks.
    #[test]
    fn ranges_past_the_address_space_end_do_not_overflow() {
        let top = usize::MAX - 16;
        writing(top, PAGE);
        title_memory_remapped(top, usize::MAX);
    }

    /// A placeholder containing a single-view `MEM_WRITE_WATCH` private block,
    /// an unmapped hole, and a `MEM_MAPPED` view tracks all three cleanly: the
    /// private block stays `PAGE_READWRITE` (zero guard faults, kernel reads
    /// succeed directly) and reports writes via `GetWriteWatch`.
    #[test]
    fn mixed_private_write_watch_and_mapped_view_coexist() {
        use std::io::{Read, Seek, Write};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER,
            MEM_RESERVE_PLACEHOLDER, MapViewOfFile3, PAGE_NOACCESS, UnmapViewOfFile2,
            VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        // SAFETY: 3-block placeholder + 1-block pagefile section.
        let (p, section) = unsafe {
            let p = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                3 * BLOCK,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(p, 0);
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                BLOCK as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            // Block 0: private `MEM_WRITE_WATCH` replacing placeholder.
            VirtualFree(p as *mut _, BLOCK, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER);
            let priv_ptr = VirtualAlloc2(
                GetCurrentProcess(),
                p as *const _,
                BLOCK,
                MEM_RESERVE | MEM_COMMIT | MEM_REPLACE_PLACEHOLDER | MEM_WRITE_WATCH,
                PAGE_READWRITE,
                std::ptr::null_mut(),
                0,
            );
            assert_eq!(priv_ptr as usize, p);
            // Block 1 stays an unmapped placeholder hole.
            // Block 2: `MEM_MAPPED` section view replacing placeholder.
            VirtualFree(
                (p + 2 * BLOCK) as *mut _,
                BLOCK,
                MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
            );
            let mapped = MapViewOfFile3(
                section,
                GetCurrentProcess(),
                (p + 2 * BLOCK) as *const _,
                0,
                BLOCK,
                MEM_REPLACE_PLACEHOLDER,
                PAGE_READWRITE,
                std::ptr::null_mut(),
                0,
            );
            assert_eq!(mapped.Value as usize, p + 2 * BLOCK);
            (p, section)
        };
        // Initial query guards block 2, notes block 1 absent, and leaves block 0 `PAGE_READWRITE`.
        let (t0, _) = writes(p, p + 3 * BLOCK, true).expect("mixed range watched");
        assert!(writable(p), "single-view MEM_WRITE_WATCH block stays PAGE_READWRITE");
        assert!(!writable(p + 2 * BLOCK), "MEM_MAPPED block is PAGE_READONLY guarded");

        // CPU write to block 0 is detected by `GetWriteWatch` without any guard fault.
        // SAFETY: block 0 is committed `PAGE_READWRITE`.
        unsafe { ((p + 16) as *mut u8).write_volatile(0x42) };
        let (t1, after_cpu) = writes(p, p + 3 * BLOCK, true).expect("mixed range after cpu write");
        assert!(t1 > t0);
        assert!(after_cpu.iter().any(|&(pg, at)| pg == p / PAGE && at == t1));
        assert!(writable(p), "block 0 still PAGE_READWRITE");

        // Kernel `ReadFile` directly into block 0 succeeds without needing `writing()`.
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(b"direct_watch").unwrap();
        file.rewind().unwrap();
        // SAFETY: 12 bytes of our committed `PAGE_READWRITE` block 0.
        let dst = unsafe { std::slice::from_raw_parts_mut((p + 128) as *mut u8, 12) };
        assert_eq!(file.read(dst).unwrap(), 12);
        assert_eq!(dst, b"direct_watch");
        let (t2, after_io) = writes(p, p + 3 * BLOCK, true).expect("mixed range after kernel read");
        assert!(after_io.iter().any(|&(pg, at)| pg == p / PAGE && at == t2));

        // Write to block 2 (`MEM_MAPPED`) still reports through the guard.
        // SAFETY: guarded `MEM_MAPPED` block 2, opened by VEH on write.
        unsafe { ((p + 2 * BLOCK + 8) as *mut u8).write_volatile(0x99) };
        let (t3, after_mapped) = writes(p, p + 3 * BLOCK, true).expect("mixed range after mapped write");
        assert!(after_mapped.iter().any(|&(pg, at)| pg == (p + 2 * BLOCK) / PAGE && at == t3));

        // Clean up: notify guards BEFORE unmapping/freeing.
        title_memory_remapped(p, 3 * BLOCK);
        // SAFETY: our private allocation, section view, placeholders, and section handle.
        unsafe {
            VirtualFree(p as *mut _, BLOCK, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER);
            UnmapViewOfFile2(
                GetCurrentProcess(),
                windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: (p + 2 * BLOCK) as *mut _,
                },
                MEM_PRESERVE_PLACEHOLDER,
            );
            for block in 0..3 {
                VirtualFree((p + block * BLOCK) as *mut _, 0, MEM_RELEASE);
            }
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
    }

    /// `title_memory_remapped` evicts freed/remapped VAs from `JOURNAL.written`
    /// so a fresh `VirtualAlloc(MEM_WRITE_WATCH)` reusing the same address
    /// starts clean without stale dirty entries.
    #[test]
    fn remapped_evicts_freed_va_from_journal() {
        // SAFETY: fresh write-watched allocation, freed after eviction check.
        let p = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                2 * PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(p, 0);
        // SAFETY: committed page of ours.
        unsafe { ((p + 8) as *mut u8).write_volatile(1) };
        let (_, written) = writes(p, p + 2 * PAGE, false).expect("watched");
        assert_eq!(written.len(), 1);
        title_memory_remapped(p, 2 * PAGE);
        // After eviction, no stale journal entries remain for `[p, p + 2 * PAGE)`.
        let (_, after) = writes(p, p + 2 * PAGE, false).expect("watched");
        assert!(after.is_empty(), "journal must not retain evicted pages");
        // SAFETY: our allocation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    /// Multiple adjacent single-page `MapViewOfFile3` `MEM_MAPPED` views (the
    /// exact layout produced by `xmem` GPU page mappings) must take the `g.clean`
    /// fast path in `writes_mixed()` with ZERO `VirtualQuery` syscalls on
    /// untouched repeat submits, while `guard_epoch()` and `writes_guarded_since`
    /// still detect CPU writes, alias writes, `writing()` (`kernel_will_write`),
    /// and `title_memory_remapped()`.
    #[test]
    fn writes_mixed_clean_mapped_views_zero_vq_and_guard_epoch_coherency() {
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER,
            MEM_RESERVE_PLACEHOLDER, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile,
            MapViewOfFile3, FILE_MAP_WRITE, PAGE_NOACCESS, SEC_COMMIT, UnmapViewOfFile,
            UnmapViewOfFile2, VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;

        const VIEWS: usize = 8;
        const TOTAL: usize = VIEWS * BLOCK;
        let proc = unsafe { GetCurrentProcess() };
        let (base, section, alias) = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE | SEC_COMMIT,
                0,
                TOTAL as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let base = VirtualAlloc2(
                proc,
                std::ptr::null(),
                TOTAL,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(base, 0);
            // Map 8 separate 64 KiB views back-to-back inside the placeholder.
            for i in 0..VIEWS {
                let va = base + i * BLOCK;
                if i + 1 < VIEWS {
                    assert_ne!(
                        VirtualFree(va as *mut _, BLOCK, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER),
                        0
                    );
                }
                let v = MapViewOfFile3(
                    section,
                    proc,
                    va as *const _,
                    (i * BLOCK) as u64,
                    BLOCK,
                    MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                );
                assert_eq!(v.Value as usize, va);
                crate::inplace::title_memory_view(va, BLOCK, section as usize, (i * BLOCK) as u64);
            }
            // Also map an external alias of view #3 (`offset = 3 * BLOCK`).
            let alias = MapViewOfFile(section, FILE_MAP_WRITE, 0, (3 * BLOCK) as u32, BLOCK).Value
                as usize;
            assert_ne!(alias, 0);
            crate::inplace::title_memory_view(alias, BLOCK, section as usize, (3 * BLOCK) as u64);
            (base, section, alias)
        };

        struct MultiViewGuard {
            base: usize,
            alias: usize,
            section: windows_sys::Win32::Foundation::HANDLE,
        }
        impl Drop for MultiViewGuard {
            fn drop(&mut self) {
                title_memory_remapped(self.base, TOTAL);
                title_memory_remapped(self.alias, BLOCK);
                unsafe {
                    let proc = GetCurrentProcess();
                    title_memory_remapped(self.base, TOTAL);
                    title_memory_remapped(self.alias, BLOCK);
                    for i in 0..VIEWS {
                        UnmapViewOfFile2(
                            proc,
                            MEMORY_MAPPED_VIEW_ADDRESS {
                                Value: (self.base + i * BLOCK) as *mut _,
                            },
                            0,
                        );
                    }
                    UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                        Value: self.alias as *mut _,
                    });
                    CloseHandle(self.section);
                }
            }
        }
        let _guard = MultiViewGuard {
            base,
            alias,
            section,
        };

        // 1. Initial `writes(..., true)` guards all 8 views (and the alias of view #3).
        let (t0, _) = writes(base, base + TOTAL, true).expect("mapped views guarded");

        // 2. Untouched repeat calls to `writes(..., true)` MUST take the `g.clean`
        //    fast path with ZERO `VirtualQuery` syscalls!
        reset_mixed_virtual_queries();
        for _ in 0..5 {
            let (t, _) = writes(base, base + TOTAL, true).expect("clean fast path");
            assert!(t > t0);
        }
        assert_eq!(
            mixed_virtual_queries(),
            0,
            "clean guarded MEM_MAPPED views must issue 0 VirtualQueries in writes_mixed"
        );

        // 3. `writes_guarded_since` on clean span returns empty vec & 0 VQs.
        let (seq0, initial_since) =
            writes_guarded_since(base, base + TOTAL, t0).expect("guarded_since clean");
        assert!(initial_since.is_empty());
        assert_eq!(mixed_virtual_queries(), 0);

        // 4. Direct CPU write to view #5 bumps `guard_epoch()` and reports ONLY page #5.
        let ep_before_cpu = guard_epoch();
        unsafe { ((base + 5 * BLOCK + 16) as *mut u32).write_volatile(0x1111_2222) };
        assert_ne!(guard_epoch(), ep_before_cpu, "CPU write fault must bump GUARD_EPOCH");
        let (seq1, after_cpu) =
            writes_guarded_since(base, base + TOTAL, seq0).expect("after cpu write");
        assert_eq!(after_cpu, vec![((base + 5 * BLOCK) / PAGE, seq1)]);

        // 5. Write through external alias of view #3 bumps `guard_epoch()` and reports ONLY page #3.
        let ep_before_alias = guard_epoch();
        unsafe { ((alias + 32) as *mut u32).write_volatile(0x3333_4444) };
        assert_ne!(guard_epoch(), ep_before_alias, "alias write fault must bump GUARD_EPOCH");
        let (seq2, after_alias) =
            writes_guarded_since(base, base + TOTAL, seq1).expect("after alias write");
        assert_eq!(after_alias, vec![((base + 3 * BLOCK) / PAGE, seq2)]);

        // 6. `writing()` (`kernel_will_write`) on view #1 bumps `guard_epoch()` and reports ONLY page #1.
        let ep_before_kw = guard_epoch();
        writing(base + BLOCK + 64, 128);
        assert_ne!(guard_epoch(), ep_before_kw, "writing() must bump GUARD_EPOCH");
        let (seq3, after_kw) =
            writes_guarded_since(base, base + TOTAL, seq2).expect("after kernel_will_write");
        assert_eq!(after_kw, vec![((base + BLOCK) / PAGE, seq3)]);

        // 7. Subsequent untouched check returns empty with 0 VQs and steady epoch!
        let ep_steady = guard_epoch();
        let (seq4, after_clean) =
            writes_guarded_since(base, base + TOTAL, seq3).expect("back to clean");
        assert!(after_clean.is_empty());
        assert_eq!(seq4, seq3);
        assert_eq!(guard_epoch(), ep_steady);
        assert_eq!(mixed_virtual_queries(), 0);
    }

    #[test]
    fn interval_set_coalesces_splits_and_extracts_gaps() {
        let mut s = IntervalSet::new();
        s.insert_range(10, 20);
        s.insert_range(20, 30);
        s.insert_range(5, 11);
        assert_eq!(s.0.len(), 1);
        assert!(s.covers(5, 30));
        assert!(!s.covers(4, 30));
        assert_eq!(s.count_in(8, 25), 17);

        let removed = s.remove_range(12, 15);
        assert_eq!(removed, vec![(12, 15)]);
        assert_eq!(s.remove_range(22, 24), vec![(22, 24)]);
        assert_eq!(s.gaps(5, 30), vec![(12, 15), (22, 24)]);
        assert_eq!(s.gaps(0, 35), vec![(0, 5), (12, 15), (22, 24), (30, 35)]);
        assert_eq!(s.count_in(5, 30), 20);
        assert_eq!(s.remove_range(0, 40), vec![(5, 12), (15, 22), (24, 30)]);
        assert!(s.0.is_empty());
    }

    /// A large `MEM_MAPPED` range (32 MiB = 512 × 64 KiB blocks) with only 2
    /// dirty blocks must return the exact 2 dirty pages in `O(dirty)` work:
    /// 0 `VirtualQuery` syscalls, 0 clean-block scans (only the 2 gap blocks
    /// visited), across both `writes_on` and `ask_once_on`, and `open_here`
    /// coalesces contiguous runs cleanly.
    #[test]
    fn large_mapped_view_with_two_dirty_blocks_is_o_dirty_and_coalesces_open_here() {
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_ALL_ACCESS, MapViewOfFile, PAGE_READWRITE, UnmapViewOfFile,
        };
        const BLOCKS: usize = 512; // 32 MiB
        const LEN: usize = BLOCKS * BLOCK;
        // SAFETY: anonymous page-file-backed section mapped and unmapped within test.
        let sec = unsafe {
            CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                LEN as u32,
                std::ptr::null(),
            )
        };
        assert!(!sec.is_null());
        // SAFETY: mapping 32 MiB anonymous section.
        let view = unsafe { MapViewOfFile(sec, FILE_MAP_ALL_ACCESS, 0, 0, LEN) };
        assert!(!view.Value.is_null());

        struct ViewCleanup {
            view: windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS,
            sec: windows_sys::Win32::Foundation::HANDLE,
        }
        impl Drop for ViewCleanup {
            fn drop(&mut self) {
                title_memory_remapped(self.view.Value as usize, LEN);
                // SAFETY: releasing test mapping & handle after clearing guards.
                unsafe {
                    UnmapViewOfFile(self.view);
                    CloseHandle(self.sec);
                }
            }
        }
        let _cleanup = ViewCleanup { view, sec };

        let start = view.Value as usize;
        let end = start + LEN;
        let first = start / BLOCK;
        let last = end / BLOCK;
        let mut j = Journal::new_for_test();

        // 1. Prime guards once across all 512 blocks.
        let (t0, initial) = writes_on(&mut j, start, end, true).expect("prime large mapped view");
        assert_eq!(initial.len(), BLOCKS);
        {
            let g = guards();
            assert!(g.known_views.covers(first, last));
            assert!(g.clean.covers(first, last));
            assert!(g.protected.covers(first, last));
        }

        // 2. Clean re-check via writes_on and ask_once_on: 0 VQs, 0 gap blocks touched.
        reset_mixed_virtual_queries();
        reset_guard_gap_blocks();
        let (t1, _) = writes_on(&mut j, start, end, true).expect("clean writes_on");
        assert!(t1 > t0);
        ask_once_on(&mut j, &[(start, end)]);
        let (t2, _) = writes_on(&mut j, start, end, true).expect("clean ask_once_on");
        ask_once_end_on(&mut j);
        assert!(t2 >= t1);
        assert_eq!(mixed_virtual_queries(), 0, "clean known view must do 0 VirtualQueries");
        assert_eq!(guard_gap_blocks(), 0, "clean known view must scan 0 blocks");

        // 3. Dirty exactly 2 blocks via CPU write faults -> writes_on touches ONLY 2 blocks.
        // SAFETY: writing within our mapped 32 MiB test view; VEH opens the 2 faulted blocks.
        unsafe {
            ((start + 17 * BLOCK + 64) as *mut u32).write_volatile(0xDEAD_0017);
            ((start + 480 * BLOCK + 128) as *mut u32).write_volatile(0xDEAD_0480);
        }
        reset_mixed_virtual_queries();
        reset_guard_gap_blocks();
        let (t3, w3) = writes_on(&mut j, start, end, true).expect("2 dirty blocks via writes_on");
        let newly_dirty: Vec<usize> = w3.into_iter().filter(|&(_, a)| a == t3).map(|(p, _)| p).collect();
        assert_eq!(newly_dirty, vec![first + 17, first + 480]);
        assert_eq!(mixed_virtual_queries(), 0, "known view with 2 dirty blocks must do 0 VQs");
        assert_eq!(guard_gap_blocks(), 2, "must touch only the 2 dirty blocks, not all 512");

        // 4. Dirty 2 different blocks -> ask_once_on touches ONLY those 2 blocks with 0 VQs.
        // SAFETY: writing within our mapped 32 MiB test view.
        unsafe {
            ((start + 3 * BLOCK + 8) as *mut u32).write_volatile(0xCAFE_0003);
            ((start + 509 * BLOCK + 16) as *mut u32).write_volatile(0xCAFE_0509);
        }
        reset_mixed_virtual_queries();
        reset_guard_gap_blocks();
        ask_once_on(&mut j, &[(start, end)]);
        let (t4, w4) = writes_on(&mut j, start, end, true).expect("2 dirty blocks via ask_once_on");
        ask_once_end_on(&mut j);
        let newly_dirty_once: Vec<usize> =
            w4.into_iter().filter(|&(_, a)| a == t4).map(|(p, _)| p).collect();
        assert_eq!(newly_dirty_once, vec![first + 3, first + 509]);
        assert_eq!(mixed_virtual_queries(), 0, "ask_once_on on known view must do 0 VQs");
        assert_eq!(guard_gap_blocks(), 2, "ask_once_on must touch only the 2 dirty blocks");

        // 5. Coalesced open_here across a 16-block contiguous run opens all 16 blocks cleanly.
        let run_start = start + 100 * BLOCK;
        let run_len = 16 * BLOCK;
        assert!(open_here(&mut guards(), run_start, run_start + run_len));
        // Every block in the opened run must be writable without faulting (i.e. already open).
        {
            let g = guards();
            assert_eq!(g.protected.count_in(first + 100, first + 116), 0);
            assert_eq!(g.clean.count_in(first + 100, first + 116), 0);
        }
        for i in 0..16 {
            // SAFETY: writing inside the opened 16-block run of our mapped test view.
            unsafe { ((run_start + i * BLOCK) as *mut u32).write_volatile(i as u32) };
        }
        reset_mixed_virtual_queries();
        reset_guard_gap_blocks();
        let (t5, w5) = writes_on(&mut j, start, end, true).expect("re-guard 16 opened blocks");
        let newly_dirty_run: Vec<usize> =
            w5.into_iter().filter(|&(_, a)| a == t5).map(|(p, _)| p).collect();
        let expected_run: Vec<usize> = (first + 100..first + 116).collect();
        assert_eq!(newly_dirty_run, expected_run);
        assert_eq!(mixed_virtual_queries(), 0);
        assert_eq!(guard_gap_blocks(), 16);
    }

    #[test]
    fn console_mem_class_classification_and_interval_splitting() {
        // Verify exact Xbox console flag/protect classification rules (#174/#176).
        assert_eq!(classify_console_memory(0x21, 0x0004), ConsoleMemClass::CpuOnly);
        assert_eq!(classify_console_memory(0x181, 0x0004), ConsoleMemClass::CpuOnly);
        assert_eq!(classify_console_memory(0x03, 0x0801), ConsoleMemClass::CpuOnly);
        assert_eq!(classify_console_memory(0x27, 0x1004), ConsoleMemClass::GpuReadOnly);
        assert_eq!(classify_console_memory(0x27, 0x1404), ConsoleMemClass::GpuReadOnly);
        assert_eq!(classify_console_memory(0x27, 0x8404), ConsoleMemClass::GpuReadOnly);
        assert_eq!(classify_console_memory(0x27, 0x2404), ConsoleMemClass::WriteCombinedGpuRw);
        assert_eq!(classify_console_memory(0x283, 0x2404), ConsoleMemClass::WriteCombinedGpuRw);
        assert_eq!(classify_console_memory(0x27, 0x2004), ConsoleMemClass::CachedGpuRw);
        assert_eq!(classify_console_memory(0x183, 0x22004), ConsoleMemClass::CoherentGpuRw);
        assert_eq!(classify_console_memory(0x283, 0x42404), ConsoleMemClass::CoherentGpuRw);
        assert_eq!(classify_console_memory(0x02, 0x0004), ConsoleMemClass::Unknown);

        // Use a high synthetic VA bucket disjoint from real host allocations.
        let base: usize = 0x7000_0000_0000;
        let mb: usize = 1 << 20;
        clear_console_memory(base, 200 * mb);

        // Composite Low-4GB-style zones: Z3 (GpuReadOnly), Z4 (WC), Z5 (Cached).
        note_console_memory(base, 64 * mb, 0x27, 0x8404);
        note_console_memory(base + 64 * mb, 64 * mb, 0x27, 0x2404);
        note_console_memory(base + 128 * mb, 46 * mb, 0x27, 0x2004);
        assert_eq!(
            console_mem_class_ranges(base, base + 174 * mb),
            vec![
                (base, base + 64 * mb, ConsoleMemClass::GpuReadOnly),
                (base + 64 * mb, base + 128 * mb, ConsoleMemClass::WriteCombinedGpuRw),
                (base + 128 * mb, base + 174 * mb, ConsoleMemClass::CachedGpuRw),
            ]
        );

        // Overwriting a middle slice splits surrounding intervals cleanly.
        note_console_memory(base + 80 * mb, 16 * mb, 0x183, 0x22004);
        assert_eq!(
            console_mem_class_ranges(base + 64 * mb, base + 128 * mb),
            vec![
                (base + 64 * mb, base + 80 * mb, ConsoleMemClass::WriteCombinedGpuRw),
                (base + 80 * mb, base + 96 * mb, ConsoleMemClass::CoherentGpuRw),
                (base + 96 * mb, base + 128 * mb, ConsoleMemClass::WriteCombinedGpuRw),
            ]
        );

        // Clearing a sub-range leaves gaps reporting `Unknown`.
        clear_console_memory(base + 80 * mb, 16 * mb);
        assert_eq!(
            console_mem_class_ranges(base + 64 * mb, base + 128 * mb),
            vec![
                (base + 64 * mb, base + 80 * mb, ConsoleMemClass::WriteCombinedGpuRw),
                (base + 80 * mb, base + 96 * mb, ConsoleMemClass::Unknown),
                (base + 96 * mb, base + 128 * mb, ConsoleMemClass::WriteCombinedGpuRw),
            ]
        );

        // Re-inserting the same class coalesces back into one interval.
        note_console_memory(base + 80 * mb, 16 * mb, 0x27, 0x2404);
        assert_eq!(
            console_mem_class_ranges(base + 64 * mb, base + 128 * mb),
            vec![(base + 64 * mb, base + 128 * mb, ConsoleMemClass::WriteCombinedGpuRw)]
        );
        clear_console_memory(base, 200 * mb);
    }

    #[test]
    fn known_holes_skips_placeholder_virtual_queries_until_remapped() {
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_RELEASE, MEM_REPLACE_PLACEHOLDER,
            MEM_RESERVE, MEM_RESERVE_PLACEHOLDER, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile3,
            PAGE_NOACCESS, PAGE_READWRITE, UnmapViewOfFile2, VirtualAlloc, VirtualAlloc2,
            VirtualFree,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;

        const TOTAL: usize = 3 * BLOCK;
        // SAFETY: 3-block placeholder reservation (block 0 mapped, blocks 1..3 remain placeholder holes).
        let base = unsafe {
            VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                TOTAL,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            )
        } as usize;
        assert_ne!(base, 0);
        // SAFETY: split block 0 off the 3-block placeholder.
        assert_ne!(
            unsafe { VirtualFree(base as *mut _, BLOCK, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER) },
            0
        );
        // SAFETY: 1-block anonymous section mapped into block 0.
        let sec = unsafe {
            CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                BLOCK as u32,
                std::ptr::null(),
            )
        };
        assert!(!sec.is_null());
        // SAFETY: map section into split placeholder block 0.
        let view = unsafe {
            MapViewOfFile3(
                sec,
                GetCurrentProcess(),
                base as *const _,
                0,
                BLOCK,
                MEM_REPLACE_PLACEHOLDER,
                PAGE_READWRITE,
                std::ptr::null_mut(),
                0,
            )
        };
        assert_eq!(view.Value as usize, base);

        // Also allocate an ordinary non-placeholder `MEM_RESERVE` block to verify
        // it is NOT cached in `held` or `known_views`.
        // SAFETY: ordinary private reservation.
        let priv_res = unsafe { VirtualAlloc(std::ptr::null(), BLOCK, MEM_RESERVE, PAGE_READWRITE) } as usize;
        assert_ne!(priv_res, 0);

        struct Cleanup {
            base: usize,
            sec: windows_sys::Win32::Foundation::HANDLE,
            priv_res: usize,
        }
        impl Drop for Cleanup {
            fn drop(&mut self) {
                KNOWN_HOLES_IN_TEST.with(|c| c.set(None));
                title_memory_remapped(self.base, TOTAL);
                title_memory_remapped(self.priv_res, BLOCK);
                // SAFETY: unmapping test view and releasing reservations.
                unsafe {
                    let proc = GetCurrentProcess();
                    UnmapViewOfFile2(
                        proc,
                        MEMORY_MAPPED_VIEW_ADDRESS { Value: self.base as *mut _ },
                        MEM_PRESERVE_PLACEHOLDER,
                    );
                    VirtualFree(self.base as *mut _, BLOCK, MEM_RELEASE);
                    VirtualFree((self.base + BLOCK) as *mut _, 2 * BLOCK, MEM_RELEASE);
                    CloseHandle(self.sec);
                    VirtualFree(self.priv_res as *mut _, 0, MEM_RELEASE);
                }
            }
        }
        let _cleanup = Cleanup { base, sec, priv_res };
        KNOWN_HOLES_IN_TEST.with(|c| c.set(Some(true)));

        let first = base / BLOCK;
        let last = (base + TOTAL) / BLOCK;
        let mut j = Journal::new_for_test();

        // Prime: block 0 guarded, blocks 1..3 recognized as placeholder holes (`held` + `known_views`).
        assert!(writes_on(&mut j, base, base + TOTAL, true).is_some());
        {
            let g = guards();
            assert!(g.known_views.covers(first, last));
            assert!(g.held.covers(first + 1, last));
        }

        // Repeat checks on `[base, base + TOTAL)` must issue 0 `VirtualQuery` syscalls!
        reset_mixed_virtual_queries();
        reset_uncommitted_vqs();
        for _ in 0..5 {
            assert!(writes_on(&mut j, base, base + TOTAL, true).is_some());
        }
        assert_eq!(mixed_virtual_queries(), 0, "known placeholder pool must do 0 writes_mixed VQs");
        assert_eq!(uncommitted_vqs(), 0, "VICTORY_KNOWN_HOLES must do 0 uncommitted_end VQs");

        // `title_memory_remapped` clears `held` and `known_views` cleanly.
        title_memory_remapped(base + BLOCK, BLOCK);
        {
            let g = guards();
            assert!(!g.held.contains(first + 1));
            assert!(!g.known_views.contains(first + 1));
        }

        // Non-placeholder `MEM_RESERVE` must NEVER enter `held` or `known_views`.
        assert!(writes_on(&mut j, priv_res, priv_res + BLOCK, true).is_some());
        {
            let g = guards();
            let pb = priv_res / BLOCK;
            assert!(g.absent.contains(pb));
            assert!(!g.held.contains(pb), "ordinary MEM_RESERVE must not enter held");
            assert!(!g.known_views.contains(pb), "ordinary MEM_RESERVE must not enter known_views");
        }
    }

    #[test]
    fn guard_rest_leaves_hot_faulted_block_open_until_rest_expires() {
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_ALL_ACCESS, MapViewOfFile, PAGE_READWRITE, UnmapViewOfFile,
        };
        const LEN: usize = 2 * BLOCK;
        // SAFETY: 2-block anonymous mapped view for rest window verification.
        let sec = unsafe {
            CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                LEN as u32,
                std::ptr::null(),
            )
        };
        assert!(!sec.is_null());
        // SAFETY: mapping 2-block section.
        let view = unsafe { MapViewOfFile(sec, FILE_MAP_ALL_ACCESS, 0, 0, LEN) };
        assert!(!view.Value.is_null());
        struct Cleanup {
            view: windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS,
            sec: windows_sys::Win32::Foundation::HANDLE,
        }
        impl Drop for Cleanup {
            fn drop(&mut self) {
                REST_IN_TEST.with(|c| c.set(Duration::ZERO));
                title_memory_remapped(self.view.Value as usize, LEN);
                // SAFETY: releasing test mapping & handle.
                unsafe {
                    UnmapViewOfFile(self.view);
                    CloseHandle(self.sec);
                }
            }
        }
        let _cleanup = Cleanup { view, sec };
        REST_IN_TEST.with(|c| c.set(Duration::from_millis(60)));

        let start = view.Value as usize;
        let end = start + LEN;
        let b0 = start / BLOCK;
        let mut j = Journal::new_for_test();

        // Prime guards on both blocks.
        assert!(writes_on(&mut j, start, end, true).is_some());
        assert!(guards().protected.contains(b0));

        // CPU write faults block 0 -> VEH opens it and records `opened[b0]`.
        // SAFETY: writing within our mapped test view.
        unsafe { (start as *mut u32).write_volatile(0x1234_5678) };

        // Immediate check during rest window: reports block 0 dirty AND leaves it unprotected!
        let (t1, w1) = writes_on(&mut j, start, end, true).expect("check during rest");
        assert!(w1.iter().any(|&(p, a)| p == b0 && a == t1));
        assert!(
            !guards().protected.contains(b0),
            "resting block must stay writable without re-guarding"
        );

        // Wait past the 60 ms rest window; next check re-guards block 0 cleanly.
        std::thread::sleep(Duration::from_millis(80));
        let (t2, w2) = writes_on(&mut j, start, end, true).expect("check after rest");
        assert!(w2.iter().any(|&(p, a)| p == b0 && a == t2));
        assert!(
            guards().protected.contains(b0),
            "block must be re-protected once rest window expires"
        );
    }
}

#[cfg(test)]
mod mapping_tests {
    use super::{title_memory_reading, title_memory_remapping};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, mpsc};
    use std::time::Duration;

    /// A change of mapping waits for a read in place that is under way, and
    /// a read in place waits for a change of mapping.
    #[test]
    fn reads_in_place_and_remappings_wait_for_each_other() {
        // A reader holds; the change does not begin until it lets go.
        let changed = Arc::new(AtomicBool::new(false));
        let reading = title_memory_reading();
        let (began, wait) = mpsc::channel();
        let change = {
            let changed = changed.clone();
            std::thread::spawn(move || {
                began.send(()).unwrap();
                let _changing = title_memory_remapping();
                changed.store(true, Ordering::SeqCst);
            })
        };
        wait.recv().unwrap();
        std::thread::sleep(Duration::from_millis(100));
        assert!(!changed.load(Ordering::SeqCst), "the mapping changed under a read in place");
        drop(reading);
        change.join().unwrap();
        assert!(changed.load(Ordering::SeqCst));
        // A change holds; the read does not begin until it is done.
        let read = Arc::new(AtomicBool::new(false));
        let changing = title_memory_remapping();
        let (began, wait) = mpsc::channel();
        let reader = {
            let read = read.clone();
            std::thread::spawn(move || {
                began.send(()).unwrap();
                let _reading = title_memory_reading();
                read.store(true, Ordering::SeqCst);
            })
        };
        wait.recv().unwrap();
        std::thread::sleep(Duration::from_millis(100));
        assert!(!read.load(Ordering::SeqCst), "a read in place began during a change of mapping");
        drop(changing);
        reader.join().unwrap();
        assert!(read.load(Ordering::SeqCst));
    }

    /// `ask_once` caches `MEM_MAPPED` guarded views in `j.ahead` so repeated
    /// `writes()` refreshes within one submission do 0 extra `VirtualQuery`
    /// probes while still immediately surfacing `wrote()` mutations.
    /// Uses a private [`Journal`] so parallel test runs never race on `JOURNAL.ahead`.
    #[test]
    fn ask_once_caches_mapped_views_and_respects_wrote() {
        use super::{
            BLOCK, Journal, PAGE, THREAD_MIXED_VQS, ask_once_end_on, ask_once_on,
            title_memory_remapped, writes_on, wrote_on,
        };
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_ALL_ACCESS, MapViewOfFile, PAGE_READWRITE, UnmapViewOfFile,
        };
        const LEN: usize = 2 * BLOCK;
        // SAFETY: anonymous page-file-backed section mapped and unmapped within test.
        let sec = unsafe {
            CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                LEN as u32,
                std::ptr::null(),
            )
        };
        assert!(!sec.is_null());
        // SAFETY: mapping 2 blocks of anonymous section.
        let view = unsafe { MapViewOfFile(sec, FILE_MAP_ALL_ACCESS, 0, 0, LEN) };
        assert!(!view.Value.is_null());
        let start = view.Value as usize;
        let end = start + LEN;
        let mut j = Journal::new_for_test();

        // Prime guards on the mapped view, then verify ask_once caches it.
        assert!(writes_on(&mut j, start, end, true).is_some());
        ask_once_on(&mut j, &[(start, end)]);
        let vq_before = THREAD_MIXED_VQS.with(|c| c.get());
        let (now1, w1) = writes_on(&mut j, start, end, true).expect("cached in ask_once");
        let (now2, w2) = writes_on(&mut j, start, end, true).expect("cached in ask_once");
        let vq_after = THREAD_MIXED_VQS.with(|c| c.get());
        assert_eq!(vq_before, vq_after, "repeated writes() under ask_once must do 0 VirtualQuery probes");
        assert_eq!(now1, now2);
        assert_eq!(w1, w2);

        // `wrote()` while ask_once stands must surface the written page immediately.
        wrote_on(&mut j, start + PAGE, PAGE);
        let (now3, w3) = writes_on(&mut j, start, end, true).expect("still cached after wrote");
        assert!(now3 > now2);
        assert!(w3.iter().any(|&(p, a)| p == (start + PAGE) / PAGE && a == now3));

        ask_once_end_on(&mut j);
        title_memory_remapped(start, LEN);
        // SAFETY: releasing test mapping & handle.
        unsafe {
            UnmapViewOfFile(view);
            CloseHandle(sec);
        }
    }
}
