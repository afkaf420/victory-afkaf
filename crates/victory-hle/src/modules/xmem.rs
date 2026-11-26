//! `xmem.dll`: console memory, served from host memory. The GPU side works
//! from CPU addresses (unified memory), so memory types and bandwidth
//! classes change nothing here; what constrains addresses (alignment,
//! 2 MiB pages, address-bit limits) is honoured.
#![allow(non_snake_case)]
// Title code calls these through its import table, passing pointers the real
// API would trust.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::alloc::{Layout, alloc, dealloc};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex, MutexGuard, OnceLock, RwLock};

use windows_sys::Win32::Foundation::{
    CloseHandle, ERROR_COMMITMENT_LIMIT, HANDLE, INVALID_HANDLE_VALUE,
};
use windows_sys::Win32::System::Memory::{
    CreateFileMappingW, MEM_ADDRESS_REQUIREMENTS, MEM_COMMIT, MEM_EXTENDED_PARAMETER,
    MEM_PRESERVE_PLACEHOLDER, MEM_RELEASE, MEM_REPLACE_PLACEHOLDER, MEM_RESERVE,
    MEM_RESERVE_PLACEHOLDER, MEMORY_BASIC_INFORMATION, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile3,
    MemExtendedParameterAddressRequirements, PAGE_NOACCESS, PAGE_READWRITE, SEC_COMMIT,
    SEC_RESERVE, UnmapViewOfFile2, VirtualAlloc, VirtualAlloc2, VirtualFree, VirtualQueryEx,
};
use windows_sys::Win32::System::ProcessStatus::{K32GetPerformanceInfo, PERFORMANCE_INFORMATION};
use windows_sys::Win32::System::Threading::GetCurrentProcess;
use windows_sys::core::BOOL;

use crate::{HleModule, hle_export};

const MIB: usize = 1 << 20;

/// Whether we run under Wine (its ntdll exports `wine_get_version`).
pub(crate) fn on_wine() -> bool {
    static WINE: LazyLock<bool> = LazyLock::new(|| {
        use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleA, GetProcAddress};
        // SAFETY: plain lookups in ntdll, always loaded.
        unsafe {
            let ntdll = GetModuleHandleA(c"ntdll.dll".as_ptr().cast());
            !ntdll.is_null() && GetProcAddress(ntdll, c"wine_get_version".as_ptr().cast()).is_some()
        }
    });
    *WINE
}

/// Standard memory the title has handed to the system.
static TRANSFERRED: AtomicUsize = AtomicUsize::new(0);

/// Bytes currently allocated through `XMemAllocDefault` (non-32-bit blocks).
static HEAP_USED: AtomicUsize = AtomicUsize::new(0);

/// Committed non-placeholder virtual ranges allocated via `XMemVirtualAlloc`,
/// stored as non-overlapping `base -> end` intervals for accurate title
/// working-set reporting (`XMemGetWorkingSetStatistics`).
static VIRTUAL_COMMITTED: LazyLock<Mutex<BTreeMap<usize, usize>>> =
    LazyLock::new(|| Mutex::new(BTreeMap::new()));

fn record_virtual_commit(base: usize, len: usize) {
    if base == 0 || len == 0 {
        return;
    }
    let end = base.saturating_add(len);
    if end <= base {
        return;
    }
    let mut map = VIRTUAL_COMMITTED.lock().unwrap_or_else(|e| e.into_inner());
    let mut lo = base;
    let mut hi = end;
    let mut remove = Vec::new();
    if let Some((&s, &e)) = map.range(..base).next_back()
        && e >= base
    {
        lo = lo.min(s);
        hi = hi.max(e);
        remove.push(s);
    }
    for (&s, &e) in map.range(base..=end) {
        lo = lo.min(s);
        hi = hi.max(e);
        remove.push(s);
    }
    for s in remove {
        map.remove(&s);
    }
    map.insert(lo, hi);
}

fn clear_virtual_commit(base: usize, len: usize) {
    if base == 0 || len == 0 {
        return;
    }
    let end = base.saturating_add(len);
    if end <= base {
        return;
    }
    let mut map = VIRTUAL_COMMITTED.lock().unwrap_or_else(|e| e.into_inner());
    let mut overlapping = Vec::new();
    if let Some((&s, &e)) = map.range(..base).next_back()
        && e > base
    {
        overlapping.push((s, e));
    }
    for (&s, &e) in map.range(base..end) {
        overlapping.push((s, e));
    }
    for &(s, _) in &overlapping {
        map.remove(&s);
    }
    for (s, e) in overlapping {
        if s < base {
            map.insert(s, base);
        }
        if e > end {
            map.insert(end, e);
        }
    }
}

/// Sums live committed bytes from `VIRTUAL_COMMITTED`, pruning sub-ranges the
/// title released directly via kernel32 `VirtualFree`.
fn live_virtual_committed() -> usize {
    let mut map = VIRTUAL_COMMITTED.lock().unwrap_or_else(|e| e.into_inner());
    if map.is_empty() {
        return 0;
    }
    let snapshot: Vec<(usize, usize)> = map.iter().map(|(&s, &e)| (s, e)).collect();
    let mut live = BTreeMap::new();
    let mut total = 0usize;
    // SAFETY: pseudo-handle for current process.
    let process = unsafe { GetCurrentProcess() };
    for (start, end) in snapshot {
        let mut p = start;
        while p < end {
            // SAFETY: plain out struct filled by `VirtualQueryEx`.
            let mut mbi: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
            let n = unsafe {
                VirtualQueryEx(
                    process,
                    p as *const c_void,
                    &mut mbi,
                    size_of::<MEMORY_BASIC_INFORMATION>(),
                )
            };
            let next = (mbi.BaseAddress as usize)
                .saturating_add(mbi.RegionSize)
                .min(end);
            if n == 0 || next <= p {
                break;
            }
            if mbi.State == MEM_COMMIT {
                total = total.saturating_add(next - p);
                if let Some((_, last_e)) = live.iter_mut().next_back()
                    && *last_e == p
                {
                    *last_e = next;
                } else {
                    live.insert(p, next);
                }
            }
            p = next;
        }
    }
    *map = live;
    total
}

/// Returns `(tracked_title_bytes, allocated_physical_pages)` counting only
/// title-owned physical pages, committed `XMemVirtualAlloc` regions, and
/// `XMemAllocDefault` blocks (excluding host layer / D3D12 commit).
fn tracked_title_memory() -> (usize, usize) {
    let phys_pages = physical().pages.len();
    let phys_bytes = phys_pages.saturating_mul(PAGE);
    let virt_bytes = live_virtual_committed();
    let heap_bytes = HEAP_USED.load(Ordering::Relaxed);
    (
        phys_bytes
            .saturating_add(virt_bytes)
            .saturating_add(heap_bytes),
        phys_pages,
    )
}

// Console additions to the host's flags.
/// 2 MiB pages; the host reads this bit as large pages, which need a
/// privilege.
const PAGES_2M: u32 = 0x2000_0000;
/// With `PAGES_2M`: 64 KiB pages; the host reads it as AWE memory.
const PAGES_64K_BIT: u32 = 0x40_0000;
const PAGES_64K: u32 = PAGES_2M | PAGES_64K_BIT;
/// The host's protection constants and `PAGE_GUARD`. Above them the console
/// adds GPU access bits (0x800-0x40000). Write-combining and uncached are
/// dropped too: our GPU side reads this memory through the CPU.
const HOST_PROTECTION: u32 = 0x1ff;
/// XMem flags of memory both processors reach: all of ours.
const GRAPHICS: u64 = 3;
/// XMem flag bit of 32-bit graphics memory.
const ADDRESS_32BIT: u64 = 4;
/// The host's write watching (not in windows-sys' memory module).
const MEM_WRITE_WATCH: u32 = 0x20_0000;

/// The addresses XMem `flags` allow, first and last: the low 4 GiB for
/// 32-bit memory; else from 4 GiB (kept for 32-bit requests, which the
/// title's big pools would otherwise crowd out) up to 46 down to 40 bits
/// from the field at bit 12, or 40 bits without one: the console keeps
/// title memory there, and its ray-tracing structures must be (
/// the title's BVH builder refuses one above; the title asks for its GPU heap
/// with no field). Decision 0001.
fn address_range(flags: u64) -> (usize, usize) {
    let bits = match (flags & ADDRESS_32BIT, flags >> 12 & 7) {
        (0, 0) => 40,
        (0, field) => 47 - field,
        _ => return (0, u32::MAX as usize),
    };
    (1 << 32, (1 << bits) - 1)
}

/// Per-VA console `(flags, raw_protect)` table preserved across placeholder
/// map/unmap cycles so `XMemMapPhysicalPages` restores each Low-4GB zone's
/// exact console protection (`0x1004`/`0x1404`/`0x8404`/`0x2404`/`0x2004`)
/// while upgrading `PAGE_GRAPHICS_NOACCESS` (`0x0801`) streaming pools to
/// `CachedGpuRw` (`0x2004`) when physical pages are mapped.
static VA_PROTECT: RwLock<BTreeMap<usize, (usize, u32, u32)>> = RwLock::new(BTreeMap::new());

fn record_va_protect(base: usize, len: usize, flags: u32, raw_protect: u32) {
    if base == 0 || len == 0 {
        return;
    }
    let end = base.saturating_add(len);
    if end <= base {
        return;
    }
    {
        let mut map = VA_PROTECT.write().unwrap_or_else(|e| e.into_inner());
        let mut overlapping: Vec<(usize, usize, u32, u32)> = Vec::new();
        if let Some((&s, &(e, f, p))) = map.range(..base).next_back()
            && e > base
        {
            overlapping.push((s, e, f, p));
        }
        for (&s, &(e, f, p)) in map.range(base..end) {
            overlapping.push((s, e, f, p));
        }
        for &(s, _, _, _) in &overlapping {
            map.remove(&s);
        }
        for (s, e, f, p) in overlapping {
            if s < base {
                map.insert(s, (base, f, p));
            }
            if e > end {
                map.insert(end, (e, f, p));
            }
        }
        map.insert(base, (end, flags, raw_protect));
    }
    victory_gpu::note_console_memory(base, len, flags, raw_protect);
}

fn clear_va_protect(base: usize, len: usize) {
    if base == 0 || len == 0 {
        return;
    }
    let end = base.saturating_add(len);
    if end <= base {
        return;
    }
    let mut map = VA_PROTECT.write().unwrap_or_else(|e| e.into_inner());
    let mut overlapping: Vec<(usize, usize, u32, u32)> = Vec::new();
    if let Some((&s, &(e, f, p))) = map.range(..base).next_back()
        && e > base
    {
        overlapping.push((s, e, f, p));
    }
    for (&s, &(e, f, p)) in map.range(base..end) {
        overlapping.push((s, e, f, p));
    }
    for &(s, _, _, _) in &overlapping {
        map.remove(&s);
    }
    for (s, e, f, p) in overlapping {
        if s < base {
            map.insert(s, (base, f, p));
        }
        if e > end {
            map.insert(end, (e, f, p));
        }
    }
}

/// Resolves the console `(lo, hi, flags, raw_protect)` slices for mapped
/// physical pages in `[base, base + len)` against `VA_PROTECT`: preserves
/// explicit zone protections (`GpuReadOnly`, `WriteCombinedGpuRw`,
/// `CachedGpuRw`, `CoherentGpuRw`) from both initial `MEM_RESERVE_PLACEHOLDER`
/// reservations (e.g. `Func9` `0x2404`) and `MEM_REPLACE_PLACEHOLDER` splits
/// (e.g. Low-4GB zones), and upgrades `PAGE_GRAPHICS_NOACCESS` (`0x0801`) or
/// unregistered placeholders to `flags = 0x2, protect = 0x2004` (`CachedGpuRw`).
fn resolve_mapped_console_memory(base: usize, len: usize) -> Vec<(usize, usize, u32, u32)> {
    const PAGE_GRAPHICS_NOACCESS: u32 = 0x800;
    const MEM_GRAPHICS: u32 = 0x2;
    const DEFAULT_MAPPED_PROTECT: u32 = 0x2004;
    if base == 0 || len == 0 {
        return Vec::new();
    }
    let end = base.saturating_add(len);
    if end <= base {
        return Vec::new();
    }
    let map = VA_PROTECT.read().unwrap_or_else(|e| e.into_inner());
    let mut out = Vec::new();
    let mut cur = base;
    if let Some((_, &(e, f, p))) = map.range(..base).next_back()
        && e > base
    {
        let hi = e.min(end);
        let (ef, ep) = if (f & MEM_GRAPHICS) != 0 && (p & PAGE_GRAPHICS_NOACCESS) == 0 {
            (f, p)
        } else {
            (MEM_GRAPHICS, DEFAULT_MAPPED_PROTECT)
        };
        out.push((base, hi, ef, ep));
        cur = hi;
    }
    if cur < end {
        for (&s, &(e, f, p)) in map.range(cur..end) {
            if s > cur {
                out.push((cur, s.min(end), MEM_GRAPHICS, DEFAULT_MAPPED_PROTECT));
            }
            let lo = s.max(cur);
            let hi = e.min(end);
            if lo < hi {
                let (ef, ep) = if (f & MEM_GRAPHICS) != 0 && (p & PAGE_GRAPHICS_NOACCESS) == 0 {
                    (f, p)
                } else {
                    (MEM_GRAPHICS, DEFAULT_MAPPED_PROTECT)
                };
                out.push((lo, hi, ef, ep));
            }
            cur = cur.max(hi);
            if cur >= end {
                break;
            }
        }
        if cur < end {
            out.push((cur, end, MEM_GRAPHICS, DEFAULT_MAPPED_PROTECT));
        }
    }
    out
}

/// Restores the console protection classification for newly mapped physical
/// pages in `[base, base + len)` from [`resolve_mapped_console_memory`].
fn note_mapped_console_memory(base: usize, len: usize) {
    for (lo, hi, flags, protect) in resolve_mapped_console_memory(base, len) {
        victory_gpu::note_console_memory(lo, hi - lo, flags, protect);
    }
}

/// Logs the 1st, 2nd, 4th... occurrence of each console `(protect, kind, flags)`
/// tuple with cumulative MiB to aid PC-native VRAM placement diagnostics.
fn note_protection(protect: u32, kind: u32, flags: u64, at: usize, size: usize) {
    type Seen = HashMap<(u32, u32, u64), (u64, usize)>;
    static SEEN: Mutex<Option<Seen>> = Mutex::new(None);
    let (n, bytes) = {
        let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
        let e = seen
            .get_or_insert_with(Default::default)
            .entry((protect, kind, flags))
            .or_default();
        *e = (e.0 + 1, e.1.saturating_add(size));
        *e
    };
    if n.is_power_of_two() {
        tracing::info!(
            "title memory asked for as protection {protect:#x}, kind {kind:#x}, flags {flags:#x}: {n} times, {} MB, last {at:#x}+{size:#x}",
            bytes >> 20
        );
    }
}

pub extern "system" fn XMemVirtualAlloc(
    base: *mut c_void,
    size: usize,
    kind: u32,
    flags: u64,
    protect: u32,
) -> *mut c_void {
    let raw_protect = protect;
    let raw_kind = kind;
    let (lowest, highest) = address_range(flags);
    let mut requirements = MEM_ADDRESS_REQUIREMENTS {
        LowestStartingAddress: lowest as *mut c_void,
        HighestEndingAddress: highest as *mut c_void,
        Alignment: if kind & (PAGES_2M | PAGES_64K_BIT) == PAGES_2M {
            2 * MIB
        } else {
            0
        },
    };
    let mut protect = match protect & HOST_PROTECTION {
        0 => PAGE_NOACCESS,
        p => p,
    };
    // A reservation of 64 KiB pages takes physical pages later
    // (`XMemMapPhysicalPages`); on the host only a placeholder can.
    // shortcut: committing part of such a reservation later would fail
    // (the title's 23 calls never do); replace that part of the
    // placeholder then.
    let placeholder = kind & PAGES_64K == PAGES_64K && kind & MEM_COMMIT == 0;
    if placeholder {
        if kind & MEM_REPLACE_PLACEHOLDER != 0 {
            // Part of a placeholder the title split: it stays one, but still
            // carries the zone's console protection (e.g. Low-4GB Zones 1..5).
            if !base.is_null() && size > 0 {
                record_va_protect(base as usize, size, flags as u32, raw_protect);
                note_protection(raw_protect, raw_kind, flags, base as usize, size);
            }
            return base;
        }
        protect = PAGE_NOACCESS;
    }
    let mut kind = kind & !(PAGES_2M | PAGES_64K_BIT);
    if placeholder {
        kind |= MEM_RESERVE | MEM_RESERVE_PLACEHOLDER;
    }
    let mut parameter = MEM_EXTENDED_PARAMETER::default();
    parameter.Anonymous1._bitfield = MemExtendedParameterAddressRequirements as u64;
    parameter.Anonymous2.Pointer = (&raw mut requirements).cast();
    // Address requirements apply to reservations at an address of our choice.
    let constrained = base.is_null();
    if constrained {
        kind |= MEM_RESERVE;
        // The GPU side copies only the pages the title wrote since its last
        // copy (`victory_gpu`, decision 0007); placeholders cannot be
        // watched (their physical-page views are copied whole).
        if kind & MEM_RESERVE_PLACEHOLDER == 0 {
            kind |= MEM_WRITE_WATCH;
        }
    }
    // SAFETY: the title's own arguments, cleaned of console bits; the
    // parameter points at `requirements`, which outlives the call.
    // Under Stage M3 (`VICTORY_SPARSE_UMA=1`), place non-32-bit reservations
    // inside the 32 GiB-aligned 24 GiB sparse VRAM aperture first, falling
    // back to the full 40-bit range if a reservation does not fit.
    let sparse_range = (constrained && flags & ADDRESS_32BIT == 0)
        .then(victory_gpu::sparse_uma::aperture)
        .flatten();
    let mut at = std::ptr::null_mut();
    if let Some((ap_base, ap_len)) = sparse_range {
        let mut ap_req = MEM_ADDRESS_REQUIREMENTS {
            LowestStartingAddress: ap_base as *mut c_void,
            HighestEndingAddress: (ap_base + ap_len - 1) as *mut c_void,
            Alignment: requirements.Alignment,
        };
        let mut ap_param = MEM_EXTENDED_PARAMETER::default();
        ap_param.Anonymous1._bitfield = MemExtendedParameterAddressRequirements as u64;
        ap_param.Anonymous2.Pointer = (&raw mut ap_req).cast();
        // SAFETY: `ap_req` outlives the call; arguments cleaned of console bits.
        at = unsafe {
            VirtualAlloc2(
                GetCurrentProcess(),
                base,
                size,
                kind,
                protect,
                &mut ap_param,
                1,
            )
        };
    }
    if at.is_null() {
        at = unsafe {
            VirtualAlloc2(
                GetCurrentProcess(),
                base,
                size,
                kind,
                protect,
                if constrained {
                    &mut parameter
                } else {
                    std::ptr::null_mut()
                },
                constrained.into(),
            )
        };
    }
    if !at.is_null() {
        record_va_protect(at as usize, size, flags as u32, raw_protect);
        note_protection(raw_protect, raw_kind, flags, at as usize, size);
        if !placeholder && (kind & MEM_COMMIT) != 0 {
            record_virtual_commit(at as usize, size);
            victory_gpu::sparse_uma::on_map(at as usize, size);
        }
    }
    if base.is_null() {
        tracing::debug!(
            size,
            kind = format_args!("{kind:#x}"),
            flags = format_args!("{flags:#x}"),
            at = format_args!("{at:p}"),
            "XMemVirtualAlloc"
        );
    }
    at
}

/// `MEMORY_BASIC_INFORMATION` (the console's first 0x30 bytes are the
/// same), then the XMem flags and tag.
#[repr(C)]
pub struct BasicInformation {
    host: MEMORY_BASIC_INFORMATION,
    flags: u64,
    tag_code: u32,
    tag: [u8; 4],
}

// shortcut: every committed page reports graphics memory and no tag, since
// all of it is GPU-reachable here (the title's BVH builder, the
// only caller, wants graphics memory); remember the flags of
// `XMemVirtualAlloc` regions if a title tells CPU-only memory apart.
pub extern "system" fn XMemVirtualQuery(
    process: HANDLE,
    address: *const c_void,
    out: *mut BasicInformation,
) -> BOOL {
    let size = size_of::<MEMORY_BASIC_INFORMATION>();
    // SAFETY: the title passes room for the whole structure; the host
    // fails on a null one.
    unsafe {
        if VirtualQueryEx(process, address, out.cast(), size) == 0 {
            return 0;
        }
        let info = &mut *out;
        info.flags = if info.host.State == MEM_COMMIT {
            GRAPHICS
        } else {
            0
        };
        info.tag_code = 0;
        info.tag = [0; 4];
    }
    1
}

// `XMemAlloc` attributes: alignment as log2 (5 bits at 24, 4 bytes and up
// from 2), then the memory type, then bit 33: a 32-bit address.
const ALIGNMENT_SHIFT: u64 = 24;
const ALLOC_32BIT: u64 = 1 << 33;
/// Our block header: the block's size and alignment, just below the
/// pointer handed out.
const HEADER: usize = 2 * size_of::<usize>();

fn layout(size: usize, attributes: u64) -> Option<Layout> {
    // An alignment of at least 16 leaves room for the header.
    let align = 1usize << (attributes >> ALIGNMENT_SHIFT & 0x1f).max(4);
    Layout::from_size_align(size.checked_add(align)?, align).ok()
}

type AllocRoutine = extern "system" fn(usize, u64) -> *mut c_void;
type FreeRoutine = extern "system" fn(*mut c_void, u64);

/// The allocation and free routines the title's exe exports as
/// `__XMemAllocationHooks__` (the title does): `XMemAlloc` and
/// `XMemFree` call them, for the title's own calls and its middleware's;
/// the `…Default` exports do not.
fn hooks() -> [usize; 2] {
    static AT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let at = match AT.get() {
        Some(&at) => at,
        // Looked up once the title is mapped, which is before it runs.
        None => match crate::runtime::main_image() {
            Some(base) => *AT.get_or_init(|| {
                use windows_sys::Win32::System::LibraryLoader::GetProcAddress;
                // SAFETY: a mapped image and a NUL-terminated name.
                let at = unsafe {
                    GetProcAddress(base as _, c"__XMemAllocationHooks__".as_ptr().cast())
                }
                .map_or(0, |f| f as usize);
                tracing::info!(found = at != 0, "xmem allocation hooks");
                at
            }),
            None => 0,
        },
    };
    if at == 0 {
        return [0; 2];
    }
    // SAFETY: the export is two routine pointers.
    unsafe { (at as *const [usize; 2]).read() }
}

/// The title's general allocator, for middleware too: its hook if it has
/// one, else [`XMemAllocDefault`].
pub extern "system" fn XMemAlloc(size: usize, attributes: u64) -> *mut c_void {
    match hooks()[0] {
        0 => XMemAllocDefault(size, attributes),
        hook => {
            // SAFETY: the title's own routine of this type.
            let alloc: AllocRoutine = unsafe { std::mem::transmute(hook) };
            alloc(size, attributes)
        }
    }
}

/// See [`XMemAlloc`].
pub extern "system" fn XMemFree(address: *mut c_void, attributes: u64) {
    match hooks()[1] {
        0 => XMemFreeDefault(address, attributes),
        hook => {
            // SAFETY: the title's own routine of this type.
            let free: FreeRoutine = unsafe { std::mem::transmute(hook) };
            free(address, attributes)
        }
    }
}

/// The system allocator behind [`XMemAlloc`].
pub extern "system" fn XMemAllocDefault(size: usize, attributes: u64) -> *mut c_void {
    if attributes & ALLOC_32BIT != 0 {
        return XMemVirtualAlloc(
            std::ptr::null_mut(),
            size,
            MEM_RESERVE | MEM_COMMIT,
            GRAPHICS | ADDRESS_32BIT,
            PAGE_READWRITE,
        );
    }
    let Some(layout) = layout(size, attributes) else {
        return std::ptr::null_mut();
    };
    // shortcut: the host heap's addresses lie above 2^40, unlike the
    // console's; nothing seen builds ray-tracing structures in this memory
    // (the title uses its GPU heap). Sub-allocate from a
    // reservation below 2^40 if one does.
    // SAFETY: the layout has a non-zero size.
    let block = unsafe { alloc(layout) };
    if block.is_null() {
        return block.cast();
    }
    // SAFETY: the block holds the alignment (at least the header) and then
    // `size` bytes.
    HEAP_USED.fetch_add(layout.size(), Ordering::Relaxed);
    unsafe {
        let user = block.add(layout.align());
        user.sub(HEADER)
            .cast::<[usize; 2]>()
            .write([layout.size(), layout.align()]);
        user.cast()
    }
}

/// Frees what [`XMemAllocDefault`] returned, given the same attributes.
pub extern "system" fn XMemFreeDefault(address: *mut c_void, attributes: u64) {
    if address.is_null() {
        return;
    }
    // SAFETY: the address came from `XMemAlloc` with these attributes: a
    // whole reservation, or a header-prefixed block.
    unsafe {
        if attributes & ALLOC_32BIT != 0 {
            let mut mbi: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
            let queried = VirtualQueryEx(
                GetCurrentProcess(),
                address,
                &mut mbi,
                size_of::<MEMORY_BASIC_INFORMATION>(),
            );
            let len = if queried != 0 { mbi.RegionSize } else { 0 };
            if len > 0 {
                clear_va_protect(address as usize, len);
                clear_virtual_commit(address as usize, len);
                victory_gpu::clear_console_memory(address as usize, len);
                victory_gpu::title_memory_remapped(address as usize, len);
            }
            VirtualFree(address, 0, MEM_RELEASE);
            return;
        }
        let user = address.cast::<u8>();
        let [size, align] = user.sub(HEADER).cast::<[usize; 2]>().read();
        let _ = HEAP_USED.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |cur| {
            Some(cur.saturating_sub(size))
        });
        dealloc(
            user.sub(align),
            Layout::from_size_align_unchecked(size, align),
        );
    }
}

/// A retail console has no tools memory.
pub extern "system" fn XMemIsToolsMemAvailable() -> BOOL {
    0
}

// Physical pages: on the console, 64 KiB frames a title maps into its own
// reservations, at more than one address if it likes. Here each allocation
// is a pagefile-backed section, its pages are 64 KiB slices of it, and a
// mapping is a view of consecutive slices in place of a host placeholder.

const PAGE: usize = 64 << 10;

/// Resolves `VICTORY_SINGLE_VIEW_WATCH`: strictly opt-in (`"1"` enables on
/// native Windows; off by default everywhere and always disabled on Wine).
/// Keeping `SEC_RESERVE` section views by default avoids asynchronously
/// unmapping live multi-megabyte single-view private allocations (`PAGE_NOACCESS`)
/// under running title CPU threads during lazy alias or DirectStorage promotion.
fn resolve_single_view_watch(env: Option<&str>, wine: bool) -> bool {
    if wine {
        return false;
    }
    env.is_some_and(|v| v.trim() == "1")
}

/// Whether single-view physical mappings use `MEM_WRITE_WATCH` private memory
/// and lazily promote to a `SEC_RESERVE` file mapping + `VIEW_GUARD` only when
/// a second virtual alias maps the same physical page
/// (`VICTORY_SINGLE_VIEW_WATCH=1` opts in; off by default).
fn single_view_watch() -> bool {
    static ON: LazyLock<bool> = LazyLock::new(|| {
        let env = std::env::var("VICTORY_SINGLE_VIEW_WATCH").ok();
        resolve_single_view_watch(env.as_deref(), on_wine())
    });
    *ON
}

/// Number of threads used for page mapping (`VICTORY_MAP_THREADS`, default `1`
/// to avoid Windows VAD address-space lock contention).
fn map_threads() -> usize {
    static N: LazyLock<usize> = LazyLock::new(|| {
        std::env::var("VICTORY_MAP_THREADS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(1)
            .clamp(1, 32)
    });
    *N
}

/// Whether already-committed section pages skip redundant `VirtualAlloc(MEM_COMMIT)`
/// calls (`VICTORY_MAP_LEAN`, default ON, `=0` disables).
fn map_lean() -> bool {
    static ON: LazyLock<bool> =
        LazyLock::new(|| std::env::var_os("VICTORY_MAP_LEAN").is_none_or(|v| v != "0"));
    *ON
}

struct Section {
    handle: HANDLE,
    pages: usize,
    /// Persistent whole-section write view (`VICTORY_SECTION_WINDOW`, default ON,
    /// `=0` disables); `0` if mapping failed or disabled.
    window: OnceLock<usize>,
    /// Bit per 64 KiB page already committed via `window()` or `commit_shown()`.
    committed: Box<[AtomicU64]>,
}

impl Section {
    fn new(handle: HANDLE, pages: usize) -> Self {
        let words = pages.div_ceil(64);
        let init = if on_wine() { u64::MAX } else { 0 };
        Self {
            handle,
            pages,
            window: OnceLock::new(),
            committed: (0..words).map(|_| AtomicU64::new(init)).collect(),
        }
    }

    /// Persistent writable address for `n` pages starting at `index`, avoiding
    /// per-write `MapViewOfFile`/`UnmapViewOfFile` TLB shootdowns
    /// (`VICTORY_SECTION_WINDOW=0` disables).
    fn window(&self, index: usize, n: usize) -> Option<usize> {
        use windows_sys::Win32::System::Memory::{FILE_MAP_WRITE, MapViewOfFile};
        static ON: LazyLock<bool> =
            LazyLock::new(|| std::env::var_os("VICTORY_SECTION_WINDOW").is_none_or(|v| v != "0"));
        if !*ON || index.checked_add(n).is_none_or(|e| e > self.pages) {
            return None;
        }
        let base = *self.window.get_or_init(|| {
            // SAFETY: mapping the whole section we own for our write window.
            unsafe { MapViewOfFile(self.handle, FILE_MAP_WRITE, 0, 0, self.pages * PAGE) }.Value
                as usize
        });
        if base == 0 {
            return None;
        }
        let missing = |page: usize| (self.committed[page / 64].load(Ordering::Acquire) >> (page % 64)) & 1 == 0;
        let mut page = index;
        while page < index + n {
            if !missing(page) {
                page += 1;
                continue;
            }
            let run = (page..index + n).take_while(|&p| missing(p)).count();
            if run == 0 {
                continue;
            }
            if !commit(base + page * PAGE, run * PAGE) {
                return None;
            }
            for p in page..page + run {
                self.committed[p / 64].fetch_or(1u64 << (p % 64), Ordering::AcqRel);
            }
            page += run;
        }
        Some(base + index * PAGE)
    }

    /// Commits `len` bytes at `base` (showing pages starting at `index`),
    /// skipping the syscall when `VICTORY_MAP_LEAN` knows they are already committed.
    fn commit_shown(&self, index: usize, base: usize, len: usize) -> bool {
        let pages = index..(index + len / PAGE).min(self.pages);
        let done = || {
            pages.clone().all(|p| {
                (self.committed[p / 64].load(Ordering::Acquire) >> (p % 64)) & 1 != 0
            })
        };
        if map_lean() && done() {
            return true;
        }
        if !commit(base, len) {
            return false;
        }
        for p in pages {
            self.committed[p / 64].fetch_or(1u64 << (p % 64), Ordering::AcqRel);
        }
        true
    }
}

// SAFETY: a section handle may be used from any thread.
unsafe impl Send for Section {}
// SAFETY: as above.
unsafe impl Sync for Section {}

impl Drop for Section {
    fn drop(&mut self) {
        use windows_sys::Win32::System::Memory::UnmapViewOfFile;
        if let Some(&base) = self.window.get()
            && base != 0
        {
            // SAFETY: our persistent window view; all holders of `Arc<Section>` are gone.
            unsafe {
                UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: base as *mut c_void,
                })
            };
        }
        victory_gpu::title_memory_section_gone(self.handle as usize);
        // SAFETY: our handle; views keep the section alive on their own.
        unsafe { CloseHandle(self.handle) };
    }
}

/// Where a page lives, or what a view starts with: a section and a page
/// index in it.
#[derive(Clone)]
struct Slice {
    section: Arc<Section>,
    index: usize,
}

impl Slice {
    /// The key in `Physical::shown` of the page `k` pages after this one.
    fn key(&self, k: usize) -> (usize, usize) {
        (Arc::as_ptr(&self.section) as usize, self.index + k)
    }
}

#[derive(Default)]
struct Physical {
    next_page: usize,
    pages: HashMap<usize, Slice>,
    /// Mapped views by address: `(len, first_slice, is_private_watch)`.
    views: BTreeMap<usize, (usize, Slice, bool)>,
    /// Where each page shows: by its section and its index there, the
    /// address of that page in every view that shows it now ([`shown_at`];
    /// as upstream's c75fdba keeps it). The title maps pages where the
    /// card reads them and has them filled afterwards by DirectStorage,
    /// which writes through views of our own.
    shown: HashMap<(usize, usize), Vec<usize>>,
    /// Physical page keys `(section_ptr, index)` that already hold live bytes
    /// in their `SEC_RESERVE` section (from prior promotion, non-zero unmap
    /// flush, or DirectStorage writes before mapping).
    backed: HashSet<(usize, usize)>,
}

static PHYSICAL: LazyLock<Mutex<Physical>> = LazyLock::new(|| {
    Mutex::new(Physical {
        // Page numbers are opaque to titles; 0 stays unused.
        next_page: 0x10_0000,
        ..Default::default()
    })
});

fn physical() -> MutexGuard<'static, Physical> {
    PHYSICAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// `count` is in and out (pages wanted, pages given); all or nothing here.
pub extern "system" fn XMemAllocatePhysicalPages(
    flags: u64,
    count: *mut usize,
    pages: *mut usize,
) -> BOOL {
    if count.is_null() || pages.is_null() {
        return 0;
    }
    // SAFETY: checked non-null.
    let n = unsafe { *count };
    let Some(size) = n.checked_mul(PAGE).filter(|&s| s > 0) else {
        return 0;
    };
    // A pagefile-backed section of `size` bytes, its pages committed when
    // first mapped or written (`commit`): the title asks for 6.1
    // GiB at once, which a busy host often cannot commit (error 1450).
    // SAFETY: plain section creation.
    let handle = unsafe {
        CreateFileMappingW(
            INVALID_HANDLE_VALUE,
            std::ptr::null(),
            // Under Wine a committed section is a sparse file, free until
            // touched, and committing a reserved one page by page costs a
            // wineserver request each that scans every range committed so
            // far: commit it whole there.
            PAGE_READWRITE | if on_wine() { SEC_COMMIT } else { SEC_RESERVE },
            (size as u64 >> 32) as u32,
            size as u32,
            std::ptr::null(),
        )
    };
    if handle.is_null() {
        let error = std::io::Error::last_os_error();
        tracing::warn!(n, %error, "XMemAllocatePhysicalPages failed");
        // SAFETY: checked non-null.
        unsafe { count.write(0) };
        return 0;
    }
    tracing::debug!(
        n,
        flags = format_args!("{flags:#x}"),
        "XMemAllocatePhysicalPages"
    );
    // For the GPU side's in-place census: only a committed section can be
    // opened as a host heap.
    victory_gpu::title_memory_section(handle as usize, size as u64, on_wine());
    let section = Arc::new(Section::new(handle, n));
    let mut physical = physical();
    let first = physical.next_page;
    physical.next_page += n * FRAMES;
    for index in 0..n {
        let slice = Slice {
            section: section.clone(),
            index,
        };
        let number = first + index * FRAMES;
        physical.pages.insert(number, slice);
        // SAFETY: the title passes room for `n` page numbers.
        unsafe { pages.add(index).write(number) };
    }
    1
}

/// Page numbers count 4 KiB frames, so a 64 KiB page's is a multiple of
/// 16: the title keeps them as 21-bit `number >> 4` and rebuilds
/// them with `<< 4` (128 GiB of pages at most).
const FRAMES: usize = PAGE >> 12;

/// Pages still mapped stay readable through their views until unmapped.
pub extern "system" fn XMemFreePhysicalPages(n: usize, pages: *const usize) -> BOOL {
    if pages.is_null() {
        return 0;
    }
    let mut gone = Vec::with_capacity(n);
    {
        let mut physical = physical();
        for i in 0..n {
            // SAFETY: the title passes `n` page numbers.
            if let Some(slice) = physical.pages.remove(&unsafe { *pages.add(i) }) {
                let key = slice.key(0);
                if !physical.shown.contains_key(&key) {
                    physical.backed.remove(&key);
                }
                gone.push(slice);
            }
        }
    }
    // Drop `Arc<Section>` outside `physical()` lock so unmapping a persistent
    // section window never runs while holding the page-table mutex.
    drop(gone);
    1
}

/// A view of our own on consecutive slices of one section, committed and
/// writable — backed by `Section::window()` (`VICTORY_SECTION_WINDOW=1`) when
/// available, or a temporary `MapViewOfFile` unmapped on drop.
enum View {
    Window {
        _section: Arc<Section>,
        ptr: usize,
        len: usize,
    },
    Temp {
        at: MEMORY_MAPPED_VIEW_ADDRESS,
        len: usize,
    },
}

impl View {
    /// Opens `n` slices from `slice` on, preferring the persistent section window.
    fn map(slice: &Slice, n: usize) -> Option<View> {
        use windows_sys::Win32::System::Memory::{FILE_MAP_WRITE, MapViewOfFile};
        let len = n * PAGE;
        if let Some(ptr) = slice.section.window(slice.index, n) {
            return Some(View::Window {
                _section: slice.section.clone(),
                ptr,
                len,
            });
        }
        let offset = (slice.index * PAGE) as u64;
        // SAFETY: slices of a section we own; the view is ours until dropped.
        let at = unsafe {
            MapViewOfFile(
                slice.section.handle,
                FILE_MAP_WRITE,
                (offset >> 32) as u32,
                offset as u32,
                len,
            )
        };
        if at.Value.is_null() {
            return None;
        }
        let view = View::Temp { at, len };
        slice
            .section
            .commit_shown(slice.index, at.Value as usize, len)
            .then_some(view)
    }

    fn bytes(&mut self) -> &mut [u8] {
        match self {
            View::Window { ptr, len, .. } => {
                // SAFETY: `ptr..ptr+len` lies inside the section's committed
                // persistent write window kept alive by `_section`.
                unsafe { std::slice::from_raw_parts_mut(*ptr as *mut u8, *len) }
            }
            View::Temp { at, len } => {
                // SAFETY: the view holds `len` committed read-write bytes, ours
                // alone while `self` lives.
                unsafe { std::slice::from_raw_parts_mut(at.Value.cast(), *len) }
            }
        }
    }
}

impl Drop for View {
    fn drop(&mut self) {
        if let View::Temp { at, .. } = self {
            use windows_sys::Win32::System::Memory::UnmapViewOfFile;
            // SAFETY: our temporary view.
            unsafe { UnmapViewOfFile(*at) };
        }
    }
}

/// The pages behind bytes `[offset, offset + len)` of `pages` (64 KiB
/// each), as runs of consecutive slices of one section: each run's first
/// slice and its length in pages. One look at the page table, which the
/// title's own mapping calls share with every DirectStorage worker. `None`
/// if a page is unknown or `pages` ends early.
fn runs(pages: &[usize], offset: usize, len: usize) -> Option<Vec<(Slice, usize)>> {
    let end = offset.checked_add(len)?;
    let pages = pages.get(offset / PAGE..end.div_ceil(PAGE))?;
    let physical = physical();
    let mut runs: Vec<(Slice, usize)> = Vec::new();
    for page in pages {
        let slice = physical.pages.get(page)?;
        match runs.last_mut() {
            Some((first, n))
                if Arc::ptr_eq(&first.section, &slice.section)
                    && slice.index == first.index + *n =>
            {
                *n += 1;
            }
            _ => runs.push((slice.clone(), 1)),
        }
    }
    Some(runs)
}

/// Whether every page under `[offset, offset + len)` is known (a write that
/// fails with them known is a host failure).
pub(crate) fn pages_known(pages: &[usize], offset: usize, len: usize) -> bool {
    runs(pages, offset, len).is_some()
}

/// Copies `data` into physical pages, from `offset` bytes into the first,
/// whether or not they are mapped anywhere (DirectStorage destinations).
/// Pages currently mapped as a single-view `MEM_WRITE_WATCH` private view are
/// written directly in place (setting hardware dirty bits with zero
/// `MapViewOfFile` TLB shootdowns); section-backed or unmapped pages write
/// through one temporary view per consecutive run. False if a page is unknown
/// or the host fails.
pub(crate) fn write_physical(pages: &[usize], offset: usize, data: &[u8]) -> bool {
    let Some(end) = offset.checked_add(data.len()) else {
        return false;
    };
    let Some(touched) = pages.get(offset / PAGE..end.div_ceil(PAGE)) else {
        return false;
    };
    if data.is_empty() {
        return pages_known(pages, offset, 0);
    }
    // Prepare any `GpuResident` (`PAGE_NOACCESS`) title blocks mapped over
    // these pages BEFORE acquiring `physical()` (zero lock inversion, zero
    // readback for full 64 KiB overwrites, edge fetch for partial blocks).
    for (at, bytes) in shown_at(pages, offset, data.len()) {
        victory_gpu::prepare_cpu_overwrite(at, bytes);
    }
    // Group section writes outside `physical()` lock while writing directly
    // into any single-view `Private` mapping while holding `physical()`.
    let mut section_writes: Vec<(Slice, usize, usize, usize, usize)> = Vec::new();
    {
        let mut physical = physical();
        let (mut within, mut done) = (offset % PAGE, 0);
        for page in touched {
            let Some(slice) = physical.pages.get(page).cloned() else {
                return false;
            };
            let k = (PAGE - within).min(data.len() - done);
            let key = slice.key(0);
            // If mapped as a single-view `Private` (`MEM_WRITE_WATCH`) page,
            // write straight into that VA while `physical` prevents unmapping.
            let private_va = physical
                .shown
                .get(&key)
                .filter(|addrs| addrs.len() == 1)
                .and_then(|addrs| {
                    let va = addrs[0];
                    physical
                        .views
                        .range(..=va)
                        .next_back()
                        .filter(|&(&base, &(size, _, is_priv))| is_priv && va < base + size)
                        .map(|_| va)
                });
            if let Some(va) = private_va {
                // SAFETY: `va..va+PAGE` is a committed `PAGE_READWRITE` private
                // view held live by the `physical` mutex; `[within..within+k]`
                // lies inside the page.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        data.as_ptr().add(done),
                        (va + within) as *mut u8,
                        k,
                    );
                }
            } else {
                physical.backed.insert(key);
                match section_writes.last_mut() {
                    Some((first, n_pages, _, prev_done, prev_len))
                        if within == 0
                            && Arc::ptr_eq(&first.section, &slice.section)
                            && slice.index == first.index + *n_pages
                            && *prev_done + *prev_len == done =>
                    {
                        *n_pages += 1;
                        *prev_len += k;
                    }
                    _ => section_writes.push((slice, 1, within, done, k)),
                }
            }
            (within, done) = (0, done + k);
        }
    }
    for (slice, n, within, done, len) in section_writes {
        let Some(mut view) = View::map(&slice, n) else {
            return false;
        };
        view.bytes()[within..within + len].copy_from_slice(&data[done..done + len]);
    }
    true
}

/// Bytes `[offset, offset + len)` of physical pages, writable in place
/// until dropped: DirectStorage decodes into it instead of a buffer it
/// would then copy.
pub(crate) struct PageWindow {
    view: View,
    within: usize,
    len: usize,
}

impl PageWindow {
    pub(crate) fn bytes(&mut self) -> &mut [u8] {
        let (within, len) = (self.within, self.len);
        &mut self.view.bytes()[within..within + len]
    }
}

/// A window on bytes `[offset, offset + len)` of `pages` when they are
/// consecutive pages of one allocation (how titles usually hand them out);
/// `None` otherwise (pages of several allocations, out of order, or
/// unknown, or a host failure), which `write_physical` then handles or
/// reports. Any page currently in a single-view private mapping is promoted to
/// its section first so the returned `PageWindow` remains valid even if the
/// title concurrently unmaps its virtual view while DirectStorage decodes.
pub(crate) fn page_window(pages: &[usize], offset: usize, len: usize) -> Option<PageWindow> {
    let end = offset.checked_add(len)?;
    let touched = pages.get(offset / PAGE..end.div_ceil(PAGE))?;
    // Prepare any `GpuResident` (`PAGE_NOACCESS`) title blocks BEFORE acquiring
    // `physical()` or decoding into the section (partial blocks fetch GPU bytes
    // first; full 64 KiB overwrites drop `GpuResident` with zero readback).
    for (at, bytes) in shown_at(pages, offset, len) {
        victory_gpu::prepare_cpu_overwrite(at, bytes);
    }
    let (slice, n) = {
        let mut physical = physical();
        let mut first: Option<(Slice, usize)> = None;
        for page in touched {
            let slice = physical.pages.get(page)?;
            match first.as_mut() {
                None => first = Some((slice.clone(), 1)),
                Some((base, n))
                    if Arc::ptr_eq(&base.section, &slice.section)
                        && slice.index == base.index + *n =>
                {
                    *n += 1;
                }
                _ => return None,
            }
        }
        let (slice, n) = first?;
        // Promote any private single-view mapping covering these pages before
        // returning a standalone section `View`. Lock order: `physical` ->
        // `title_memory_remapping` (same as `XMemMapPhysicalPages`).
        let to_promote: Vec<usize> = (0..n)
            .filter_map(|k| physical.shown.get(&slice.key(k)))
            .flatten()
            .copied()
            .filter(|&va| {
                physical
                    .views
                    .range(..=va)
                    .next_back()
                    .is_some_and(|(&base, &(size, _, is_priv))| is_priv && va < base + size)
            })
            .collect();
        if !to_promote.is_empty() {
            let _changing = victory_gpu::title_memory_remapping();
            for va in to_promote {
                if !promote_view_containing(&mut physical, va) {
                    return None;
                }
            }
        }
        for k in 0..n {
            physical.backed.insert(slice.key(k));
        }
        (slice, n)
    };
    Some(PageWindow {
        view: View::map(&slice, n)?,
        within: offset % PAGE,
        len,
    })
}

/// Writes a 64 KiB block into the section backing a `MEM_MAPPED` title VA `va`
/// through a temporary alias view (`View::map`), allowing `lazy_readback` to
/// populate physical RAM *while `va` remains `PAGE_NOACCESS`* (zero torn-read
/// window). Uses bounded non-blocking `PHYSICAL.try_lock()` retries (5 ms cap)
/// so concurrent unrelated `XMemMapPhysicalPages` calls do not spuriously
/// defeat alias writes, while still preventing any AB-BA deadlock with
/// `title_memory_remapped -> lazy_readback::evict_range`.
pub fn write_va_via_alias(va: usize, data: &[u8; PAGE]) -> bool {
    if !va.is_multiple_of(PAGE) {
        return false;
    }
    let page_slice = {
        let mut deadline = None;
        let mut spins = 0u32;
        let physical = loop {
            match PHYSICAL.try_lock() {
                Ok(guard) => break guard,
                Err(std::sync::TryLockError::Poisoned(p)) => break p.into_inner(),
                Err(std::sync::TryLockError::WouldBlock) => {
                    if spins < 16 {
                        spins += 1;
                        std::hint::spin_loop();
                        continue;
                    }
                    let until = *deadline.get_or_insert_with(|| {
                        std::time::Instant::now() + std::time::Duration::from_millis(5)
                    });
                    if std::time::Instant::now() >= until {
                        return false;
                    }
                    std::thread::yield_now();
                }
            }
        };
        let Some((&base, &(size, ref slice, is_priv))) = physical.views.range(..=va).next_back()
        else {
            return false;
        };
        if is_priv || va < base || va.saturating_add(PAGE) > base.saturating_add(size) {
            return false;
        }
        Slice {
            section: slice.section.clone(),
            index: slice.index + (va - base) / PAGE,
        }
    };
    let Some(mut view) = View::map(&page_slice, 1) else {
        return false;
    };
    view.bytes()[..PAGE].copy_from_slice(data);
    true
}

/// The title addresses that show bytes `[offset, offset + len)` of `pages`
/// now, byte for byte, as runs (address, bytes) in address order: a write
/// to the pages through a view of our own ([`page_window`],
/// [`write_physical`]) lands there too, and nothing that watches those
/// addresses sees it (the GPU side's guards fault only on a write at the
/// guarded address itself). Empty if no view shows them. Byte for byte,
/// not by whole pages: the title packs several small textures into one
/// page and fills each with a request of its own.
pub(crate) fn shown_at(pages: &[usize], offset: usize, len: usize) -> Vec<(usize, usize)> {
    let Some(end) = offset.checked_add(len) else {
        return Vec::new();
    };
    let first = offset / PAGE;
    let Some(touched) = pages.get(first..end.div_ceil(PAGE)) else {
        return Vec::new();
    };
    let mut pieces: Vec<(usize, usize)> = Vec::new();
    {
        let physical = physical();
        for (i, page) in touched.iter().enumerate() {
            let Some(addresses) = physical.pages.get(page).and_then(|slice| physical.shown.get(&slice.key(0))) else {
                continue;
            };
            // The bytes of this page that the write covers.
            let lo = if i == 0 { offset % PAGE } else { 0 };
            let hi = (end - (first + i) * PAGE).min(PAGE);
            pieces.extend(addresses.iter().map(|&at| (at + lo, hi - lo)));
        }
    }
    pieces.sort_unstable();
    let mut runs: Vec<(usize, usize)> = Vec::new();
    for (at, bytes) in pieces {
        match runs.last_mut() {
            Some((start, n)) if *start + *n == at => *n += bytes,
            _ => runs.push((at, bytes)),
        }
    }
    runs
}

/// Bytes `[offset, offset + len)` of `pages` have been written through a
/// view of our own: the GPU side is told at every title address that
/// shows them (`victory_gpu::title_memory_written`; what it does with it
/// is `VICTORY_PAGE_WRITES`' to say, and it counts either way). Call when
/// the bytes are there, and without the page table's lock ([`shown_at`]
/// takes and leaves it).
pub(crate) fn pages_written(pages: &[usize], offset: usize, len: usize) {
    for (at, bytes) in shown_at(pages, offset, len) {
        victory_gpu::title_memory_written(at, bytes, true);
    }
}

/// Maps `n` pages at `address`, in place of what is mapped there; null
/// `pages` unmaps. Returns `address`, or null if a page or the host fails.
pub extern "system" fn XMemMapPhysicalPages(
    address: *mut c_void,
    n: usize,
    pages: *const usize,
) -> *mut c_void {
    let base = address as usize;
    let mut physical = physical();
    // Until the pages are mapped again: `unmap` takes whole views away and
    // maps what stays of them again, and a mapped view is a reservation
    // until it is committed. A read of title memory in place must not fall
    // into that gap (`victory_gpu::title_memory_remapping`).
    let _changing = victory_gpu::title_memory_remapping();
    unmap(&mut physical, base, n * PAGE);
    if pages.is_null() {
        return address;
    }
    // SAFETY: the title passes `n` page numbers.
    let pages = unsafe { std::slice::from_raw_parts(pages, n) };
    let mut i = 0;
    let _threads = map_threads();
    while i < n {
        let Some(first) = physical.pages.get(&pages[i]).cloned() else {
            if i > 0 {
                note_mapped_console_memory(base, i * PAGE);
            }
            tracing::warn!(page = pages[i], "XMemMapPhysicalPages: unknown page");
            return std::ptr::null_mut();
        };
        // One view per run of consecutive slices of one section.
        let run = 1 + pages[i + 1..]
            .iter()
            .zip(1..)
            .take_while(|&(page, k)| {
                physical.pages.get(page).is_some_and(|s| {
                    Arc::ptr_eq(&s.section, &first.section) && s.index == first.index + k
                })
            })
            .count();
        if !map(&mut physical, base + i * PAGE, run * PAGE, first) {
            if i > 0 {
                note_mapped_console_memory(base, i * PAGE);
            }
            return std::ptr::null_mut();
        }
        i += run;
    }
    note_mapped_console_memory(base, n * PAGE);
    address
}

/// One page at each address; null `pages`, or a page number 0, unmaps
/// (the title passes 0 for addresses to clear).
pub extern "system" fn XMemMapPhysicalPagesScatter(
    addresses: *const *mut c_void,
    n: usize,
    pages: *const usize,
) -> BOOL {
    if addresses.is_null() {
        return 0;
    }
    for i in 0..n {
        // SAFETY: the title passes `n` addresses and, unless null, `n` pages.
        let (address, page) = unsafe {
            (
                *addresses.add(i),
                if pages.is_null() || *pages.add(i) == 0 {
                    std::ptr::null()
                } else {
                    pages.add(i)
                },
            )
        };
        if XMemMapPhysicalPages(address, 1, page).is_null() {
            return 0;
        }
    }
    1
}

/// Maps the physical pages shown at `[from, from + len)` at `to` as well,
/// in place of what `to` shows; `from` 0 unmaps. The GPU's page-mapping
/// copies (`victory_gpu`, queue `CopyPageMappingsBatchX`):
/// the title maps pages into a pool view, then has the GPU map them into its tiled
/// heap. Page by page; false if a source page is not mapped physical memory.
pub fn map_alias(to: u64, from: u64, len: u64) -> bool {
    let (to, from, len) = (to as usize, from as usize, len as usize);
    let mut physical = physical();
    // As in `XMemMapPhysicalPages`: no read in place meets the gap.
    let _changing = victory_gpu::title_memory_remapping();
    unmap(&mut physical, to, len);
    for k in (0..len).step_by(PAGE) {
        if from == 0 {
            break;
        }
        let at = from + k;
        let Some(slice) = physical
            .views
            .range(..=at)
            .next_back()
            .filter(|(base, (size, _, _))| at < *base + size)
            .map(|(base, (_, s, _))| Slice {
                index: s.index + (at - base) / PAGE,
                section: s.section.clone(),
            })
        else {
            return false;
        };
        if !map(&mut physical, to + k, PAGE, slice) {
            return false;
        }
    }
    if from != 0 {
        note_mapped_console_memory(to, len);
    }
    true
}

/// Lazily promotes a single-view `MEM_WRITE_WATCH` private mapping containing
/// `addr` to a `SEC_RESERVE` file-mapping section view so a second virtual
/// alias can share the exact same physical pages with `VIEW_GUARD` coherence.
/// Caller MUST hold both `physical` and `victory_gpu::title_memory_remapping()`.
fn promote_view_containing(physical: &mut Physical, addr: usize) -> bool {
    let Some((view_base, view_len, slice, is_private)) = physical
        .views
        .range(..=addr)
        .next_back()
        .filter(|&(&base, &(size, _, _))| addr < base + size)
        .map(|(&base, &(size, ref slice, is_priv))| (base, size, slice.clone(), is_priv))
    else {
        return true;
    };
    if !is_private {
        return true;
    }
    // Preserve live CPU-written bytes before turning the private allocation
    // back into a placeholder.
    // SAFETY: `view_base..view_base+view_len` is our committed `PAGE_READWRITE`
    // private view and `title_memory_remapping` blocks concurrent GPU reads.
    let saved = unsafe { std::slice::from_raw_parts(view_base as *const u8, view_len) }.to_vec();
    victory_gpu::title_memory_remapped(view_base, view_len);
    // SAFETY: release our entire `MEM_REPLACE_PLACEHOLDER` private allocation
    // back to a placeholder of exact size `view_len`.
    let freed = unsafe {
        VirtualFree(
            view_base as *mut c_void,
            view_len,
            MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
        )
    };
    if freed == 0 {
        tracing::warn!(
            address = format_args!("{view_base:#x}"),
            len = view_len,
            error = %std::io::Error::last_os_error(),
            "promoting private physical view failed on VirtualFree(MEM_PRESERVE_PLACEHOLDER)"
        );
        return false;
    }
    // Helper for recovering `physical` metadata after `VirtualFree` already
    // turned `view_base..view_base+view_len` back into a placeholder: try to
    // restore the private mapping with `saved`, and if even that fails, evict
    // the torn entry from `views`/`shown` so we never leave `is_private = true`
    // pointing at an unmapped placeholder.
    let rollback_after_free = |physical: &mut Physical, saved: &[u8]| {
        physical.views.remove(&view_base);
        for k in 0..view_len.div_ceil(PAGE) {
            let key = slice.key(k);
            if let Some(addresses) = physical.shown.get_mut(&key) {
                addresses.retain(|&a| a != view_base + k * PAGE);
                if addresses.is_empty() {
                    physical.shown.remove(&key);
                }
            }
        }
        if map_private(physical, view_base, view_len, slice.clone()) {
            // SAFETY: freshly restored committed `PAGE_READWRITE` private view.
            unsafe {
                std::ptr::copy_nonoverlapping(saved.as_ptr(), view_base as *mut u8, view_len);
            }
        }
    };
    // SAFETY: map the `SEC_RESERVE` section into the restored placeholder.
    let view = unsafe {
        MapViewOfFile3(
            slice.section.handle,
            GetCurrentProcess(),
            view_base as *const c_void,
            (slice.index * PAGE) as u64,
            view_len,
            MEM_REPLACE_PLACEHOLDER,
            PAGE_READWRITE,
            std::ptr::null_mut(),
            0,
        )
    };
    if view.Value as usize != view_base {
        tracing::warn!(
            address = format_args!("{view_base:#x}"),
            len = view_len,
            error = %std::io::Error::last_os_error(),
            "promoting private physical view failed on MapViewOfFile3"
        );
        rollback_after_free(physical, &saved);
        return false;
    }
    if !slice.section.commit_shown(slice.index, view_base, view_len) {
        // SAFETY: unmap view back to placeholder on commit failure.
        unsafe { UnmapViewOfFile2(GetCurrentProcess(), view, MEM_PRESERVE_PLACEHOLDER) };
        rollback_after_free(physical, &saved);
        return false;
    }
    // SAFETY: freshly mapped & committed `PAGE_READWRITE` section view.
    unsafe {
        std::ptr::copy_nonoverlapping(saved.as_ptr(), view_base as *mut u8, view_len);
    }
    victory_gpu::title_memory_remapped(view_base, view_len);
    victory_gpu::title_memory_view(
        view_base,
        view_len,
        slice.section.handle as usize,
        (slice.index * PAGE) as u64,
    );
    for k in 0..view_len.div_ceil(PAGE) {
        physical.backed.insert(slice.key(k));
    }
    physical.views.insert(view_base, (view_len, slice, false));
    true
}

/// Maps `[base, base + len)` as a single-view `MEM_WRITE_WATCH` private
/// allocation in place of a placeholder (zero `PAGE_READONLY` guards / SEH
/// faults while only one virtual address shows these physical pages).
fn map_private(physical: &mut Physical, base: usize, len: usize, slice: Slice) -> bool {
    map_private_ex(physical, base, len, slice, true)
}

fn map_private_ex(
    physical: &mut Physical,
    base: usize,
    len: usize,
    slice: Slice,
    notify_remap: bool,
) -> bool {
    // SAFETY: splits placeholder (failing harmlessly if exact size) and
    // replaces it with committed `MEM_WRITE_WATCH` private memory.
    let ptr = unsafe {
        VirtualFree(
            base as *mut c_void,
            len,
            MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
        );
        VirtualAlloc2(
            GetCurrentProcess(),
            base as *const c_void,
            len,
            MEM_RESERVE | MEM_COMMIT | MEM_REPLACE_PLACEHOLDER | MEM_WRITE_WATCH,
            PAGE_READWRITE,
            std::ptr::null_mut(),
            0,
        )
    };
    if ptr as usize != base {
        return false;
    }
    if notify_remap {
        victory_gpu::title_memory_remapped(base, len);
        victory_gpu::sparse_uma::on_map(base, len);
    }
    for k in 0..len.div_ceil(PAGE) {
        physical.shown.entry(slice.key(k)).or_default().push(base + k * PAGE);
    }
    physical.views.insert(base, (len, slice, true));
    true
}

/// Maps `slice` and the slices after it over `[base, base + len)`, which
/// must lie in one placeholder. Uses `MEM_WRITE_WATCH` private backing when
/// `VICTORY_SINGLE_VIEW_WATCH` is enabled and every page is single-view and
/// unbacked in the section; otherwise lazily promotes any existing private
/// alias and maps the `SEC_RESERVE` section view.
fn map(physical: &mut Physical, base: usize, len: usize, slice: Slice) -> bool {
    let n_pages = len.div_ceil(PAGE);
    let can_private = single_view_watch()
        && (0..n_pages).all(|k| {
            let key = slice.key(k);
            !physical.shown.contains_key(&key) && !physical.backed.contains(&key)
        });
    if can_private && map_private(physical, base, len, slice.clone()) {
        return true;
    }
    // Lazily promote any existing single-view private alias of these pages
    // before mapping the second view from the shared `SEC_RESERVE` section.
    let aliases: Vec<usize> = (0..n_pages)
        .filter_map(|k| physical.shown.get(&slice.key(k)))
        .flatten()
        .copied()
        .collect();
    for alias_va in aliases {
        if !promote_view_containing(physical, alias_va) {
            return false;
        }
    }
    map_section(physical, base, len, slice)
}

/// Maps `slice` and the slices after it from their `SEC_RESERVE` section over
/// `[base, base + len)`, which must lie in one placeholder.
fn map_section(physical: &mut Physical, base: usize, len: usize, slice: Slice) -> bool {
    map_section_ex(physical, base, len, slice, true)
}

/// Like [`map_section`], emitting `title_memory_remapped` + `sparse_uma::on_map`
/// only when `notify_remap` is true (skipped when restoring unchanged surviving
/// head/tail slices of a split view inside [`unmap`]).
fn map_section_ex(
    physical: &mut Physical,
    base: usize,
    len: usize,
    slice: Slice,
    notify_remap: bool,
) -> bool {
    // SAFETY: splits the placeholder around the range (failing harmlessly
    // when the range is the whole placeholder), then maps the section
    // there; both act only on placeholders.
    let view = unsafe {
        VirtualFree(
            base as *mut c_void,
            len,
            MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
        );
        MapViewOfFile3(
            slice.section.handle,
            GetCurrentProcess(),
            base as *const c_void,
            (slice.index * PAGE) as u64,
            len,
            MEM_REPLACE_PLACEHOLDER,
            PAGE_READWRITE,
            std::ptr::null_mut(),
            0,
        )
    };
    if view.Value as usize != base {
        tracing::warn!(
            address = format_args!("{base:#x}"),
            len,
            error = %std::io::Error::last_os_error(),
            "mapping physical pages failed (not a placeholder?)"
        );
        return false;
    }
    if !slice.section.commit_shown(slice.index, base, len) {
        // SAFETY: the view just mapped, back to a placeholder.
        unsafe { UnmapViewOfFile2(GetCurrentProcess(), view, MEM_PRESERVE_PLACEHOLDER) };
        return false;
    }
    if notify_remap {
        // New contents, read-write: the GPU side's copies no longer guard it.
        victory_gpu::title_memory_remapped(base, len);
        victory_gpu::sparse_uma::on_map(base, len);
    }
    // What the range shows now, for the GPU side's in-place census.
    static ALIAS_HOOK: std::sync::Once = std::sync::Once::new();
    ALIAS_HOOK.call_once(|| victory_gpu::use_alias_writer(write_va_via_alias));
    victory_gpu::title_memory_view(base, len, slice.section.handle as usize, (slice.index * PAGE) as u64);
    for k in 0..len.div_ceil(PAGE) {
        let key = slice.key(k);
        physical.shown.entry(key).or_default().push(base + k * PAGE);
        physical.backed.insert(key);
    }
    physical.views.insert(base, (len, slice, false));
    true
}

/// Commits the section pages a view shows (once per page, whichever view).
fn commit(base: usize, len: usize) -> bool {
    if on_wine() {
        return true; // sections are committed whole there
    }
    // SAFETY: `[base, base + len)` is a view of a reserved section.
    if !unsafe { VirtualAlloc(base as *const c_void, len, MEM_COMMIT, PAGE_READWRITE) }.is_null() {
        return true;
    }
    let error = std::io::Error::last_os_error();
    if error.raw_os_error() == Some(ERROR_COMMITMENT_LIMIT as i32) {
        let mut info = PERFORMANCE_INFORMATION {
            cb: size_of::<PERFORMANCE_INFORMATION>() as u32,
            ..Default::default()
        };
        // SAFETY: the buffer and size describe a writable PERFORMANCE_INFORMATION.
        let queried = unsafe { K32GetPerformanceInfo(&mut info, info.cb) } != 0;
        victory_loader::log_crash(&commit_limit_message(len, &error, queried.then_some(&info)));
    } else {
        tracing::warn!(len, %error, "committing physical pages failed");
    }
    false
}

fn commit_limit_message(
    len: usize,
    error: &std::io::Error,
    info: Option<&PERFORMANCE_INFORMATION>,
) -> String {
    let mut message = format!(
        "VIctory: Windows commit limit reached while committing {len} bytes of physical pages: {error}.\n"
    );
    if let Some(info) = info {
        use std::fmt::Write;

        let committed = info.CommitTotal * info.PageSize / MIB;
        let limit = info.CommitLimit * info.PageSize / MIB;
        let _ = writeln!(
            message,
            "System commit: {committed} MiB of {limit} MiB in use."
        );
    }
    message.push_str(
        "Close other applications or increase the Windows paging file size, then retry.\n",
    );
    message
}

#[cfg(test)]
thread_local! {
    static UNMAP_REMAP_LOG: std::cell::RefCell<Option<Vec<(usize, usize)>>> =
        const { std::cell::RefCell::new(None) };
}

#[inline]
fn notify_unmap_remapped(at: usize, len: usize) {
    if len == 0 {
        return;
    }
    #[cfg(test)]
    UNMAP_REMAP_LOG.with(|log| {
        if let Some(log) = log.borrow_mut().as_mut() {
            log.push((at, len));
        }
    });
    victory_gpu::title_memory_remapped(at, len);
}

/// Returns `[base, base + len)` to placeholders; views reaching past it
/// are mapped again outside it.
fn unmap(physical: &mut Physical, base: usize, len: usize) {
    // No bytes: no view to return. The search below would still find the
    // view holding `base` and cut it in two there, and its range of views
    // starting inside would run backwards (which `BTreeMap` panics on).
    if len == 0 {
        return;
    }
    let end = base + len;
    // Views do not overlap: the one holding `base`, then those starting
    // inside (the GPU's page mappings make one view per page).
    let first = physical
        .views
        .range(..=base)
        .next_back()
        .filter(|(at, (size, _, _))| *at + size > base)
        .map(|(at, _)| *at);
    let hit: Vec<usize> = first
        .into_iter()
        .chain(physical.views.range(base + 1..end).map(|(at, _)| *at))
        .collect();
    for at in hit {
        let Some((size, slice, is_private)) = physical.views.remove(&at) else {
            continue;
        };
        let unmap_start = base.max(at);
        let unmap_end = end.min(at + size);
        if unmap_start < unmap_end {
            victory_gpu::clear_console_memory(unmap_start, unmap_end - unmap_start);
            victory_gpu::sparse_uma::on_unmap(unmap_start, unmap_end - unmap_start);
        }
        for k in 0..size.div_ceil(PAGE) {
            let key = slice.key(k);
            if let Some(addresses) = physical.shown.get_mut(&key) {
                addresses.retain(|&a| a != at + k * PAGE);
                if addresses.is_empty() {
                    physical.shown.remove(&key);
                }
            }
        }
        if is_private {
            // Save head and tail that remain mapped so they stay single-view
            // `MEM_WRITE_WATCH` private allocations without touching the section.
            let head_saved = (at < base).then(|| {
                // SAFETY: committed `PAGE_READWRITE` private view.
                unsafe { std::slice::from_raw_parts(at as *const u8, base - at) }.to_vec()
            });
            let tail_saved = (at + size > end).then(|| {
                // SAFETY: committed `PAGE_READWRITE` private view.
                unsafe { std::slice::from_raw_parts(end as *const u8, at + size - end) }.to_vec()
            });
            // Preserve any non-zero unmapped middle pages into their backing
            // `SEC_RESERVE` section so a later remap sees their exact bytes.
            let mid_start = base.max(at);
            let mid_end = end.min(at + size);
            for va in (mid_start..mid_end).step_by(PAGE) {
                // SAFETY: committed `PAGE_READWRITE` private page.
                let page_bytes = unsafe { std::slice::from_raw_parts(va as *const u8, PAGE) };
                if page_bytes.iter().any(|&b| b != 0) {
                    let k = (va - at) / PAGE;
                    let page_slice = Slice {
                        index: slice.index + k,
                        section: slice.section.clone(),
                    };
                    if let Some(mut v) = View::map(&page_slice, 1) {
                        v.bytes().copy_from_slice(page_bytes);
                        physical.backed.insert(slice.key(k));
                    }
                }
            }
            if unmap_start < unmap_end {
                notify_unmap_remapped(unmap_start, unmap_end - unmap_start);
            }
            // SAFETY: release our entire `MEM_REPLACE_PLACEHOLDER` private
            // allocation back into a placeholder of exact size `size`.
            let freed = unsafe {
                VirtualFree(
                    at as *mut c_void,
                    size,
                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
                )
            };
            if freed == 0 {
                tracing::warn!(
                    address = format_args!("{at:#x}"),
                    len = size,
                    error = %std::io::Error::last_os_error(),
                    "unmapping private physical view failed on VirtualFree(MEM_PRESERVE_PLACEHOLDER)"
                );
                // The private allocation is still live at `[at, at + size)`:
                // restore tracking so `views`/`shown` stay consistent.
                for k in 0..size.div_ceil(PAGE) {
                    physical
                        .shown
                        .entry(slice.key(k))
                        .or_default()
                        .push(at + k * PAGE);
                }
                physical.views.insert(at, (size, slice, true));
                continue;
            }
            if let Some(bytes) = head_saved {
                let len = base - at;
                if map_private_ex(physical, at, len, slice.clone(), false)
                    || map_section_ex(physical, at, len, slice.clone(), false)
                {
                    // SAFETY: newly mapped `PAGE_READWRITE` head.
                    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), at as *mut u8, len) };
                }
            }
            if let Some(bytes) = tail_saved {
                let len = at + size - end;
                let index = slice.index + (end - at) / PAGE;
                let tail_slice = Slice { index, ..slice };
                if map_private_ex(physical, end, len, tail_slice.clone(), false)
                    || map_section_ex(physical, end, len, tail_slice, false)
                {
                    // SAFETY: newly mapped `PAGE_READWRITE` tail.
                    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), end as *mut u8, len) };
                }
            }
            continue;
        }
        if unmap_start < unmap_end {
            notify_unmap_remapped(unmap_start, unmap_end - unmap_start);
        }
        // SAFETY: a view we mapped; it becomes a placeholder again.
        unsafe {
            UnmapViewOfFile2(
                GetCurrentProcess(),
                MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: at as *mut c_void,
                },
                MEM_PRESERVE_PLACEHOLDER,
            )
        };
        if at < base {
            map_section_ex(physical, at, base - at, slice.clone(), false);
        }
        if at + size > end {
            let index = slice.index + (end - at) / PAGE;
            map_section_ex(physical, end, at + size - end, Slice { index, ..slice }, false);
        }
    }
}

/// The oldest layout we know; newer titles pass other flag values (Minecraft
/// 0x20000001) and read the same leading fields.
#[repr(C)]
#[derive(Debug, Default)]
pub struct WorkingSet {
    pub title_limit: usize,
    pub title_used: usize,
    pub tools_limit: usize,
    pub tools_used: usize,
    pub stack_used: u32,
    pub physical_pages: u32,
    pub gpu_optimal_limit: usize,
    pub gpu_optimal_used: usize,
    pub tools_gpu_optimal_limit: usize,
    pub tools_gpu_optimal_used: usize,
    pub kernel_pool_used: u32,
    pub kernel_pool_limit: u32,
}

/// The title's budget and actual tracked title memory usage (allocated
/// physical pages, committed `XMemVirtualAlloc` regions, and `XMemAllocDefault`
/// heap blocks — excluding VIctory's own host/D3D12 commit). No tools memory:
/// a retail console.
pub extern "system" fn XMemGetWorkingSetStatistics(_which: u32, out: *mut WorkingSet) -> BOOL {
    if out.is_null() {
        return 0;
    }
    let (tracked, phys_pages) = tracked_title_memory();
    let (gpu_optimal, standard) = victory_gpu::console::current().title_memory();
    let stats = WorkingSet {
        title_limit: gpu_optimal + standard - TRANSFERRED.load(Ordering::Relaxed),
        title_used: tracked.max(PAGE),
        physical_pages: phys_pages.min(u32::MAX as usize) as u32,
        gpu_optimal_limit: gpu_optimal,
        ..Default::default()
    };
    static ASKED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let asked = ASKED.fetch_add(1, Ordering::Relaxed);
    if asked.is_multiple_of(1024) {
        tracing::info!(
            asked = asked + 1,
            which = _which,
            limit_mb = stats.title_limit >> 20,
            used_mb = stats.title_used >> 20,
            phys_pages,
            "the title asks how much memory it uses"
        );
    }
    // SAFETY: checked non-null; the title passes room for the structure.
    unsafe { out.write(stats) };
    1
}

/// Hands title memory (standard, in 2 MiB pages) to the system. The other
/// transfers involve tools memory, which a retail console has none of.
pub extern "system" fn XMemTransferMemory(kind: u32, size: usize) -> BOOL {
    const TITLE_TO_SYSTEM: u32 = 0;
    tracing::info!(kind, size, "XMemTransferMemory");
    if size == 0 {
        return 1;
    }
    let Some(size) = size.checked_next_multiple_of(2 * MIB) else {
        return 0;
    };
    let standard = victory_gpu::console::current().title_memory().1;
    let moved = kind == TITLE_TO_SYSTEM
        && TRANSFERRED
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |t| {
                t.checked_add(size).filter(|&t| t <= standard)
            })
            .is_ok();
    moved.into()
}

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "xmem",
        dlls: &["xmem.dll"],
        exports: vec![
            hle_export!(XMemAlloc),
            hle_export!(XMemFree),
            hle_export!(XMemAllocDefault),
            hle_export!(XMemFreeDefault),
            hle_export!(XMemIsToolsMemAvailable),
            hle_export!(XMemAllocatePhysicalPages),
            hle_export!(XMemFreePhysicalPages),
            hle_export!(XMemMapPhysicalPages),
            hle_export!(XMemMapPhysicalPagesScatter),
            hle_export!(XMemVirtualAlloc),
            hle_export!(XMemVirtualQuery),
            hle_export!(XMemGetWorkingSetStatistics),
            hle_export!(XMemTransferMemory),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn query(address: *const c_void) -> BasicInformation {
        // SAFETY: plain data; zero is a valid value.
        let mut info: BasicInformation = unsafe { std::mem::zeroed() };
        // SAFETY: a pseudo handle needs no closing.
        let process = unsafe { GetCurrentProcess() };
        assert_eq!(XMemVirtualQuery(process, address, &mut info), 1);
        info
    }

    #[test]
    fn alloc_honours_alignment_and_frees() {
        for (field, align) in [(0, 16), (4, 16), (0xc, 4096), (0x10, 64 * 1024)] {
            let attributes = field << ALIGNMENT_SHIFT | 1 << 29; // a graphics type
            let p = XMemAlloc(100, attributes);
            assert!(!p.is_null());
            assert_eq!(p as usize % align, 0, "field {field:#x}");
            // SAFETY: 100 bytes were allocated.
            unsafe { p.cast::<u8>().write_bytes(0xab, 100) };
            XMemFree(p, attributes);
        }
        let p = XMemAlloc(4096, ALLOC_32BIT);
        assert!(!p.is_null() && (p as usize) < 1 << 32);
        XMemFree(p, ALLOC_32BIT);
    }

    #[test]
    fn virtual_alloc_meets_console_constraints() {
        const GPU_READ_WRITE: u32 = 0x2000;
        let p = XMemVirtualAlloc(
            std::ptr::null_mut(),
            4 * MIB,
            PAGES_2M | MEM_RESERVE | MEM_COMMIT,
            GRAPHICS | 7 << 12, // 40-bit addresses
            PAGE_READWRITE | GPU_READ_WRITE,
        );
        assert!(!p.is_null());
        assert_eq!(p as usize % (2 * MIB), 0);
        assert!((p as usize) < 1 << 40);
        let info = query(p);
        assert_eq!(info.host.State, MEM_COMMIT);
        assert_eq!(info.host.Protect, PAGE_READWRITE);
        assert_eq!(info.flags & GRAPHICS, GRAPHICS);
        // Write-watched, for the GPU side's copies.
        let (mut found, mut count, mut granularity) = ([0usize; 4], 4, 0);
        // SAFETY: room for `count` addresses.
        let watched = unsafe {
            windows_sys::Win32::System::Memory::GetWriteWatch(
                0,
                p,
                4 * MIB,
                found.as_mut_ptr().cast(),
                &mut count,
                &mut granularity,
            )
        };
        assert_eq!(watched, 0);
        // SAFETY: our own reservation.
        assert_ne!(unsafe { VirtualFree(p, 0, MEM_RELEASE) }, 0);
        assert_eq!(query(p).flags, 0);
        // No address field: above the 32-bit range and below 2^40, where
        // the console keeps it.
        let p = XMemVirtualAlloc(
            std::ptr::null_mut(),
            16 << 30,
            MEM_RESERVE,
            GRAPHICS,
            PAGE_NOACCESS,
        );
        assert!(!p.is_null() && (p as usize) >= 1 << 32, "{p:p}");
        assert!((p as usize) + (16 << 30) <= 1 << 40, "{p:p}");
        // SAFETY: our own reservation.
        assert_ne!(unsafe { VirtualFree(p, 0, MEM_RELEASE) }, 0);
    }

    #[test]
    fn transfers_shrink_the_title_budget() {
        let mut before = WorkingSet::default();
        assert_eq!(XMemGetWorkingSetStatistics(0x2000_0001, &mut before), 1);
        assert!(before.title_used > 0);
        assert_eq!(XMemTransferMemory(0, 128 * MIB - 1), 1);
        assert_eq!(XMemTransferMemory(2, MIB), 0, "no tools memory");
        let standard = victory_gpu::console::current().title_memory().1;
        assert_eq!(XMemTransferMemory(0, standard), 0, "over budget");
        let mut after = WorkingSet::default();
        XMemGetWorkingSetStatistics(0x2000_0001, &mut after);
        assert_eq!(before.title_limit - after.title_limit, 128 * MIB);
    }

    #[test]
    fn statistics_have_the_console_layout() {
        assert_eq!(size_of::<WorkingSet>(), 0x50);
        assert_eq!(size_of::<BasicInformation>(), 0x40);
        assert_eq!(std::mem::offset_of!(WorkingSet, gpu_optimal_limit), 0x28);
    }

    #[test]
    fn commit_limit_report_includes_request_and_system_commit() {
        let error = std::io::Error::from_raw_os_error(ERROR_COMMITMENT_LIMIT as i32);
        let info = PERFORMANCE_INFORMATION {
            CommitTotal: 512,
            CommitLimit: 768,
            PageSize: 4096,
            ..Default::default()
        };
        let report = commit_limit_message(4 * MIB, &error, Some(&info));
        assert!(report.contains("4194304 bytes"));
        assert!(report.contains("2 MiB of 3 MiB in use"));
        let fallback = commit_limit_message(4 * MIB, &error, None);
        for report in [&report, &fallback] {
            assert!(report.contains("os error 1455"));
            assert!(report.contains("Windows paging file size"));
        }
        assert!(!fallback.contains("System commit:"));
    }

    #[test]
    fn physical_pages_map_alias_and_unmap() {
        let mut pages = [0usize; 3];
        let mut n = 3;
        assert_eq!(XMemAllocatePhysicalPages(0, &mut n, pages.as_mut_ptr()), 1);
        assert_eq!(n, 3);
        // Packed into 21 bits as `number >> 4`, as the title keeps them.
        for p in pages {
            assert_eq!((p >> 4 & 0x1f_ffff) << 4, p);
        }
        // Reservations of 64 KiB pages, as titles make them for mapping.
        let reserve = || {
            XMemVirtualAlloc(
                std::ptr::null_mut(),
                3 * PAGE,
                MEM_RESERVE | PAGES_64K,
                0,
                PAGE_READWRITE,
            )
        };
        let (a, b) = (reserve(), reserve());
        assert!(!a.is_null() && !b.is_null());
        let at = |p: *mut c_void, page: usize| p.wrapping_byte_add(page * PAGE);
        assert_eq!(XMemMapPhysicalPages(a, 3, pages.as_ptr()), a);

        // The same page at a second address shows the same bytes.
        assert_eq!(
            XMemMapPhysicalPages(at(b, 1), 1, pages[2..].as_ptr()),
            at(b, 1)
        );
        // SAFETY: both pages are mapped read-write.
        unsafe {
            at(a, 2).cast::<u8>().write(7);
            assert_eq!(at(b, 1).cast::<u8>().read(), 7);
            a.cast::<u8>().write(1);
        }
        // Writes into pages (DirectStorage's) show through their views.
        assert!(write_physical(&pages[2..], 1, &[9]));
        // SAFETY: mapped read-write.
        assert_eq!(unsafe { at(b, 1).cast::<u8>().add(1).read() }, 9);

        // Unmapping the middle page keeps its neighbours.
        assert_eq!(
            XMemMapPhysicalPages(at(a, 1), 1, std::ptr::null()),
            at(a, 1)
        );
        assert_eq!(query(at(a, 1)).host.State, MEM_RESERVE);
        assert_eq!(query(a).host.State, MEM_COMMIT);
        assert_eq!(query(at(a, 2)).host.State, MEM_COMMIT);
        // SAFETY: still mapped.
        assert_eq!(unsafe { at(a, 2).cast::<u8>().read() }, 7);

        // A scattered page lands in the hole.
        let addresses = [at(a, 1)];
        assert_eq!(
            XMemMapPhysicalPagesScatter(addresses.as_ptr(), 1, pages.as_ptr()),
            1
        );
        // SAFETY: page 0 is now mapped at both `a` and `a + PAGE`.
        assert_eq!(unsafe { at(a, 1).cast::<u8>().read() }, 1);
        // Page number 0 unmaps.
        assert_eq!(
            XMemMapPhysicalPagesScatter(addresses.as_ptr(), 1, [0usize].as_ptr()),
            1
        );
        assert_eq!(query(at(a, 1)).host.State, MEM_RESERVE);
        // The GPU's page-mapping copy: what `a + 2 pages` shows, at `b`.
        assert!(map_alias(b as u64, at(a, 2) as u64, PAGE as u64));
        // SAFETY: mapped read-write.
        assert_eq!(unsafe { b.cast::<u8>().read() }, 7);
        assert!(map_alias(b as u64, 0, PAGE as u64));
        assert_eq!(query(b).host.State, MEM_RESERVE);
        assert!(!map_alias(b as u64, at(a, 1) as u64, PAGE as u64));
        assert_eq!(XMemFreePhysicalPages(3, pages.as_ptr()), 1);
        assert_eq!(XMemIsToolsMemAvailable(), 0);
    }

    /// A thread that reads a mapped page in place, as the GPU side reads a
    /// texture, never meets the moment in which the runtime has taken the
    /// page's view away to map it again: mapping the same pages at the same
    /// address unmaps the view and maps a new one, and without the lock the
    /// reader faults on a page "reserved, never committed" (the crashes of
    /// runs 71 and 74).
    #[test]
    fn a_read_in_place_never_meets_the_gap_of_a_remapping() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let mut pages = [0usize; 3];
        let mut n = 3;
        assert_eq!(XMemAllocatePhysicalPages(0, &mut n, pages.as_mut_ptr()), 1);
        let a = XMemVirtualAlloc(
            std::ptr::null_mut(),
            3 * PAGE,
            MEM_RESERVE | PAGES_64K,
            0,
            PAGE_READWRITE,
        );
        assert!(!a.is_null());
        assert_eq!(XMemMapPhysicalPages(a, 3, pages.as_ptr()), a);
        // SAFETY: mapped read-write.
        unsafe { a.cast::<u8>().write(1) };
        let first = a as usize;
        let stop = std::sync::Arc::new(AtomicBool::new(false));
        let reader = {
            let stop = stop.clone();
            std::thread::spawn(move || {
                let mut reads = 0u64;
                while !stop.load(Ordering::Relaxed) {
                    let _reading = victory_gpu::title_memory_reading();
                    // SAFETY: the page is mapped whenever no change of
                    // mapping is under way, and none is while `_reading`
                    // is held.
                    assert_eq!(unsafe { (first as *const u8).read_volatile() }, 1);
                    reads += 1;
                }
                reads
            })
        };
        for _ in 0..2000 {
            // The same pages at the same address: the view goes and comes
            // back, with the same bytes.
            assert_eq!(XMemMapPhysicalPages(a, 3, pages.as_ptr()), a);
        }
        stop.store(true, Ordering::Relaxed);
        assert!(reader.join().expect("the reader never faulted") > 0);
        assert_eq!(XMemMapPhysicalPages(a, 3, std::ptr::null()), a);
        assert_eq!(XMemFreePhysicalPages(3, pages.as_ptr()), 1);
    }

    #[test]
    fn mapping_no_pages_changes_nothing() {
        let pages = allocate(2);
        let a = XMemVirtualAlloc(
            std::ptr::null_mut(),
            2 * PAGE,
            MEM_RESERVE | PAGES_64K,
            0,
            PAGE_READWRITE,
        );
        assert!(!a.is_null());
        assert_eq!(XMemMapPhysicalPages(a, 2, pages.as_ptr()), a);
        // SAFETY: both pages are mapped read-write.
        unsafe { a.cast::<u8>().write(5) };
        let second = a.wrapping_byte_add(PAGE);

        // A count of 0 at the start of a view, inside it and past it: with
        // pages, without (an unmap), and as the GPU's page-mapping copy.
        for at in [a, second, a.wrapping_byte_add(2 * PAGE)] {
            assert_eq!(XMemMapPhysicalPages(at, 0, pages.as_ptr()), at);
            assert_eq!(XMemMapPhysicalPages(at, 0, std::ptr::null()), at);
            assert!(map_alias(at as u64, a as u64, 0));
            assert!(map_alias(at as u64, 0, 0));
        }

        // The view is still the one view it was.
        let view = physical().views.get(&(a as usize)).map(|v| v.0);
        assert_eq!(view, Some(2 * PAGE));
        assert_eq!(query(a).host.State, MEM_COMMIT);
        assert_eq!(query(second).host.State, MEM_COMMIT);
        // SAFETY: still mapped.
        assert_eq!(unsafe { a.cast::<u8>().read() }, 5);

        // A count of 1 still unmaps that page alone.
        assert_eq!(XMemMapPhysicalPages(second, 1, std::ptr::null()), second);
        assert_eq!(query(second).host.State, MEM_RESERVE);
        assert_eq!(query(a).host.State, MEM_COMMIT);
        assert_eq!(XMemMapPhysicalPages(a, 1, std::ptr::null()), a);
        assert_eq!(XMemFreePhysicalPages(2, pages.as_ptr()), 1);
    }

    /// `n` pages of one new allocation.
    fn allocate(n: usize) -> Vec<usize> {
        let (mut pages, mut given) = (vec![0; n], n);
        assert_eq!(
            XMemAllocatePhysicalPages(0, &mut given, pages.as_mut_ptr()),
            1
        );
        assert_eq!(given, n);
        pages
    }

    /// The bytes of one page, through a view of our own.
    fn page_bytes(page: usize) -> Vec<u8> {
        let slice = physical().pages.get(&page).cloned().expect("known page");
        View::map(&slice, 1).expect("mapped").bytes().to_vec()
    }

    #[test]
    fn runs_group_consecutive_pages_of_one_allocation() {
        let (a, b) = (allocate(3), allocate(1));
        let lengths = |pages: &[usize], offset, len| {
            runs(pages, offset, len).map(|r| r.iter().map(|&(_, n)| n).collect::<Vec<_>>())
        };
        assert_eq!(lengths(&a, 0, 3 * PAGE), Some(vec![3]));
        // Only the pages the bytes reach.
        assert_eq!(lengths(&a, PAGE + 5, 10), Some(vec![1]));
        assert_eq!(lengths(&a, PAGE - 1, 2), Some(vec![2]));
        assert_eq!(lengths(&a, 0, 0), Some(vec![]));
        // Out of order, or across allocations: one run each.
        assert_eq!(
            lengths(&[a[0], a[2], a[1]], 0, 3 * PAGE),
            Some(vec![1, 1, 1])
        );
        assert_eq!(lengths(&[a[1], a[2], b[0]], 0, 3 * PAGE), Some(vec![2, 1]));
        // A slice that follows on in another section starts a run.
        assert_eq!(lengths(&[b[0], a[1]], 0, 2 * PAGE), Some(vec![1, 1]));
        // Too few pages for the bytes, or an unknown one.
        assert_eq!(lengths(&a, 0, 3 * PAGE + 1), None);
        assert_eq!(lengths(&[a[0], 0x10], 0, 2 * PAGE), None);
        assert_eq!(lengths(&a, usize::MAX, 2), None);
        assert_eq!(XMemFreePhysicalPages(3, a.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(1, b.as_ptr()), 1);
    }

    #[test]
    fn writes_cross_runs_and_allocations() {
        let (a, b) = (allocate(2), allocate(1));
        let pages = [a[0], a[1], b[0]];
        let data: Vec<u8> = (0..2 * PAGE + 64).map(|i| (i * 7 % 251) as u8).collect();
        let offset = PAGE - 32;
        // Past the third page: the second run is too short.
        assert!(!write_physical(&pages, offset, &data));
        assert!(write_physical(&pages, offset, &data[..2 * PAGE]));
        let seen: Vec<u8> = pages.iter().flat_map(|&p| page_bytes(p)).collect();
        assert!(seen[..offset].iter().all(|&b| b == 0), "before the offset");
        assert_eq!(seen[offset..offset + 2 * PAGE], data[..2 * PAGE]);
        assert!(seen[offset + 2 * PAGE..].iter().all(|&b| b == 0), "after");
        assert!(
            !write_physical(&[a[0], 0x10], 0, &data[..PAGE + 1]),
            "unknown page"
        );
        assert_eq!(XMemFreePhysicalPages(2, a.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(1, b.as_ptr()), 1);
    }

    /// Pages say where the title has them mapped, byte for byte (after
    /// fork PR #571's test; its runs are whole pages).
    #[test]
    fn pages_say_where_they_are_mapped() {
        let pages = allocate(3);
        let reserve = |n: usize| {
            let at = XMemVirtualAlloc(std::ptr::null_mut(), n * PAGE, MEM_RESERVE | PAGES_64K, 0, PAGE_READWRITE);
            assert!(!at.is_null());
            at
        };
        // `b` keeps a page unmapped at either end, so that no run at `b`
        // ever touches one at `a`, wherever the host puts the two.
        let (a, b) = (reserve(3), reserve(4));
        assert!(shown_at(&pages, 0, 3 * PAGE).is_empty(), "mapped nowhere");
        // All three at `a`; the last two at `b`, from its second page on.
        assert_eq!(XMemMapPhysicalPages(a, 3, pages.as_ptr()), a);
        let second = b.wrapping_byte_add(PAGE);
        assert_eq!(XMemMapPhysicalPages(second, 2, pages[1..].as_ptr()), second);
        let (a, second) = (a as usize, second as usize);
        let sorted = |mut runs: Vec<(usize, usize)>| {
            runs.sort_unstable();
            runs
        };
        assert_eq!(shown_at(&pages, 0, PAGE), vec![(a, PAGE)]);
        // A request for part of a page names those bytes alone.
        assert_eq!(shown_at(&pages, 0x4000, 0x3000), vec![(a + 0x4000, 0x3000)]);
        // Across two pages, shown at two addresses each: one run at each.
        assert_eq!(
            shown_at(&pages, PAGE + 1, PAGE),
            sorted(vec![(a + PAGE + 1, PAGE), (second + 1, PAGE)])
        );
        // A page unmapped in the middle of a view: the rest still shows.
        let middle = (a + PAGE) as *mut c_void;
        assert_eq!(XMemMapPhysicalPages(middle, 1, std::ptr::null()), middle);
        assert_eq!(
            shown_at(&pages, 0, 3 * PAGE),
            sorted(vec![(a, PAGE), (a + 2 * PAGE, PAGE), (second, 2 * PAGE)])
        );
        // Too few pages for the bytes: nothing (the write fails too).
        assert!(shown_at(&pages, 0, 3 * PAGE + 1).is_empty());
        for (at, n) in [(a, 3), (second, 2)] {
            let at = at as *mut c_void;
            assert_eq!(XMemMapPhysicalPages(at, n, std::ptr::null()), at);
        }
        assert!(shown_at(&pages, 0, 3 * PAGE).is_empty());
        assert_eq!(XMemFreePhysicalPages(3, pages.as_ptr()), 1);
    }

    #[test]
    fn page_windows_cover_one_run_in_place() {
        let (a, b) = (allocate(3), allocate(1));
        let mut window = page_window(&a, PAGE + 3, PAGE).expect("one run");
        assert_eq!(window.bytes().len(), PAGE);
        window.bytes().fill(0x5a);
        drop(window);
        let (second, third) = (page_bytes(a[1]), page_bytes(a[2]));
        assert!(second[..3].iter().all(|&b| b == 0));
        assert!(second[3..].iter().all(|&b| b == 0x5a));
        assert!(third[..3].iter().all(|&b| b == 0x5a));
        assert!(third[3..].iter().all(|&b| b == 0));
        // Several runs, or an unknown page: no window.
        assert!(page_window(&[a[0], b[0]], 0, PAGE + 1).is_none());
        assert!(page_window(&[a[1], a[0]], 0, PAGE + 1).is_none());
        assert!(page_window(&[0x10], 0, 1).is_none());
        // One page of several is still one run.
        assert!(page_window(&[a[0], b[0]], 0, PAGE).is_some());
        assert_eq!(XMemFreePhysicalPages(3, a.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(1, b.as_ptr()), 1);
    }

    #[test]
    fn single_view_uses_write_watch_and_lazily_promotes_on_second_alias() {
        use std::io::{Read, Seek, Write};
        use windows_sys::Win32::System::Memory::{GetWriteWatch, MEM_MAPPED, MEM_PRIVATE};
        const WRITE_WATCH_FLAG_RESET: u32 = 1;

        // Verify env switch resolver (strictly opt-in: default OFF, "1" ON on native Windows, always OFF on Wine).
        assert!(!resolve_single_view_watch(None, false));
        assert!(resolve_single_view_watch(Some("1"), false));
        assert!(resolve_single_view_watch(Some(" 1 "), false));
        assert!(!resolve_single_view_watch(Some("0"), false));
        assert!(!resolve_single_view_watch(Some(" 0 "), false));
        assert!(!resolve_single_view_watch(None, true));
        assert!(!resolve_single_view_watch(Some("1"), true));

        let pages = allocate(4);
        let reserve = |n: usize| {
            let p = XMemVirtualAlloc(
                std::ptr::null_mut(),
                n * PAGE,
                MEM_RESERVE | PAGES_64K,
                0,
                PAGE_READWRITE,
            );
            assert!(!p.is_null());
            p
        };
        let (a, b, c, d) = (reserve(2), reserve(1), reserve(1), reserve(1));

        // 1. By default (`single_view_watch() == false`), single-view mappings
        //    use `MEM_MAPPED` (`SEC_RESERVE`) directly; when opted in (`=1`),
        //    they use `MEM_PRIVATE | MEM_WRITE_WATCH`.
        assert_eq!(XMemMapPhysicalPages(a, 2, pages.as_ptr()), a);
        if single_view_watch() {
            assert_eq!(query(a).host.Type, MEM_PRIVATE);
            assert!(physical().views.get(&(a as usize)).is_some_and(|v| v.2));

            // Clear any initial dirty bits, then write via CPU and verify hardware `GetWriteWatch`.
            let mut found = [std::ptr::null_mut::<c_void>(); 64];
            let mut count = found.len();
            let mut gran = 0u32;
            // SAFETY: valid GetWriteWatch query on our private write-watched view.
            assert_eq!(
                unsafe {
                    GetWriteWatch(
                        WRITE_WATCH_FLAG_RESET,
                        a,
                        2 * PAGE,
                        found.as_mut_ptr(),
                        &mut count,
                        &mut gran,
                    )
                },
                0
            );
            // SAFETY: `a` is committed `PAGE_READWRITE`.
            unsafe { a.cast::<u8>().add(0x100).write_volatile(0xAB) };
            count = found.len();
            assert_eq!(
                unsafe {
                    GetWriteWatch(
                        WRITE_WATCH_FLAG_RESET,
                        a,
                        2 * PAGE,
                        found.as_mut_ptr(),
                        &mut count,
                        &mut gran,
                    )
                },
                0
            );
            assert!(count >= 1 && found[..count].contains(&a));

            // DirectStorage `write_physical` writes directly into single-view private `p1`
            // without promoting it, and triggers `GetWriteWatch` on `a + PAGE`.
            assert!(write_physical(&pages[1..], 0x20, &[0xCD, 0xEF]));
            // SAFETY: mapped read-write.
            assert_eq!(unsafe { a.cast::<u8>().add(PAGE + 0x20).read() }, 0xCD);
            assert_eq!(unsafe { a.cast::<u8>().add(PAGE + 0x21).read() }, 0xEF);
            assert!(physical().views.get(&(a as usize)).is_some_and(|v| v.2));
            count = found.len();
            assert_eq!(
                unsafe {
                    GetWriteWatch(
                        WRITE_WATCH_FLAG_RESET,
                        a,
                        2 * PAGE,
                        found.as_mut_ptr(),
                        &mut count,
                        &mut gran,
                    )
                },
                0
            );
            assert!(count >= 1 && found[..count].contains(&a.wrapping_byte_add(PAGE)));

            // Kernel `ReadFile` directly into single-view private mapping succeeds cleanly.
            let mut file = tempfile::tempfile().unwrap();
            file.write_all(b"kernel_ok").unwrap();
            file.rewind().unwrap();
            // SAFETY: 9 bytes inside `a`.
            let dst = unsafe { std::slice::from_raw_parts_mut(a.cast::<u8>().add(0x300), 9) };
            assert_eq!(file.read(dst).unwrap(), 9);
            assert_eq!(dst, b"kernel_ok");
        } else {
            assert_eq!(query(a).host.Type, MEM_MAPPED);
            assert!(physical().views.get(&(a as usize)).is_some_and(|v| !v.2));
            // SAFETY: still write the expected bytes under the default `MEM_MAPPED` path.
            unsafe { a.cast::<u8>().add(0x100).write_volatile(0xAB) };
            assert!(write_physical(&pages[1..], 0x20, &[0xCD, 0xEF]));
        }

        // 2. Mapping `pages[0]` at a 2nd virtual alias `b` shares the physical
        //    bytes via `MEM_MAPPED` (`SEC_RESERVE` section), promoting if needed.
        assert_eq!(XMemMapPhysicalPages(b, 1, pages.as_ptr()), b);
        assert_eq!(query(a).host.Type, MEM_MAPPED);
        assert_eq!(query(b).host.Type, MEM_MAPPED);
        assert!(physical().views.get(&(a as usize)).is_some_and(|v| !v.2));
        // SAFETY: both `a` and `b` are mapped read-write to the same physical page.
        unsafe {
            assert_eq!(b.cast::<u8>().add(0x100).read(), 0xAB);
            assert_eq!(a.cast::<u8>().add(PAGE + 0x20).read(), 0xCD);
            b.cast::<u8>().add(0x200).write_volatile(0x55);
            assert_eq!(a.cast::<u8>().add(0x200).read(), 0x55);
        }

        // 3. `page_window` preserves prior CPU bytes while landing DirectStorage decoded bytes.
        assert_eq!(XMemMapPhysicalPages(c, 1, pages[2..].as_ptr()), c);
        // SAFETY: `c` is mapped read-write.
        unsafe { c.cast::<u8>().write_volatile(0x77) };
        {
            let mut win = page_window(&pages[2..], 16, 4).expect("page_window on mapped page");
            win.bytes().copy_from_slice(&[1, 2, 3, 4]);
        }
        // SAFETY: `c` stays mapped & coherent after DirectStorage write.
        unsafe {
            assert_eq!(c.cast::<u8>().read(), 0x77);
            assert_eq!(
                std::slice::from_raw_parts(c.cast::<u8>().add(16), 4),
                &[1, 2, 3, 4]
            );
        }

        // 4. Directly exercise the opt-in `map_private` + `promote_view_containing`
        //    transition on native Windows even when default is OFF.
        if !on_wine() {
            {
                let mut phys = physical();
                let _changing = victory_gpu::title_memory_remapping();
                let slice = phys.pages.get(&pages[3]).cloned().unwrap();
                assert!(map_private(&mut phys, d as usize, PAGE, slice));
                assert!(phys.views.get(&(d as usize)).is_some_and(|v| v.2));
            }
            assert_eq!(query(d).host.Type, MEM_PRIVATE);
            // SAFETY: committed private write-watched view.
            unsafe { d.cast::<u8>().add(0x40).write_volatile(0x9E) };
            {
                let mut phys = physical();
                let _changing = victory_gpu::title_memory_remapping();
                assert!(promote_view_containing(&mut phys, d as usize));
                assert!(phys.views.get(&(d as usize)).is_some_and(|v| !v.2));
            }
            assert_eq!(query(d).host.Type, MEM_MAPPED);
            // SAFETY: promoted `MEM_MAPPED` view preserved written byte.
            assert_eq!(unsafe { d.cast::<u8>().add(0x40).read() }, 0x9E);
        }

        assert_eq!(XMemMapPhysicalPages(a, 2, std::ptr::null()), a);
        assert_eq!(XMemMapPhysicalPages(b, 1, std::ptr::null()), b);
        assert_eq!(XMemMapPhysicalPages(c, 1, std::ptr::null()), c);
        assert_eq!(XMemMapPhysicalPages(d, 1, std::ptr::null()), d);
        assert_eq!(XMemFreePhysicalPages(4, pages.as_ptr()), 1);
    }

    #[test]
    fn initial_placeholder_preserves_wc_protect_and_alias_writes_while_noaccess() {
        use windows_sys::Win32::System::Memory::VirtualProtect;

        const WC_GPU_RW: u32 = 0x2404; // PAGE_GRAPHICS_READWRITE | PAGE_WRITECOMBINE | PAGE_READWRITE
        const POOL_FLAGS: u64 = 0x23;
        const KIND: u32 = PAGES_64K | MEM_RESERVE;

        // Initial `MEM_RESERVE_PLACEHOLDER` reservation (exact Func9 pattern).
        let base_ptr = XMemVirtualAlloc(std::ptr::null_mut(), 4 * PAGE, KIND, POOL_FLAGS, WC_GPU_RW);
        assert!(!base_ptr.is_null());
        let base = base_ptr as usize;

        assert_eq!(
            resolve_mapped_console_memory(base, 4 * PAGE),
            vec![(base, base + 4 * PAGE, POOL_FLAGS as u32, WC_GPU_RW)],
            "initial MEM_RESERVE_PLACEHOLDER must record 0x2404 in VA_PROTECT (not default to 0x2004)"
        );

        let mut count = 2usize;
        let mut pages = [0usize; 2];
        assert_eq!(XMemAllocatePhysicalPages(GRAPHICS, &mut count, pages.as_mut_ptr()), 1);
        assert_eq!(count, 2);

        // Map -> partial unmap -> remap cycle must keep 0x2404 intact.
        assert_eq!(XMemMapPhysicalPages(base_ptr, 2, pages.as_ptr()), base_ptr);
        assert_eq!(XMemMapPhysicalPages(base_ptr, 1, std::ptr::null()), base_ptr);
        assert_eq!(XMemMapPhysicalPages(base_ptr, 1, pages.as_ptr()), base_ptr);
        assert_eq!(
            resolve_mapped_console_memory(base, 2 * PAGE),
            vec![(base, base + 2 * PAGE, POOL_FLAGS as u32, WC_GPU_RW)]
        );

        // `write_va_via_alias` must populate physical RAM on a `MEM_MAPPED` view
        // even while the title VA is `PAGE_NOACCESS` (zero torn-read window).
        let mut old = 0u32;
        // SAFETY: committed 64 KiB `MEM_MAPPED` view at `base`.
        assert_ne!(unsafe { VirtualProtect(base_ptr, PAGE, PAGE_NOACCESS, &mut old) }, 0);
        let payload = [0xCCu8; PAGE];
        assert!(
            write_va_via_alias(base, &payload),
            "write_va_via_alias must succeed on section-backed MEM_MAPPED view while PAGE_NOACCESS"
        );
        // SAFETY: restoring `PAGE_READWRITE` and verifying all 64 KiB landed.
        unsafe {
            assert_ne!(VirtualProtect(base_ptr, PAGE, PAGE_READWRITE, &mut old), 0);
            let got = std::slice::from_raw_parts(base_ptr.cast::<u8>(), PAGE);
            assert!(got.iter().all(|&b| b == 0xCC));
        }

        assert_eq!(XMemMapPhysicalPages(base_ptr, 2, std::ptr::null()), base_ptr);
        assert_eq!(XMemFreePhysicalPages(2, pages.as_ptr()), 1);
        clear_va_protect(base, 4 * PAGE);
        // After `clear_va_protect`, unregistered placeholder resolves to default 0x2004.
        assert_eq!(
            resolve_mapped_console_memory(base, PAGE),
            vec![(base, base + PAGE, 0x2, 0x2004)]
        );
        // SAFETY: releasing our test reservation.
        unsafe { VirtualFree(base_ptr, 0, MEM_RELEASE) };
    }

    #[test]
    fn unmap_split_notifies_only_unmapped_subrange() {
        let pages = allocate(4);
        let a = XMemVirtualAlloc(
            std::ptr::null_mut(),
            4 * PAGE,
            MEM_RESERVE | PAGES_64K,
            0,
            PAGE_READWRITE,
        );
        assert!(!a.is_null());
        assert_eq!(XMemMapPhysicalPages(a, 4, pages.as_ptr()), a);
        let base = a as usize;
        // Write distinct markers into head (page 0) and tail (pages 2..3).
        // SAFETY: all 4 pages are mapped `PAGE_READWRITE`.
        unsafe {
            (base as *mut u8).write(0x11);
            ((base + 2 * PAGE) as *mut u8).write(0x33);
            ((base + 3 * PAGE) as *mut u8).write(0x44);
        }

        // Unmap ONLY page 1 in the middle of the 4-page view.
        UNMAP_REMAP_LOG.with(|log| *log.borrow_mut() = Some(Vec::new()));
        let middle = (base + PAGE) as *mut c_void;
        assert_eq!(XMemMapPhysicalPages(middle, 1, std::ptr::null()), middle);
        let events = UNMAP_REMAP_LOG.with(|log| log.borrow_mut().take().unwrap());

        assert_eq!(
            events,
            vec![(base + PAGE, PAGE)],
            "splitting a 4-page view to unmap page 1 must emit title_memory_remapped ONLY for [base+PAGE, PAGE), not head or tail"
        );
        assert_eq!(query(a).host.State, MEM_COMMIT);
        assert_eq!(query(middle).host.State, MEM_RESERVE);
        assert_eq!(query((base + 2 * PAGE) as *const c_void).host.State, MEM_COMMIT);
        assert_eq!(query((base + 3 * PAGE) as *const c_void).host.State, MEM_COMMIT);
        // SAFETY: head and tail remain mapped with their original bytes intact.
        unsafe {
            assert_eq!((base as *const u8).read(), 0x11);
            assert_eq!(((base + 2 * PAGE) as *const u8).read(), 0x33);
            assert_eq!(((base + 3 * PAGE) as *const u8).read(), 0x44);
        }

        assert_eq!(XMemMapPhysicalPages(a, 4, std::ptr::null()), a);
        assert_eq!(XMemFreePhysicalPages(4, pages.as_ptr()), 1);
        // SAFETY: releasing our test reservation.
        unsafe { VirtualFree(a, 0, MEM_RELEASE) };
    }

    #[test]
    fn working_set_tracks_title_memory_and_ignores_host_allocations() {
        let mut before = WorkingSet::default();
        assert_eq!(XMemGetWorkingSetStatistics(0, &mut before), 1);
        assert!(before.title_used >= PAGE);

        // Use large enough title allocations (16 + 4 + 2 = 22 MiB) and host
        // noise (64 MiB) that concurrent sibling unit tests (which touch a few
        // hundred KiB at most) cannot perturb the assertions.
        const PHYS_COUNT: usize = 256; // 16 MiB
        const VIRT_BYTES: usize = 4 * MIB;
        const HEAP_BYTES: usize = 2 * MIB;
        const HOST_NOISE_BYTES: usize = 64 * MIB;

        let phys = allocate(PHYS_COUNT);
        let virt = XMemVirtualAlloc(
            std::ptr::null_mut(),
            VIRT_BYTES,
            MEM_RESERVE | MEM_COMMIT,
            GRAPHICS,
            PAGE_READWRITE,
        );
        assert!(!virt.is_null());
        let heap = XMemAllocDefault(HEAP_BYTES, 0);
        assert!(!heap.is_null());

        let mut after = WorkingSet::default();
        assert_eq!(XMemGetWorkingSetStatistics(0, &mut after), 1);
        assert!(
            after.title_used >= before.title_used.saturating_add(16 * MIB),
            "expected tracked title_used to rise by >= 16 MiB (before={}, after={})",
            before.title_used,
            after.title_used
        );
        assert!(after.physical_pages >= PHYS_COUNT as u32);
        assert!(
            VIRTUAL_COMMITTED.lock().unwrap().contains_key(&(virt as usize)),
            "committed XMemVirtualAlloc region must be tracked"
        );

        // Touching a 64 MiB host Rust allocation commits 64 MiB in Windows
        // `PagefileUsage` but must NOT inflate `title_used`.
        let host_noise = vec![0xA5u8; HOST_NOISE_BYTES];
        std::hint::black_box(&host_noise);
        let mut with_noise = WorkingSet::default();
        assert_eq!(XMemGetWorkingSetStatistics(0, &mut with_noise), 1);
        assert!(
            with_noise.title_used < after.title_used.saturating_add(16 * MIB),
            "64 MiB host-side allocation must not inflate title_used (after={}, with_noise={})",
            after.title_used,
            with_noise.title_used
        );
        drop(host_noise);

        // Free all title allocations (including direct kernel32 VirtualFree on `virt`).
        XMemFreeDefault(heap, 0);
        // SAFETY: releasing our committed virtual region.
        assert_ne!(unsafe { VirtualFree(virt, 0, MEM_RELEASE) }, 0);
        assert_eq!(XMemFreePhysicalPages(PHYS_COUNT, phys.as_ptr()), 1);

        let mut freed = WorkingSet::default();
        assert_eq!(XMemGetWorkingSetStatistics(0, &mut freed), 1);
        assert!(
            freed.title_used.saturating_add(16 * MIB) <= after.title_used,
            "freeing 22 MiB of title memory must reduce tracked title_used (after={}, freed={})",
            after.title_used,
            freed.title_used
        );
        assert!(
            !VIRTUAL_COMMITTED.lock().unwrap().contains_key(&(virt as usize)),
            "kernel32 VirtualFree'd region must be pruned from VIRTUAL_COMMITTED"
        );
    }
}
