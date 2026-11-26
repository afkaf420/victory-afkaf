//! PC-Native VRAM Promotion — Move 1 (`VICTORY_NATIVE_VRAM`, default safe mode `1`).
//!
//! Queries [`crate::watch::console_mem_class_ranges`] for every declared UAV
//! write range on a card-backed [`crate::shadow::Shadow`] in
//! [`super::Host::card_before`]:
//! - Mode `1` (default when unset/empty/`1`/`true`): drops **ONLY**
//!   [`ConsoleMemClass::GpuReadOnly`] (`0x1004` / `0x1404` / `0x8404`) sub-ranges,
//!   keeping `WriteCombinedGpuRw` (`0x2404`) intact to avoid partial 4 KiB page
//!   CPU-write clobber and stale CPU-side reads (`Step::Texture`, `ValueWait`, etc.).
//! - Mode `2` (opt-in `VICTORY_NATIVE_VRAM=2` or `wc`): drops both
//!   [`ConsoleMemClass::GpuReadOnly`] and [`ConsoleMemClass::WriteCombinedGpuRw`].
//! - Mode `0` (`0` / `off` / `false`): disabled (all declared card UAV ranges
//!   are compared and read back).

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, RwLock};

use crate::shadow::Shadow;
use crate::watch::ConsoleMemClass;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    /// Disabled (`0` / `off` / `false`).
    Off,
    /// Safe default (`1` / unset): drop `GpuReadOnly` only.
    ReadOnly,
    /// Aggressive opt-in (`2` / `wc`): drop `GpuReadOnly` + `WriteCombinedGpuRw`.
    ReadOnlyAndWc,
}

impl Mode {
    fn code(self) -> u32 {
        match self {
            Mode::Off => 0,
            Mode::ReadOnly => 1,
            Mode::ReadOnlyAndWc => 2,
        }
    }
}

/// Resolves `VICTORY_NATIVE_VRAM`: default safe mode `1` (`ReadOnly`);
/// `2` / `wc` opts into dropping `WriteCombinedGpuRw` too; `0` / `off` / `false`
/// disables.
fn resolve_mode(env: Option<&str>) -> Mode {
    match env.map(|v| v.trim().to_ascii_lowercase()).as_deref() {
        Some("0" | "off" | "false") => Mode::Off,
        Some("2" | "wc") => Mode::ReadOnlyAndWc,
        _ => Mode::ReadOnly,
    }
}

/// Active `VICTORY_NATIVE_VRAM` mode. Logs a startup summary on first query.
fn mode() -> Mode {
    static MODE: OnceLock<Mode> = OnceLock::new();
    *MODE.get_or_init(|| {
        let env = std::env::var("VICTORY_NATIVE_VRAM").ok();
        let active = resolve_mode(env.as_deref());
        tracing::info!(
            mode = active.code(),
            "PC-native VRAM promotion (VICTORY_NATIVE_VRAM={}): {}",
            active.code(),
            match active {
                Mode::Off => "OFF (comparing and reading back all declared card UAV ranges)",
                Mode::ReadOnly => "SAFE (dropping GpuReadOnly sub-ranges only; WriteCombinedGpuRw kept intact)",
                Mode::ReadOnlyAndWc => "AGGRESSIVE (dropping GpuReadOnly and WriteCombinedGpuRw sub-ranges from Card::before, GPU compare, and PCIe readback)",
            }
        );
        active
    })
}

#[inline]
fn drop_readback(class: ConsoleMemClass, mode: Mode) -> bool {
    match mode {
        Mode::Off => false,
        Mode::ReadOnly => class == ConsoleMemClass::GpuReadOnly,
        Mode::ReadOnlyAndWc => matches!(
            class,
            ConsoleMemClass::GpuReadOnly | ConsoleMemClass::WriteCombinedGpuRw
        ),
    }
}

/// Per-interval counters reset on each periodic summary log:
/// `[0]` dropped `GpuReadOnly` ranges, `[1]` dropped `GpuReadOnly` bytes,
/// `[2]` dropped `WriteCombinedGpuRw` ranges, `[3]` dropped `WriteCombinedGpuRw` bytes,
/// `[4]` kept ranges, `[5]` kept bytes, `[6]` shadows fully dropped (no `Card::before`).
static PERIOD: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];

/// Cumulative counters since process start:
/// `[0]` total dropped ranges, `[1]` total dropped bytes, `[2]` total shadows skipped.
static TOTAL: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// Filters `[from, to)` against console memory classifications under `mode`.
/// Adjacent kept sub-ranges are coalesced.
fn filter_range(from: usize, to: usize, active: Mode, record_stats: bool) -> Vec<(usize, usize)> {
    if from >= to {
        return Vec::new();
    }
    if active == Mode::Off {
        return vec![(from, to)];
    }
    let parts = crate::watch::console_mem_class_ranges(from, to);
    let mut kept: Vec<(usize, usize)> = Vec::new();
    let (mut drop_ro_n, mut drop_ro_b) = (0u64, 0u64);
    let (mut drop_wc_n, mut drop_wc_b) = (0u64, 0u64);
    let mut kept_b = 0u64;
    for (lo, hi, class) in parts {
        let len = (hi - lo) as u64;
        if drop_readback(class, active) {
            if class == ConsoleMemClass::GpuReadOnly {
                drop_ro_n += 1;
                drop_ro_b += len;
            } else {
                drop_wc_n += 1;
                drop_wc_b += len;
            }
            continue;
        }
        kept_b += len;
        if let Some(last) = kept.last_mut()
            && last.1 == lo
        {
            last.1 = hi;
        } else {
            kept.push((lo, hi));
        }
    }
    if record_stats {
        if drop_ro_n > 0 {
            PERIOD[0].fetch_add(drop_ro_n, Ordering::Relaxed);
            PERIOD[1].fetch_add(drop_ro_b, Ordering::Relaxed);
        }
        if drop_wc_n > 0 {
            PERIOD[2].fetch_add(drop_wc_n, Ordering::Relaxed);
            PERIOD[3].fetch_add(drop_wc_b, Ordering::Relaxed);
        }
        let dropped_n = drop_ro_n + drop_wc_n;
        let dropped_b = drop_ro_b + drop_wc_b;
        if dropped_n > 0 {
            TOTAL[0].fetch_add(dropped_n, Ordering::Relaxed);
            TOTAL[1].fetch_add(dropped_b, Ordering::Relaxed);
        }
        if !kept.is_empty() {
            PERIOD[4].fetch_add(kept.len() as u64, Ordering::Relaxed);
            PERIOD[5].fetch_add(kept_b, Ordering::Relaxed);
        }
    }
    kept
}

/// Tracks which carded `Shadow::id()` currently owns each `GpuResident` 64 KiB
/// block (`block = va >> 16`) so `Shadow::update` / `merge_under_writes` skip
/// refreshes ONLY for the owning shadow, while any overlapping sub-buffer
/// shadow still materializes fresh GPU bytes via `ensure_cpu_resident`.
/// Queried ONLY on the slow path after lock-free `lazy_readback::is_gpu_resident(block)`.
static LAZY_OWNERS: OnceLock<RwLock<HashMap<usize, u64>>> = OnceLock::new();

fn lazy_owners() -> &'static RwLock<HashMap<usize, u64>> {
    LAZY_OWNERS.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Returns `true` if `block` (`va >> 16`) is currently `GpuResident` AND owned
/// by `shadow_id`. Fast path is a ~1 ns lock-free atomic bitmap check when
/// `block` is not `GpuResident`.
#[inline]
pub(crate) fn is_lazy_owner(block: usize, shadow_id: u64) -> bool {
    if !super::lazy_readback::is_gpu_resident(block) {
        return false;
    }
    lazy_owners()
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .get(&block)
        .copied()
        == Some(shadow_id)
}

/// Records `shadow_id` as the owner of every 64 KiB block in `before` that
/// `lazy_readback::claim_gpu_resident` promoted (i.e. now `is_gpu_resident` and
/// no longer present in `remaining`).
fn record_lazy_owners(shadow_id: u64, before: &[(usize, usize)], remaining: &[(usize, usize)]) {
    let block_sz = super::lazy_readback::BLOCK;
    let mut claimed = Vec::new();
    for &(from, to) in before {
        if from >= to {
            continue;
        }
        for block in from / block_sz..to.div_ceil(block_sz) {
            let b_lo = block * block_sz;
            let b_hi = b_lo.saturating_add(block_sz);
            if !super::touches(remaining, b_lo, b_hi) && super::lazy_readback::is_gpu_resident(block) {
                claimed.push(block);
            }
        }
    }
    if !claimed.is_empty() {
        let mut map = lazy_owners().write().unwrap_or_else(|e| e.into_inner());
        for block in claimed {
            map.insert(block, shadow_id);
        }
    }
}

/// Removes ownership entries for `shadow_id` when a shadow is dropped.
pub(crate) fn forget_lazy_owner(shadow_id: u64) {
    if let Some(lock) = LAZY_OWNERS.get() {
        let mut map = lock.write().unwrap_or_else(|e| e.into_inner());
        if !map.is_empty() {
            map.retain(|_, &mut id| id != shadow_id);
        }
    }
}

/// Coalesces each shadow's declared UAV write ranges and filters them under
/// Move 1 (`VICTORY_NATIVE_VRAM`, dropping `GpuReadOnly` outright) and Move 2
/// (`VICTORY_LAZY_READBACK`, claiming eligible `WriteCombinedGpuRw` /
/// `CachedGpuRw` blocks as `GpuResident` up to `flight_fence`). Shadows whose
/// ranges drop/claim completely are removed from `declared` so they never
/// allocate `Card::before`, never run GPU compare, and never read back over PCIe.
pub(super) fn filter_declared(
    declared: &mut Vec<(Arc<Shadow>, Vec<(usize, usize)>)>,
    flight_fence: u64,
) {
    if declared.is_empty() {
        return;
    }
    let active = mode();
    let lazy_on = super::lazy_readback::on();
    if active == Mode::Off && !lazy_on {
        return;
    }
    let mut skipped_shadows = 0u64;
    for (shadow, ranges) in declared.iter_mut() {
        super::join_ranges(ranges, std::iter::empty());
        if active != Mode::Off {
            let mut filtered = Vec::new();
            for &(from, to) in ranges.iter() {
                filtered.extend(filter_range(from, to, active, true));
            }
            *ranges = filtered;
        }
        if lazy_on && !ranges.is_empty() {
            let before = std::mem::take(ranges);
            let remaining = super::lazy_readback::claim_gpu_resident(shadow, &before, flight_fence);
            record_lazy_owners(shadow.id(), &before, &remaining);
            *ranges = remaining;
        }
        if ranges.is_empty() {
            skipped_shadows += 1;
        }
    }
    if skipped_shadows > 0 {
        PERIOD[6].fetch_add(skipped_shadows, Ordering::Relaxed);
        TOTAL[2].fetch_add(skipped_shadows, Ordering::Relaxed);
        declared.retain(|(_, ranges)| !ranges.is_empty());
    }
}

/// Logs the periodic summary of native-VRAM skipped bytes and ranges (called
/// from `Meter` every 5 seconds).
pub(super) fn log_periodic() {
    let active = mode();
    if active == Mode::Off {
        return;
    }
    let p: [u64; 7] = std::array::from_fn(|i| PERIOD[i].swap(0, Ordering::Relaxed));
    let total_ranges = TOTAL[0].load(Ordering::Relaxed);
    let total_bytes = TOTAL[1].load(Ordering::Relaxed);
    let total_shadows = TOTAL[2].load(Ordering::Relaxed);
    if p[0] + p[2] + p[4] == 0 && total_ranges == 0 {
        return;
    }
    tracing::info!(
        "PC-native VRAM promotion (VICTORY_NATIVE_VRAM={}): skipped {:.1} MB across {} ranges \
         (GpuReadOnly {:.1} MB / {}, WriteCombinedGpuRw {:.1} MB / {}, {} pure buffers omitted), \
         kept {:.1} MB / {} ranges; cumulative skipped {:.1} MB ({} ranges, {} pure buffers)",
        active.code(),
        (p[1] + p[3]) as f64 / (1024.0 * 1024.0),
        p[0] + p[2],
        p[1] as f64 / (1024.0 * 1024.0),
        p[0],
        p[3] as f64 / (1024.0 * 1024.0),
        p[2],
        p[6],
        p[5] as f64 / (1024.0 * 1024.0),
        p[4],
        total_bytes as f64 / (1024.0 * 1024.0),
        total_ranges,
        total_shadows,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    const MEM_GRAPHICS: u32 = 0x2;
    const PAGE_READWRITE: u32 = 0x4;
    const PAGE_WRITECOMBINE: u32 = 0x400;
    const PAGE_GRAPHICS_READONLY: u32 = 0x1000;
    const PAGE_GRAPHICS_READWRITE: u32 = 0x2000;
    const PAGE_GRAPHICS_EXECUTE_READ: u32 = 0x8000;
    const PAGE_GRAPHICS_COHERENT: u32 = 0x20000;
    const PAGE_GRAPHICS_NOACCESS: u32 = 0x800;

    #[test]
    fn native_vram_mode_defaults_safe_and_honors_overrides() {
        assert_eq!(resolve_mode(None), Mode::ReadOnly);
        assert_eq!(resolve_mode(Some("")), Mode::ReadOnly);
        assert_eq!(resolve_mode(Some("1")), Mode::ReadOnly);
        assert_eq!(resolve_mode(Some("true")), Mode::ReadOnly);
        assert_eq!(resolve_mode(Some("2")), Mode::ReadOnlyAndWc);
        assert_eq!(resolve_mode(Some("wc")), Mode::ReadOnlyAndWc);
        assert_eq!(resolve_mode(Some("WC")), Mode::ReadOnlyAndWc);
        assert_eq!(resolve_mode(Some("0")), Mode::Off);
        assert_eq!(resolve_mode(Some("off")), Mode::Off);
        assert_eq!(resolve_mode(Some("OFF")), Mode::Off);
        assert_eq!(resolve_mode(Some("false")), Mode::Off);
    }

    #[test]
    fn mode_1_drops_only_gpu_read_only_and_mode_2_also_drops_wc_gpu_rw() {
        let base = 0x0000_5100_0000_0000usize;
        let m = 1usize << 20;
        // 0x1004 (RO), 0x1404 (WC+RO), 0x8404 (WC+ExecRead), 0x2404 (WC+RW).
        crate::watch::note_console_memory(base, 4 * m, MEM_GRAPHICS, PAGE_GRAPHICS_READONLY | PAGE_READWRITE);
        crate::watch::note_console_memory(
            base + 4 * m,
            4 * m,
            MEM_GRAPHICS,
            PAGE_GRAPHICS_READONLY | PAGE_WRITECOMBINE | PAGE_READWRITE,
        );
        crate::watch::note_console_memory(
            base + 8 * m,
            4 * m,
            MEM_GRAPHICS,
            PAGE_GRAPHICS_EXECUTE_READ | PAGE_WRITECOMBINE | PAGE_READWRITE,
        );
        crate::watch::note_console_memory(
            base + 12 * m,
            4 * m,
            MEM_GRAPHICS,
            PAGE_GRAPHICS_READWRITE | PAGE_WRITECOMBINE | PAGE_READWRITE,
        );

        // Mode 0 (Off): keeps everything.
        assert_eq!(
            filter_range(base, base + 16 * m, Mode::Off, false),
            vec![(base, base + 16 * m)]
        );
        // Mode 1 (Safe default): drops the 3 GpuReadOnly ranges [0..12M), keeps WC+RW [12M..16M).
        assert_eq!(
            filter_range(base, base + 16 * m, Mode::ReadOnly, false),
            vec![(base + 12 * m, base + 16 * m)]
        );
        // Pure GpuReadOnly drops completely in Mode 1.
        assert!(filter_range(base, base + 12 * m, Mode::ReadOnly, false).is_empty());
        // Mode 2 (Aggressive opt-in): drops both GpuReadOnly and WriteCombinedGpuRw.
        assert!(filter_range(base, base + 16 * m, Mode::ReadOnlyAndWc, false).is_empty());

        crate::watch::clear_console_memory(base, 16 * m);
    }

    #[test]
    fn mixed_174mb_super_buffer_clips_by_mode() {
        let base = 0x0000_5200_0000_0000usize;
        let m = 1usize << 20;
        let z3_len = 60 * m;
        let z4_len = 64 * m;
        let z5_len = 50 * m;
        let total = z3_len + z4_len + z5_len; // 174 MiB

        // Zone 3: GpuReadOnly (0x1404)
        crate::watch::note_console_memory(
            base,
            z3_len,
            MEM_GRAPHICS,
            PAGE_GRAPHICS_READONLY | PAGE_WRITECOMBINE | PAGE_READWRITE,
        );
        // Zone 4: WriteCombinedGpuRw (0x2404)
        crate::watch::note_console_memory(
            base + z3_len,
            z4_len,
            MEM_GRAPHICS,
            PAGE_GRAPHICS_READWRITE | PAGE_WRITECOMBINE | PAGE_READWRITE,
        );
        // Zone 5: CachedGpuRw (0x2004)
        crate::watch::note_console_memory(
            base + z3_len + z4_len,
            z5_len,
            MEM_GRAPHICS,
            PAGE_GRAPHICS_READWRITE | PAGE_READWRITE,
        );

        // Mode 1 (Safe default): drops Z3, keeps coalesced Z4+Z5.
        assert_eq!(
            filter_range(base, base + total, Mode::ReadOnly, false),
            vec![(base + z3_len, base + total)]
        );
        // Mode 2 (Aggressive opt-in): drops Z3+Z4, clips strictly to Z5.
        assert_eq!(
            filter_range(base, base + total, Mode::ReadOnlyAndWc, false),
            vec![(base + z3_len + z4_len, base + total)]
        );
        // Mode 0 (Off): keeps the entire 174 MiB span untouched.
        assert_eq!(
            filter_range(base, base + total, Mode::Off, false),
            vec![(base, base + total)]
        );

        crate::watch::clear_console_memory(base, total);
    }

    #[test]
    fn coherent_cached_unknown_and_cpu_only_are_preserved_and_merged() {
        let base = 0x0000_5300_0000_0000usize;
        let m = 1usize << 20;

        // [0..2M): CachedGpuRw (keep)
        crate::watch::note_console_memory(base, 2 * m, MEM_GRAPHICS, PAGE_GRAPHICS_READWRITE | PAGE_READWRITE);
        // [2M..4M): CoherentGpuRw (keep, merges with [0..2M))
        crate::watch::note_console_memory(
            base + 2 * m,
            2 * m,
            MEM_GRAPHICS,
            PAGE_GRAPHICS_COHERENT | PAGE_GRAPHICS_READWRITE | PAGE_READWRITE,
        );
        // [4M..6M): Unknown gap (unregistered -> keep, merges into [0..6M))
        // [6M..8M): WriteCombinedGpuRw (kept in Mode 1, dropped in Mode 2)
        crate::watch::note_console_memory(
            base + 6 * m,
            2 * m,
            MEM_GRAPHICS,
            PAGE_GRAPHICS_READWRITE | PAGE_WRITECOMBINE | PAGE_READWRITE,
        );
        // [8M..10M): CpuOnly (keep conservatively)
        crate::watch::note_console_memory(
            base + 8 * m,
            2 * m,
            MEM_GRAPHICS,
            PAGE_GRAPHICS_NOACCESS | PAGE_READWRITE,
        );

        assert_eq!(
            filter_range(base, base + 10 * m, Mode::ReadOnly, false),
            vec![(base, base + 10 * m)]
        );
        assert_eq!(
            filter_range(base, base + 10 * m, Mode::ReadOnlyAndWc, false),
            vec![(base, base + 6 * m), (base + 8 * m, base + 10 * m)]
        );

        crate::watch::clear_console_memory(base, 10 * m);
    }

    /// Live GPU D3D12 Debug Layer verification (`--features diagnostics`,
    /// `VICTORY_D3D12_DEBUG=1`, `--ignored`): exercises BOTH:
    /// 1. `lazy_readback`'s private `victory-lazy-fetch` COPY queue
    ///    `Wait(&main_fence, flight_fence)` + 64 KiB readback triggered by a
    ///    real Windows VEH `PAGE_NOACCESS` read fault on `WriteCombinedGpuRw`
    ///    (`0x2404`) memory backed by a mirror-free ReBAR-streamed card shadow;
    /// 2. `FillTo::Buffer` (`HostList::upload`) and `FillTo::BufferOffset`
    ///    (`HostList::shared_scratch`) at `VICTORY_CARD_STEPS=1` via
    ///    `card_steps` GPU copy + barrier transitions.
    /// Asserts exact data roundtrip and ZERO D3D12 Debug Layer errors.
    #[test]
    #[ignore]
    fn debug_layer_lazy_readback_copy_queue_and_card_steps_buffer_fills() {
        use windows::Win32::Graphics::Direct3D12::ID3D12CommandList;
        use windows::core::Interface;
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
        };
        const MEM_WRITE_WATCH: u32 = 0x20_0000;
        const BLOCK: usize = super::super::lazy_readback::BLOCK;

        let Some(host) = super::super::host() else {
            panic!("host D3D12 device required for live Debug Layer test");
        };
        // Drain any earlier messages so we strictly check this test's commands.
        let _ = host.log_messages();

        // --- Part 1: Real VEH fault -> `victory-lazy-fetch` COPY queue Wait + 64 KiB readback ---
        // SAFETY: fresh 64 KiB write-watched title-memory allocation freed at test end.
        let mem = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                BLOCK,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(mem, 0);
        struct MemGuard(usize);
        impl Drop for MemGuard {
            fn drop(&mut self) {
                crate::watch::title_memory_remapped(self.0, BLOCK);
                crate::watch::clear_console_memory(self.0, BLOCK);
                // SAFETY: owned VirtualAlloc region.
                unsafe { VirtualFree(self.0 as *mut _, 0, MEM_RELEASE) };
            }
        }
        let _mem_guard = MemGuard(mem);
        crate::watch::note_console_memory(
            mem,
            BLOCK,
            MEM_GRAPHICS,
            PAGE_GRAPHICS_READWRITE | PAGE_WRITECOMBINE | PAGE_READWRITE,
        );

        let shadow = crate::shadow::test_card_shadow_at(mem as u64, BLOCK as u64)
            .expect("mirror-free card shadow");
        assert!(shadow.is_rebar_streaming());

        // Populate `shadow.buffer` on the GPU with 0x5A pattern via main queue.
        let (src_upload, src_ptr) = host.upload_buffer(BLOCK as u64).expect("upload buffer");
        // SAFETY: mapped `BLOCK` writable upload bytes.
        unsafe { std::ptr::write_bytes(src_ptr, 0x5A, BLOCK) };
        let copy_list = host.new_list().expect("host list");
        // SAFETY: open command list and live buffers.
        unsafe {
            copy_list.list.CopyBufferRegion(&shadow.buffer, 0, &src_upload, 0, BLOCK as u64);
        }
        copy_list.close();
        let flight_fence = {
            let mut submitted = host.submitted.lock().unwrap_or_else(|e| e.into_inner());
            *submitted += 1;
            let val = *submitted;
            let cmd: ID3D12CommandList = copy_list.list.cast().expect("cast cmd list");
            // SAFETY: closed command list and live main queue/fence.
            unsafe {
                host.queue.ExecuteCommandLists(&[Some(cmd)]);
                host.queue.Signal(&host.fence, val).expect("signal main fence");
            }
            val
        };

        // Claim the 64 KiB block as `GpuResident` up to `flight_fence` -> flips `PAGE_NOACCESS`.
        let rem = super::super::lazy_readback::claim_gpu_resident(
            &shadow,
            &[(mem, mem + BLOCK)],
            flight_fence,
        );
        assert!(
            rem.is_empty(),
            "WriteCombinedGpuRw 64 KiB block must be claimed by lazy_readback (remaining={rem:?})"
        );
        assert!(super::super::lazy_readback::is_gpu_resident(mem / BLOCK));

        // Direct CPU read triggers Windows VEH -> `victory-lazy-fetch` private COPY queue
        // `Wait(&main_fence, flight_fence)` -> 64 KiB GPU->CPU readback -> restores Title RAM!
        // SAFETY: committed 64 KiB block guarded by `lazy_readback` VEH.
        let first_byte = unsafe { (mem as *const u8).read_volatile() };
        assert_eq!(first_byte, 0x5A, "VEH demand readback must deliver GPU bytes");
        // SAFETY: block is now restored to `PAGE_READWRITE`.
        let full = unsafe { std::slice::from_raw_parts(mem as *const u8, BLOCK) };
        assert!(full.iter().all(|&b| b == 0x5A), "all 64 KiB must match GPU 0x5A bytes");
        assert!(!super::super::lazy_readback::is_gpu_resident(mem / BLOCK));

        // --- Part 2: `FillTo::Buffer` + `FillTo::BufferOffset` at `stage = 1` ---
        assert!(super::super::card_steps::stage() >= 1);
        let fill_list = host.new_list().expect("fill list");
        let (buf_a, ptr_a) = fill_list.upload(host, 512).expect("shared upload buffer at stage=1");
        let (buf_b, off_b, _gpu_b, ptr_b) = fill_list.shared_scratch(256).expect("shared scratch");
        let (buf_b2, off_b2, _gpu_b2, ptr_b2) = fill_list.shared_scratch(256).expect("shared scratch 2");
        assert_eq!(off_b, 0);
        assert_eq!(off_b2, 256);
        let (src_shared, src_shared_ptr) = host.shared_buffer(1024).expect("src shared");
        // SAFETY: mapped 1024-byte shared buffer.
        unsafe {
            std::ptr::write_bytes(src_shared_ptr, 0xA1, 512);
            std::ptr::write_bytes(src_shared_ptr.add(512), 0xB2, 256);
            std::ptr::write_bytes(src_shared_ptr.add(768), 0xC3, 256);
        }
        let between = host.uav_buffer(4096).expect("between uav");
        let own_list = host.new_list().expect("own list");
        super::super::card_steps::record_for_test(
            &own_list.list,
            &between,
            &[(&src_shared, 0, 0, 512), (&src_shared, 512, 512, 256), (&src_shared, 768, 768, 256)],
            &[(&buf_a, 0, 0, 512), (&buf_b, off_b, 512, 256), (&buf_b2, off_b2, 768, 256)],
        );
        own_list.close();
        fill_list.close();
        host.execute(&[&own_list, &fill_list]);
        // SAFETY: GPU execution finished; buffers are mapped `shared_buffer`s.
        unsafe {
            assert!(std::slice::from_raw_parts(ptr_a, 512).iter().all(|&b| b == 0xA1));
            assert!(std::slice::from_raw_parts(ptr_b, 256).iter().all(|&b| b == 0xB2));
            assert!(std::slice::from_raw_parts(ptr_b2, 256).iter().all(|&b| b == 0xC3));
        }

        if let Some(err) = host.log_messages() {
            panic!("D3D12 Debug Layer reported error: {err}");
        }
    }
}
