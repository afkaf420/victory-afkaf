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

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

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

/// Coalesces each shadow's declared UAV write ranges and filters them under
/// `VICTORY_NATIVE_VRAM`. Shadows whose ranges drop completely are removed
/// from `declared` so they never allocate `Card::before`, never run GPU
/// compare, and never read back over PCIe.
pub(super) fn filter_declared(declared: &mut Vec<(Arc<Shadow>, Vec<(usize, usize)>)>) {
    let active = mode();
    if declared.is_empty() || active == Mode::Off {
        return;
    }
    let mut skipped_shadows = 0u64;
    for (_, ranges) in declared.iter_mut() {
        super::join_ranges(ranges, std::iter::empty());
        let mut filtered = Vec::new();
        for &(from, to) in ranges.iter() {
            filtered.extend(filter_range(from, to, active, true));
        }
        *ranges = filtered;
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
}
