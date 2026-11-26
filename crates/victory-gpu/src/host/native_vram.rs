//! PC-Native VRAM Promotion — Move 1 (`VICTORY_NATIVE_VRAM`, default ON).
//!
//! Queries [`crate::watch::console_mem_class_ranges`] for every declared UAV
//! write range on a card-backed [`crate::shadow::Shadow`] in
//! [`super::Host::card_before`], dropping sub-ranges classified as:
//! - [`ConsoleMemClass::GpuReadOnly`] (`0x1004` / `0x1404` / `0x8404`)
//! - [`ConsoleMemClass::WriteCombinedGpuRw`] (`0x2404`)
//!
//! Pure `GpuReadOnly` and `WriteCombinedGpuRw` card buffers therefore never
//! allocate `Card::before`, never run GPU rehash/compare passes, and never
//! read back over PCIe. Mixed super-buffers (such as a Low-4GB `Z3+Z4+Z5`
//! heap) clip GPU compare and PCIe readback strictly to Zone 5
//! (`CachedGpuRw`), `CoherentGpuRw`, and any unclassified (`Unknown`) spans.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use crate::shadow::Shadow;
use crate::watch::ConsoleMemClass;

/// Resolves `VICTORY_NATIVE_VRAM`: enabled by default; `0`, `off`, or `false`
/// disables it.
fn resolve_on(env: Option<&str>) -> bool {
    match env.map(|v| v.trim().to_ascii_lowercase()) {
        Some(ref s) if s == "0" || s == "off" || s == "false" => false,
        _ => true,
    }
}

/// Whether Move 1 (`VICTORY_NATIVE_VRAM`) is enabled. Logs a startup summary
/// on first evaluation.
pub(super) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let env = std::env::var("VICTORY_NATIVE_VRAM").ok();
        let active = resolve_on(env.as_deref());
        tracing::info!(
            active,
            "PC-native VRAM promotion (VICTORY_NATIVE_VRAM): {}",
            if active {
                "ON (dropping GpuReadOnly and WriteCombinedGpuRw sub-ranges from Card::before, GPU compare, and PCIe readback)"
            } else {
                "OFF (comparing and reading back all declared card UAV ranges)"
            }
        );
        active
    })
}

#[inline]
fn drop_readback(class: ConsoleMemClass) -> bool {
    matches!(class, ConsoleMemClass::GpuReadOnly | ConsoleMemClass::WriteCombinedGpuRw)
}

/// Per-interval counters reset on each periodic summary log:
/// `[0]` dropped `GpuReadOnly` ranges, `[1]` dropped `GpuReadOnly` bytes,
/// `[2]` dropped `WriteCombinedGpuRw` ranges, `[3]` dropped `WriteCombinedGpuRw` bytes,
/// `[4]` kept ranges, `[5]` kept bytes, `[6]` shadows fully dropped (no `Card::before`).
static PERIOD: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];

/// Cumulative counters since process start:
/// `[0]` total dropped ranges, `[1]` total dropped bytes, `[2]` total shadows skipped.
static TOTAL: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// Filters `[from, to)` against console memory classifications. When `enabled`
/// is true, sub-ranges classified as `GpuReadOnly` or `WriteCombinedGpuRw` are
/// dropped; adjacent kept sub-ranges are coalesced.
fn filter_range(from: usize, to: usize, enabled: bool, record_stats: bool) -> Vec<(usize, usize)> {
    if from >= to {
        return Vec::new();
    }
    if !enabled {
        return vec![(from, to)];
    }
    let parts = crate::watch::console_mem_class_ranges(from, to);
    let mut kept: Vec<(usize, usize)> = Vec::new();
    let (mut drop_ro_n, mut drop_ro_b) = (0u64, 0u64);
    let (mut drop_wc_n, mut drop_wc_b) = (0u64, 0u64);
    let mut kept_b = 0u64;
    for (lo, hi, class) in parts {
        let len = (hi - lo) as u64;
        if drop_readback(class) {
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

/// Coalesces each shadow's declared UAV write ranges and, when
/// `VICTORY_NATIVE_VRAM` is enabled, drops `GpuReadOnly` and
/// `WriteCombinedGpuRw` sub-ranges. Shadows whose ranges drop completely are
/// removed from `declared` so they never allocate `Card::before`, never run
/// GPU compare, and never read back over PCIe.
pub(super) fn filter_declared(declared: &mut Vec<(Arc<Shadow>, Vec<(usize, usize)>)>) {
    if declared.is_empty() || !on() {
        return;
    }
    let mut skipped_shadows = 0u64;
    for (_, ranges) in declared.iter_mut() {
        super::join_ranges(ranges, std::iter::empty());
        let mut filtered = Vec::new();
        for &(from, to) in ranges.iter() {
            filtered.extend(filter_range(from, to, true, true));
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
    if !on() {
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
        "PC-native VRAM promotion (VICTORY_NATIVE_VRAM=1): skipped {:.1} MB across {} ranges \
         (GpuReadOnly {:.1} MB / {}, WriteCombinedGpuRw {:.1} MB / {}, {} pure buffers omitted), \
         kept {:.1} MB / {} ranges; cumulative skipped {:.1} MB ({} ranges, {} pure buffers)",
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
    fn native_vram_defaults_on_and_honors_overrides() {
        assert!(resolve_on(None));
        assert!(resolve_on(Some("")));
        assert!(resolve_on(Some("1")));
        assert!(resolve_on(Some("true")));
        assert!(!resolve_on(Some("0")));
        assert!(!resolve_on(Some("off")));
        assert!(!resolve_on(Some("OFF")));
        assert!(!resolve_on(Some("false")));
    }

    #[test]
    fn pure_gpu_read_only_and_wc_gpu_rw_drop_completely() {
        // High synthetic addresses isolated from other tests.
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

        assert!(filter_range(base, base + 16 * m, true, false).is_empty());
        assert_eq!(filter_range(base, base + 16 * m, false, false), vec![(base, base + 16 * m)]);

        crate::watch::clear_console_memory(base, 16 * m);
    }

    #[test]
    fn mixed_174mb_super_buffer_clips_strictly_to_zone_5() {
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

        let kept = filter_range(base, base + total, true, false);
        assert_eq!(kept, vec![(base + z3_len + z4_len, base + total)]);
        // Disabled: keeps the entire 174 MiB span untouched.
        assert_eq!(filter_range(base, base + total, false, false), vec![(base, base + total)]);

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
        // [6M..8M): WriteCombinedGpuRw (DROP!)
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

        let kept = filter_range(base, base + 10 * m, true, false);
        assert_eq!(kept, vec![(base, base + 6 * m), (base + 8 * m, base + 10 * m)]);

        crate::watch::clear_console_memory(base, 10 * m);
    }
}
