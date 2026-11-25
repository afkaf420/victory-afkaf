//! Opt-in, build-verified runtime hooks for unlocking frame rates while
//! preserving real-time simulation speed (clearing fixed min-delta-time floor
//! clamping and CPU software frame limiters while keeping the engine's own
//! hard min/max delta-time bounds intact).
//!
//! All patches are strictly opt-in (`VICTORY_UNLOCK_FPS=1` or
//! `VICTORY_TARGET_FPS=<N>`), gated on the executable's BLAKE3 digest, and
//! byte-verified at every patch site before any memory is modified.

const MIN_TARGET_FPS: u32 = 1;
const MAX_TARGET_FPS: u32 = 1_000;

/// BLAKE3 digest of console-x64 build 1.0.2.1.
pub const BUILD_1_0_2_1_BLAKE3: &str =
    "b99f876b74d4a77f03af5f878223b27bd647b5c35a79d5decd9f716504239e93";

/// Configuration parsed from `VICTORY_UNLOCK_FPS` and `VICTORY_TARGET_FPS`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UnlockConfig {
    /// Unlock frame pacing (`VICTORY_UNLOCK_FPS=1` or `true`), disabling
    /// console min-dt floor clamping and engine busy-spin limiters.
    pub unlock_fps: bool,
    /// Optional target FPS (`VICTORY_TARGET_FPS=1..=1000`). Defaults to 1000
    /// when `unlock_fps` is set without an explicit cap.
    pub target_fps: Option<u32>,
}

impl UnlockConfig {
    /// Parses `VICTORY_UNLOCK_FPS` and `VICTORY_TARGET_FPS` from raw values.
    pub fn parse(unlock: Option<&str>, target: Option<&str>) -> Self {
        let unlock_fps = unlock.is_some_and(|v| {
            let t = v.trim();
            t == "1" || t.eq_ignore_ascii_case("true")
        });
        let target_fps = target
            .and_then(|v| v.trim().parse::<u32>().ok())
            .filter(|&fps| (MIN_TARGET_FPS..=MAX_TARGET_FPS).contains(&fps));
        Self {
            unlock_fps,
            target_fps,
        }
    }

    /// Reads the active configuration from the process environment.
    pub fn from_env() -> Self {
        let unlock = std::env::var("VICTORY_UNLOCK_FPS").ok();
        let target = std::env::var("VICTORY_TARGET_FPS").ok();
        Self::parse(unlock.as_deref(), target.as_deref())
    }

    /// True when any runtime frame-rate hook has been requested.
    pub const fn is_active(self) -> bool {
        self.unlock_fps || self.target_fps.is_some()
    }

    /// Target FPS fed into the engine's own interval calculation (1..=1000).
    pub const fn effective_target_fps(self) -> u32 {
        match self.target_fps {
            Some(fps) if fps >= MIN_TARGET_FPS && fps <= MAX_TARGET_FPS => fps,
            _ => MAX_TARGET_FPS,
        }
    }
}

/// One byte-verified code patch within a mapped executable image.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CodePatch {
    /// Short identifier for logging and diagnostics.
    pub name: &'static str,
    /// Relative virtual address within the executable image.
    pub rva: u32,
    /// Exact original instruction bytes that must be present before patching.
    pub expected: &'static [u8],
    /// Replacement bytes of identical length.
    pub replacement: Vec<u8>,
}

/// Error returned when a patch site does not match the expected image bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PatchMismatch {
    pub name: &'static str,
    pub rva: u32,
}

impl std::fmt::Display for PatchMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "byte signature mismatch for hook '{}' at RVA {:#x}",
            self.name, self.rva
        )
    }
}

impl std::error::Error for PatchMismatch {}

/// Builds the ordered list of byte-verified patches for `blake3_hex` under
/// `cfg`, or `None` if unlocking is disabled or the build is unknown.
pub fn plan(blake3_hex: &str, cfg: UnlockConfig) -> Option<Vec<CodePatch>> {
    if !cfg.is_active() || !blake3_hex.eq_ignore_ascii_case(BUILD_1_0_2_1_BLAKE3) {
        return None;
    }

    let fps = cfg.effective_target_fps();
    let fps_le = fps.to_le_bytes();
    let fps_f32_le = (fps as f32).to_bits().to_le_bytes();

    // 1. Timer init: clear the min-dt floor-clamp flag (`0` instead of `sete`)
    //    so frame deltas fall through to the engine's 1/3000 s floor while
    //    preserving the 1/15 s upper clamp and sub-millisecond accumulator.
    //    `mov byte ptr [rip + 0x522db17], 0` keeps the exact 7-byte length and
    //    RIP-relative displacement of `sete byte ptr [rip + 0x522db17]`.
    let mut patches = vec![
        CodePatch {
            name: "timer_init_min_dt_clamp",
            rva: 0x0222_61A2,
            expected: &[0x0F, 0x94, 0x05, 0x17, 0xDB, 0x22, 0x05],
            replacement: vec![0xC6, 0x05, 0x17, 0xDB, 0x22, 0x05, 0x00],
        },
        // 2. Prevent runtime configuration callbacks from re-enabling the floor clamp.
        CodePatch {
            name: "timer_clamp_setter",
            rva: 0x0222_6360,
            expected: &[0x88, 0x0D, 0x5A, 0xD9, 0x22, 0x05, 0xC3],
            replacement: vec![0xC3, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90],
        },
        // 3. Keep the CPU software frame-limiter toggle disabled (`0` instead of `1`)
        //    so it does not busy-spin against host frame pacing.
        CodePatch {
            name: "sw_frame_limiter_enable",
            rva: 0x05EA_7CD0,
            expected: &[0xC6, 0x05, 0xF2, 0xC1, 0x5A, 0x01, 0x01, 0xC3],
            replacement: vec![0xC6, 0x05, 0xF2, 0xC1, 0x5A, 0x01, 0x00, 0xC3],
        },
        // 4. Feed `fps` directly into both engine target-interval writers
        //    (`mov eax, fps; nop` replacing `mov eax, [reg + 0x210]`).
        CodePatch {
            name: "target_interval_writer_a",
            rva: 0x01E0_FC62,
            expected: &[0x8B, 0x87, 0x10, 0x02, 0x00, 0x00],
            replacement: vec![0xB8, fps_le[0], fps_le[1], fps_le[2], fps_le[3], 0x90],
        },
        CodePatch {
            name: "target_interval_writer_b",
            rva: 0x0224_359A,
            expected: &[0x8B, 0x86, 0x10, 0x02, 0x00, 0x00],
            replacement: vec![0xB8, fps_le[0], fps_le[1], fps_le[2], fps_le[3], 0x90],
        },
        // 5. Return `(float)fps` from the target-FPS getter so derived `1 / fps`
        //    timings stay consistent (`mov eax, bits; movd xmm0, eax; ret; nop`).
        CodePatch {
            name: "target_fps_getter",
            rva: 0x0224_1FD0,
            expected: &[
                0x56, 0x57, 0x48, 0x83, 0xEC, 0x28, 0xE8, 0xD5, 0xA1, 0xE0, 0x02,
            ],
            replacement: vec![
                0xB8,
                fps_f32_le[0],
                fps_f32_le[1],
                fps_f32_le[2],
                fps_f32_le[3],
                0x66,
                0x0F,
                0x6E,
                0xC0,
                0xC3,
                0x90,
            ],
        },
    ];

    // 6. When unlocked/VRR mode is active, disable the engine's QPC busy-spin
    //    wait helper so host presentation and `frame.rs` own pacing cleanly.
    if cfg.unlock_fps {
        patches.push(CodePatch {
            name: "qpc_busy_spin_wait",
            rva: 0x0248_2BC0,
            expected: &[
                0x56, 0x57, 0x53, 0x48, 0x83, 0xEC, 0x40, 0xC5, 0xF8, 0x29, 0x74, 0x24, 0x30,
            ],
            replacement: vec![
                0xC3, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90,
            ],
        });
    }

    Some(patches)
}

/// Checks that every patch fits within `image` and matches `expected` bytes
/// before mutating anything, then applies all patches transactionally.
pub fn apply_to_slice(image: &mut [u8], patches: &[CodePatch]) -> Result<usize, PatchMismatch> {
    for p in patches {
        let start = p.rva as usize;
        let end = start.checked_add(p.expected.len()).ok_or(PatchMismatch {
            name: p.name,
            rva: p.rva,
        })?;
        if p.expected.len() != p.replacement.len() || image.get(start..end) != Some(p.expected) {
            return Err(PatchMismatch {
                name: p.name,
                rva: p.rva,
            });
        }
    }
    for p in patches {
        let start = p.rva as usize;
        let end = start + p.replacement.len();
        image[start..end].copy_from_slice(&p.replacement);
    }
    Ok(patches.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unlock_config_parses_env_values() {
        let off = UnlockConfig::parse(None, None);
        assert!(!off.is_active());
        assert_eq!(off.effective_target_fps(), 1000);

        let unlocked = UnlockConfig::parse(Some("1"), None);
        assert!(unlocked.is_active());
        assert!(unlocked.unlock_fps);
        assert_eq!(unlocked.effective_target_fps(), 1000);

        let capped = UnlockConfig::parse(Some("false"), Some("120"));
        assert!(capped.is_active());
        assert!(!capped.unlock_fps);
        assert_eq!(capped.effective_target_fps(), 120);

        let both = UnlockConfig::parse(Some("TRUE"), Some("144"));
        assert!(both.is_active());
        assert!(both.unlock_fps);
        assert_eq!(both.effective_target_fps(), 144);

        let invalid = UnlockConfig::parse(Some("0"), Some("0"));
        assert!(!invalid.is_active());
        let too_high = UnlockConfig::parse(None, Some("1001"));
        assert!(!too_high.is_active());
    }

    #[test]
    fn plan_gates_on_build_hash_and_active_flag() {
        let off = UnlockConfig::default();
        assert!(plan(BUILD_1_0_2_1_BLAKE3, off).is_none());

        let on = UnlockConfig {
            unlock_fps: true,
            target_fps: Some(120),
        };
        assert!(plan("0000000000000000000000000000000000000000000000000000000000000000", on).is_none());

        let patches = plan(&BUILD_1_0_2_1_BLAKE3.to_ascii_uppercase(), on).unwrap();
        assert_eq!(patches.len(), 7);
        for p in &patches {
            assert_eq!(p.expected.len(), p.replacement.len(), "patch {}", p.name);
        }
        // Verify exact encoded target FPS (120) and float bits (120.0f).
        let writer = patches
            .iter()
            .find(|p| p.name == "target_interval_writer_a")
            .unwrap();
        assert_eq!(&writer.replacement[1..5], &120u32.to_le_bytes());
        let getter = patches.iter().find(|p| p.name == "target_fps_getter").unwrap();
        assert_eq!(&getter.replacement[1..5], &120.0f32.to_bits().to_le_bytes());

        // Target-only mode omits the QPC spin-wait patch.
        let target_only = UnlockConfig {
            unlock_fps: false,
            target_fps: Some(60),
        };
        let patches_60 = plan(BUILD_1_0_2_1_BLAKE3, target_only).unwrap();
        assert_eq!(patches_60.len(), 6);
        assert!(patches_60.iter().all(|p| p.name != "qpc_busy_spin_wait"));
    }

    #[test]
    fn apply_to_slice_is_transactional_and_byte_verified() {
        let patches = vec![
            CodePatch {
                name: "first",
                rva: 4,
                expected: &[0x0F, 0x94, 0x05],
                replacement: vec![0xC6, 0x05, 0x00],
            },
            CodePatch {
                name: "second",
                rva: 12,
                expected: &[0x88, 0x0D, 0xC3],
                replacement: vec![0xC3, 0x90, 0x90],
            },
        ];

        // Mismatch on second patch leaves first patch untouched.
        let mut buf = vec![0xAAu8; 32];
        buf[4..7].copy_from_slice(&[0x0F, 0x94, 0x05]);
        buf[12..15].copy_from_slice(&[0x88, 0x0D, 0xFF]);
        let before = buf.clone();
        let err = apply_to_slice(&mut buf, &patches).unwrap_err();
        assert_eq!(err.name, "second");
        assert_eq!(err.rva, 12);
        assert_eq!(buf, before, "failed verification must not mutate memory");

        // Matching bytes apply cleanly.
        buf[12..15].copy_from_slice(&[0x88, 0x0D, 0xC3]);
        assert_eq!(apply_to_slice(&mut buf, &patches).unwrap(), 2);
        assert_eq!(&buf[4..7], &[0xC6, 0x05, 0x00]);
        assert_eq!(&buf[12..15], &[0xC3, 0x90, 0x90]);
    }
}
