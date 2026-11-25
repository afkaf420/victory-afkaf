//! Applies opt-in, build-verified frame-rate unlock hooks from
//! [`victory_game::frame_unlock`] to a freshly mapped executable image before
//! any title code runs.

use std::ffi::c_void;

use anyhow::{Result, bail};
use victory_game::frame_unlock::{self, CodePatch, UnlockConfig};
use windows_sys::Win32::System::Diagnostics::Debug::FlushInstructionCache;
use windows_sys::Win32::System::Memory::{PAGE_EXECUTE_READWRITE, VirtualProtect};
use windows_sys::Win32::System::Threading::GetCurrentProcess;

/// Applies opt-in frame-rate unlock patches to the mapped image at `base` if
/// `VICTORY_UNLOCK_FPS=1` or `VICTORY_TARGET_FPS=<N>` is set and
/// `original_hash` matches a known build. Returns the number of patches
/// applied.
///
/// # Safety
/// `base..base + image_size` must be a mapped executable image whose code has
/// not started executing yet.
pub unsafe fn apply_if_enabled(
    base: usize,
    image_size: usize,
    original_hash: &blake3::Hash,
) -> Result<usize> {
    let cfg = UnlockConfig::from_env();
    // SAFETY: forwarded from caller.
    unsafe { apply_with_config(base, image_size, &original_hash.to_hex(), cfg) }
}

/// Same as [`apply_if_enabled`] with an explicit [`UnlockConfig`] (for tests).
///
/// # Safety
/// Same requirements as [`apply_if_enabled`].
pub unsafe fn apply_with_config(
    base: usize,
    image_size: usize,
    blake3_hex: &str,
    cfg: UnlockConfig,
) -> Result<usize> {
    if !cfg.is_active() {
        return Ok(0);
    }
    let Some(patches) = frame_unlock::plan(blake3_hex, cfg) else {
        tracing::warn!(
            blake3 = blake3_hex,
            unlock_fps = cfg.unlock_fps,
            target_fps = ?cfg.target_fps,
            "frame-rate unlock requested, but no hook profile matches this build"
        );
        return Ok(0);
    };
    // SAFETY: forwarded from caller.
    let count = unsafe { apply_patches(base, image_size, &patches) }?;
    tracing::info!(
        applied = count,
        unlock_fps = cfg.unlock_fps,
        target_fps = cfg.effective_target_fps(),
        "applied opt-in simulation-safe frame-rate unlock hooks"
    );
    Ok(count)
}

/// Verifies every patch's exact byte signature inside `base..base + image_size`
/// before modifying any page, then writes each patch atomically with
/// `VirtualProtect` + `FlushInstructionCache`.
///
/// # Safety
/// `base..base + image_size` must be mapped, readable memory for an image that
/// is not yet executing.
pub unsafe fn apply_patches(base: usize, image_size: usize, patches: &[CodePatch]) -> Result<usize> {
    // Phase 1: verify every site before writing a single byte.
    for p in patches {
        let len = p.expected.len();
        if len == 0 || len != p.replacement.len() {
            bail!("invalid patch length for '{}'", p.name);
        }
        let end = (p.rva as usize)
            .checked_add(len)
            .filter(|&e| e <= image_size)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "hook '{}' at RVA {:#x} ({len} B) lies outside image size {image_size:#x}",
                    p.name,
                    p.rva
                )
            })?;
        let _ = end;
        // SAFETY: `p.rva..p.rva + len` is within `image_size` of the mapped image.
        let current = unsafe { std::slice::from_raw_parts((base + p.rva as usize) as *const u8, len) };
        if current != p.expected {
            bail!(
                "byte signature mismatch for hook '{}' at RVA {:#x}: expected {:02x?}, found {:02x?}",
                p.name,
                p.rva,
                p.expected,
                current
            );
        }
    }

    // Phase 2: patch each verified site and flush the instruction cache.
    // SAFETY: pseudo-handle for the current process.
    let process = unsafe { GetCurrentProcess() };
    for p in patches {
        let addr = (base + p.rva as usize) as *mut u8;
        let len = p.replacement.len();
        let mut old = 0u32;
        // SAFETY: verified inside the mapped image above.
        if unsafe { VirtualProtect(addr as *const c_void, len, PAGE_EXECUTE_READWRITE, &mut old) }
            == 0
        {
            bail!(
                "VirtualProtect(hook '{}' at RVA {:#x}) failed: {}",
                p.name,
                p.rva,
                std::io::Error::last_os_error()
            );
        }
        // SAFETY: writable inside the mapped image; lengths verified equal above.
        unsafe { std::ptr::copy_nonoverlapping(p.replacement.as_ptr(), addr, len) };
        // SAFETY: current process and modified code range.
        unsafe { FlushInstructionCache(process, addr as *const c_void, len) };
        let mut ignored = 0u32;
        // SAFETY: restoring original page protection.
        unsafe { VirtualProtect(addr as *const c_void, len, old, &mut ignored) };
    }

    Ok(patches.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_EXECUTE_READ, PAGE_READWRITE, VirtualAlloc,
        VirtualFree,
    };

    #[test]
    fn mapped_code_pages_are_verified_before_patching() {
        let size = 4096usize;
        // SAFETY: private allocation owned by this test.
        let mem = unsafe { VirtualAlloc(std::ptr::null(), size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE) };
        assert!(!mem.is_null());
        let base = mem as usize;

        let patches = vec![
            CodePatch {
                name: "site_a",
                rva: 0x10,
                expected: &[0x0F, 0x94, 0x05, 0x17, 0xDB, 0x22, 0x05],
                replacement: vec![0xC6, 0x05, 0x17, 0xDB, 0x22, 0x05, 0x00],
            },
            CodePatch {
                name: "site_b",
                rva: 0x40,
                expected: &[0x88, 0x0D, 0x5A, 0xD9, 0x22, 0x05, 0xC3],
                replacement: vec![0xC3, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90],
            },
        ];

        // Seed only site_a, leave site_b mismatched, and mark page RX.
        // SAFETY: `mem` is writable for `size` bytes.
        unsafe {
            std::ptr::copy_nonoverlapping(patches[0].expected.as_ptr(), (base + 0x10) as *mut u8, 7);
            let mut old = 0;
            assert_ne!(VirtualProtect(mem, size, PAGE_EXECUTE_READ, &mut old), 0);
        }

        // Mismatch must fail transactionally without touching site_a.
        // SAFETY: `base..base + size` is mapped and not executing.
        assert!(unsafe { apply_patches(base, size, &patches) }.is_err());
        // SAFETY: readable page.
        let site_a = unsafe { std::slice::from_raw_parts((base + 0x10) as *const u8, 7) };
        assert_eq!(site_a, patches[0].expected);

        // Now seed site_b properly, switch back to RX, and apply.
        // SAFETY: test-owned mapping.
        unsafe {
            let mut old = 0;
            assert_ne!(VirtualProtect(mem, size, PAGE_READWRITE, &mut old), 0);
            std::ptr::copy_nonoverlapping(patches[1].expected.as_ptr(), (base + 0x40) as *mut u8, 7);
            assert_ne!(VirtualProtect(mem, size, PAGE_EXECUTE_READ, &mut old), 0);
        }
        // SAFETY: `base..base + size` is mapped and not executing.
        assert_eq!(unsafe { apply_patches(base, size, &patches) }.unwrap(), 2);
        // SAFETY: readable page.
        let site_a_after = unsafe { std::slice::from_raw_parts((base + 0x10) as *const u8, 7) };
        let site_b_after = unsafe { std::slice::from_raw_parts((base + 0x40) as *const u8, 7) };
        assert_eq!(site_a_after, patches[0].replacement.as_slice());
        assert_eq!(site_b_after, patches[1].replacement.as_slice());

        // Unknown build hash is a safe no-op even when enabled.
        let cfg = UnlockConfig {
            unlock_fps: true,
            target_fps: Some(120),
        };
        // SAFETY: mapped buffer.
        assert_eq!(unsafe { apply_with_config(base, size, "deadbeef", cfg) }.unwrap(), 0);

        // SAFETY: releasing the test allocation.
        unsafe { VirtualFree(mem, 0, MEM_RELEASE) };
    }
}
