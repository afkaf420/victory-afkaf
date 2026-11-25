//! `amd_fidelityfx_loader_xbox.dll`: PC-native upscaler HLE bridge.
//!
//! Intercepts the four static FidelityFX API imports (`ffxCreateContext`,
//! `ffxDestroyContext`, `ffxQuery`, `ffxDispatch`) and routes upscaling to a
//! PC-native runtime on the host D3D12 device instead of loading the console's
//! fixed-resolution upscaler DLLs:
//!
//! - **Backend A (PC AMD FSR 3.1 / 4.x):** dynamically loads
//!   `amd_fidelityfx_loader_dx12.dll` (and sibling
//!   `amd_fidelityfx_upscaler_dx12.dll`).
//! - **Backend B (NVIDIA DLSS / DLAA via Streamline):** dynamically loads
//!   `sl.interposer.dll` + `sl.dlss.dll` (`slInit`, `slSetD3DDevice`,
//!   `slSetConstants`, `slSetTagForFrame`, `slEvaluateFeature`).
//!
//! Environment switches:
//! - `VICTORY_FIDELITYFX=auto|fsr3|dlss|console` (default: `auto`; `console`
//!   omits this HLE module so the title's shipped DLLs load untouched; `auto`
//!   prefers DLSS when Streamline DLLs are found, then PC FSR 3.1, else falls
//!   back cleanly to the console DLLs with a setup log hint).
//! - `VICTORY_FIDELITYFX_DIR=<path>` (directory containing PC upscaler DLLs;
//!   also probes `<exe_dir>/fidelityfx`, `<exe_dir>`, sibling `data/fidelityfx`,
//!   and `%LOCALAPPDATA%/victory/fidelityfx`).
//! - `VICTORY_UPSCALE_OUTPUT=<W>x<H>` (optional output resolution override,
//!   e.g. `2560x1440` or `3840x2160`, lifting console output resolution limits).
//! - `VICTORY_UPSCALE_MODE=dlaa|quality|balanced|performance|ultra_performance`
//!   (optional quality-mode override).
#![allow(non_snake_case)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::{c_char, c_void};
use std::mem::offset_of;
use std::path::{Path, PathBuf};
use std::ptr::{null, null_mut};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, OnceLock};

use victory_gpu::upscaler_bridge::{
    TextureDesc, host_command_list_prepare, host_device_raw, host_texture_for_upscaler,
    host_upscaler_post_dispatch,
};

use crate::{HleModule, hle_export};

// ---------------------------------------------------------------------------
// FidelityFX API return codes, descriptor type IDs, enums, and ABI structs
// ---------------------------------------------------------------------------

pub const FFX_API_RETURN_OK: u32 = 0;
pub const FFX_API_RETURN_ERROR: u32 = 1;
pub const FFX_API_RETURN_ERROR_UNKNOWN_DESCTYPE: u32 = 2;
pub const FFX_API_RETURN_ERROR_RUNTIME_ERROR: u32 = 3;
pub const FFX_API_RETURN_NO_PROVIDER: u32 = 4;
pub const FFX_API_RETURN_ERROR_PARAMETER: u32 = 6;

pub const FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12: u64 = 0x0000_0002;
pub const FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_CONSOLE_ALT: u64 = 0x0100_0002;
pub const FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_XBOX: u64 = 0x0200_0002;
pub const FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12_ALLOC_CB: u64 = 0x0000_0003;
pub const FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_CONSOLE_ALLOC_CB: u64 = 0x0100_0003;
pub const FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_XBOX_ALLOC_CB: u64 = 0x0200_0003;

pub const FFX_API_QUERY_DESC_TYPE_GET_VERSIONS: u64 = 0x0000_0004;
pub const FFX_API_DESC_TYPE_OVERRIDE_VERSION: u64 = 0x0000_0005;
pub const FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION: u64 = 0x0000_0006;

pub const FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE: u64 = 0x0001_0000;
pub const FFX_API_DISPATCH_DESC_TYPE_UPSCALE: u64 = 0x0001_0001;
pub const FFX_API_QUERY_DESC_TYPE_UPSCALE_GETUPSCALERATIOFROMQUALITYMODE: u64 = 0x0001_0002;
pub const FFX_API_QUERY_DESC_TYPE_UPSCALE_GETRENDERRESOLUTIONFROMQUALITYMODE: u64 = 0x0001_0003;
pub const FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTERPHASECOUNT: u64 = 0x0001_0004;
pub const FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTEROFFSET: u64 = 0x0001_0005;
pub const FFX_API_DISPATCH_DESC_TYPE_UPSCALE_GENERATEREACTIVEMASK: u64 = 0x0001_0006;
pub const FFX_API_QUERY_DESC_TYPE_UPSCALE_GPU_MEMORY_USAGE: u64 = 0x0001_0008;
pub const FFX_API_QUERY_DESC_TYPE_UPSCALE_GPU_MEMORY_USAGE_V2: u64 = 0x0001_0009;
pub const FFX_API_QUERY_DESC_TYPE_UPSCALE_GET_RESOURCE_REQUIREMENTS: u64 = 0x0001_000A;
pub const FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE_VERSION: u64 = 0x0001_000B;

pub const FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE: u32 = 1 << 0;
pub const FFX_UPSCALE_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS: u32 = 1 << 1;
pub const FFX_UPSCALE_ENABLE_MOTION_VECTORS_JITTER_CANCELLATION: u32 = 1 << 2;
pub const FFX_UPSCALE_ENABLE_DEPTH_INVERTED: u32 = 1 << 3;
pub const FFX_UPSCALE_ENABLE_DEPTH_INFINITE: u32 = 1 << 4;
pub const FFX_UPSCALE_ENABLE_AUTO_EXPOSURE: u32 = 1 << 5;

pub const FFX_API_RESOURCE_STATE_COMMON: u32 = 1 << 0;
pub const FFX_API_RESOURCE_STATE_UNORDERED_ACCESS: u32 = 1 << 1;
pub const FFX_API_RESOURCE_STATE_COMPUTE_READ: u32 = 1 << 2;
pub const FFX_API_RESOURCE_STATE_PIXEL_READ: u32 = 1 << 3;
pub const FFX_API_RESOURCE_STATE_COPY_SRC: u32 = 1 << 4;
pub const FFX_API_RESOURCE_STATE_COPY_DEST: u32 = 1 << 5;
pub const FFX_API_RESOURCE_STATE_INDIRECT_ARGUMENT: u32 = 1 << 6;
pub const FFX_API_RESOURCE_STATE_PRESENT: u32 = 1 << 7;
pub const FFX_API_RESOURCE_STATE_RENDER_TARGET: u32 = 1 << 8;
pub const FFX_API_RESOURCE_STATE_DEPTH_ATTACHMENT: u32 = 1 << 9;

// D3D12_RESOURCE_STATES bitmasks used with victory_gpu::upscaler_bridge.
const D3D12_STATE_COMMON: u32 = 0x0;
const D3D12_STATE_RENDER_TARGET: u32 = 0x4;
const D3D12_STATE_UNORDERED_ACCESS: u32 = 0x8;
const D3D12_STATE_DEPTH_WRITE: u32 = 0x10;
const D3D12_STATE_NON_PIXEL_SHADER_RESOURCE: u32 = 0x40;
const D3D12_STATE_PIXEL_SHADER_RESOURCE: u32 = 0x80;
const D3D12_STATE_INDIRECT_ARGUMENT: u32 = 0x200;
const D3D12_STATE_COPY_DEST: u32 = 0x400;
const D3D12_STATE_COPY_SOURCE: u32 = 0x800;

type FfxApiMessage = extern "system" fn(msg_type: u32, message: *const u16);

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FfxApiHeader {
    pub type_: u64,
    pub p_next: *mut FfxApiHeader,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FfxApiDimensions2D {
    pub width: u32,
    pub height: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FfxApiFloatCoords2D {
    pub x: f32,
    pub y: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FfxApiResourceDescription {
    pub type_: u32,
    pub format: u32,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub mip_count: u32,
    pub flags: u32,
    pub usage: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FfxApiResource {
    pub resource: *mut c_void,
    pub description: FfxApiResourceDescription,
    pub state: u32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FfxCreateContextDescUpscale {
    pub header: FfxApiHeader,
    pub flags: u32,
    pub max_render_size: FfxApiDimensions2D,
    pub max_upscale_size: FfxApiDimensions2D,
    pub _pad: u32,
    pub fp_message: Option<FfxApiMessage>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FfxCreateBackendDX12Desc {
    pub header: FfxApiHeader,
    pub device: *mut c_void,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FfxDispatchDescUpscale {
    pub header: FfxApiHeader,
    pub command_list: *mut c_void,
    pub color: FfxApiResource,
    pub depth: FfxApiResource,
    pub motion_vectors: FfxApiResource,
    pub exposure: FfxApiResource,
    pub reactive: FfxApiResource,
    pub transparency_and_composition: FfxApiResource,
    pub output: FfxApiResource,
    pub jitter_offset: FfxApiFloatCoords2D,
    pub motion_vector_scale: FfxApiFloatCoords2D,
    pub render_size: FfxApiDimensions2D,
    pub upscale_size: FfxApiDimensions2D,
    pub enable_sharpening: u8,
    pub _pad0: [u8; 3],
    pub sharpness: f32,
    pub frame_time_delta: f32,
    pub pre_exposure: f32,
    pub reset: u8,
    pub _pad1: [u8; 3],
    pub camera_near: f32,
    pub camera_far: f32,
    pub camera_fov_angle_vertical: f32,
    pub view_space_to_meters_factor: f32,
    pub flags: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FfxDispatchDescGenerateReactiveMask {
    pub header: FfxApiHeader,
    pub command_list: *mut c_void,
    pub color_opaque_only: FfxApiResource,
    pub color_pre_upscale: FfxApiResource,
    pub out_reactive: FfxApiResource,
    pub render_size: FfxApiDimensions2D,
    pub scale: f32,
    pub cutoff_threshold: f32,
    pub binary_value: f32,
    pub flags: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FfxQueryDescGetVersions {
    pub header: FfxApiHeader,
    pub create_desc_type: u64,
    pub device: *mut c_void,
    pub output_count: *mut u64,
    pub version_ids: *mut u64,
    pub version_names: *mut *const c_char,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FfxQueryGetProviderVersion {
    pub header: FfxApiHeader,
    pub version_id: u64,
    pub version_name: *const c_char,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FfxQueryDescUpscaleGetRatio {
    pub header: FfxApiHeader,
    pub quality_mode: u32,
    pub _pad: u32,
    pub out_upscale_ratio: *mut f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FfxQueryDescUpscaleGetRenderRes {
    pub header: FfxApiHeader,
    pub display_width: u32,
    pub display_height: u32,
    pub quality_mode: u32,
    pub _pad: u32,
    pub out_render_width: *mut u32,
    pub out_render_height: *mut u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FfxQueryDescUpscaleGetJitterPhaseCount {
    pub header: FfxApiHeader,
    pub render_width: u32,
    pub display_width: u32,
    pub out_phase_count: *mut i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FfxQueryDescUpscaleGetJitterOffset {
    pub header: FfxApiHeader,
    pub index: i32,
    pub phase_count: i32,
    pub out_x: *mut f32,
    pub out_y: *mut f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct FfxApiEffectMemoryUsage {
    pub total_usage_in_bytes: u64,
    pub aliasable_usage_in_bytes: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FfxQueryDescUpscaleGetGpuMemoryUsage {
    pub header: FfxApiHeader,
    pub gpu_memory_usage_upscaler: *mut FfxApiEffectMemoryUsage,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FfxQueryDescUpscaleGetGpuMemoryUsageV2 {
    pub header: FfxApiHeader,
    pub device: *mut c_void,
    pub max_render_size: FfxApiDimensions2D,
    pub max_upscale_size: FfxApiDimensions2D,
    pub flags: u32,
    pub _pad: u32,
    pub gpu_memory_usage_upscaler: *mut FfxApiEffectMemoryUsage,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FfxQueryDescUpscaleGetResourceRequirements {
    pub header: FfxApiHeader,
    pub required_resources: u64,
    pub optional_resources: u64,
}

const _: () = {
    assert!(size_of::<FfxApiHeader>() == 0x10);
    assert!(size_of::<FfxApiResourceDescription>() == 0x20);
    assert!(size_of::<FfxApiResource>() == 0x30);
    assert!(offset_of!(FfxApiResource, state) == 0x28);
    assert!(size_of::<FfxCreateContextDescUpscale>() == 0x30);
    assert!(size_of::<FfxCreateBackendDX12Desc>() == 0x18);
    assert!(size_of::<FfxDispatchDescUpscale>() == 0x1B0);
    assert!(offset_of!(FfxDispatchDescUpscale, command_list) == 0x10);
    assert!(offset_of!(FfxDispatchDescUpscale, color) == 0x18);
    assert!(offset_of!(FfxDispatchDescUpscale, output) == 0x138);
    assert!(offset_of!(FfxDispatchDescUpscale, jitter_offset) == 0x168);
    assert!(offset_of!(FfxDispatchDescUpscale, enable_sharpening) == 0x188);
    assert!(offset_of!(FfxDispatchDescUpscale, reset) == 0x198);
    assert!(offset_of!(FfxDispatchDescUpscale, flags) == 0x1AC);
};

// ---------------------------------------------------------------------------
// Configuration parsing (`VICTORY_FIDELITYFX*`, `VICTORY_UPSCALE_*`)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendMode {
    Auto,
    Fsr3,
    Dlss,
    Console,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UpscaleQuality {
    Dlaa,
    Quality,
    Balanced,
    Performance,
    UltraPerformance,
}

impl UpscaleQuality {
    pub const fn ffx_mode(self) -> u32 {
        match self {
            Self::Dlaa => 0,
            Self::Quality => 1,
            Self::Balanced => 2,
            Self::Performance => 3,
            Self::UltraPerformance => 4,
        }
    }

    pub const fn sl_dlss_mode(self) -> u32 {
        match self {
            Self::Dlaa => 6,             // DLSSMode::eDLAA
            Self::Quality => 3,          // DLSSMode::eMaxQuality
            Self::Balanced => 2,         // DLSSMode::eBalanced
            Self::Performance => 1,      // DLSSMode::eMaxPerformance
            Self::UltraPerformance => 4, // DLSSMode::eUltraPerformance
        }
    }

    pub fn from_ratio(ratio: f32) -> Self {
        if ratio <= 1.05 {
            Self::Dlaa
        } else if ratio <= 1.60 {
            Self::Quality
        } else if ratio <= 1.85 {
            Self::Balanced
        } else if ratio <= 2.35 {
            Self::Performance
        } else {
            Self::UltraPerformance
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UpscalerConfig {
    pub mode: BackendMode,
    pub dir: Option<PathBuf>,
    pub output_override: Option<(u32, u32)>,
    pub quality_override: Option<UpscaleQuality>,
}

impl UpscalerConfig {
    pub fn parse(
        mode_raw: Option<&str>,
        dir_raw: Option<&str>,
        output_raw: Option<&str>,
        quality_raw: Option<&str>,
    ) -> Self {
        let mode = match mode_raw.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
            Some("console" | "off" | "0") => BackendMode::Console,
            Some("fsr3" | "fsr" | "amd") => BackendMode::Fsr3,
            Some("dlss" | "dlaa" | "streamline" | "nvidia") => BackendMode::Dlss,
            _ => BackendMode::Auto,
        };
        let dir = dir_raw
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(PathBuf::from);
        let output_override = parse_output_resolution(output_raw);
        let quality_override = match quality_raw
            .map(|s| s.trim().to_ascii_lowercase())
            .as_deref()
        {
            Some("dlaa" | "native" | "nativeaa") => Some(UpscaleQuality::Dlaa),
            Some("quality") => Some(UpscaleQuality::Quality),
            Some("balanced") => Some(UpscaleQuality::Balanced),
            Some("performance") => Some(UpscaleQuality::Performance),
            Some("ultra_performance" | "ultraperformance") => {
                Some(UpscaleQuality::UltraPerformance)
            }
            _ => None,
        };
        Self {
            mode,
            dir,
            output_override,
            quality_override,
        }
    }

    pub fn from_env() -> Self {
        Self::parse(
            std::env::var("VICTORY_FIDELITYFX").ok().as_deref(),
            std::env::var("VICTORY_FIDELITYFX_DIR").ok().as_deref(),
            std::env::var("VICTORY_UPSCALE_OUTPUT").ok().as_deref(),
            std::env::var("VICTORY_UPSCALE_MODE").ok().as_deref(),
        )
    }

    /// True if `VICTORY_FIDELITYFX=console` explicitly opted out of HLE interception.
    pub const fn is_console_opt_out(&self) -> bool {
        matches!(self.mode, BackendMode::Console)
    }
}

/// Parses `<W>x<H>` (640..=7680 by 360..=4320) from `VICTORY_UPSCALE_OUTPUT`.
pub fn parse_output_resolution(raw: Option<&str>) -> Option<(u32, u32)> {
    let raw = raw?.trim();
    let (w, h) = raw.split_once(['x', 'X'])?;
    let w = w.trim().parse::<u32>().ok()?;
    let h = h.trim().parse::<u32>().ok()?;
    ((640..=7680).contains(&w) && (360..=4320).contains(&h)).then_some((w, h))
}

fn config() -> &'static UpscalerConfig {
    static CONFIG: OnceLock<UpscalerConfig> = OnceLock::new();
    CONFIG.get_or_init(UpscalerConfig::from_env)
}

// ---------------------------------------------------------------------------
// Analytical Halton(2, 3) jitter & quality-mode ratio helpers
// ---------------------------------------------------------------------------

/// Radical inverse of `index` in `base` (base 2 or base 3 for Halton sequences).
pub fn halton(mut index: i32, base: i32) -> f32 {
    if index <= 0 || base <= 1 {
        return 0.0;
    }
    let mut f = 1.0f32;
    let mut r = 0.0f32;
    let b = base as f32;
    while index > 0 {
        f /= b;
        r += f * (index % base) as f32;
        index /= base;
    }
    r
}

/// Standard FSR / DLSS jitter phase count: `ceil(8 * (displayW / renderW)^2)`.
pub fn jitter_phase_count(render_width: u32, display_width: u32) -> i32 {
    let rw = render_width.max(1) as f32;
    let dw = display_width.max(1) as f32;
    let ratio = dw / rw;
    (8.0f32 * ratio * ratio).ceil().max(1.0) as i32
}

/// Subpixel jitter offset `(x, y)` in `[-0.5, 0.5]` using Halton(2, 3).
pub fn jitter_offset(index: i32, phase_count: i32) -> (f32, f32) {
    let phases = phase_count.max(1);
    let sample = index.rem_euclid(phases) + 1;
    (halton(sample, 2) - 0.5, halton(sample, 3) - 0.5)
}

/// Per-dimension upscaling ratio for `FfxApiUpscaleQualityMode`.
pub fn upscale_ratio_from_quality_mode(quality_mode: u32) -> Option<f32> {
    match quality_mode {
        0 => Some(1.0), // NativeAA / DLAA
        1 => Some(1.5), // Quality
        2 => Some(1.7), // Balanced
        3 => Some(2.0), // Performance
        4 => Some(3.0), // Ultra Performance
        _ => None,
    }
}

/// Render resolution `(w, h)` from display dimensions and quality mode.
pub fn render_resolution_from_quality_mode(
    display_width: u32,
    display_height: u32,
    quality_mode: u32,
) -> Option<(u32, u32)> {
    let ratio = upscale_ratio_from_quality_mode(quality_mode)?;
    let rw = ((display_width as f32) / ratio).round().max(1.0) as u32;
    let rh = ((display_height as f32) / ratio).round().max(1.0) as u32;
    Some((rw, rh))
}

// ---------------------------------------------------------------------------
// `ffxApiHeader` pNext chain translation (console -> PC DX12)
// ---------------------------------------------------------------------------

/// Owned PC-compatible create-context descriptor chain produced from a console
/// `ffxCreateContext` call.
pub struct RewrittenCreateChain {
    pub upscale: FfxCreateContextDescUpscale,
    pub backend: FfxCreateBackendDX12Desc,
    pub stripped_nodes: usize,
    pub saw_backend_desc: bool,
    pub original_backend_type: Option<u64>,
}

impl RewrittenCreateChain {
    /// Links `self.upscale.header.p_next -> &mut self.backend.header` in place
    /// and returns the head pointer to hand to PC `ffxCreateContext`.
    pub fn link_head(&mut self) -> *mut FfxApiHeader {
        self.backend.header.p_next = null_mut();
        self.upscale.header.p_next = &raw mut self.backend.header;
        &raw mut self.upscale.header
    }
}

extern "system" fn ffx_log_callback(msg_type: u32, message: *const u16) {
    if message.is_null() {
        return;
    }
    // SAFETY: runtime passes a NUL-terminated UTF-16 message string.
    let mut len = 0usize;
    unsafe {
        while *message.add(len) != 0 && len < 1024 {
            len += 1;
        }
    }
    // SAFETY: bounded above by the NUL terminator.
    let text = String::from_utf16_lossy(unsafe { std::slice::from_raw_parts(message, len) });
    if msg_type == 0 {
        tracing::warn!(target: "victory_hle::fidelityfx", "{text}");
    } else {
        tracing::debug!(target: "victory_hle::fidelityfx", "{text}");
    }
}

/// Walks a console `ffxCreateContext` `pNext` chain, translates console DX12
/// backend descriptors (`0x0100_0002` and `0x0200_0002`) into PC
/// `FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12` (`0x0000_0002`) carrying
/// `host_device`, strips console-only override/version/allocator nodes, and
/// applies any `VICTORY_UPSCALE_OUTPUT` override.
///
/// # Safety
/// `head` must either be null or point to a valid `FfxApiHeader` linked list
/// matching the FidelityFX API C ABI.
pub unsafe fn rewrite_create_chain(
    head: *const FfxApiHeader,
    host_device: *mut c_void,
    cfg: &UpscalerConfig,
) -> Result<RewrittenCreateChain, u32> {
    if head.is_null() {
        return Err(FFX_API_RETURN_ERROR_PARAMETER);
    }

    let mut upscale: Option<FfxCreateContextDescUpscale> = None;
    let mut saw_backend_desc = false;
    let mut original_backend_type = None;
    let mut stripped_nodes = 0usize;

    let mut cur = head;
    for _ in 0..32 {
        if cur.is_null() {
            break;
        }
        // SAFETY: caller guarantees valid `FfxApiHeader` nodes in the chain.
        let hdr = unsafe { *cur };
        match hdr.type_ {
            FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE => {
                // SAFETY: `type_` identifies `FfxCreateContextDescUpscale`.
                let mut desc = unsafe { *(cur.cast::<FfxCreateContextDescUpscale>()) };
                desc.header.p_next = null_mut();
                if let Some((w, h)) = cfg.output_override {
                    desc.max_upscale_size = FfxApiDimensions2D {
                        width: w,
                        height: h,
                    };
                    desc.max_render_size.width = desc.max_render_size.width.min(w);
                    desc.max_render_size.height = desc.max_render_size.height.min(h);
                }
                desc.fp_message = Some(ffx_log_callback);
                upscale = Some(desc);
            }
            FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12
            | FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_CONSOLE_ALT
            | FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_XBOX => {
                saw_backend_desc = true;
                original_backend_type = Some(hdr.type_);
            }
            FFX_API_DESC_TYPE_OVERRIDE_VERSION
            | FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE_VERSION
            | FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12_ALLOC_CB
            | FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_CONSOLE_ALLOC_CB
            | FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_XBOX_ALLOC_CB => {
                stripped_nodes += 1;
            }
            other => {
                tracing::debug!(
                    desc_type = format_args!("{other:#x}"),
                    "stripping unrecognised create-context descriptor from FidelityFX pNext chain"
                );
                stripped_nodes += 1;
            }
        }
        cur = hdr.p_next.cast_const();
    }

    let upscale = upscale.ok_or(FFX_API_RETURN_ERROR_UNKNOWN_DESCTYPE)?;
    let backend = FfxCreateBackendDX12Desc {
        header: FfxApiHeader {
            type_: FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12,
            p_next: null_mut(),
        },
        device: host_device,
    };

    Ok(RewrittenCreateChain {
        upscale,
        backend,
        stripped_nodes,
        saw_backend_desc,
        original_backend_type,
    })
}

/// Converts a `FfxApiResourceState` bitmask to host `D3D12_RESOURCE_STATES`.
pub fn ffx_state_to_d3d12(state: u32) -> u32 {
    if state == 0 || state == FFX_API_RESOURCE_STATE_COMMON || state == FFX_API_RESOURCE_STATE_PRESENT {
        return D3D12_STATE_COMMON;
    }
    let mut out = 0u32;
    if state & FFX_API_RESOURCE_STATE_UNORDERED_ACCESS != 0 {
        out |= D3D12_STATE_UNORDERED_ACCESS;
    }
    if state & FFX_API_RESOURCE_STATE_COMPUTE_READ != 0 {
        out |= D3D12_STATE_NON_PIXEL_SHADER_RESOURCE;
    }
    if state & FFX_API_RESOURCE_STATE_PIXEL_READ != 0 {
        out |= D3D12_STATE_PIXEL_SHADER_RESOURCE;
    }
    if state & FFX_API_RESOURCE_STATE_COPY_SRC != 0 {
        out |= D3D12_STATE_COPY_SOURCE;
    }
    if state & FFX_API_RESOURCE_STATE_COPY_DEST != 0 {
        out |= D3D12_STATE_COPY_DEST;
    }
    if state & FFX_API_RESOURCE_STATE_INDIRECT_ARGUMENT != 0 {
        out |= D3D12_STATE_INDIRECT_ARGUMENT;
    }
    if state & FFX_API_RESOURCE_STATE_RENDER_TARGET != 0 {
        out |= D3D12_STATE_RENDER_TARGET;
    }
    if state & FFX_API_RESOURCE_STATE_DEPTH_ATTACHMENT != 0 {
        out |= D3D12_STATE_DEPTH_WRITE;
    }
    out
}

/// Converts a `DXGI_FORMAT` raw value to `FfxApiSurfaceFormat` (matching
/// `ffxApiGetSurfaceFormatDX12` in FidelityFX API).
pub fn dxgi_to_ffx_surface_format(dxgi: u32) -> u32 {
    match dxgi {
        1 => 1,   // R32G32B32A32_TYPELESS
        2 => 3,   // R32G32B32A32_FLOAT
        3 => 2,   // R32G32B32A32_UINT
        9 => 30,  // R16G16B16A16_TYPELESS
        10 => 4,  // R16G16B16A16_FLOAT
        15 => 31, // R32G32_TYPELESS
        16 => 6,  // R32G32_FLOAT
        17 => 38, // R32G32_UINT
        19 | 20 | 21 => 28, // R32G8X24_TYPELESS / D32_FLOAT_S8X24_UINT / R32_FLOAT_X8X24
        22 | 47 => 7,       // X32_TYPELESS_G8X24_UINT / X24_TYPELESS_G8_UINT -> R8_UINT
        23 => 32, // R10G10B10A2_TYPELESS
        24 => 17, // R10G10B10A2_UNORM
        26 => 16, // R11G11B10_FLOAT
        27 => 9,  // R8G8B8A8_TYPELESS
        28 => 10, // R8G8B8A8_UNORM
        29 => 12, // R8G8B8A8_UNORM_SRGB
        31 => 11, // R8G8B8A8_SNORM
        33 => 33, // R16G16_TYPELESS
        34 => 18, // R16G16_FLOAT
        36 => 19, // R16G16_UINT
        39 => 37, // R32_TYPELESS
        40 | 41 => 28, // D32_FLOAT / R32_FLOAT
        42 => 8,  // R32_UINT
        44 | 45 | 46 => 8, // R24G8_TYPELESS / D24_UNORM_S8_UINT / R24_UNORM_X8 -> R32_UINT
        48 => 36, // R8G8_TYPELESS
        49 => 26, // R8G8_UNORM
        50 => 27, // R8G8_UINT
        53 => 34, // R16_TYPELESS
        54 => 21, // R16_FLOAT
        55 | 56 => 23, // D16_UNORM / R16_UNORM
        57 => 22, // R16_UINT
        58 => 24, // R16_SNORM
        60 => 35, // R8_TYPELESS
        61 | 65 => 25, // R8_UNORM / A8_UNORM
        62 => 7,  // R8_UINT
        63 => 39, // R8_SNORM
        67 => 29, // R9G9B9E5_SHAREDEXP
        87 => 14, // B8G8R8A8_UNORM
        90 => 13, // B8G8R8A8_TYPELESS
        91 => 15, // B8G8R8A8_UNORM_SRGB
        _ => 0,   // UNKNOWN
    }
}

/// Builds a PC `FfxApiResourceDescription` from a host [`TextureDesc`].
pub fn ffx_desc_from_host(desc: TextureDesc) -> FfxApiResourceDescription {
    const D3D12_FLAG_ALLOW_UAV: u32 = 0x4;
    if desc.dimension == 1 {
        return FfxApiResourceDescription {
            type_: 0, // BUFFER
            format: dxgi_to_ffx_surface_format(desc.dxgi_format),
            width: desc.width as u32,
            height: desc.height,
            depth: 0,
            mip_count: 0,
            flags: 0,
            usage: 1 << 1, // UAV
        };
    }
    let mut usage = match desc.dxgi_format {
        40 | 55 => 1 << 2,      // D32_FLOAT / D16_UNORM -> DEPTHTARGET
        20 | 45 => (1 << 2) | (1 << 5), // D32S8 / D24S8 -> DEPTH | STENCIL
        _ => 0,                 // READ_ONLY
    };
    if desc.flags & D3D12_FLAG_ALLOW_UAV != 0 {
        usage |= 1 << 1; // UAV
    }
    let type_ = match desc.dimension {
        2 => 1, // TEXTURE1D
        3 if desc.depth_or_array_size == 6 => 3, // TEXTURE_CUBE
        3 => 2, // TEXTURE2D
        4 => 4, // TEXTURE3D
        _ => 2,
    };
    FfxApiResourceDescription {
        type_,
        format: dxgi_to_ffx_surface_format(desc.dxgi_format),
        width: desc.width as u32,
        height: desc.height,
        depth: u32::from(desc.depth_or_array_size),
        mip_count: u32::from(desc.mip_levels),
        flags: 0,
        usage,
    }
}

// ---------------------------------------------------------------------------
// Search paths & DLL discovery
// ---------------------------------------------------------------------------

const FSR_LOADER_DLL: &str = "amd_fidelityfx_loader_dx12.dll";
const SL_INTERPOSER_DLL: &str = "sl.interposer.dll";
const SL_DLSS_DLL: &str = "sl.dlss.dll";

/// Ordered search directories for PC upscaler binaries.
pub fn search_dirs(cfg: &UpscalerConfig) -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    let mut push_unique = |p: PathBuf| {
        if !dirs.contains(&p) {
            dirs.push(p);
        }
    };
    if let Some(dir) = &cfg.dir {
        push_unique(dir.clone());
    }
    if let Ok(exe) = std::env::current_exe()
        && let Some(exe_dir) = exe.parent()
    {
        push_unique(exe_dir.join("fidelityfx"));
        push_unique(exe_dir.to_path_buf());
        for ancestor in exe_dir.ancestors().skip(1).take(5) {
            push_unique(ancestor.join("data").join("fidelityfx"));
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        for ancestor in cwd.ancestors().take(4) {
            push_unique(ancestor.join("data").join("fidelityfx"));
        }
    }
    if let Ok(local) = std::env::var("LOCALAPPDATA") {
        push_unique(PathBuf::from(local).join("victory").join("fidelityfx"));
    }
    dirs
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SelectedBackend {
    Dlss(PathBuf),
    Fsr3(PathBuf),
}

/// Resolves which PC backend should handle `amd_fidelityfx_loader_xbox.dll`
/// under `cfg`, scanning `dirs` in priority order.
pub fn discover_backend_in(cfg: &UpscalerConfig, dirs: &[PathBuf]) -> Option<SelectedBackend> {
    let find_dlss = || {
        dirs.iter().find_map(|d| {
            let interposer = d.join(SL_INTERPOSER_DLL);
            let dlss = d.join(SL_DLSS_DLL);
            (interposer.is_file() && dlss.is_file()).then_some(SelectedBackend::Dlss(interposer))
        })
    };
    let find_fsr3 = || {
        dirs.iter().find_map(|d| {
            let loader = d.join(FSR_LOADER_DLL);
            loader.is_file().then_some(SelectedBackend::Fsr3(loader))
        })
    };
    match cfg.mode {
        BackendMode::Console => None,
        BackendMode::Dlss => find_dlss(),
        BackendMode::Fsr3 => find_fsr3(),
        BackendMode::Auto => find_dlss().or_else(find_fsr3),
    }
}

/// Decides whether `modules::all()` should register the HLE FidelityFX bridge.
///
/// - `VICTORY_FIDELITYFX=console`: returns `false` (console DLLs load untouched).
/// - `VICTORY_FIDELITYFX=auto`: returns `true` when PC Streamline/DLSS or PC
///   FSR 3.1 DLLs are found; otherwise logs a setup hint once and returns
///   `false` so console DLLs load as fallback.
/// - `VICTORY_FIDELITYFX=fsr3|dlss`: always returns `true` (explicit request).
pub fn should_register() -> bool {
    let cfg = config();
    if cfg.is_console_opt_out() {
        tracing::info!("VICTORY_FIDELITYFX=console: using console upscaler DLLs");
        return false;
    }
    let dirs = search_dirs(cfg);
    match discover_backend_in(cfg, &dirs) {
        Some(SelectedBackend::Dlss(path)) => {
            tracing::info!(
                dll = %path.display(),
                output_override = ?cfg.output_override,
                quality_override = ?cfg.quality_override,
                "PC-native upscaler active: NVIDIA DLSS (Streamline)"
            );
            true
        }
        Some(SelectedBackend::Fsr3(path)) => {
            tracing::info!(
                dll = %path.display(),
                output_override = ?cfg.output_override,
                quality_override = ?cfg.quality_override,
                "PC-native upscaler active: AMD FidelityFX FSR 3.1"
            );
            true
        }
        None => match cfg.mode {
            BackendMode::Auto => {
                tracing::info!(
                    searched = ?dirs,
                    "no PC upscaler DLLs (sl.interposer.dll+sl.dlss.dll or amd_fidelityfx_loader_dx12.dll) found; \
                     falling back to console DLLs (set VICTORY_FIDELITYFX_DIR or drop DLLs into data/fidelityfx/)"
                );
                false
            }
            BackendMode::Fsr3 | BackendMode::Dlss => {
                tracing::warn!(
                    mode = ?cfg.mode,
                    searched = ?dirs,
                    "explicit PC upscaler mode requested but DLLs were not found"
                );
                true
            }
            BackendMode::Console => false,
        },
    }
}

// ---------------------------------------------------------------------------
// Windows runtime backends (PC AMD FSR 3.1 + NVIDIA Streamline DLSS)
// ---------------------------------------------------------------------------

#[cfg(windows)]
mod runtime {
    use super::*;
    use std::ffi::CString;
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::System::LibraryLoader::{
        GetProcAddress, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR,
        LoadLibraryExW,
    };

    // -- Backend A: PC AMD FidelityFX FSR 3.1 (`amd_fidelityfx_loader_dx12.dll`) --

    type PfnFfxCreateContext = unsafe extern "system" fn(
        context: *mut *mut c_void,
        desc: *mut FfxApiHeader,
        mem_cb: *const c_void,
    ) -> u32;
    type PfnFfxDestroyContext =
        unsafe extern "system" fn(context: *mut *mut c_void, mem_cb: *const c_void) -> u32;
    type PfnFfxQuery =
        unsafe extern "system" fn(context: *mut *mut c_void, desc: *mut FfxApiHeader) -> u32;
    type PfnFfxDispatch =
        unsafe extern "system" fn(context: *mut *mut c_void, desc: *const FfxApiHeader) -> u32;

    struct Fsr3Runtime {
        _module: usize,
        create_context: PfnFfxCreateContext,
        destroy_context: PfnFfxDestroyContext,
        query: PfnFfxQuery,
        dispatch: PfnFfxDispatch,
    }

    // SAFETY: DLL module handle and exported C function pointers are process-wide.
    unsafe impl Send for Fsr3Runtime {}
    unsafe impl Sync for Fsr3Runtime {}

    // -- Backend B: NVIDIA Streamline DLSS (`sl.interposer.dll` + `sl.dlss.dll`) --

    const SL_SDK_VERSION: u64 = (2u64 << 48) | (14u64 << 32) | (16u64 << 16) | 0xfedc;
    const SL_STRUCT_V1: usize = 1;
    const SL_STRUCT_V2: usize = 2;
    const SL_STRUCT_V3: usize = 3;
    const SL_FEATURE_DLSS: u32 = 0;
    const SL_RESULT_OK: i32 = 0;
    const SL_BOOL_FALSE: u8 = 0;
    const SL_BOOL_TRUE: u8 = 1;
    const SL_INVALID_FLOAT: f32 = 3.40282346638528859811704183484516925440e38;

    const SL_BUF_DEPTH: u32 = 0;
    const SL_BUF_MOTION_VECTORS: u32 = 1;
    const SL_BUF_SCALING_INPUT_COLOR: u32 = 3;
    const SL_BUF_SCALING_OUTPUT_COLOR: u32 = 4;
    const SL_BUF_EXPOSURE: u32 = 13;
    const SL_BUF_REACTIVE_MASK_HINT: u32 = 36;
    const SL_LIFECYCLE_VALID_UNTIL_EVALUATE: u32 = 2;
    const SL_PREF_DISABLE_CL_STATE_TRACKING: u64 = 1 << 0;
    const SL_PREF_USE_FRAME_BASED_TAGGING: u64 = 1 << 7;

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct SlStructType {
        data1: u32,
        data2: u16,
        data3: u16,
        data4: [u8; 8],
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct SlBaseStructure {
        next: *mut SlBaseStructure,
        struct_type: SlStructType,
        struct_version: usize,
    }

    const SL_GUID_PREFERENCES: SlStructType = SlStructType {
        data1: 0x1ca1_0965,
        data2: 0xbf8e,
        data3: 0x432b,
        data4: [0x8d, 0xa1, 0x67, 0x16, 0xd8, 0x79, 0xfb, 0x14],
    };
    const SL_GUID_CONSTANTS: SlStructType = SlStructType {
        data1: 0xdcd3_5ad7,
        data2: 0x4e4a,
        data3: 0x4bad,
        data4: [0xa9, 0x0c, 0xe0, 0xc4, 0x9e, 0xb2, 0x3a, 0xfe],
    };
    const SL_GUID_RESOURCE: SlStructType = SlStructType {
        data1: 0x3a9d_70cf,
        data2: 0x2418,
        data3: 0x4b72,
        data4: [0x83, 0x91, 0x13, 0xf8, 0x72, 0x1c, 0x72, 0x61],
    };
    const SL_GUID_RESOURCE_TAG: SlStructType = SlStructType {
        data1: 0x4c6a_5aad,
        data2: 0xb445,
        data3: 0x496c,
        data4: [0x87, 0xff, 0x1a, 0xf3, 0x84, 0x5b, 0xe6, 0x53],
    };
    const SL_GUID_VIEWPORT_HANDLE: SlStructType = SlStructType {
        data1: 0x171b_6435,
        data2: 0x9b3c,
        data3: 0x4fc8,
        data4: [0x99, 0x94, 0xfb, 0xe5, 0x25, 0x69, 0xaa, 0xa4],
    };
    const SL_GUID_DLSS_OPTIONS: SlStructType = SlStructType {
        data1: 0x6ac8_26e4,
        data2: 0x4c61,
        data3: 0x4101,
        data4: [0xa9, 0x2d, 0x63, 0x8d, 0x42, 0x10, 0x57, 0xb8],
    };
    const SL_GUID_DLSS_STATE: SlStructType = SlStructType {
        data1: 0x9366_b056,
        data2: 0x8c01,
        data3: 0x463c,
        data4: [0xbb, 0x91, 0xe6, 0x87, 0x82, 0x63, 0x6c, 0xe9],
    };

    #[repr(C)]
    struct SlPreferences {
        base: SlBaseStructure,
        show_console: u8,
        _pad0: [u8; 3],
        log_level: u32,
        paths_to_plugins: *const *const u16,
        num_paths_to_plugins: u32,
        _pad1: u32,
        path_to_logs_and_data: *const u16,
        allocate_callback: *const c_void,
        release_callback: *const c_void,
        log_message_callback: *const c_void,
        flags: u64,
        features_to_load: *const u32,
        num_features_to_load: u32,
        application_id: u32,
        engine: u32,
        _pad2: u32,
        engine_version: *const c_char,
        project_id: *const c_char,
        render_api: u32,
        _pad3: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct SlExtent {
        top: u32,
        left: u32,
        width: u32,
        height: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct SlViewportHandle {
        base: SlBaseStructure,
        value: u32,
        _pad: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct SlResource {
        base: SlBaseStructure,
        type_: u8,
        _pad0: [u8; 7],
        native: *mut c_void,
        memory: *mut c_void,
        view: *mut c_void,
        state: u32,
        width: u32,
        height: u32,
        native_format: u32,
        mip_levels: u32,
        array_layers: u32,
        gpu_virtual_address: u64,
        flags: u32,
        usage: u32,
        internal_flags: u16,
        reserved: u16,
        _pad1: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct SlResourceTag {
        base: SlBaseStructure,
        resource: *mut SlResource,
        type_: u32,
        lifecycle: u32,
        extent: SlExtent,
    }

    #[repr(C)]
    struct SlConstants {
        base: SlBaseStructure,
        camera_view_to_clip: [[f32; 4]; 4],
        clip_to_camera_view: [[f32; 4]; 4],
        clip_to_lens_clip: [[f32; 4]; 4],
        clip_to_prev_clip: [[f32; 4]; 4],
        prev_clip_to_clip: [[f32; 4]; 4],
        jitter_offset: [f32; 2],
        mvec_scale: [f32; 2],
        camera_pinhole_offset: [f32; 2],
        camera_pos: [f32; 3],
        camera_up: [f32; 3],
        camera_right: [f32; 3],
        camera_fwd: [f32; 3],
        camera_near: f32,
        camera_far: f32,
        camera_fov: f32,
        camera_aspect_ratio: f32,
        motion_vectors_invalid_value: f32,
        depth_inverted: u8,
        camera_motion_included: u8,
        motion_vectors_3d: u8,
        reset: u8,
        orthographic_projection: u8,
        motion_vectors_dilated: u8,
        motion_vectors_jittered: u8,
        _pad0: u8,
        min_relative_linear_depth_object_separation: f32,
    }

    #[repr(C)]
    struct SlDlssOptions {
        base: SlBaseStructure,
        mode: u32,
        output_width: u32,
        output_height: u32,
        sharpness: f32,
        pre_exposure: f32,
        exposure_scale: f32,
        color_buffers_hdr: u8,
        indicator_invert_axis_x: u8,
        indicator_invert_axis_y: u8,
        _pad0: u8,
        dlaa_preset: u32,
        quality_preset: u32,
        balanced_preset: u32,
        performance_preset: u32,
        ultra_performance_preset: u32,
        ultra_quality_preset: u32,
        use_auto_exposure: u8,
        alpha_upscaling_enabled: u8,
        _pad1: [u8; 2],
    }

    #[repr(C)]
    struct SlDlssState {
        base: SlBaseStructure,
        estimated_vram_usage_in_bytes: u64,
    }

    const _: () = {
        assert!(size_of::<SlBaseStructure>() == 32);
        assert!(size_of::<SlViewportHandle>() == 40);
        assert!(size_of::<SlDlssState>() == 40);
    };

    type PfnSlInit = unsafe extern "system" fn(pref: *const SlPreferences, sdk_ver: u64) -> i32;
    type PfnSlSetD3DDevice = unsafe extern "system" fn(device: *mut c_void) -> i32;
    type PfnSlGetNewFrameToken =
        unsafe extern "system" fn(token: *mut *mut c_void, frame_index: *const u32) -> i32;
    type PfnSlSetConstants = unsafe extern "system" fn(
        values: *const SlConstants,
        frame: *const c_void,
        viewport: *const SlViewportHandle,
    ) -> i32;
    type PfnSlSetTagForFrame = unsafe extern "system" fn(
        frame: *const c_void,
        viewport: *const SlViewportHandle,
        tags: *const SlResourceTag,
        num_tags: u32,
        cmd_buffer: *mut c_void,
    ) -> i32;
    type PfnSlSetTag = unsafe extern "system" fn(
        viewport: *const SlViewportHandle,
        tags: *const SlResourceTag,
        num_tags: u32,
        cmd_buffer: *mut c_void,
    ) -> i32;
    type PfnSlEvaluateFeature = unsafe extern "system" fn(
        feature: u32,
        frame: *const c_void,
        inputs: *const *const SlBaseStructure,
        num_inputs: u32,
        cmd_buffer: *mut c_void,
    ) -> i32;
    type PfnSlGetFeatureFunction = unsafe extern "system" fn(
        feature: u32,
        func_name: *const c_char,
        out_fn: *mut *mut c_void,
    ) -> i32;
    type PfnSlFreeResources =
        unsafe extern "system" fn(feature: u32, viewport: *const SlViewportHandle) -> i32;
    type PfnSlDlssSetOptions = unsafe extern "system" fn(
        viewport: *const SlViewportHandle,
        options: *const SlDlssOptions,
    ) -> i32;
    type PfnSlDlssGetState = unsafe extern "system" fn(
        viewport: *const SlViewportHandle,
        state: *mut SlDlssState,
    ) -> i32;

    struct StreamlineRuntime {
        _module: usize,
        plugin_dir_wide: Vec<u16>,
        sl_init: PfnSlInit,
        sl_set_d3d_device: PfnSlSetD3DDevice,
        sl_get_new_frame_token: PfnSlGetNewFrameToken,
        sl_set_constants: PfnSlSetConstants,
        sl_set_tag_for_frame: Option<PfnSlSetTagForFrame>,
        sl_set_tag: Option<PfnSlSetTag>,
        sl_evaluate_feature: PfnSlEvaluateFeature,
        sl_get_feature_function: PfnSlGetFeatureFunction,
        sl_free_resources: Option<PfnSlFreeResources>,
        device_state: Mutex<Option<StreamlineDevice>>,
        next_viewport: AtomicU32,
        frame_counter: AtomicU32,
    }

    struct StreamlineDevice {
        device_raw: usize,
        set_options: Option<PfnSlDlssSetOptions>,
        get_state: Option<PfnSlDlssGetState>,
    }

    // SAFETY: process-wide DLL function pointers; mutable init guarded by Mutex.
    unsafe impl Send for StreamlineRuntime {}
    unsafe impl Sync for StreamlineRuntime {}

    enum BackendRuntime {
        Fsr3(Fsr3Runtime),
        Dlss(StreamlineRuntime),
    }

    enum UpscalerContext {
        Fsr3 {
            pc_ctx: *mut c_void,
        },
        Dlss {
            viewport_id: u32,
            flags: u32,
            max_upscale: FfxApiDimensions2D,
        },
    }

    unsafe fn load_dll(path: &Path) -> Option<usize> {
        let wide: Vec<u16> = path
            .as_os_str()
            .encode_wide()
            .chain(std::iter::once(0))
            .collect();
        // SAFETY: NUL-terminated absolute path; searches the DLL's own folder
        // first so sibling upscaler/Streamline DLLs resolve cleanly.
        let handle = unsafe {
            LoadLibraryExW(
                wide.as_ptr(),
                null_mut(),
                LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS,
            )
        } as usize;
        (handle != 0).then_some(handle)
    }

    unsafe fn proc_addr<T>(module: usize, name: &str) -> Option<T> {
        let cname = CString::new(name).ok()?;
        // SAFETY: valid HMODULE and NUL-terminated export name.
        let sym = unsafe { GetProcAddress(module as _, cname.as_ptr().cast()) }?;
        // SAFETY: caller supplies the exact extern "system" signature for `name`.
        Some(unsafe { std::mem::transmute_copy::<_, T>(&sym) })
    }

    fn load_fsr3(path: &Path) -> Option<Fsr3Runtime> {
        // SAFETY: `path` is a user-supplied PC FidelityFX loader DLL.
        let module = unsafe { load_dll(path) }?;
        // SAFETY: standard C exports of `amd_fidelityfx_loader_dx12.dll`.
        let (create_context, destroy_context, query, dispatch) = unsafe {
            (
                proc_addr(module, "ffxCreateContext")?,
                proc_addr(module, "ffxDestroyContext")?,
                proc_addr(module, "ffxQuery")?,
                proc_addr(module, "ffxDispatch")?,
            )
        };
        Some(Fsr3Runtime {
            _module: module,
            create_context,
            destroy_context,
            query,
            dispatch,
        })
    }

    fn load_streamline(path: &Path) -> Option<StreamlineRuntime> {
        // SAFETY: `path` is `sl.interposer.dll`.
        let module = unsafe { load_dll(path) }?;
        let plugin_dir = path.parent().unwrap_or_else(|| Path::new("."));
        let plugin_dir_wide: Vec<u16> = plugin_dir
            .as_os_str()
            .encode_wide()
            .chain(std::iter::once(0))
            .collect();
        // SAFETY: standard C exports of `sl.interposer.dll`.
        unsafe {
            Some(StreamlineRuntime {
                _module: module,
                plugin_dir_wide,
                sl_init: proc_addr(module, "slInit")?,
                sl_set_d3d_device: proc_addr(module, "slSetD3DDevice")?,
                sl_get_new_frame_token: proc_addr(module, "slGetNewFrameToken")?,
                sl_set_constants: proc_addr(module, "slSetConstants")?,
                sl_set_tag_for_frame: proc_addr(module, "slSetTagForFrame"),
                sl_set_tag: proc_addr(module, "slSetTag"),
                sl_evaluate_feature: proc_addr(module, "slEvaluateFeature")?,
                sl_get_feature_function: proc_addr(module, "slGetFeatureFunction")?,
                sl_free_resources: proc_addr(module, "slFreeResources"),
                device_state: Mutex::new(None),
                next_viewport: AtomicU32::new(1),
                frame_counter: AtomicU32::new(0),
            })
        }
    }

    fn active_runtime() -> Option<&'static BackendRuntime> {
        static RUNTIME: OnceLock<Option<BackendRuntime>> = OnceLock::new();
        RUNTIME
            .get_or_init(|| {
                let cfg = config();
                let dirs = search_dirs(cfg);
                match discover_backend_in(cfg, &dirs)? {
                    SelectedBackend::Dlss(path) => match load_streamline(&path) {
                        Some(rt) => Some(BackendRuntime::Dlss(rt)),
                        None => {
                            tracing::error!(
                                dll = %path.display(),
                                "failed to load Streamline interposer DLL"
                            );
                            None
                        }
                    },
                    SelectedBackend::Fsr3(path) => match load_fsr3(&path) {
                        Some(rt) => Some(BackendRuntime::Fsr3(rt)),
                        None => {
                            tracing::error!(
                                dll = %path.display(),
                                "failed to load PC FidelityFX DX12 loader DLL"
                            );
                            None
                        }
                    },
                }
            })
            .as_ref()
    }

    impl StreamlineRuntime {
        fn ensure_device(&self, host_device: *mut c_void) -> Result<(Option<PfnSlDlssSetOptions>, Option<PfnSlDlssGetState>), u32> {
            let mut guard = self.device_state.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(state) = guard.as_ref() {
                if state.device_raw == host_device as usize {
                    return Ok((state.set_options, state.get_state));
                }
            }
            let dir_ptr = self.plugin_dir_wide.as_ptr();
            let features = [SL_FEATURE_DLSS];
            let prefs = SlPreferences {
                base: SlBaseStructure {
                    next: null_mut(),
                    struct_type: SL_GUID_PREFERENCES,
                    struct_version: SL_STRUCT_V1,
                },
                show_console: 0,
                _pad0: [0; 3],
                log_level: 1, // eDefault
                paths_to_plugins: &raw const dir_ptr,
                num_paths_to_plugins: 1,
                _pad1: 0,
                path_to_logs_and_data: null(),
                allocate_callback: null(),
                release_callback: null(),
                log_message_callback: null(),
                flags: SL_PREF_DISABLE_CL_STATE_TRACKING | SL_PREF_USE_FRAME_BASED_TAGGING,
                features_to_load: features.as_ptr(),
                num_features_to_load: features.len() as u32,
                application_id: 231313132, // Generic custom engine app ID
                engine: 0,                 // eCustom
                _pad2: 0,
                engine_version: c"1.0.0".as_ptr(),
                project_id: null(),
                render_api: 1, // eD3D12
                _pad3: 0,
            };
            // SAFETY: valid Streamline Preferences struct and SDK version.
            let r_init = unsafe { (self.sl_init)(&raw const prefs, SL_SDK_VERSION) };
            if r_init != SL_RESULT_OK {
                tracing::error!(result = r_init, "slInit failed");
                return Err(FFX_API_RETURN_ERROR_RUNTIME_ERROR);
            }
            // SAFETY: borrowed active host `ID3D12Device*`.
            let r_dev = unsafe { (self.sl_set_d3d_device)(host_device) };
            if r_dev != SL_RESULT_OK {
                tracing::error!(result = r_dev, "slSetD3DDevice failed");
                return Err(FFX_API_RETURN_ERROR_RUNTIME_ERROR);
            }
            let mut fn_set_opts: *mut c_void = null_mut();
            let mut fn_get_state: *mut c_void = null_mut();
            // SAFETY: querying DLSS plugin feature exports after device bind.
            let set_options = unsafe {
                if (self.sl_get_feature_function)(
                    SL_FEATURE_DLSS,
                    c"slDLSSSetOptions".as_ptr(),
                    &raw mut fn_set_opts,
                ) == SL_RESULT_OK
                    && !fn_set_opts.is_null()
                {
                    Some(std::mem::transmute::<*mut c_void, PfnSlDlssSetOptions>(
                        fn_set_opts,
                    ))
                } else {
                    None
                }
            };
            // SAFETY: same as above.
            let get_state = unsafe {
                if (self.sl_get_feature_function)(
                    SL_FEATURE_DLSS,
                    c"slDLSSGetState".as_ptr(),
                    &raw mut fn_get_state,
                ) == SL_RESULT_OK
                    && !fn_get_state.is_null()
                {
                    Some(std::mem::transmute::<*mut c_void, PfnSlDlssGetState>(
                        fn_get_state,
                    ))
                } else {
                    None
                }
            };
            *guard = Some(StreamlineDevice {
                device_raw: host_device as usize,
                set_options,
                get_state,
            });
            Ok((set_options, get_state))
        }
    }

    fn make_sl_resource(ptr: *mut c_void, desc: TextureDesc, d3d12_state: u32) -> SlResource {
        SlResource {
            base: SlBaseStructure {
                next: null_mut(),
                struct_type: SL_GUID_RESOURCE,
                struct_version: SL_STRUCT_V1,
            },
            type_: 0, // ResourceType::eTex2d
            _pad0: [0; 7],
            native: ptr,
            memory: null_mut(),
            view: null_mut(),
            state: d3d12_state,
            width: desc.width as u32,
            height: desc.height,
            native_format: desc.dxgi_format,
            mip_levels: u32::from(desc.mip_levels),
            array_layers: u32::from(desc.depth_or_array_size),
            gpu_virtual_address: 0,
            flags: 0,
            usage: 0,
            internal_flags: 0,
            reserved: 0,
            _pad1: 0,
        }
    }

    fn make_sl_tag(
        res: *mut SlResource,
        buf_type: u32,
        width: u32,
        height: u32,
    ) -> SlResourceTag {
        SlResourceTag {
            base: SlBaseStructure {
                next: null_mut(),
                struct_type: SL_GUID_RESOURCE_TAG,
                struct_version: SL_STRUCT_V1,
            },
            resource: res,
            type_: buf_type,
            lifecycle: SL_LIFECYCLE_VALID_UNTIL_EVALUATE,
            extent: SlExtent {
                top: 0,
                left: 0,
                width,
                height,
            },
        }
    }

    fn unwrap_ffx_resource(
        console_list: *const c_void,
        console_res: FfxApiResource,
        default_d3d_state: u32,
        tracked: &mut Vec<(*const c_void, u32)>,
    ) -> FfxApiResource {
        if console_res.resource.is_null() {
            return console_res;
        }
        let wanted = if console_res.state != 0 {
            ffx_state_to_d3d12(console_res.state)
        } else {
            default_d3d_state
        };
        let this = console_res.resource.cast_const();
        let Some((host_ptr, desc)) = host_texture_for_upscaler(console_list, this, wanted) else {
            return FfxApiResource {
                resource: null_mut(),
                description: FfxApiResourceDescription::default(),
                state: 0,
                _pad: 0,
            };
        };
        tracked.push((this, wanted));
        FfxApiResource {
            resource: host_ptr,
            description: ffx_desc_from_host(desc),
            state: console_res.state,
            _pad: 0,
        }
    }

    pub fn create_context(context: *mut *mut c_void, desc: *mut FfxApiHeader) -> u32 {
        if context.is_null() || desc.is_null() {
            return FFX_API_RETURN_ERROR_PARAMETER;
        }
        let Some(host_device) = host_device_raw() else {
            tracing::error!("ffxCreateContext called before host D3D12 device was created");
            return FFX_API_RETURN_ERROR_RUNTIME_ERROR;
        };
        let Some(rt) = active_runtime() else {
            return FFX_API_RETURN_NO_PROVIDER;
        };
        // SAFETY: title passes a valid `FfxApiHeader` chain.
        let mut chain = match unsafe { rewrite_create_chain(desc.cast_const(), host_device, config()) } {
            Ok(c) => c,
            Err(code) => return code,
        };
        let max_render = chain.upscale.max_render_size;
        let max_upscale = chain.upscale.max_upscale_size;
        let flags = chain.upscale.flags;

        match rt {
            BackendRuntime::Fsr3(fsr) => {
                let mut pc_ctx: *mut c_void = null_mut();
                let head = chain.link_head();
                // SAFETY: `head` points into `chain` (`upscale -> backend`), mem_cb = null.
                let rc = unsafe { (fsr.create_context)(&raw mut pc_ctx, head, null()) };
                if rc != FFX_API_RETURN_OK {
                    tracing::error!(
                        rc,
                        render = ?max_render,
                        upscale = ?max_upscale,
                        flags = format_args!("{flags:#x}"),
                        "PC FSR 3.1 ffxCreateContext failed"
                    );
                    return rc;
                }
                let boxed = Box::new(UpscalerContext::Fsr3 { pc_ctx });
                // SAFETY: checked non-null.
                unsafe { *context = Box::into_raw(boxed).cast() };
                tracing::info!(
                    render = ?max_render,
                    upscale = ?max_upscale,
                    stripped_nodes = chain.stripped_nodes,
                    "created PC FSR 3.1 upscaler context"
                );
                FFX_API_RETURN_OK
            }
            BackendRuntime::Dlss(sl) => {
                let (set_options, _) = match sl.ensure_device(host_device) {
                    Ok(fns) => fns,
                    Err(code) => return code,
                };
                let viewport_id = sl.next_viewport.fetch_add(1, Ordering::Relaxed);
                let ratio = (max_upscale.width.max(1) as f32) / (max_render.width.max(1) as f32);
                let quality = config()
                    .quality_override
                    .unwrap_or_else(|| UpscaleQuality::from_ratio(ratio));
                if let Some(set_opts) = set_options {
                    let vp = SlViewportHandle {
                        base: SlBaseStructure {
                            next: null_mut(),
                            struct_type: SL_GUID_VIEWPORT_HANDLE,
                            struct_version: SL_STRUCT_V1,
                        },
                        value: viewport_id,
                        _pad: 0,
                    };
                    let opts = SlDlssOptions {
                        base: SlBaseStructure {
                            next: null_mut(),
                            struct_type: SL_GUID_DLSS_OPTIONS,
                            struct_version: SL_STRUCT_V3,
                        },
                        mode: quality.sl_dlss_mode(),
                        output_width: max_upscale.width,
                        output_height: max_upscale.height,
                        sharpness: 0.0,
                        pre_exposure: 1.0,
                        exposure_scale: 1.0,
                        color_buffers_hdr: u8::from(
                            flags & FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE != 0,
                        ),
                        indicator_invert_axis_x: SL_BOOL_FALSE,
                        indicator_invert_axis_y: SL_BOOL_FALSE,
                        _pad0: 0,
                        dlaa_preset: 0,
                        quality_preset: 0,
                        balanced_preset: 0,
                        performance_preset: 0,
                        ultra_performance_preset: 0,
                        ultra_quality_preset: 0,
                        use_auto_exposure: u8::from(flags & FFX_UPSCALE_ENABLE_AUTO_EXPOSURE != 0),
                        alpha_upscaling_enabled: SL_BOOL_FALSE,
                        _pad1: [0; 2],
                    };
                    // SAFETY: valid viewport handle and DLSSOptions struct.
                    let r = unsafe { set_opts(&raw const vp, &raw const opts) };
                    if r != SL_RESULT_OK {
                        tracing::error!(result = r, "slDLSSSetOptions failed");
                        return FFX_API_RETURN_ERROR_RUNTIME_ERROR;
                    }
                }
                let boxed = Box::new(UpscalerContext::Dlss {
                    viewport_id,
                    flags,
                    max_upscale,
                });
                // SAFETY: checked non-null.
                unsafe { *context = Box::into_raw(boxed).cast() };
                tracing::info!(
                    viewport_id,
                    ?quality,
                    render = ?max_render,
                    upscale = ?max_upscale,
                    "created NVIDIA DLSS (Streamline) upscaler context"
                );
                FFX_API_RETURN_OK
            }
        }
    }

    pub fn destroy_context(context: *mut *mut c_void) -> u32 {
        if context.is_null() {
            return FFX_API_RETURN_ERROR_PARAMETER;
        }
        // SAFETY: checked non-null; null inner pointer is tolerated cleanly.
        let raw = unsafe { *context };
        if raw.is_null() {
            return FFX_API_RETURN_OK;
        }
        // SAFETY: allocated by `Box::into_raw` in `create_context`.
        let boxed = unsafe { Box::from_raw(raw.cast::<UpscalerContext>()) };
        // SAFETY: checked non-null.
        unsafe { *context = null_mut() };

        match *boxed {
            UpscalerContext::Fsr3 { mut pc_ctx, .. } => {
                if let Some(BackendRuntime::Fsr3(fsr)) = active_runtime() {
                    // SAFETY: valid PC FSR context handle.
                    return unsafe { (fsr.destroy_context)(&raw mut pc_ctx, null()) };
                }
                FFX_API_RETURN_OK
            }
            UpscalerContext::Dlss { viewport_id, .. } => {
                if let Some(BackendRuntime::Dlss(sl)) = active_runtime()
                    && let Some(free_fn) = sl.sl_free_resources
                {
                    let vp = SlViewportHandle {
                        base: SlBaseStructure {
                            next: null_mut(),
                            struct_type: SL_GUID_VIEWPORT_HANDLE,
                            struct_version: SL_STRUCT_V1,
                        },
                        value: viewport_id,
                        _pad: 0,
                    };
                    // SAFETY: valid viewport handle.
                    let _ = unsafe { free_fn(SL_FEATURE_DLSS, &raw const vp) };
                }
                FFX_API_RETURN_OK
            }
        }
    }

    pub fn query(context: *mut *mut c_void, desc: *mut FfxApiHeader) -> u32 {
        if desc.is_null() {
            return FFX_API_RETURN_ERROR_PARAMETER;
        }
        // SAFETY: checked non-null.
        let desc_type = unsafe { (*desc).type_ };

        // Apply optional quality-mode override before answering ratio/resolution queries.
        let quality_override = config().quality_override.map(UpscaleQuality::ffx_mode);

        // If PC FSR 3.1 is active, forward queries directly whenever possible
        // (rewriting any console device pointer on global version/memory queries).
        if let Some(BackendRuntime::Fsr3(fsr)) = active_runtime() {
            let mut pc_ctx = if !context.is_null() {
                // SAFETY: checked non-null.
                let raw = unsafe { *context };
                if !raw.is_null() {
                    // SAFETY: allocated by `create_context`.
                    match unsafe { &*(raw.cast::<UpscalerContext>()) } {
                        UpscalerContext::Fsr3 { pc_ctx, .. } => *pc_ctx,
                        UpscalerContext::Dlss { .. } => null_mut(),
                    }
                } else {
                    null_mut()
                }
            } else {
                null_mut()
            };
            let ctx_ptr = if pc_ctx.is_null() {
                null_mut()
            } else {
                &raw mut pc_ctx
            };

            let fsr_rc = match desc_type {
                FFX_API_QUERY_DESC_TYPE_GET_VERSIONS => {
                    // SAFETY: `desc_type` identifies `FfxQueryDescGetVersions`.
                    let mut copy = unsafe { *(desc.cast::<FfxQueryDescGetVersions>()) };
                    copy.header.p_next = null_mut();
                    copy.device = host_device_raw().unwrap_or(null_mut());
                    // SAFETY: valid copied query struct.
                    unsafe { (fsr.query)(ctx_ptr, &raw mut copy.header) }
                }
                FFX_API_QUERY_DESC_TYPE_UPSCALE_GPU_MEMORY_USAGE_V2 => {
                    // SAFETY: `desc_type` identifies `FfxQueryDescUpscaleGetGpuMemoryUsageV2`.
                    let mut copy =
                        unsafe { *(desc.cast::<FfxQueryDescUpscaleGetGpuMemoryUsageV2>()) };
                    copy.header.p_next = null_mut();
                    copy.device = host_device_raw().unwrap_or(null_mut());
                    if let Some((w, h)) = config().output_override {
                        copy.max_upscale_size = FfxApiDimensions2D {
                            width: w,
                            height: h,
                        };
                        copy.max_render_size.width = copy.max_render_size.width.min(w);
                        copy.max_render_size.height = copy.max_render_size.height.min(h);
                    }
                    // SAFETY: valid copied query struct.
                    unsafe { (fsr.query)(ctx_ptr, &raw mut copy.header) }
                }
                FFX_API_QUERY_DESC_TYPE_UPSCALE_GETUPSCALERATIOFROMQUALITYMODE => {
                    // SAFETY: `desc_type` identifies ratio query.
                    let mut copy = unsafe { *(desc.cast::<FfxQueryDescUpscaleGetRatio>()) };
                    if let Some(q) = quality_override {
                        copy.quality_mode = q;
                    }
                    // SAFETY: valid copied query struct.
                    unsafe { (fsr.query)(ctx_ptr, &raw mut copy.header) }
                }
                FFX_API_QUERY_DESC_TYPE_UPSCALE_GETRENDERRESOLUTIONFROMQUALITYMODE => {
                    // SAFETY: `desc_type` identifies render-res query.
                    let mut copy = unsafe { *(desc.cast::<FfxQueryDescUpscaleGetRenderRes>()) };
                    if let Some(q) = quality_override {
                        copy.quality_mode = q;
                    }
                    // SAFETY: valid copied query struct.
                    unsafe { (fsr.query)(ctx_ptr, &raw mut copy.header) }
                }
                _ => {
                    // SAFETY: `desc` has the exact same ABI on PC for jitter, memory usage,
                    // provider version, and resource requirements.
                    unsafe { (fsr.query)(ctx_ptr, desc) }
                }
            };
            if fsr_rc == FFX_API_RETURN_OK || !ctx_ptr.is_null() {
                return fsr_rc;
            }
        }

        // Analytical / Streamline DLSS query implementation.
        match desc_type {
            FFX_API_QUERY_DESC_TYPE_UPSCALE_GETUPSCALERATIOFROMQUALITYMODE => {
                // SAFETY: `desc_type` identifies ratio query.
                let q = unsafe { &*(desc.cast::<FfxQueryDescUpscaleGetRatio>()) };
                if q.out_upscale_ratio.is_null() {
                    return FFX_API_RETURN_ERROR_PARAMETER;
                }
                let mode = quality_override.unwrap_or(q.quality_mode);
                let Some(ratio) = upscale_ratio_from_quality_mode(mode) else {
                    return FFX_API_RETURN_ERROR_PARAMETER;
                };
                // SAFETY: checked non-null.
                unsafe { *q.out_upscale_ratio = ratio };
                FFX_API_RETURN_OK
            }
            FFX_API_QUERY_DESC_TYPE_UPSCALE_GETRENDERRESOLUTIONFROMQUALITYMODE => {
                // SAFETY: `desc_type` identifies render-res query.
                let q = unsafe { &*(desc.cast::<FfxQueryDescUpscaleGetRenderRes>()) };
                if q.out_render_width.is_null() || q.out_render_height.is_null() {
                    return FFX_API_RETURN_ERROR_PARAMETER;
                }
                let mode = quality_override.unwrap_or(q.quality_mode);
                let Some((rw, rh)) =
                    render_resolution_from_quality_mode(q.display_width, q.display_height, mode)
                else {
                    return FFX_API_RETURN_ERROR_PARAMETER;
                };
                // SAFETY: checked non-null.
                unsafe {
                    *q.out_render_width = rw;
                    *q.out_render_height = rh;
                }
                FFX_API_RETURN_OK
            }
            FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTERPHASECOUNT => {
                // SAFETY: `desc_type` identifies jitter-phase query.
                let q = unsafe { &*(desc.cast::<FfxQueryDescUpscaleGetJitterPhaseCount>()) };
                if q.out_phase_count.is_null() {
                    return FFX_API_RETURN_ERROR_PARAMETER;
                }
                // SAFETY: checked non-null.
                unsafe { *q.out_phase_count = jitter_phase_count(q.render_width, q.display_width) };
                FFX_API_RETURN_OK
            }
            FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTEROFFSET => {
                // SAFETY: `desc_type` identifies jitter-offset query.
                let q = unsafe { &*(desc.cast::<FfxQueryDescUpscaleGetJitterOffset>()) };
                if q.out_x.is_null() || q.out_y.is_null() {
                    return FFX_API_RETURN_ERROR_PARAMETER;
                }
                let (x, y) = jitter_offset(q.index, q.phase_count);
                // SAFETY: checked non-null.
                unsafe {
                    *q.out_x = x;
                    *q.out_y = y;
                }
                FFX_API_RETURN_OK
            }
            FFX_API_QUERY_DESC_TYPE_UPSCALE_GPU_MEMORY_USAGE => {
                // SAFETY: `desc_type` identifies GPU memory usage query.
                let q = unsafe { &*(desc.cast::<FfxQueryDescUpscaleGetGpuMemoryUsage>()) };
                if q.gpu_memory_usage_upscaler.is_null() {
                    return FFX_API_RETURN_ERROR_PARAMETER;
                }
                let mut total = 64u64 << 20;
                if let Some(BackendRuntime::Dlss(sl)) = active_runtime()
                    && !context.is_null()
                {
                    // SAFETY: checked non-null.
                    let raw = unsafe { *context };
                    if !raw.is_null()
                        && let UpscalerContext::Dlss { viewport_id, .. } =
                            unsafe { &*(raw.cast::<UpscalerContext>()) }
                        && let Some(dev) = host_device_raw()
                        && let Ok((_, Some(get_state))) = sl.ensure_device(dev)
                    {
                        let vp = SlViewportHandle {
                            base: SlBaseStructure {
                                next: null_mut(),
                                struct_type: SL_GUID_VIEWPORT_HANDLE,
                                struct_version: SL_STRUCT_V1,
                            },
                            value: *viewport_id,
                            _pad: 0,
                        };
                        let mut state = SlDlssState {
                            base: SlBaseStructure {
                                next: null_mut(),
                                struct_type: SL_GUID_DLSS_STATE,
                                struct_version: SL_STRUCT_V1,
                            },
                            estimated_vram_usage_in_bytes: 0,
                        };
                        // SAFETY: valid viewport and output state struct.
                        if unsafe { get_state(&raw const vp, &raw mut state) } == SL_RESULT_OK {
                            total = state.estimated_vram_usage_in_bytes;
                        }
                    }
                }
                // SAFETY: checked non-null.
                unsafe {
                    *q.gpu_memory_usage_upscaler = FfxApiEffectMemoryUsage {
                        total_usage_in_bytes: total,
                        aliasable_usage_in_bytes: 0,
                    };
                }
                FFX_API_RETURN_OK
            }
            FFX_API_QUERY_DESC_TYPE_UPSCALE_GET_RESOURCE_REQUIREMENTS => {
                // SAFETY: `desc_type` identifies resource requirements query.
                let q = unsafe { &mut *(desc.cast::<FfxQueryDescUpscaleGetResourceRequirements>()) };
                // Required: Color | Depth | MotionVectors. Optional: Exposure | Reactive | T&C.
                q.required_resources = 0x7;
                q.optional_resources = 0x38;
                FFX_API_RETURN_OK
            }
            FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION => {
                // SAFETY: `desc_type` identifies provider-version query.
                let q = unsafe { &mut *(desc.cast::<FfxQueryGetProviderVersion>()) };
                q.version_id = 0x0003_0001_0005;
                q.version_name = c"victory-pc-dlss".as_ptr();
                FFX_API_RETURN_OK
            }
            FFX_API_QUERY_DESC_TYPE_GET_VERSIONS => {
                // SAFETY: `desc_type` identifies get-versions query.
                let q = unsafe { &*(desc.cast::<FfxQueryDescGetVersions>()) };
                if q.output_count.is_null() {
                    return FFX_API_RETURN_ERROR_PARAMETER;
                }
                // SAFETY: checked non-null.
                let cap = unsafe { *q.output_count };
                if cap >= 1 {
                    if !q.version_ids.is_null() {
                        // SAFETY: caller provided space for at least 1 entry.
                        unsafe { *q.version_ids = 0x0003_0001_0005 };
                    }
                    if !q.version_names.is_null() {
                        // SAFETY: caller provided space for at least 1 entry.
                        unsafe { *q.version_names = c"victory-pc-dlss".as_ptr() };
                    }
                }
                // SAFETY: checked non-null.
                unsafe { *q.output_count = 1 };
                FFX_API_RETURN_OK
            }
            _ => FFX_API_RETURN_ERROR_UNKNOWN_DESCTYPE,
        }
    }

    pub fn dispatch(context: *mut *mut c_void, desc: *const FfxApiHeader) -> u32 {
        if context.is_null() || desc.is_null() {
            return FFX_API_RETURN_ERROR_PARAMETER;
        }
        // SAFETY: checked non-null.
        let raw_ctx = unsafe { *context };
        if raw_ctx.is_null() {
            return FFX_API_RETURN_ERROR_PARAMETER;
        }
        // SAFETY: allocated by `create_context`.
        let ctx = unsafe { &mut *(raw_ctx.cast::<UpscalerContext>()) };
        // SAFETY: checked non-null.
        let desc_type = unsafe { (*desc).type_ };

        match (ctx, active_runtime()) {
            (UpscalerContext::Fsr3 { pc_ctx, .. }, Some(BackendRuntime::Fsr3(fsr))) => {
                match desc_type {
                    FFX_API_DISPATCH_DESC_TYPE_UPSCALE => {
                        // SAFETY: `desc_type` identifies `FfxDispatchDescUpscale`.
                        let in_desc = unsafe { *(desc.cast::<FfxDispatchDescUpscale>()) };
                        let console_list = in_desc.command_list.cast_const();
                        let Some(host_cmd_list) = host_command_list_prepare(console_list) else {
                            return FFX_API_RETURN_ERROR_RUNTIME_ERROR;
                        };
                        let mut tracked = Vec::with_capacity(7);
                        let mut pc_desc = in_desc;
                        pc_desc.header.p_next = null_mut();
                        pc_desc.command_list = host_cmd_list;
                        pc_desc.color = unwrap_ffx_resource(
                            console_list,
                            in_desc.color,
                            D3D12_STATE_NON_PIXEL_SHADER_RESOURCE,
                            &mut tracked,
                        );
                        pc_desc.depth = unwrap_ffx_resource(
                            console_list,
                            in_desc.depth,
                            D3D12_STATE_NON_PIXEL_SHADER_RESOURCE,
                            &mut tracked,
                        );
                        pc_desc.motion_vectors = unwrap_ffx_resource(
                            console_list,
                            in_desc.motion_vectors,
                            D3D12_STATE_NON_PIXEL_SHADER_RESOURCE,
                            &mut tracked,
                        );
                        pc_desc.exposure = unwrap_ffx_resource(
                            console_list,
                            in_desc.exposure,
                            D3D12_STATE_NON_PIXEL_SHADER_RESOURCE,
                            &mut tracked,
                        );
                        pc_desc.reactive = unwrap_ffx_resource(
                            console_list,
                            in_desc.reactive,
                            D3D12_STATE_NON_PIXEL_SHADER_RESOURCE,
                            &mut tracked,
                        );
                        pc_desc.transparency_and_composition = unwrap_ffx_resource(
                            console_list,
                            in_desc.transparency_and_composition,
                            D3D12_STATE_NON_PIXEL_SHADER_RESOURCE,
                            &mut tracked,
                        );
                        pc_desc.output = unwrap_ffx_resource(
                            console_list,
                            in_desc.output,
                            D3D12_STATE_UNORDERED_ACCESS,
                            &mut tracked,
                        );
                        if let Some((w, h)) = config().output_override {
                            // Never exceed the actual host output texture bounds.
                            let max_w = pc_desc.output.description.width.max(1);
                            let max_h = pc_desc.output.description.height.max(1);
                            pc_desc.upscale_size = FfxApiDimensions2D {
                                width: w.min(max_w),
                                height: h.min(max_h),
                            };
                        }
                        // SAFETY: all handles unwrapped to host D3D12 interfaces on `host_cmd_list`.
                        let rc = unsafe { (fsr.dispatch)(pc_ctx, &raw const pc_desc.header) };
                        host_upscaler_post_dispatch(console_list, &tracked);
                        rc
                    }
                    FFX_API_DISPATCH_DESC_TYPE_UPSCALE_GENERATEREACTIVEMASK => {
                        // SAFETY: `desc_type` identifies reactive-mask dispatch.
                        let in_desc =
                            unsafe { *(desc.cast::<FfxDispatchDescGenerateReactiveMask>()) };
                        let console_list = in_desc.command_list.cast_const();
                        let Some(host_cmd_list) = host_command_list_prepare(console_list) else {
                            return FFX_API_RETURN_ERROR_RUNTIME_ERROR;
                        };
                        let mut tracked = Vec::with_capacity(3);
                        let mut pc_desc = in_desc;
                        pc_desc.header.p_next = null_mut();
                        pc_desc.command_list = host_cmd_list;
                        pc_desc.color_opaque_only = unwrap_ffx_resource(
                            console_list,
                            in_desc.color_opaque_only,
                            D3D12_STATE_NON_PIXEL_SHADER_RESOURCE,
                            &mut tracked,
                        );
                        pc_desc.color_pre_upscale = unwrap_ffx_resource(
                            console_list,
                            in_desc.color_pre_upscale,
                            D3D12_STATE_NON_PIXEL_SHADER_RESOURCE,
                            &mut tracked,
                        );
                        pc_desc.out_reactive = unwrap_ffx_resource(
                            console_list,
                            in_desc.out_reactive,
                            D3D12_STATE_UNORDERED_ACCESS,
                            &mut tracked,
                        );
                        // SAFETY: unwrapped host D3D12 resources & command list.
                        let rc = unsafe { (fsr.dispatch)(pc_ctx, &raw const pc_desc.header) };
                        host_upscaler_post_dispatch(console_list, &tracked);
                        rc
                    }
                    _ => FFX_API_RETURN_ERROR_UNKNOWN_DESCTYPE,
                }
            }
            (
                UpscalerContext::Dlss {
                    viewport_id,
                    flags: ctx_flags,
                    max_upscale,
                    ..
                },
                Some(BackendRuntime::Dlss(sl)),
            ) => {
                if desc_type != FFX_API_DISPATCH_DESC_TYPE_UPSCALE {
                    // Reactive-mask helper passes are optional for DLSS.
                    return if desc_type == FFX_API_DISPATCH_DESC_TYPE_UPSCALE_GENERATEREACTIVEMASK {
                        FFX_API_RETURN_OK
                    } else {
                        FFX_API_RETURN_ERROR_UNKNOWN_DESCTYPE
                    };
                }
                // SAFETY: `desc_type` identifies `FfxDispatchDescUpscale`.
                let in_desc = unsafe { *(desc.cast::<FfxDispatchDescUpscale>()) };
                let console_list = in_desc.command_list.cast_const();
                let Some(host_cmd_list) = host_command_list_prepare(console_list) else {
                    return FFX_API_RETURN_ERROR_RUNTIME_ERROR;
                };
                let mut tracked = Vec::with_capacity(6);
                let unwrap_sl =
                    |res: FfxApiResource, wanted: u32, trk: &mut Vec<(*const c_void, u32)>| {
                        if res.resource.is_null() {
                            return None;
                        }
                        let this = res.resource.cast_const();
                        let (ptr, td) = host_texture_for_upscaler(console_list, this, wanted)?;
                        trk.push((this, wanted));
                        Some((make_sl_resource(ptr, td, wanted), td))
                    };
                let Some((mut sl_color, color_td)) =
                    unwrap_sl(in_desc.color, D3D12_STATE_NON_PIXEL_SHADER_RESOURCE, &mut tracked)
                else {
                    return FFX_API_RETURN_ERROR_PARAMETER;
                };
                let Some((mut sl_depth, depth_td)) =
                    unwrap_sl(in_desc.depth, D3D12_STATE_NON_PIXEL_SHADER_RESOURCE, &mut tracked)
                else {
                    return FFX_API_RETURN_ERROR_PARAMETER;
                };
                let Some((mut sl_mvec, mvec_td)) = unwrap_sl(
                    in_desc.motion_vectors,
                    D3D12_STATE_NON_PIXEL_SHADER_RESOURCE,
                    &mut tracked,
                ) else {
                    return FFX_API_RETURN_ERROR_PARAMETER;
                };
                let Some((mut sl_out, out_td)) =
                    unwrap_sl(in_desc.output, D3D12_STATE_UNORDERED_ACCESS, &mut tracked)
                else {
                    return FFX_API_RETURN_ERROR_PARAMETER;
                };
                let mut sl_exposure = unwrap_sl(
                    in_desc.exposure,
                    D3D12_STATE_NON_PIXEL_SHADER_RESOURCE,
                    &mut tracked,
                );
                let mut sl_reactive = unwrap_sl(
                    in_desc.reactive,
                    D3D12_STATE_NON_PIXEL_SHADER_RESOURCE,
                    &mut tracked,
                );

                let frame_idx = sl.frame_counter.fetch_add(1, Ordering::Relaxed);
                let mut frame_token: *mut c_void = null_mut();
                // SAFETY: valid out pointer and frame index.
                if unsafe { (sl.sl_get_new_frame_token)(&raw mut frame_token, &raw const frame_idx) }
                    != SL_RESULT_OK
                    || frame_token.is_null()
                {
                    host_upscaler_post_dispatch(console_list, &tracked);
                    return FFX_API_RETURN_ERROR_RUNTIME_ERROR;
                }

                let vp = SlViewportHandle {
                    base: SlBaseStructure {
                        next: null_mut(),
                        struct_type: SL_GUID_VIEWPORT_HANDLE,
                        struct_version: SL_STRUCT_V1,
                    },
                    value: *viewport_id,
                    _pad: 0,
                };
                let rw = in_desc.render_size.width.max(1);
                let rh = in_desc.render_size.height.max(1);
                let ow = if in_desc.upscale_size.width > 0 {
                    in_desc.upscale_size.width
                } else {
                    max_upscale.width
                }
                .min(out_td.width as u32);
                let oh = if in_desc.upscale_size.height > 0 {
                    in_desc.upscale_size.height
                } else {
                    max_upscale.height
                }
                .min(out_td.height);

                let ident = [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ];
                let consts = SlConstants {
                    base: SlBaseStructure {
                        next: null_mut(),
                        struct_type: SL_GUID_CONSTANTS,
                        struct_version: SL_STRUCT_V2,
                    },
                    camera_view_to_clip: ident,
                    clip_to_camera_view: ident,
                    clip_to_lens_clip: ident,
                    clip_to_prev_clip: ident,
                    prev_clip_to_clip: ident,
                    jitter_offset: [in_desc.jitter_offset.x, in_desc.jitter_offset.y],
                    // FFX motionVectorScale converts MV -> pixels; Streamline expects [-1, 1] UV scale.
                    mvec_scale: [
                        in_desc.motion_vector_scale.x / (rw as f32),
                        in_desc.motion_vector_scale.y / (rh as f32),
                    ],
                    camera_pinhole_offset: [0.0, 0.0],
                    camera_pos: [0.0, 0.0, 0.0],
                    camera_up: [0.0, 1.0, 0.0],
                    camera_right: [1.0, 0.0, 0.0],
                    camera_fwd: [0.0, 0.0, 1.0],
                    camera_near: in_desc.camera_near,
                    camera_far: in_desc.camera_far,
                    camera_fov: in_desc.camera_fov_angle_vertical,
                    camera_aspect_ratio: (rw as f32) / (rh as f32),
                    motion_vectors_invalid_value: SL_INVALID_FLOAT,
                    depth_inverted: u8::from(*ctx_flags & FFX_UPSCALE_ENABLE_DEPTH_INVERTED != 0),
                    camera_motion_included: SL_BOOL_TRUE,
                    motion_vectors_3d: SL_BOOL_FALSE,
                    reset: u8::from(in_desc.reset != 0),
                    orthographic_projection: SL_BOOL_FALSE,
                    motion_vectors_dilated: u8::from(
                        *ctx_flags & FFX_UPSCALE_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS != 0,
                    ),
                    motion_vectors_jittered: u8::from(
                        *ctx_flags & FFX_UPSCALE_ENABLE_MOTION_VECTORS_JITTER_CANCELLATION != 0,
                    ),
                    _pad0: 0,
                    min_relative_linear_depth_object_separation: 40.0,
                };
                // SAFETY: valid Streamline constants, frame token, and viewport.
                let r_const = unsafe { (sl.sl_set_constants)(&raw const consts, frame_token, &raw const vp) };
                if r_const != SL_RESULT_OK {
                    tracing::error!(result = r_const, "slSetConstants failed");
                    host_upscaler_post_dispatch(console_list, &tracked);
                    return FFX_API_RETURN_ERROR_RUNTIME_ERROR;
                }

                let mut tags = Vec::with_capacity(6);
                tags.push(make_sl_tag(
                    &raw mut sl_color,
                    SL_BUF_SCALING_INPUT_COLOR,
                    rw.min(color_td.width as u32),
                    rh.min(color_td.height),
                ));
                tags.push(make_sl_tag(
                    &raw mut sl_depth,
                    SL_BUF_DEPTH,
                    rw.min(depth_td.width as u32),
                    rh.min(depth_td.height),
                ));
                tags.push(make_sl_tag(
                    &raw mut sl_mvec,
                    SL_BUF_MOTION_VECTORS,
                    rw.min(mvec_td.width as u32),
                    rh.min(mvec_td.height),
                ));
                tags.push(make_sl_tag(
                    &raw mut sl_out,
                    SL_BUF_SCALING_OUTPUT_COLOR,
                    ow,
                    oh,
                ));
                if let Some((ref mut exp, exp_td)) = sl_exposure {
                    tags.push(make_sl_tag(
                        exp,
                        SL_BUF_EXPOSURE,
                        exp_td.width as u32,
                        exp_td.height,
                    ));
                }
                if let Some((ref mut react, react_td)) = sl_reactive {
                    tags.push(make_sl_tag(
                        react,
                        SL_BUF_REACTIVE_MASK_HINT,
                        rw.min(react_td.width as u32),
                        rh.min(react_td.height),
                    ));
                }

                let r_tag = if let Some(tag_for_frame) = sl.sl_set_tag_for_frame {
                    // SAFETY: valid tags array, frame token, viewport, and open command list.
                    unsafe {
                        tag_for_frame(
                            frame_token,
                            &raw const vp,
                            tags.as_ptr(),
                            tags.len() as u32,
                            host_cmd_list,
                        )
                    }
                } else if let Some(tag_fn) = sl.sl_set_tag {
                    // SAFETY: legacy tag API fallback.
                    unsafe {
                        tag_fn(
                            &raw const vp,
                            tags.as_ptr(),
                            tags.len() as u32,
                            host_cmd_list,
                        )
                    }
                } else {
                    -1
                };
                if r_tag != SL_RESULT_OK {
                    tracing::error!(result = r_tag, "slSetTagForFrame failed");
                    host_upscaler_post_dispatch(console_list, &tracked);
                    return FFX_API_RETURN_ERROR_RUNTIME_ERROR;
                }

                let inputs: [*const SlBaseStructure; 1] = [&raw const vp.base];
                // SAFETY: valid Streamline inputs and host `ID3D12GraphicsCommandList*`.
                let r_eval = unsafe {
                    (sl.sl_evaluate_feature)(
                        SL_FEATURE_DLSS,
                        frame_token,
                        inputs.as_ptr(),
                        inputs.len() as u32,
                        host_cmd_list,
                    )
                };
                host_upscaler_post_dispatch(console_list, &tracked);
                if r_eval == SL_RESULT_OK {
                    FFX_API_RETURN_OK
                } else {
                    tracing::error!(result = r_eval, "slEvaluateFeature(DLSS) failed");
                    FFX_API_RETURN_ERROR_RUNTIME_ERROR
                }
            }
            _ => FFX_API_RETURN_NO_PROVIDER,
        }
    }
}

// ---------------------------------------------------------------------------
// HLE Exports (exact 4 static imports of `amd_fidelityfx_loader_xbox.dll`)
// ---------------------------------------------------------------------------

pub extern "system" fn ffxCreateContext(
    context: *mut *mut c_void,
    desc: *mut FfxApiHeader,
    _mem_cb: *const c_void,
) -> u32 {
    #[cfg(windows)]
    {
        runtime::create_context(context, desc)
    }
    #[cfg(not(windows))]
    {
        let _ = (context, desc);
        FFX_API_RETURN_NO_PROVIDER
    }
}

pub extern "system" fn ffxDestroyContext(
    context: *mut *mut c_void,
    _mem_cb: *const c_void,
) -> u32 {
    #[cfg(windows)]
    {
        runtime::destroy_context(context)
    }
    #[cfg(not(windows))]
    {
        let _ = context;
        FFX_API_RETURN_OK
    }
}

pub extern "system" fn ffxQuery(context: *mut *mut c_void, desc: *mut FfxApiHeader) -> u32 {
    #[cfg(windows)]
    {
        runtime::query(context, desc)
    }
    #[cfg(not(windows))]
    {
        let _ = (context, desc);
        FFX_API_RETURN_NO_PROVIDER
    }
}

pub extern "system" fn ffxDispatch(context: *mut *mut c_void, desc: *const FfxApiHeader) -> u32 {
    #[cfg(windows)]
    {
        runtime::dispatch(context, desc)
    }
    #[cfg(not(windows))]
    {
        let _ = (context, desc);
        FFX_API_RETURN_NO_PROVIDER
    }
}

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "fidelityfx",
        dlls: &["amd_fidelityfx_loader_xbox.dll"],
        exports: vec![
            hle_export!(ffxCreateContext),
            hle_export!(ffxDestroyContext),
            hle_export!(ffxQuery),
            hle_export!(ffxDispatch),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[repr(C)]
    struct ConsoleVersionDesc {
        header: FfxApiHeader,
        version: u32,
        _pad: u32,
    }

    #[repr(C)]
    struct OverrideVersionDesc {
        header: FfxApiHeader,
        version_id: u64,
    }

    #[test]
    fn rewrites_console_backend_desc_and_strips_console_only_nodes() {
        for console_backend_type in [
            FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_CONSOLE_ALT,
            FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_XBOX,
        ] {
            let mut ov = OverrideVersionDesc {
                header: FfxApiHeader {
                    type_: FFX_API_DESC_TYPE_OVERRIDE_VERSION,
                    p_next: null_mut(),
                },
                version_id: 0x1234,
            };
            let mut ver = ConsoleVersionDesc {
                header: FfxApiHeader {
                    type_: FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE_VERSION,
                    p_next: &raw mut ov.header,
                },
                version: 0x0100_1001,
                _pad: 0,
            };
            let mut backend = FfxCreateBackendDX12Desc {
                header: FfxApiHeader {
                    type_: console_backend_type,
                    p_next: &raw mut ver.header,
                },
                device: 0xDEAD_BEEFusize as *mut c_void,
            };
            let upscale = FfxCreateContextDescUpscale {
                header: FfxApiHeader {
                    type_: FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE,
                    p_next: &raw mut backend.header,
                },
                flags: FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE | FFX_UPSCALE_ENABLE_DEPTH_INVERTED,
                max_render_size: FfxApiDimensions2D {
                    width: 1280,
                    height: 720,
                },
                max_upscale_size: FfxApiDimensions2D {
                    width: 1920,
                    height: 1080,
                },
                _pad: 0,
                fp_message: None,
            };

            let host_dev = 0xCAFE_BABEusize as *mut c_void;
            let cfg = UpscalerConfig::parse(Some("fsr3"), None, Some("3840x2160"), Some("quality"));
            // SAFETY: local stack-allocated FidelityFX descriptor chain.
            let mut rewritten =
                unsafe { rewrite_create_chain(&raw const upscale.header, host_dev, &cfg) }.unwrap();
            let head = rewritten.link_head();

            assert!(rewritten.saw_backend_desc);
            assert_eq!(rewritten.original_backend_type, Some(console_backend_type));
            assert_eq!(rewritten.stripped_nodes, 2);
            assert_eq!(rewritten.upscale.max_render_size, FfxApiDimensions2D { width: 1280, height: 720 });
            assert_eq!(rewritten.upscale.max_upscale_size, FfxApiDimensions2D { width: 3840, height: 2160 });
            assert!(rewritten.upscale.fp_message.is_some());
            assert_eq!(rewritten.backend.header.type_, FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12);
            assert_eq!(rewritten.backend.device, host_dev);
            assert!(rewritten.backend.header.p_next.is_null());

            // Verify linked list terminates cleanly after 2 nodes (`upscale -> backend`).
            // SAFETY: `head` points to `rewritten.upscale.header`.
            let next = unsafe { (*head).p_next };
            assert_eq!(next, &raw mut rewritten.backend.header);
            // SAFETY: `next` points to `rewritten.backend.header`.
            assert!(unsafe { (*next).p_next }.is_null());
        }
    }

    #[test]
    fn analytical_halton_jitter_and_upscale_ratios() {
        assert_eq!(upscale_ratio_from_quality_mode(0), Some(1.0));
        assert_eq!(upscale_ratio_from_quality_mode(1), Some(1.5));
        assert_eq!(upscale_ratio_from_quality_mode(2), Some(1.7));
        assert_eq!(upscale_ratio_from_quality_mode(3), Some(2.0));
        assert_eq!(upscale_ratio_from_quality_mode(4), Some(3.0));
        assert_eq!(upscale_ratio_from_quality_mode(99), None);

        assert_eq!(render_resolution_from_quality_mode(3840, 2160, 0), Some((3840, 2160)));
        assert_eq!(render_resolution_from_quality_mode(3840, 2160, 1), Some((2560, 1440)));
        assert_eq!(render_resolution_from_quality_mode(3840, 2160, 3), Some((1920, 1080)));
        assert_eq!(render_resolution_from_quality_mode(1920, 1080, 1), Some((1280, 720)));

        // Quality (1.5x): ceil(8 * 1.5^2) = 18; Performance (2.0x): ceil(8 * 4) = 32; DLAA: 8.
        assert_eq!(jitter_phase_count(1280, 1920), 18);
        assert_eq!(jitter_phase_count(1920, 3840), 32);
        assert_eq!(jitter_phase_count(3840, 3840), 8);

        // Halton(1, 2) = 0.5 -> 0.0; Halton(1, 3) = 1/3 -> -1/6.
        let (x0, y0) = jitter_offset(0, 18);
        assert!((x0 - 0.0).abs() < 1e-6);
        assert!((y0 - (1.0 / 3.0 - 0.5)).abs() < 1e-6);
        // Every offset stays within [-0.5, 0.5] and wraps cleanly with `phase_count`.
        assert_eq!(jitter_offset(0, 18), jitter_offset(18, 18));
        for i in 0..32 {
            let (x, y) = jitter_offset(i, 32);
            assert!((-0.5..=0.5).contains(&x) && (-0.5..=0.5).contains(&y));
        }
    }

    #[test]
    fn config_parsing_output_override_and_console_opt_out() {
        let console = UpscalerConfig::parse(Some("console"), None, None, None);
        assert!(console.is_console_opt_out());
        assert_eq!(discover_backend_in(&console, &[]), None);

        let auto = UpscalerConfig::parse(None, Some("C:\\upscalers"), Some("2560x1440"), Some("dlaa"));
        assert!(!auto.is_console_opt_out());
        assert_eq!(auto.mode, BackendMode::Auto);
        assert_eq!(auto.output_override, Some((2560, 1440)));
        assert_eq!(auto.quality_override, Some(UpscaleQuality::Dlaa));

        assert_eq!(parse_output_resolution(Some("3840X2160")), Some((3840, 2160)));
        assert_eq!(parse_output_resolution(Some("320x240")), None);
        assert_eq!(parse_output_resolution(Some("bad")), None);

        // Auto prefers DLSS over FSR3 when both exist in the same directory.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join(FSR_LOADER_DLL), b"").unwrap();
        assert_eq!(
            discover_backend_in(&auto, &[dir.path().to_path_buf()]),
            Some(SelectedBackend::Fsr3(dir.path().join(FSR_LOADER_DLL)))
        );
        std::fs::write(dir.path().join(SL_INTERPOSER_DLL), b"").unwrap();
        std::fs::write(dir.path().join(SL_DLSS_DLL), b"").unwrap();
        assert_eq!(
            discover_backend_in(&auto, &[dir.path().to_path_buf()]),
            Some(SelectedBackend::Dlss(dir.path().join(SL_INTERPOSER_DLL)))
        );
        let fsr_only = UpscalerConfig::parse(Some("fsr3"), None, None, None);
        assert_eq!(
            discover_backend_in(&fsr_only, &[dir.path().to_path_buf()]),
            Some(SelectedBackend::Fsr3(dir.path().join(FSR_LOADER_DLL)))
        );
    }

    #[test]
    fn hle_module_exports_exact_four_static_symbols() {
        let m = module();
        assert!(m.serves("amd_fidelityfx_loader_xbox.dll"));
        let names: Vec<&str> = m.exports.iter().map(|e| e.name).collect();
        assert_eq!(
            names,
            vec![
                "ffxCreateContext",
                "ffxDestroyContext",
                "ffxQuery",
                "ffxDispatch",
            ]
        );
    }

    #[test]
    fn state_and_surface_format_conversions_match_dx12() {
        assert_eq!(ffx_state_to_d3d12(FFX_API_RESOURCE_STATE_COMMON), D3D12_STATE_COMMON);
        assert_eq!(
            ffx_state_to_d3d12(FFX_API_RESOURCE_STATE_UNORDERED_ACCESS),
            D3D12_STATE_UNORDERED_ACCESS
        );
        assert_eq!(
            ffx_state_to_d3d12(FFX_API_RESOURCE_STATE_COMPUTE_READ | FFX_API_RESOURCE_STATE_PIXEL_READ),
            D3D12_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_STATE_PIXEL_SHADER_RESOURCE
        );
        assert_eq!(dxgi_to_ffx_surface_format(10), 4); // R16G16B16A16_FLOAT
        assert_eq!(dxgi_to_ffx_surface_format(40), 28); // D32_FLOAT -> R32_FLOAT
        assert_eq!(dxgi_to_ffx_surface_format(34), 18); // R16G16_FLOAT
        assert_eq!(dxgi_to_ffx_surface_format(61), 25); // R8_UNORM

        let td = TextureDesc {
            dimension: 3,
            width: 3840,
            height: 2160,
            depth_or_array_size: 1,
            mip_levels: 1,
            dxgi_format: 10,
            flags: 0x4, // ALLOW_UNORDERED_ACCESS
        };
        let desc = ffx_desc_from_host(td);
        assert_eq!(desc.type_, 2);
        assert_eq!(desc.format, 4);
        assert_eq!((desc.width, desc.height), (3840, 2160));
        assert_eq!(desc.usage, 1 << 1);
    }

    #[cfg(windows)]
    #[test]
    fn live_pc_fsr3_context_create_query_destroy() {
        let cfg = config();
        let dirs = search_dirs(cfg);
        let Some(SelectedBackend::Fsr3(loader_path)) = discover_backend_in(cfg, &dirs) else {
            eprintln!("skipping live PC FSR3 smoke test: amd_fidelityfx_loader_dx12.dll not found");
            return;
        };
        if host_device_raw().is_none() {
            // Initialise the host D3D12 device through the standard console entry.
            let mut dev = null();
            let _ = victory_gpu::device::create(
                null(),
                null(),
                &raw const victory_gpu::device::IID,
                &raw mut dev,
            );
        }
        if host_device_raw().is_none() {
            eprintln!("skipping live PC FSR3 smoke test: no host D3D12 adapter available");
            return;
        }

        let mut override_ver = OverrideVersionDesc {
            header: FfxApiHeader {
                type_: FFX_API_DESC_TYPE_OVERRIDE_VERSION,
                p_next: null_mut(),
            },
            version_id: 0x0003_0001_0005,
        };
        let mut console_ver = ConsoleVersionDesc {
            header: FfxApiHeader {
                type_: FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE_VERSION,
                p_next: &raw mut override_ver.header,
            },
            version: 0x00C0_1005,
            _pad: 0,
        };
        let mut backend = FfxCreateBackendDX12Desc {
            header: FfxApiHeader {
                type_: FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_CONSOLE_ALT,
                p_next: &raw mut console_ver.header,
            },
            device: 0x1234 as *mut c_void,
        };
        // Exercise 1280x720 -> 2560x1440 (above typical 1080p console targets).
        let mut upscale = FfxCreateContextDescUpscale {
            header: FfxApiHeader {
                type_: FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE,
                p_next: &raw mut backend.header,
            },
            flags: FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE
                | FFX_UPSCALE_ENABLE_DEPTH_INVERTED
                | FFX_UPSCALE_ENABLE_DEPTH_INFINITE,
            max_render_size: FfxApiDimensions2D {
                width: 1280,
                height: 720,
            },
            max_upscale_size: FfxApiDimensions2D {
                width: 2560,
                height: 1440,
            },
            _pad: 0,
            fp_message: None,
        };

        let mut ctx: *mut c_void = null_mut();
        let rc = ffxCreateContext(&raw mut ctx, &raw mut upscale.header, null());
        assert_eq!(
            rc,
            FFX_API_RETURN_OK,
            "PC FSR3 ffxCreateContext failed (loader = {})",
            loader_path.display()
        );
        assert!(!ctx.is_null());

        let mut provider = FfxQueryGetProviderVersion {
            header: FfxApiHeader {
                type_: FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION,
                p_next: null_mut(),
            },
            version_id: 0,
            version_name: null(),
        };
        assert_eq!(
            ffxQuery(&raw mut ctx, &raw mut provider.header),
            FFX_API_RETURN_OK
        );
        let provider_name = if provider.version_name.is_null() {
            "<null>".to_string()
        } else {
            // SAFETY: provider writes a static NUL-terminated C string.
            unsafe { std::ffi::CStr::from_ptr(provider.version_name) }
                .to_string_lossy()
                .into_owned()
        };
        eprintln!(
            "live PC FSR3 provider: id={:#x} name={provider_name} (loader={})",
            provider.version_id,
            loader_path.display()
        );

        let mut phase_count = 0i32;
        let mut q_phase = FfxQueryDescUpscaleGetJitterPhaseCount {
            header: FfxApiHeader {
                type_: FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTERPHASECOUNT,
                p_next: null_mut(),
            },
            render_width: 1280,
            display_width: 2560,
            out_phase_count: &raw mut phase_count,
        };
        assert_eq!(
            ffxQuery(&raw mut ctx, &raw mut q_phase.header),
            FFX_API_RETURN_OK
        );
        assert_eq!(phase_count, 32);

        let (mut jx, mut jy) = (0.0f32, 0.0f32);
        let mut q_jitter = FfxQueryDescUpscaleGetJitterOffset {
            header: FfxApiHeader {
                type_: FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTEROFFSET,
                p_next: null_mut(),
            },
            index: 0,
            phase_count,
            out_x: &raw mut jx,
            out_y: &raw mut jy,
        };
        assert_eq!(
            ffxQuery(&raw mut ctx, &raw mut q_jitter.header),
            FFX_API_RETURN_OK
        );
        assert!((-0.5..=0.5).contains(&jx) && (-0.5..=0.5).contains(&jy));

        let mut mem = FfxApiEffectMemoryUsage::default();
        let mut q_mem = FfxQueryDescUpscaleGetGpuMemoryUsage {
            header: FfxApiHeader {
                type_: FFX_API_QUERY_DESC_TYPE_UPSCALE_GPU_MEMORY_USAGE,
                p_next: null_mut(),
            },
            gpu_memory_usage_upscaler: &raw mut mem,
        };
        assert_eq!(
            ffxQuery(&raw mut ctx, &raw mut q_mem.header),
            FFX_API_RETURN_OK
        );
        assert!(mem.total_usage_in_bytes > 0);
        eprintln!(
            "live PC FSR3 1440p context VRAM: {} MiB (aliasable {} MiB)",
            mem.total_usage_in_bytes >> 20,
            mem.aliasable_usage_in_bytes >> 20
        );

        assert_eq!(ffxDestroyContext(&raw mut ctx, null()), FFX_API_RETURN_OK);
        assert!(ctx.is_null());
    }
}
