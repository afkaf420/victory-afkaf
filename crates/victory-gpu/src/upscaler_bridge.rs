//! Host D3D12 handle bridge for PC-native upscalers (AMD FidelityFX FSR 3.1 /
//! NVIDIA Streamline DLSS) invoked from `victory-hle`.
//!
//! Translates console device, command-list, and texture pointers into live host
//! D3D12 COM interfaces while keeping VIctory's resource-state tracking, depth
//! colour/plane twins, resource lifetimes, and cached command-list bindings
//! consistent across the external dispatch boundary.

use std::ffi::c_void;

#[cfg(windows)]
use crate::object::This;

/// Portable subset of `D3D12_RESOURCE_DESC` describing a host texture handed
/// to a PC upscaler.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TextureDesc {
    /// `D3D12_RESOURCE_DIMENSION` (1 = Buffer, 2 = Tex1D, 3 = Tex2D, 4 = Tex3D).
    pub dimension: u32,
    pub width: u64,
    pub height: u32,
    pub depth_or_array_size: u16,
    pub mip_levels: u16,
    /// `DXGI_FORMAT` raw numeric value.
    pub dxgi_format: u32,
    /// `D3D12_RESOURCE_FLAGS` raw bitmask.
    pub flags: u32,
}

#[cfg(windows)]
impl From<windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_DESC> for TextureDesc {
    fn from(d: windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_DESC) -> Self {
        Self {
            dimension: d.Dimension.0 as u32,
            width: d.Width,
            height: d.Height,
            depth_or_array_size: d.DepthOrArraySize,
            mip_levels: d.MipLevels,
            dxgi_format: d.Format.0 as u32,
            flags: d.Flags.0 as u32,
        }
    }
}

/// Returns the active host `ID3D12Device*` (borrowed, non-owning) if a host
/// D3D12 device has been initialised.
pub fn host_device_raw() -> Option<*mut c_void> {
    #[cfg(windows)]
    {
        crate::host::device_raw()
    }
    #[cfg(not(windows))]
    {
        None
    }
}

/// Prepares `console_list` for an inline external PC upscaler pass and returns
/// its underlying host `ID3D12GraphicsCommandList*` (borrowed, non-owning).
pub fn host_command_list_prepare(console_list: *const c_void) -> Option<*mut c_void> {
    #[cfg(windows)]
    {
        crate::list::host_list_raw(console_list.cast())
    }
    #[cfg(not(windows))]
    {
        let _ = console_list;
        None
    }
}

/// Unwraps a console texture resource for an external PC upscaler dispatch on
/// `console_list`:
///
/// 1. Flushes any dirty colour/plane depth twin inline so depth reads are up
///    to date before the upscaler executes.
/// 2. Transitions the host texture to `wanted_state` (recording a barrier on
///    `console_list` or queueing it in the list prelude) and pins the host COM
///    resource for the lifetime of the list (`HostList::keep`).
/// 3. Returns the borrowed host `ID3D12Resource*` alongside its
///    [`TextureDesc`].
pub fn host_texture_for_upscaler(
    console_list: *const c_void,
    console_res: *const c_void,
    wanted_state: u32,
) -> Option<(*mut c_void, TextureDesc)> {
    #[cfg(windows)]
    {
        use windows::core::Interface;

        if console_list.is_null() || console_res.is_null() {
            return None;
        }
        let list_this: This = console_list.cast();
        let res_this: This = console_res.cast();
        let list = crate::list::host_list(list_this)?;
        let (_copy, desc) = crate::resource::host_copy_for_external(res_this)?;
        // Ensure depth textures with a dirty colour/plane twin are copied
        // before the upscaler samples them.
        let _ = list.flush_depth_twin_if_dirty(res_this);
        // Transition to the state the PC upscaler expects and pin lifetime.
        let host_copy = list.ensure_texture_state(res_this, wanted_state)?;
        Some((host_copy.texture.as_raw(), TextureDesc::from(desc)))
    }
    #[cfg(not(windows))]
    {
        let _ = (console_list, console_res, wanted_state);
        None
    }
}

/// Completes an external PC upscaler pass recorded onto `console_list`:
///
/// 1. Updates VIctory's tracked end-of-list state for every `(console_res,
///    after_state)` pair touched by the upscaler.
/// 2. Invalidates VIctory's cached descriptor-heap, root-signature, and PSO
///    bindings on `console_list` (replaying non-pipeline state) so the next
///    console draw or dispatch re-binds VIctory's own shader-visible heaps and
///    pipeline cleanly.
pub fn host_upscaler_post_dispatch(console_list: *const c_void, resources: &[(*const c_void, u32)]) {
    #[cfg(windows)]
    {
        if console_list.is_null() {
            return;
        }
        let list_this: This = console_list.cast();
        if let Some(list) = crate::list::host_list(list_this) {
            for &(res, after_state) in resources {
                if !res.is_null() {
                    list.record_external_state(res.cast(), after_state);
                }
            }
        }
        crate::list::invalidate_cached_bindings(list_this);
    }
    #[cfg(not(windows))]
    {
        let _ = (console_list, resources);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_handles_are_tolerated_without_panicking() {
        let null = std::ptr::null();
        assert!(host_command_list_prepare(null).is_none());
        assert!(host_texture_for_upscaler(null, null, 0x40).is_none());
        host_upscaler_post_dispatch(null, &[(null, 0x40)]);
    }

    #[cfg(windows)]
    #[test]
    fn texture_desc_converts_from_d3d12_resource_desc() {
        use windows::Win32::Graphics::Direct3D12::{
            D3D12_RESOURCE_DESC, D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        };
        use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R16G16B16A16_FLOAT;

        let d3d = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 2560,
            Height: 1440,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_R16G16B16A16_FLOAT,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let desc = TextureDesc::from(d3d);
        assert_eq!(desc.dimension, 3);
        assert_eq!((desc.width, desc.height), (2560, 1440));
        assert_eq!((desc.depth_or_array_size, desc.mip_levels), (1, 1));
        assert_eq!(desc.dxgi_format, DXGI_FORMAT_R16G16B16A16_FLOAT.0 as u32);
        assert_eq!(desc.flags, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 as u32);
    }

    #[cfg(windows)]
    #[test]
    fn post_dispatch_invalidates_cached_bindings_on_console_list() {
        use crate::list::{create_allocator, create_list};
        use crate::object::{S_OK, guid, slot};

        let alloc_iid = guid(
            0x6102dee4,
            0xaf59,
            0x4b09,
            [0xb9, 0x99, 0xb4, 0x4d, 0x73, 0xf0, 0x9b, 0x24],
        );
        let list_iid = guid(
            0x7116d91c,
            0xe7e4,
            0x47ce,
            [0xb8, 0xc6, 0xec, 0x81, 0x68, 0xf4, 0x37, 0xe5],
        );
        let mut allocator = std::ptr::null();
        assert_eq!(
            create_allocator(std::ptr::null(), 0, &alloc_iid, &mut allocator),
            S_OK
        );
        let mut cmd_list = std::ptr::null();
        assert_eq!(
            create_list(
                std::ptr::null(),
                0,
                0,
                allocator,
                std::ptr::null(),
                &list_iid,
                &mut cmd_list,
            ),
            S_OK
        );
        assert!(!cmd_list.is_null());

        if crate::host::host().is_some() {
            assert!(host_device_raw().is_some());
            assert!(host_command_list_prepare(cmd_list.cast()).is_some());
            // Should cleanly invalidate cached bindings and replay without error.
            host_upscaler_post_dispatch(cmd_list.cast(), &[]);
        }

        type Release = extern "system" fn(This) -> u32;
        slot::<Release>(cmd_list, 0x10)(cmd_list);
        slot::<Release>(allocator, 0x10)(allocator);
    }
}
