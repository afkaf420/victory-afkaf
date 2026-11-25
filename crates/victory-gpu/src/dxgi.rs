//! The DXGI side of the device: one adapter with one output, reached from
//! the device by interface query. Tables follow the PC interfaces' order
//! (IUnknown, the object methods, then each interface's methods).
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use crate::object::{
    self, ADAPTER, DXGI_DEVICE, E_INVALIDARG, FACTORY, HRESULT, Iid, MORE_DATA, NOT_FOUND, OUTPUT,
    Object, S_OK, Table, This, guid,
};

/// The DXGI device interfaces, base first (as on PC).
pub(crate) const DEVICE_IIDS: [Iid; 3] = [
    guid(
        0x54ec77fa,
        0x1377,
        0x44e6,
        [0x8c, 0x32, 0x88, 0xfd, 0x5f, 0x44, 0xc8, 0x4c],
    ),
    guid(
        0x77db970f,
        0x6276,
        0x48ba,
        [0xba, 0x28, 0x07, 0x01, 0x43, 0xb4, 0x39, 0x2c],
    ),
    guid(
        0x05008617,
        0xfbfd,
        0x4051,
        [0xa7, 0x90, 0x14, 0x48, 0x84, 0xb4, 0xf6, 0xa9],
    ),
];
const ADAPTER_IIDS: [Iid; 3] = [
    guid(
        0x2411e7e1,
        0x12ac,
        0x4ccf,
        [0xbd, 0x14, 0x97, 0x98, 0xe8, 0x53, 0x4d, 0xc0],
    ),
    guid(
        0x29038f61,
        0x3839,
        0x4626,
        [0x91, 0xfd, 0x08, 0x68, 0x79, 0x01, 0x1a, 0x05],
    ),
    guid(
        0x0aa1ae0a,
        0xfa0e,
        0x4b84,
        [0x86, 0x44, 0xe0, 0x5f, 0xf8, 0xe5, 0xac, 0xb5],
    ),
];
const FACTORY_IIDS: [Iid; 3] = [
    guid(
        0x7b7166ec,
        0x21c7,
        0x44ae,
        [0xb2, 0x1a, 0xc9, 0xae, 0x32, 0x1a, 0xe3, 0x69],
    ),
    guid(
        0x770aae78,
        0xf26f,
        0x4dba,
        [0xa8, 0x29, 0x25, 0x3c, 0x83, 0xd1, 0xb3, 0x87],
    ),
    guid(
        0x50c83a1c,
        0xe072,
        0x4c48,
        [0x87, 0xb0, 0x36, 0x30, 0xfa, 0x36, 0xa6, 0xd0],
    ),
];
const OUTPUT_IIDS: [Iid; 1] = [guid(
    0xae02eedb,
    0xc735,
    0x4690,
    [0x8d, 0x52, 0x5a, 0x8d, 0xc2, 0x02, 0x13, 0xaa],
)];

static DEVICE: Object = Object::new(&DEVICE_FUNCTIONS);
static ADAPTER_OBJECT: Object = Object::new(&ADAPTER_FUNCTIONS);
static FACTORY_OBJECT: Object = Object::new(&FACTORY_FUNCTIONS);
static OUTPUT_OBJECT: Object = Object::new(&OUTPUT_FUNCTIONS);

pub(crate) fn device() -> This {
    &DEVICE
}

/// `found` if `iid` is one of `known`.
fn query(kind: usize, iid: *const Iid, known: &[Iid], found: This, out: *mut This) -> HRESULT {
    object::answer(kind, iid, object::is(iid, known).then_some(found), out)
}

/// The DXGI device also answers for the D3D12 device.
extern "system" fn device_query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    if crate::device::is_device(iid) {
        return object::answer(DXGI_DEVICE, iid, Some(crate::device::device()), out);
    }
    query(DXGI_DEVICE, iid, &DEVICE_IIDS, this, out)
}
extern "system" fn device_parent(_: This, iid: *const Iid, out: *mut This) -> HRESULT {
    query(DXGI_DEVICE, iid, &ADAPTER_IIDS, &ADAPTER_OBJECT, out)
}
extern "system" fn device_adapter(_: This, out: *mut This) -> HRESULT {
    object::answer(DXGI_DEVICE, &ADAPTER_IIDS[0], Some(&ADAPTER_OBJECT), out)
}

extern "system" fn adapter_query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    query(ADAPTER, iid, &ADAPTER_IIDS, this, out)
}
extern "system" fn adapter_parent(_: This, iid: *const Iid, out: *mut This) -> HRESULT {
    query(ADAPTER, iid, &FACTORY_IIDS, &FACTORY_OBJECT, out)
}
extern "system" fn adapter_outputs(_: This, index: u32, out: *mut This) -> HRESULT {
    if index != 0 {
        return NOT_FOUND;
    }
    object::answer(ADAPTER, &OUTPUT_IIDS[0], Some(&OUTPUT_OBJECT), out)
}

#[repr(C)]
struct AdapterDescription {
    name: [u16; 128],
    vendor: u32,
    device: u32,
    subsystem: u32,
    revision: u32,
    dedicated_video_memory: usize,
    dedicated_system_memory: usize,
    shared_system_memory: usize,
    luid: [u32; 2],
}
const _: () = assert!(size_of::<AdapterDescription>() == 0x130);

const AMD: u32 = 0x1002;

extern "system" fn adapter_description(_: This, out: *mut AdapterDescription) -> HRESULT {
    if out.is_null() {
        return E_INVALIDARG;
    }
    let mut name = [0; 128];
    for (slot, unit) in name.iter_mut().zip("VIctory console GPU".encode_utf16()) {
        *slot = unit;
    }
    // shortcut: memory sizes are the title's split (GPU-optimal, standard),
    // not measured; revisit once memory is emulated.
    let (gpu_optimal, standard) = crate::console::current().title_memory();
    let description = AdapterDescription {
        name,
        vendor: AMD,
        device: 0,
        subsystem: 0,
        revision: 0,
        dedicated_video_memory: gpu_optimal,
        dedicated_system_memory: 0,
        shared_system_memory: standard,
        luid: [1, 0],
    };
    // SAFETY: checked non-null; the title passes room for the description.
    unsafe { out.write(description) };
    S_OK
}

extern "system" fn factory_query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    query(FACTORY, iid, &FACTORY_IIDS, this, out)
}
extern "system" fn factory_adapters(_: This, index: u32, out: *mut This) -> HRESULT {
    if index != 0 {
        return NOT_FOUND;
    }
    object::answer(FACTORY, &ADAPTER_IIDS[0], Some(&ADAPTER_OBJECT), out)
}

extern "system" fn output_query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    query(OUTPUT, iid, &OUTPUT_IIDS, this, out)
}
extern "system" fn output_parent(_: This, iid: *const Iid, out: *mut This) -> HRESULT {
    query(OUTPUT, iid, &ADAPTER_IIDS, &ADAPTER_OBJECT, out)
}

/// IUnknown 3, object 4, device 5 + 2 + 3.
static DEVICE_FUNCTIONS: Table<32> = {
    let mut f = object::missing_table::<DXGI_DEVICE, 32>();
    f[0] = device_query as *const ();
    f[6] = device_parent as *const (); // 0x30
    f[7] = device_adapter as *const (); // 0x38
    Table(f)
};

/// IUnknown 3, object 4, adapter 3 + 1 + 1.
static ADAPTER_FUNCTIONS: Table<24> = {
    let mut f = object::missing_table::<ADAPTER, 24>();
    f[0] = adapter_query as *const ();
    f[6] = adapter_parent as *const (); // 0x30
    f[7] = adapter_outputs as *const (); // 0x38
    f[8] = adapter_description as *const (); // 0x40
    Table(f)
};

/// IUnknown 3, object 4, factory 5 + 2 + 11.
static FACTORY_FUNCTIONS: Table<32> = {
    let mut f = object::missing_table::<FACTORY, 32>();
    f[0] = factory_query as *const ();
    f[7] = factory_adapters as *const (); // 0x38
    f[12] = factory_adapters as *const (); // 0x60, the same for version 1
    Table(f)
};

/// A display mode as the console lists it: size, refresh rate (numerator,
/// denominator), format, scanline order, scaling, flags.
#[repr(C)]
#[derive(Clone, Copy)]
struct Mode {
    width: u32,
    height: u32,
    refresh: [u32; 2],
    format: u32,
    scanlines: u32,
    scaling: u32,
    flags: u32,
}
const _: () = assert!(size_of::<Mode>() == 32);

/// The size of the display the console reports: `VICTORY_DISPLAY_MODE=WxH`
/// if set (as a console reports the television it is plugged into: a
/// Series X on a 1080p set lists a 1080p mode), else 3840x2160.
/// shortcut: not yet the host display's own size by default; make it so
/// once a title is seen to render for the mode it is given.
fn display_size() -> (u32, u32) {
    static SIZE: std::sync::OnceLock<(u32, u32)> = std::sync::OnceLock::new();
    *SIZE.get_or_init(|| {
        let asked = std::env::var("VICTORY_DISPLAY_MODE").ok().and_then(|v| {
            let (w, h) = v.trim().split_once(['x', 'X'])?;
            Some((w.trim().parse::<u32>().ok()?, h.trim().parse::<u32>().ok()?))
        });
        let size = match asked {
            Some((w, h)) if (640..=7680).contains(&w) && (360..=4320).contains(&h) => (w, h),
            _ => (3840, 2160),
        };
        tracing::info!(width = size.0, height = size.1, "display mode reported to the title");
        size
    })
}

/// Minimum and maximum display refresh rates accepted from `VICTORY_REFRESH_HZ`.
const MIN_REFRESH_HZ: u32 = 30;
const MAX_REFRESH_HZ: u32 = 480;

/// Parses `VICTORY_REFRESH_HZ` into the list of refresh rates reported to the
/// title. Unset or empty defaults to `[60, 120]` so titles that probe for a
/// 120 Hz output mode can unlock 120 Hz frame intervals while preserving 60 Hz.
fn parse_refresh_rates(raw: Option<&str>) -> Vec<u32> {
    let mut rates = Vec::new();
    if let Some(raw) = raw {
        for part in raw.split(',') {
            if let Ok(hz) = part.trim().parse::<u32>()
                && (MIN_REFRESH_HZ..=MAX_REFRESH_HZ).contains(&hz)
                && !rates.contains(&hz)
            {
                rates.push(hz);
            }
        }
    }
    if rates.is_empty() {
        rates.extend_from_slice(&[60, 120]);
    }
    rates
}

/// The refresh rates reported by `IDXGIOutput::GetDisplayModeList`.
/// Controlled by `VICTORY_REFRESH_HZ` (single rate such as `60` or `120`, or a
/// comma-separated list such as `60,120,144`); defaults to `[60, 120]`.
fn refresh_rates() -> &'static [u32] {
    static RATES: std::sync::OnceLock<Vec<u32>> = std::sync::OnceLock::new();
    RATES.get_or_init(|| {
        let raw = std::env::var("VICTORY_REFRESH_HZ").ok();
        let rates = parse_refresh_rates(raw.as_deref());
        tracing::info!(?rates, "display refresh rates reported to the title");
        rates
    })
}

/// The console's mode list: one mode per configured refresh rate (60 Hz and
/// 120 Hz by default so titles that probe for 120 Hz can select 120 Hz frame
/// intervals; override with `VICTORY_REFRESH_HZ`).
extern "system" fn modes(
    _: This,
    _format: u32,
    _flags: u32,
    count: *mut u32,
    out: *mut Mode,
) -> HRESULT {
    // SAFETY: the title passes its count (null is tolerated).
    let Some(count) = (unsafe { count.as_mut() }) else {
        return E_INVALIDARG;
    };
    let rates = refresh_rates();
    let total = rates.len() as u32;
    if out.is_null() {
        *count = total;
        return S_OK;
    }
    if *count < total {
        return MORE_DATA;
    }
    *count = total;
    let (width, height) = display_size();
    for (i, &hz) in rates.iter().enumerate() {
        let mode = Mode {
            width,
            height,
            refresh: [hz, 1],
            format: 0,
            scanlines: 1, // progressive
            scaling: 0,
            flags: 0,
        };
        // SAFETY: checked non-null and `*count >= total > i`.
        unsafe { out.add(i).write(mode) };
    }
    S_OK
}

/// Queries whether the host DXGI factory supports variable-refresh tearing
/// (`DXGI_FEATURE_PRESENT_ALLOW_TEARING`).
#[cfg(windows)]
pub fn host_supports_tearing(factory: &windows::Win32::Graphics::Dxgi::IDXGIFactory2) -> bool {
    use windows::Win32::Graphics::Dxgi::{DXGI_FEATURE_PRESENT_ALLOW_TEARING, IDXGIFactory5};
    use windows::core::Interface;

    let Ok(factory5) = factory.cast::<IDXGIFactory5>() else {
        return false;
    };
    let mut allow: u32 = 0;
    // SAFETY: `allow` is a valid 4-byte out parameter matching `DXGI_FEATURE_PRESENT_ALLOW_TEARING`.
    let hr = unsafe {
        factory5.CheckFeatureSupport(
            DXGI_FEATURE_PRESENT_ALLOW_TEARING,
            (&raw mut allow).cast(),
            size_of::<u32>() as u32,
        )
    };
    hr.is_ok() && allow != 0
}

/// Swap-chain creation flags for flip-discard presentation given whether
/// unlocked/VRR tearing is requested and supported by the host DXGI factory.
#[cfg(windows)]
pub fn swap_chain_flags_for(unlock_fps: bool, tearing_supported: bool) -> u32 {
    use windows::Win32::Graphics::Dxgi::DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING;
    if unlock_fps && tearing_supported {
        DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING.0 as u32
    } else {
        0
    }
}

/// Swap-chain creation flags matching the active `VICTORY_UNLOCK_FPS` setting.
#[cfg(windows)]
pub fn swap_chain_flags(tearing_supported: bool) -> u32 {
    swap_chain_flags_for(crate::frame::allow_tearing(), tearing_supported)
}

/// `IDXGISwapChain::Present(0, flags)` flags given whether unlocked/VRR
/// tearing is requested and supported on the swap chain.
#[cfg(windows)]
pub fn present_flags_for(
    unlock_fps: bool,
    tearing_supported: bool,
) -> windows::Win32::Graphics::Dxgi::DXGI_PRESENT {
    use windows::Win32::Graphics::Dxgi::{DXGI_PRESENT, DXGI_PRESENT_ALLOW_TEARING};
    if unlock_fps && tearing_supported {
        DXGI_PRESENT_ALLOW_TEARING
    } else {
        DXGI_PRESENT(0)
    }
}

/// `IDXGISwapChain::Present(0, flags)` flags matching the active
/// `VICTORY_UNLOCK_FPS` setting.
#[cfg(windows)]
pub fn present_flags(tearing_supported: bool) -> windows::Win32::Graphics::Dxgi::DXGI_PRESENT {
    present_flags_for(crate::frame::allow_tearing(), tearing_supported)
}

/// IUnknown 3, object 4, output 15 (12 as on PC, then 3 console ones).
static OUTPUT_FUNCTIONS: Table<32> = {
    let mut f = object::missing_table::<OUTPUT, 32>();
    f[0] = output_query as *const ();
    f[6] = output_parent as *const (); // 0x30
    f[21] = modes as *const (); // 0xa8
    Table(f)
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::{E_NOINTERFACE, slot};

    type Query = extern "system" fn(This, *const Iid, *mut This) -> HRESULT;
    type Get = extern "system" fn(This, *mut This) -> HRESULT;
    type Indexed = extern "system" fn(This, u32, *mut This) -> HRESULT;
    type Describe = extern "system" fn(This, *mut AdapterDescription) -> HRESULT;

    /// Minecraft's startup: device → DXGI device (newest first) → adapter →
    /// description, factory, first output.
    #[test]
    fn startup_walks_from_device_to_output() {
        let d3d = crate::device::device();
        let newest_pc = guid(
            0x6007896c,
            0x3244,
            0x4afd,
            [0xbf, 0x18, 0xa6, 0xd3, 0xbe, 0xda, 0x50, 0x23],
        );
        let mut dxgi = std::ptr::null();
        let qi = slot::<Query>(d3d, 0);
        assert_eq!(qi(d3d, &newest_pc, &mut dxgi), E_NOINTERFACE);
        assert!(dxgi.is_null());
        assert_eq!(qi(d3d, &DEVICE_IIDS[2], &mut dxgi), S_OK);
        assert_eq!(dxgi, device());

        let mut adapter = std::ptr::null();
        assert_eq!(slot::<Get>(dxgi, 0x38)(dxgi, &mut adapter), S_OK);
        let mut description = std::mem::MaybeUninit::<AdapterDescription>::uninit();
        assert_eq!(
            slot::<Describe>(adapter, 0x40)(adapter, description.as_mut_ptr()),
            S_OK
        );
        // SAFETY: written above.
        assert_eq!(unsafe { description.assume_init() }.vendor, AMD);

        let mut factory = std::ptr::null();
        let parent = slot::<Query>(adapter, 0x30);
        assert_eq!(parent(adapter, &FACTORY_IIDS[2], &mut factory), S_OK);
        let mut again = std::ptr::null();
        assert_eq!(slot::<Indexed>(factory, 0x38)(factory, 0, &mut again), S_OK);
        assert_eq!(again, adapter);

        let mut output = std::ptr::null();
        let outputs = slot::<Indexed>(adapter, 0x38);
        assert_eq!(outputs(adapter, 0, &mut output), S_OK);
        assert!(!output.is_null());
        assert_eq!(outputs(adapter, 1, &mut output), NOT_FOUND);
    }

    #[test]
    fn parse_refresh_rates_handles_defaults_lists_and_bounds() {
        assert_eq!(parse_refresh_rates(None), vec![60, 120]);
        assert_eq!(parse_refresh_rates(Some("")), vec![60, 120]);
        assert_eq!(parse_refresh_rates(Some("60")), vec![60]);
        assert_eq!(parse_refresh_rates(Some("120")), vec![120]);
        assert_eq!(parse_refresh_rates(Some("60, 120, 144, 120")), vec![60, 120, 144]);
        assert_eq!(parse_refresh_rates(Some("0, 9999, nope")), vec![60, 120]);
    }

    #[test]
    fn modes_expose_60_and_120_hz_by_default() {
        type GetModes = extern "system" fn(This, u32, u32, *mut u32, *mut Mode) -> HRESULT;
        let get_modes = slot::<GetModes>(&OUTPUT_OBJECT, 0xa8);

        assert_eq!(get_modes(&OUTPUT_OBJECT, 0, 0, std::ptr::null_mut(), std::ptr::null_mut()), E_INVALIDARG);

        let mut count = 0u32;
        assert_eq!(get_modes(&OUTPUT_OBJECT, 0, 0, &mut count, std::ptr::null_mut()), S_OK);
        let expected = refresh_rates();
        assert_eq!(count as usize, expected.len());
        assert!(expected.contains(&60));
        assert!(expected.contains(&120));

        if count > 1 {
            let mut too_small = count - 1;
            let mut one = std::mem::MaybeUninit::<Mode>::uninit();
            assert_eq!(get_modes(&OUTPUT_OBJECT, 0, 0, &mut too_small, one.as_mut_ptr()), MORE_DATA);
        }

        let mut list = vec![
            Mode {
                width: 0,
                height: 0,
                refresh: [0, 0],
                format: 0,
                scanlines: 0,
                scaling: 0,
                flags: 0,
            };
            count as usize
        ];
        assert_eq!(get_modes(&OUTPUT_OBJECT, 0, 0, &mut count, list.as_mut_ptr()), S_OK);
        let got_hz: Vec<u32> = list.iter().map(|m| m.refresh[0]).collect();
        assert_eq!(got_hz, expected);
        assert!(list.iter().all(|m| m.refresh[1] == 1 && m.scanlines == 1));
    }

    #[cfg(windows)]
    #[test]
    fn tearing_flags_match_unlock_and_host_capability() {
        use windows::Win32::Graphics::Dxgi::{
            CreateDXGIFactory2, DXGI_CREATE_FACTORY_FLAGS, DXGI_PRESENT,
            DXGI_PRESENT_ALLOW_TEARING, DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING, IDXGIFactory2,
        };
        assert_eq!(swap_chain_flags_for(false, true), 0);
        assert_eq!(swap_chain_flags_for(true, false), 0);
        assert_eq!(
            swap_chain_flags_for(true, true),
            DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING.0 as u32
        );
        assert_eq!(present_flags_for(false, true), DXGI_PRESENT(0));
        assert_eq!(present_flags_for(true, false), DXGI_PRESENT(0));
        assert_eq!(present_flags_for(true, true), DXGI_PRESENT_ALLOW_TEARING);

        // SAFETY: creating a read-only DXGI factory to probe feature support.
        if let Ok(factory) = unsafe { CreateDXGIFactory2::<IDXGIFactory2>(DXGI_CREATE_FACTORY_FLAGS(0)) } {
            let _supported = host_supports_tearing(&factory);
        }
    }
}
