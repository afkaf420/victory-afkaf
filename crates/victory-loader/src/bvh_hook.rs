//! Transparent opt-in (`VICTORY_DXR_RAYQUERY=1`) shadow interception of
//! `XboxBvhBuilder.dll` exports for Tier-2 hardware DXR 1.1 BLAS/TLAS mirroring.
//!
//! Every intercepted export forwards to the real `XboxBvhBuilder.dll` function
//! **first** (so console BVH4 buffers remain 100% valid for any unlifted shader)
//! and then notifies the host DXR 1.1 shadow builder (`victory-gpu::dxr`) via a
//! process-scoped named section (`Local\victory_dxr_bridge_<pid>`).

use std::ffi::c_void;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

use victory_pe::ImportSymbol;

const BRIDGE_MAGIC: u64 = 0x5649_4354_4458_5231;
const BRIDGE_VERSION: u32 = 1;

const DLL_NAME: &str = "XboxBvhBuilder.dll";

/// Verified C++ mangled export names from `XboxBvhBuilder.dll`.
pub const SYM_BUILD_SINGLE: &str = "?BuildRaytracingAccelerationStructure@XboxBvhBuilder@@QEAAXPEAUID3D12GraphicsCommandList4@@PEBUD3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC@@IPEBUD3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC@@@Z";
pub const SYM_BUILD_BATCHED_2X: &str = "?BuildRaytracingAccelerationStructureBatched2X@XboxBvhBuilder@@QEAAXPEAUID3D12GraphicsCommandList6@@IPEBUD3D12XBOX_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_BATCHED_DESC2@@@Z";
pub const SYM_REHYDRATE_BATCHED_2X: &str = "?RehydrateRaytracingAccelerationStructureBatched2X@XboxBvhBuilder@@QEAAJPEAUID3D12GraphicsCommandList8@@IPEBUD3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC@@PEB_K@Z";
pub const SYM_REHYDRATE_BATCHED_X: &str = "?RehydrateRaytracingAccelerationStructureBatchedX@XboxBvhBuilder@@QEAAJPEAUID3D12GraphicsCommandList8@@IPEBUD3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC@@@Z";
pub const SYM_COPY_AS: &str = "?CopyRaytracingAccelerationStructure@XboxBvhBuilder@@QEAAXPEAUID3D12GraphicsCommandList4@@_K1W4D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE@@@Z";

/// ABI-stable shadow callback table published by `victory-gpu::dxr` into
/// `Local\victory_dxr_bridge_<pid>`.
#[repr(C)]
pub struct BvhShadowHooks {
    pub magic: u64,
    pub version: u32,
    pub pid: u32,
    pub on_build_single: extern "system" fn(*mut c_void, *const c_void),
    pub on_build_batched_2x: extern "system" fn(*mut c_void, u32, *const c_void),
    pub on_rehydrate_batched: extern "system" fn(*mut c_void, u32, *const c_void, *const u64),
    pub on_copy_as: extern "system" fn(*mut c_void, u64, u64, u32),
}

static HOOKS: AtomicPtr<BvhShadowHooks> = AtomicPtr::new(std::ptr::null_mut());

static REAL_BUILD_SINGLE: AtomicUsize = AtomicUsize::new(0);
static REAL_BUILD_BATCHED_2X: AtomicUsize = AtomicUsize::new(0);
static REAL_REHYDRATE_BATCHED_2X: AtomicUsize = AtomicUsize::new(0);
static REAL_REHYDRATE_BATCHED_X: AtomicUsize = AtomicUsize::new(0);
static REAL_COPY_AS: AtomicUsize = AtomicUsize::new(0);

/// Resolves `VICTORY_DXR_RAYQUERY`: strictly opt-in (`"1"` enables).
fn resolve_dxr_rayquery(env: Option<&str>) -> bool {
    env.is_some_and(|v| v.trim() == "1")
}

/// Whether `XboxBvhBuilder.dll` shadow interception is enabled
/// (`VICTORY_DXR_RAYQUERY=1`).
pub fn active() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let env = std::env::var("VICTORY_DXR_RAYQUERY").ok();
        resolve_dxr_rayquery(env.as_deref())
    })
}

/// Directly registers a static [`BvhShadowHooks`] table (optional override for
/// embedding callers or tests).
pub fn register_hooks(hooks: &'static BvhShadowHooks) {
    HOOKS.store(std::ptr::from_ref(hooks).cast_mut(), Ordering::Release);
}

/// Discovers the `victory-gpu::dxr` shadow callback table from the process-local
/// `Local\victory_dxr_bridge_<pid>` mapping, caching the validated pointer once
/// discovered.
fn resolve_hooks() -> Option<&'static BvhShadowHooks> {
    let ptr = HOOKS.load(Ordering::Acquire);
    if !ptr.is_null() {
        // SAFETY: `HOOKS` holds either a `&'static BvhShadowHooks` or a
        // validated process-lifetime read-only file-mapping view.
        return Some(unsafe { &*ptr });
    }
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::Memory::{FILE_MAP_READ, MapViewOfFile, OpenFileMappingW};
    use windows_sys::Win32::System::Threading::GetCurrentProcessId;

    // SAFETY: opening and mapping a read-only process-scoped section published
    // by `victory-gpu::dxr::publish_loader_hooks`.
    unsafe {
        let pid = GetCurrentProcessId();
        let name: Vec<u16> = format!("Local\\victory_dxr_bridge_{pid}\0")
            .encode_utf16()
            .collect();
        let handle = OpenFileMappingW(FILE_MAP_READ, 0, name.as_ptr());
        if handle.is_null() {
            return None;
        }
        let view = MapViewOfFile(handle, FILE_MAP_READ, 0, 0, size_of::<BvhShadowHooks>());
        CloseHandle(handle);
        if view.Value.is_null() {
            return None;
        }
        let block = view.Value.cast::<BvhShadowHooks>();
        let hdr = &*block;
        if hdr.magic != BRIDGE_MAGIC || hdr.version != BRIDGE_VERSION || hdr.pid != pid {
            return None;
        }
        // Cache for O(1) atomic loads on all subsequent BVH builds; the view
        // intentionally lives for the remainder of the process.
        match HOOKS.compare_exchange(
            std::ptr::null_mut(),
            block,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => Some(&*block),
            Err(existing) if !existing.is_null() => Some(&*existing),
            Err(_) => Some(&*block),
        }
    }
}

/// Wraps a resolved `XboxBvhBuilder.dll` export with a transparent forward-first
/// shadow trampoline when `VICTORY_DXR_RAYQUERY=1`. Returns `real_addr`
/// unchanged when disabled or for non-intercepted exports/DLLs.
pub(crate) fn wrap_export(dll: &str, symbol: &ImportSymbol, real_addr: usize) -> usize {
    wrap_export_if(active(), dll, symbol, real_addr)
}

fn wrap_export_if(enabled: bool, dll: &str, symbol: &ImportSymbol, real_addr: usize) -> usize {
    if !enabled || real_addr == 0 || !dll.eq_ignore_ascii_case(DLL_NAME) {
        return real_addr;
    }
    let ImportSymbol::Name { name, .. } = symbol else {
        return real_addr;
    };
    let wrapped = match name.as_str() {
        SYM_BUILD_SINGLE => {
            REAL_BUILD_SINGLE.store(real_addr, Ordering::Release);
            thunk_build_single as *const () as usize
        }
        SYM_BUILD_BATCHED_2X => {
            REAL_BUILD_BATCHED_2X.store(real_addr, Ordering::Release);
            thunk_build_batched_2x as *const () as usize
        }
        SYM_REHYDRATE_BATCHED_2X => {
            REAL_REHYDRATE_BATCHED_2X.store(real_addr, Ordering::Release);
            thunk_rehydrate_batched_2x as *const () as usize
        }
        SYM_REHYDRATE_BATCHED_X => {
            REAL_REHYDRATE_BATCHED_X.store(real_addr, Ordering::Release);
            thunk_rehydrate_batched_x as *const () as usize
        }
        SYM_COPY_AS => {
            REAL_COPY_AS.store(real_addr, Ordering::Release);
            thunk_copy_as as *const () as usize
        }
        _ => return real_addr,
    };
    tracing::info!(
        dll,
        symbol = name.as_str(),
        real = format_args!("{real_addr:#x}"),
        "installed DXR 1.1 shadow trampoline"
    );
    wrapped
}

extern "system" fn thunk_build_single(
    this: *mut c_void,
    list: *mut c_void,
    desc: *const c_void,
    num_postbuild: u32,
    postbuild: *const c_void,
) {
    let real = REAL_BUILD_SINGLE.load(Ordering::Acquire);
    if real != 0 {
        // SAFETY: exact x64 C++ member ABI of `XboxBvhBuilder::BuildRaytracingAccelerationStructure`.
        let f: extern "system" fn(*mut c_void, *mut c_void, *const c_void, u32, *const c_void) =
            unsafe { std::mem::transmute(real) };
        f(this, list, desc, num_postbuild, postbuild);
    }
    if let Some(hooks) = resolve_hooks() {
        (hooks.on_build_single)(list, desc);
    }
}

extern "system" fn thunk_build_batched_2x(
    this: *mut c_void,
    list: *mut c_void,
    count: u32,
    descs: *const c_void,
) {
    let real = REAL_BUILD_BATCHED_2X.load(Ordering::Acquire);
    if real != 0 {
        // SAFETY: exact x64 C++ member ABI of `XboxBvhBuilder::BuildRaytracingAccelerationStructureBatched2X`.
        let f: extern "system" fn(*mut c_void, *mut c_void, u32, *const c_void) =
            unsafe { std::mem::transmute(real) };
        f(this, list, count, descs);
    }
    if let Some(hooks) = resolve_hooks() {
        (hooks.on_build_batched_2x)(list, count, descs);
    }
}

extern "system" fn thunk_rehydrate_batched_2x(
    this: *mut c_void,
    list: *mut c_void,
    count: u32,
    descs: *const c_void,
    src_vas: *const u64,
) -> i32 {
    let real = REAL_REHYDRATE_BATCHED_2X.load(Ordering::Acquire);
    let hr = if real != 0 {
        // SAFETY: exact x64 C++ member ABI of `XboxBvhBuilder::RehydrateRaytracingAccelerationStructureBatched2X`.
        let f: extern "system" fn(
            *mut c_void,
            *mut c_void,
            u32,
            *const c_void,
            *const u64,
        ) -> i32 = unsafe { std::mem::transmute(real) };
        f(this, list, count, descs, src_vas)
    } else {
        0
    };
    if hr >= 0
        && let Some(hooks) = resolve_hooks()
    {
        (hooks.on_rehydrate_batched)(list, count, descs, src_vas);
    }
    hr
}

extern "system" fn thunk_rehydrate_batched_x(
    this: *mut c_void,
    list: *mut c_void,
    count: u32,
    descs: *const c_void,
) -> i32 {
    let real = REAL_REHYDRATE_BATCHED_X.load(Ordering::Acquire);
    let hr = if real != 0 {
        // SAFETY: exact x64 C++ member ABI of `XboxBvhBuilder::RehydrateRaytracingAccelerationStructureBatchedX`.
        let f: extern "system" fn(*mut c_void, *mut c_void, u32, *const c_void) -> i32 =
            unsafe { std::mem::transmute(real) };
        f(this, list, count, descs)
    } else {
        0
    };
    if hr >= 0
        && let Some(hooks) = resolve_hooks()
    {
        (hooks.on_rehydrate_batched)(list, count, descs, std::ptr::null());
    }
    hr
}

extern "system" fn thunk_copy_as(
    this: *mut c_void,
    list: *mut c_void,
    dst_va: u64,
    src_va: u64,
    mode: u32,
) {
    let real = REAL_COPY_AS.load(Ordering::Acquire);
    if real != 0 {
        // SAFETY: exact x64 C++ member ABI of `XboxBvhBuilder::CopyRaytracingAccelerationStructure`.
        let f: extern "system" fn(*mut c_void, *mut c_void, u64, u64, u32) =
            unsafe { std::mem::transmute(real) };
        f(this, list, dst_va, src_va, mode);
    }
    if let Some(hooks) = resolve_hooks() {
        (hooks.on_copy_as)(list, dst_va, src_va, mode);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;

    static LOG: Mutex<Vec<&'static str>> = Mutex::new(Vec::new());
    static TEST_GUARD: Mutex<()> = Mutex::new(());

    extern "system" fn fake_real_batched(
        _this: *mut c_void,
        _list: *mut c_void,
        count: u32,
        _descs: *const c_void,
    ) {
        assert_eq!(count, 3);
        LOG.lock().unwrap().push("real_batched");
    }

    extern "system" fn fake_real_rehydrate(
        _this: *mut c_void,
        _list: *mut c_void,
        count: u32,
        _descs: *const c_void,
        src_vas: *const u64,
    ) -> i32 {
        assert_eq!(count, 2);
        assert!(!src_vas.is_null());
        LOG.lock().unwrap().push("real_rehydrate");
        0x1234
    }

    extern "system" fn fake_real_copy(
        _this: *mut c_void,
        _list: *mut c_void,
        dst: u64,
        src: u64,
        mode: u32,
    ) {
        assert_eq!((dst, src, mode), (0x2000, 0x1000, 1));
        LOG.lock().unwrap().push("real_copy");
    }

    extern "system" fn shadow_single(_list: *mut c_void, _desc: *const c_void) {
        LOG.lock().unwrap().push("shadow_single");
    }

    extern "system" fn shadow_batched(_list: *mut c_void, count: u32, _descs: *const c_void) {
        assert_eq!(count, 3);
        LOG.lock().unwrap().push("shadow_batched");
    }

    extern "system" fn shadow_rehydrate(
        _list: *mut c_void,
        count: u32,
        _descs: *const c_void,
        src_vas: *const u64,
    ) {
        assert_eq!(count, 2);
        assert!(!src_vas.is_null());
        LOG.lock().unwrap().push("shadow_rehydrate");
    }

    extern "system" fn shadow_copy(_list: *mut c_void, dst: u64, src: u64, mode: u32) {
        assert_eq!((dst, src, mode), (0x2000, 0x1000, 1));
        LOG.lock().unwrap().push("shadow_copy");
    }

    #[test]
    fn env_parsing_is_strictly_opt_in() {
        assert!(!resolve_dxr_rayquery(None));
        assert!(!resolve_dxr_rayquery(Some("0")));
        assert!(!resolve_dxr_rayquery(Some("yes")));
        assert!(resolve_dxr_rayquery(Some("1")));
        assert!(resolve_dxr_rayquery(Some(" 1 ")));
    }

    #[test]
    fn wraps_xbox_bvh_builder_exports_and_forwards_real_first_then_shadow() {
        let _lock = TEST_GUARD.lock().unwrap();
        LOG.lock().unwrap().clear();

        // Disabled -> returns original address unchanged.
        let sym_batched = ImportSymbol::Name {
            hint: 0,
            name: SYM_BUILD_BATCHED_2X.to_owned(),
        };
        let real_b = fake_real_batched as *const () as usize;
        assert_eq!(wrap_export_if(false, DLL_NAME, &sym_batched, real_b), real_b);

        // Other DLL -> untouched even when enabled.
        assert_eq!(wrap_export_if(true, "other.dll", &sym_batched, real_b), real_b);

        // Unrelated export on XboxBvhBuilder -> untouched.
        let sym_ctor = ImportSymbol::Name {
            hint: 0,
            name: "??0XboxBvhBuilder@@QEAA@PEAUID3D12Device11@@_N@Z".to_owned(),
        };
        assert_eq!(wrap_export_if(true, DLL_NAME, &sym_ctor, 0xDEAD_BEEF), 0xDEAD_BEEF);

        static TEST_HOOKS: BvhShadowHooks = BvhShadowHooks {
            magic: BRIDGE_MAGIC,
            version: BRIDGE_VERSION,
            pid: 0,
            on_build_single: shadow_single,
            on_build_batched_2x: shadow_batched,
            on_rehydrate_batched: shadow_rehydrate,
            on_copy_as: shadow_copy,
        };
        register_hooks(&TEST_HOOKS);

        let wrapped_b = wrap_export_if(true, "xboxbvhbuilder.dll", &sym_batched, real_b);
        assert_ne!(wrapped_b, real_b);

        let sym_rehydrate = ImportSymbol::Name {
            hint: 0,
            name: SYM_REHYDRATE_BATCHED_2X.to_owned(),
        };
        let real_r = fake_real_rehydrate as *const () as usize;
        let wrapped_r = wrap_export_if(true, DLL_NAME, &sym_rehydrate, real_r);
        assert_ne!(wrapped_r, real_r);

        let sym_copy = ImportSymbol::Name {
            hint: 0,
            name: SYM_COPY_AS.to_owned(),
        };
        let real_c = fake_real_copy as *const () as usize;
        let wrapped_c = wrap_export_if(true, DLL_NAME, &sym_copy, real_c);
        assert_ne!(wrapped_c, real_c);

        // Invoke all three trampolines and verify strict real-first -> shadow-second order.
        let dummy_src: u64 = 0x5555;
        // SAFETY: trampolines match exact signatures of the fake exports above.
        unsafe {
            let fb: extern "system" fn(*mut c_void, *mut c_void, u32, *const c_void) =
                std::mem::transmute(wrapped_b);
            fb(std::ptr::null_mut(), std::ptr::null_mut(), 3, std::ptr::null());

            let fr: extern "system" fn(
                *mut c_void,
                *mut c_void,
                u32,
                *const c_void,
                *const u64,
            ) -> i32 = std::mem::transmute(wrapped_r);
            let hr = fr(
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                2,
                std::ptr::null(),
                &raw const dummy_src,
            );
            assert_eq!(hr, 0x1234, "must preserve real XboxBvhBuilder HRESULT");

            let fc: extern "system" fn(*mut c_void, *mut c_void, u64, u64, u32) =
                std::mem::transmute(wrapped_c);
            fc(std::ptr::null_mut(), std::ptr::null_mut(), 0x2000, 0x1000, 1);
        }

        let got = LOG.lock().unwrap().clone();
        assert_eq!(
            got,
            vec![
                "real_batched",
                "shadow_batched",
                "real_rehydrate",
                "shadow_rehydrate",
                "real_copy",
                "shadow_copy",
            ]
        );
        HOOKS.store(std::ptr::null_mut(), Ordering::Release);
    }

    #[test]
    fn discovers_hooks_across_local_named_bridge_section() {
        use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, PAGE_READWRITE, UnmapViewOfFile,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcessId;

        let _lock = TEST_GUARD.lock().unwrap();
        HOOKS.store(std::ptr::null_mut(), Ordering::Release);
        LOG.lock().unwrap().clear();

        // SAFETY: creating a per-PID anonymous named section matching `victory-gpu::dxr`.
        unsafe {
            let pid = GetCurrentProcessId();
            let name: Vec<u16> = format!("Local\\victory_dxr_bridge_{pid}\0")
                .encode_utf16()
                .collect();
            let handle = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                size_of::<BvhShadowHooks>() as u32,
                name.as_ptr(),
            );
            assert!(!handle.is_null(), "CreateFileMappingW failed");
            let view = MapViewOfFile(handle, FILE_MAP_WRITE, 0, 0, size_of::<BvhShadowHooks>());
            assert!(!view.Value.is_null(), "MapViewOfFile failed");
            view.Value.cast::<BvhShadowHooks>().write(BvhShadowHooks {
                magic: BRIDGE_MAGIC,
                version: BRIDGE_VERSION,
                pid,
                on_build_single: shadow_single,
                on_build_batched_2x: shadow_batched,
                on_rehydrate_batched: shadow_rehydrate,
                on_copy_as: shadow_copy,
            });

            let discovered = resolve_hooks().expect("resolve_hooks must find named bridge");
            (discovered.on_build_single)(std::ptr::null_mut(), std::ptr::null());
            assert_eq!(LOG.lock().unwrap().as_slice(), &["shadow_single"]);

            let cached = HOOKS.swap(std::ptr::null_mut(), Ordering::AcqRel);
            if !cached.is_null() {
                UnmapViewOfFile(windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: cached.cast(),
                });
            }
            UnmapViewOfFile(view);
            CloseHandle(handle);
        }
    }
}
