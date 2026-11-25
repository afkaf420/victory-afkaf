//! Phase 5A: Tier-2 Hardware DXR 1.1 Shadow BLAS/TLAS Builder & `SLOT_TLAS_MAP`
//! (`t251`) for inline SM 6.6 `RayQuery` execution on hardware RT Cores
//! (`VICTORY_DXR_RAYQUERY=1`, strictly opt-in).
//!
//! When enabled alongside `XboxBvhBuilder.dll` shadow interception
//! (`victory-loader::bvh_hook`), every console BVH4 build, rehydration, and copy
//! continues to run through the real `XboxBvhBuilder.dll` (preserving 100%
//! fallback parity for unlifted shaders) while this module mirrors the exact
//! triangle/AABB geometry and instance tables into native PC DXR 1.1 BLAS/TLAS
//! acceleration structures in 24 GB VRAM `DEFAULT` pools.
//!
//! Built TLASes are published as `D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE`
//! descriptors inside the fixed 64-slot run `TLAS_BASE..TLAS_BASE+64` right
//! below `SPARSE_BASE` (`null_slot - 16 - 64`) in every shader-visible view
//! heap, and indexed via the root SRV table `SLOT_TLAS_MAP` (`t251`):
//! 64 entries × 16 bytes `[tlas_va_shr8_lo, tlas_va_shr8_hi, host_tlas_srv_slot, valid]`.

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::Interface;

/// Root SRV register `t251` in `ROOT_CBV_SPACE` holding the console-TLAS-VA to
/// host descriptor-heap slot lookup table (shared contract with Track 5B).
pub const SLOT_TLAS_MAP: u8 = 251;

/// Fixed descriptor-heap slots reserved right below `SPARSE_BASE` for host
/// `D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE` SRVs.
pub const TLAS_SLOTS: u32 = 64;

const _: () = assert!(SLOT_TLAS_MAP == victory_shader::emit::SLOT_TLAS_MAP);
const _: () = assert!(TLAS_SLOTS == victory_shader::emit::TLAS_MAP_ENTRIES);

/// Size of each `SLOT_TLAS_MAP` entry in bytes:
/// `[tlas_va_shr8_lo: u32, tlas_va_shr8_hi: u32, host_tlas_srv_slot: u32, valid: u32]`.
pub const TLAS_MAP_ENTRY_BYTES: usize = 16;

/// Total size of the `SLOT_TLAS_MAP` GPU table in bytes (`64 × 16 = 1024 B`).
pub const TLAS_MAP_BYTES: usize = (TLAS_SLOTS as usize) * TLAS_MAP_ENTRY_BYTES;

/// Required byte alignment for DXR acceleration structure GPU virtual addresses.
pub const AS_ALIGNMENT: u64 = 256;

/// Size of each `DEFAULT` heap chunk used to sub-allocate BLAS/TLAS buffers (64 MiB).
const AS_POOL_CHUNK_BYTES: u64 = 64 << 20;

/// Minimum size of the pooled `DEFAULT` scratch buffer (16 MiB).
const SCRATCH_MIN_BYTES: u64 = 16 << 20;

/// Valid PC `D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS` bitmask
/// (`ALLOW_UPDATE..MINIMIZE_MEMORY`), stripping any console-private upper bits.
const PC_BUILD_FLAGS_MASK: i32 = 0x3F;

/// Magic header shared with `victory-loader::bvh_hook` across the per-PID
/// `Local\victory_dxr_bridge_<pid>` named section (`"VICTDXR1"`).
const BRIDGE_MAGIC: u64 = 0x5649_4354_4458_5231;
const BRIDGE_VERSION: u32 = 1;

/// Counters logged periodically by `Meter::frame`:
/// `[0]` BLAS builds, `[1]` TLAS builds, `[2]` AS copies, `[3]` evictions,
/// `[4]` live BLAS count, `[5]` live TLAS count, `[6]` AS pool bytes alive.
static STATS: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];

/// Monotonic generation bumped whenever any TLAS slot changes, so shader-visible
/// view heaps (`HostHeap::sync_tlas_views`) update descriptors only when needed.
static TLAS_GEN: AtomicU64 = AtomicU64::new(0);

/// Resolves `VICTORY_DXR_RAYQUERY`: strictly opt-in (`"1"` enables; unset or
/// any other value keeps all Tier-2 DXR hooks inactive).
fn resolve_dxr_rayquery(env: Option<&str>) -> bool {
    env.is_some_and(|v| v.trim() == "1")
}

/// Whether Tier-2 DXR 1.1 shadow BLAS/TLAS building & `SLOT_TLAS_MAP` binding
/// are enabled (`VICTORY_DXR_RAYQUERY=1`).
pub fn active() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let env = std::env::var("VICTORY_DXR_RAYQUERY").ok();
        resolve_dxr_rayquery(env.as_deref())
    })
}

/// Whether the active host GPU supports DXR Tier 1.1 (`RayQuery`).
pub fn raytracing_tier_1_1() -> bool {
    crate::host::raytracing_tier_1_1()
}

/// Fixed base slot of the 64 host TLAS SRVs right below `SPARSE_BASE`
/// (`null_slot - 16 - 64`), identical across all shader-visible view heaps.
pub fn tlas_base_slot() -> Option<u32> {
    let null = victory_shader::emit::view_heap_limit();
    if null == 0 {
        return None;
    }
    null.checked_sub(crate::sparse_uma::SPARSE_SLOTS)?
        .checked_sub(TLAS_SLOTS)
}

/// Current TLAS descriptor generation counter (0 until the first TLAS is built).
pub(crate) fn tlas_generation() -> u64 {
    TLAS_GEN.load(Ordering::Acquire)
}

/// Aligns `value` up to a power-of-two `align`.
#[inline]
fn align_up(value: u64, align: u64) -> u64 {
    debug_assert!(align.is_power_of_two());
    (value + (align - 1)) & !(align - 1)
}

/// Console `D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC` (48 bytes),
/// verified bit-identical to PC D3D12.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ConsoleBuildDesc {
    pub dest_as_va: u64,
    pub inputs_type: i32,
    pub inputs_flags: i32,
    pub num_descs: u32,
    pub descs_layout: i32,
    /// `InstanceDescs` GPU VA (TLAS), `pGeometryDescs` (ARRAY), or
    /// `ppGeometryDescs` (ARRAY_OF_POINTERS).
    pub descs_ptr_or_va: u64,
    pub source_as_va: u64,
    pub scratch_as_va: u64,
}

/// Console `D3D12XBOX_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_BATCHED_DESC2`
/// (80 bytes = `0x50`, verified from `XboxBvhBuilder.dll` entry disassembly).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ConsoleBatchedDesc2 {
    pub desc: ConsoleBuildDesc,
    pub num_postbuild_info_descs: u32,
    pub _pad: u32,
    pub postbuild_info_descs: u64,
    pub _xbox_extra: [u64; 2],
}

/// 64-byte `D3D12_RAYTRACING_INSTANCE_DESC` with explicit raw fields so bytes
/// `0..56` (`Transform`, `InstanceID`, `InstanceMask`,
/// `InstanceContributionToHitGroupIndex`, `Flags`) are preserved bit-for-bit.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InstanceDesc {
    pub transform: [[f32; 4]; 3],
    pub instance_id_and_mask: u32,
    pub contribution_and_flags: u32,
    pub acceleration_structure: u64,
}

impl InstanceDesc {
    /// Packs `InstanceID` (low 24 bits) and `InstanceMask` (high 8 bits).
    pub const fn pack_id_mask(instance_id: u32, instance_mask: u8) -> u32 {
        (instance_id & 0x00FF_FFFF) | ((instance_mask as u32) << 24)
    }

    /// Packs `InstanceContributionToHitGroupIndex` (low 24 bits) and `Flags` (high 8 bits).
    pub const fn pack_contrib_flags(contribution: u32, flags: u8) -> u32 {
        (contribution & 0x00FF_FFFF) | ((flags as u32) << 24)
    }
}

const _: () = assert!(size_of::<ConsoleBuildDesc>() == 0x30);
const _: () = assert!(size_of::<ConsoleBatchedDesc2>() == 0x50);
const _: () = assert!(size_of::<InstanceDesc>() == 64);
const _: () = assert!(size_of::<InstanceDesc>() == size_of::<D3D12_RAYTRACING_INSTANCE_DESC>());

/// Packs the 4 dwords of one `SLOT_TLAS_MAP` entry.
#[inline]
pub const fn encode_tlas_map_entry(console_tlas_va: u64, host_tlas_srv_slot: u32) -> [u32; 4] {
    let shr8 = console_tlas_va >> 8;
    [shr8 as u32, (shr8 >> 32) as u32, host_tlas_srv_slot, 1]
}

/// Helper: byte size of vertex/index DXGI formats commonly passed in BLAS descs.
fn dxgi_format_bytes(fmt: DXGI_FORMAT) -> u64 {
    match fmt {
        DXGI_FORMAT_R32G32B32_FLOAT => 12,
        DXGI_FORMAT_R32G32_FLOAT | DXGI_FORMAT_R16G16B16A16_FLOAT
        | DXGI_FORMAT_R16G16B16A16_SNORM | DXGI_FORMAT_R16G16B16A16_UNORM => 8,
        DXGI_FORMAT_R16G16_FLOAT | DXGI_FORMAT_R16G16_SNORM | DXGI_FORMAT_R16G16_UNORM
        | DXGI_FORMAT_R32_UINT => 4,
        DXGI_FORMAT_R16_UINT => 2,
        DXGI_FORMAT_R32G32B32A32_FLOAT => 16,
        _ => 12,
    }
}

/// Allocates a `DEFAULT` buffer resource with the given size, flags, and initial state.
fn create_default_buffer(
    device: &ID3D12Device,
    bytes: u64,
    flags: D3D12_RESOURCE_FLAGS,
    initial_state: D3D12_RESOURCE_STATES,
) -> Option<ID3D12Resource> {
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: bytes.max(AS_ALIGNMENT),
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: flags,
        ..Default::default()
    };
    let mut res: Option<ID3D12Resource> = None;
    // SAFETY: valid D3D12 committed buffer parameters.
    unsafe {
        device
            .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, initial_state, None, &mut res)
            .ok()?;
    }
    res
}

/// Allocates a persistently mapped `UPLOAD` buffer resource (`GENERIC_READ`).
fn create_upload_buffer(device: &ID3D12Device, bytes: u64) -> Option<(ID3D12Resource, u64, *mut u8)> {
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_UPLOAD,
        ..Default::default()
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: bytes.max(256),
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    let mut res: Option<ID3D12Resource> = None;
    let mut mapped = std::ptr::null_mut();
    let read_none = D3D12_RANGE { Begin: 0, End: 0 };
    // SAFETY: valid upload buffer creation and write-only map.
    unsafe {
        device
            .CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_GENERIC_READ,
                None,
                &mut res,
            )
            .ok()?;
        let buf = res?;
        buf.Map(0, Some(&read_none), Some(&mut mapped)).ok()?;
        let gpu = buf.GetGPUVirtualAddress();
        Some((buf, gpu, mapped.cast()))
    }
}

/// Emits a global UAV barrier (`pResource = None`) on a `ID3D12GraphicsCommandList4`.
fn emit_uav_barrier(list: &ID3D12GraphicsCommandList4) {
    let barrier = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                pResource: std::mem::ManuallyDrop::new(None),
            }),
        },
    };
    // SAFETY: open command list and valid global UAV barrier.
    unsafe { list.ResourceBarrier(&[barrier]) };
}

/// A 256-byte-aligned slice sub-allocated inside a `DEFAULT` AS pool chunk.
#[derive(Clone)]
struct AsAlloc {
    chunk_id: usize,
    resource: ID3D12Resource,
    offset: u64,
    size: u64,
    gpu_va: u64,
}

struct AsChunk {
    id: usize,
    resource: ID3D12Resource,
    gpu_base: u64,
    size: u64,
    /// Coalesced free spans `(offset, len)` sorted by `offset`.
    free: Vec<(u64, u64)>,
    used_bytes: u64,
}

struct RetiredAs {
    retire_fence: u64,
    chunk_id: usize,
    offset: u64,
    size: u64,
    /// Keeps the chunk resource alive until the GPU passes `retire_fence`.
    _keep: ID3D12Resource,
}

/// Sub-allocates 256-byte-aligned BLAS/TLAS buffers out of 64 MiB `DEFAULT`
/// `D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE` VRAM chunks.
struct AsPool {
    next_id: usize,
    chunks: Vec<AsChunk>,
    retired: Vec<RetiredAs>,
}

// SAFETY: D3D12 COM interfaces are free-threaded; mutated only under `DxrEngine` lock.
unsafe impl Send for AsPool {}

impl AsPool {
    fn new() -> Self {
        Self {
            next_id: 1,
            chunks: Vec::new(),
            retired: Vec::new(),
        }
    }

    fn total_bytes(&self) -> u64 {
        self.chunks.iter().map(|c| c.size).sum()
    }

    fn alloc(&mut self, device: &ID3D12Device, raw_size: u64) -> Option<AsAlloc> {
        let size = align_up(raw_size.max(AS_ALIGNMENT), AS_ALIGNMENT);
        for chunk in &mut self.chunks {
            if let Some(pos) = chunk.free.iter().position(|&(_, len)| len >= size) {
                let (off, len) = chunk.free[pos];
                if len == size {
                    chunk.free.remove(pos);
                } else {
                    chunk.free[pos] = (off + size, len - size);
                }
                chunk.used_bytes += size;
                return Some(AsAlloc {
                    chunk_id: chunk.id,
                    resource: chunk.resource.clone(),
                    offset: off,
                    size,
                    gpu_va: chunk.gpu_base + off,
                });
            }
        }
        let chunk_bytes = align_up(size.max(AS_POOL_CHUNK_BYTES), 64 << 10);
        let resource = create_default_buffer(
            device,
            chunk_bytes,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
        )?;
        crate::host::name(&resource, || {
            format!("DXR 1.1 AS pool chunk #{} ({} MiB)", self.next_id, chunk_bytes >> 20)
        });
        // SAFETY: live committed buffer.
        let gpu_base = unsafe { resource.GetGPUVirtualAddress() };
        let id = self.next_id;
        self.next_id += 1;
        let mut free = Vec::new();
        if chunk_bytes > size {
            free.push((size, chunk_bytes - size));
        }
        let alloc = AsAlloc {
            chunk_id: id,
            resource: resource.clone(),
            offset: 0,
            size,
            gpu_va: gpu_base,
        };
        self.chunks.push(AsChunk {
            id,
            resource,
            gpu_base,
            size: chunk_bytes,
            free,
            used_bytes: size,
        });
        STATS[6].store(self.total_bytes(), Ordering::Relaxed);
        Some(alloc)
    }

    fn defer_free(&mut self, alloc: AsAlloc, retire_fence: u64) {
        self.retired.push(RetiredAs {
            retire_fence,
            chunk_id: alloc.chunk_id,
            offset: alloc.offset,
            size: alloc.size,
            _keep: alloc.resource,
        });
    }

    fn reclaim(&mut self, completed_fence: u64) -> usize {
        let mut reclaimed = 0usize;
        let mut pending = Vec::new();
        for item in self.retired.drain(..) {
            if item.retire_fence <= completed_fence {
                reclaimed += 1;
                if let Some(chunk) = self.chunks.iter_mut().find(|c| c.id == item.chunk_id) {
                    chunk.used_bytes = chunk.used_bytes.saturating_sub(item.size);
                    insert_coalesced(&mut chunk.free, item.offset, item.size);
                }
            } else {
                pending.push(item);
            }
        }
        self.retired = pending;
        // Release excess completely empty chunks (keep at most 1 warm 64 MiB chunk).
        let mut kept_empty = false;
        self.chunks.retain(|c| {
            if c.used_bytes > 0 {
                true
            } else if !kept_empty && c.size == AS_POOL_CHUNK_BYTES {
                kept_empty = true;
                true
            } else {
                false
            }
        });
        STATS[6].store(self.total_bytes(), Ordering::Relaxed);
        reclaimed
    }
}

fn insert_coalesced(free: &mut Vec<(u64, u64)>, mut off: u64, mut len: u64) {
    let idx = free.partition_point(|&(o, _)| o < off);
    if idx > 0 {
        let (po, pl) = free[idx - 1];
        if po + pl == off {
            off = po;
            len += pl;
            free.remove(idx - 1);
            let next_idx = idx - 1;
            if next_idx < free.len() && off + len == free[next_idx].0 {
                len += free[next_idx].1;
                free[next_idx] = (off, len);
            } else {
                free.insert(next_idx, (off, len));
            }
            return;
        }
    }
    if idx < free.len() && off + len == free[idx].0 {
        len += free[idx].1;
        free[idx] = (off, len);
    } else {
        free.insert(idx, (off, len));
    }
}

/// Scratch buffer pool in `DEFAULT` (`UNORDERED_ACCESS`) VRAM with deferred
/// retirement when grown.
struct ScratchPool {
    buffer: Option<(ID3D12Resource, u64, u64)>,
    retired: Vec<(u64, ID3D12Resource)>,
}

// SAFETY: D3D12 COM resources are free-threaded.
unsafe impl Send for ScratchPool {}

impl ScratchPool {
    fn new() -> Self {
        Self {
            buffer: None,
            retired: Vec::new(),
        }
    }

    fn ensure(
        &mut self,
        device: &ID3D12Device,
        needed: u64,
        retire_fence: u64,
    ) -> Option<(ID3D12Resource, u64)> {
        let req = align_up(needed.max(AS_ALIGNMENT), AS_ALIGNMENT);
        if let Some((ref buf, gpu, cap)) = self.buffer
            && cap >= req
        {
            return Some((buf.clone(), gpu));
        }
        let alloc_bytes = align_up(req.max(SCRATCH_MIN_BYTES), 64 << 10);
        let buf = create_default_buffer(
            device,
            alloc_bytes,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        )?;
        crate::host::name(&buf, || {
            format!("DXR 1.1 scratch buffer ({} MiB)", alloc_bytes >> 20)
        });
        // SAFETY: live buffer.
        let gpu = unsafe { buf.GetGPUVirtualAddress() };
        if let Some((old, _, _)) = self.buffer.replace((buf.clone(), gpu, alloc_bytes)) {
            self.retired.push((retire_fence, old));
        }
        Some((buf, gpu))
    }

    fn reclaim(&mut self, completed_fence: u64) {
        self.retired.retain(|(f, _)| *f > completed_fence);
    }
}

struct BlasEntry {
    alloc: AsAlloc,
    allow_update: bool,
}

struct TlasSlot {
    console_tlas_va: u64,
    alloc: AsAlloc,
    allow_update: bool,
    last_touch: u64,
}

/// Core hardware DXR 1.1 shadow BLAS/TLAS & `SLOT_TLAS_MAP` engine.
pub(crate) struct DxrEngine {
    device: ID3D12Device,
    device5: ID3D12Device5,
    tlas_base: u32,
    as_pool: AsPool,
    scratch: ScratchPool,
    /// `console_blas_va -> host BLAS entry`.
    blas_map: BTreeMap<u64, BlasEntry>,
    /// Fixed 64-slot table matching `TLAS_BASE..TLAS_BASE+64` and `SLOT_TLAS_MAP`.
    tlas_slots: Vec<Option<TlasSlot>>,
    /// `console_tlas_va -> slot index (0..64)`.
    tlas_by_va: BTreeMap<u64, usize>,
    touch_clock: u64,
    /// Persistent 1,024-byte upload buffer bound as root SRV `SLOT_TLAS_MAP` (`t251`).
    tlas_map_buf: ID3D12Resource,
    tlas_map_gpu: u64,
    tlas_map_ptr: usize,
    /// Transient upload buffers (patched instance descs / staged geometry) retired by fence.
    retired_uploads: Vec<(u64, ID3D12Resource)>,
}

// SAFETY: D3D12 COM objects are free-threaded; `tlas_map_ptr` is persistently
// mapped for `tlas_map_buf`'s lifetime and accessed only under `DxrEngine` lock.
unsafe impl Send for DxrEngine {}

impl DxrEngine {
    pub(crate) fn new(device: &ID3D12Device, tlas_base: u32) -> Option<Self> {
        let mut o5 = D3D12_FEATURE_DATA_D3D12_OPTIONS5::default();
        // SAFETY: plain capability query.
        let ok = unsafe {
            device
                .CheckFeatureSupport(
                    D3D12_FEATURE_D3D12_OPTIONS5,
                    (&raw mut o5).cast(),
                    size_of_val(&o5) as u32,
                )
                .is_ok()
        };
        if !ok || o5.RaytracingTier.0 < D3D12_RAYTRACING_TIER_1_1.0 {
            tracing::warn!(
                tier = o5.RaytracingTier.0,
                "DXR 1.1 not supported by device; shadow BLAS/TLAS engine disabled"
            );
            return None;
        }
        let device5: ID3D12Device5 = device.cast().ok()?;
        let (tlas_map_buf, tlas_map_gpu, ptr) = create_upload_buffer(device, TLAS_MAP_BYTES as u64)?;
        // SAFETY: `ptr` maps `TLAS_MAP_BYTES` bytes.
        unsafe { std::ptr::write_bytes(ptr, 0, TLAS_MAP_BYTES) };
        crate::host::name(&tlas_map_buf, || "DXR 1.1 SLOT_TLAS_MAP (t251)".to_owned());
        let mut tlas_slots = Vec::with_capacity(TLAS_SLOTS as usize);
        for _ in 0..TLAS_SLOTS {
            tlas_slots.push(None);
        }
        Some(Self {
            device: device.clone(),
            device5,
            tlas_base,
            as_pool: AsPool::new(),
            scratch: ScratchPool::new(),
            blas_map: BTreeMap::new(),
            tlas_slots,
            tlas_by_va: BTreeMap::new(),
            touch_clock: 0,
            tlas_map_buf,
            tlas_map_gpu,
            tlas_map_ptr: ptr as usize,
            retired_uploads: Vec::new(),
        })
    }

    pub(crate) fn tlas_map_buffer(&self) -> (u64, ID3D12Resource) {
        (self.tlas_map_gpu, self.tlas_map_buf.clone())
    }

    fn write_tlas_map_slot(&self, slot_idx: usize, entry: [u32; 4]) {
        debug_assert!(slot_idx < TLAS_SLOTS as usize);
        let base = self.tlas_map_ptr as *mut u32;
        // SAFETY: `tlas_map_ptr` holds `TLAS_SLOTS * 4` dwords in upload memory.
        unsafe {
            let dst = base.add(slot_idx * 4);
            for (k, &word) in entry.iter().enumerate() {
                dst.add(k).write_volatile(word);
            }
        }
    }

    /// Reads the current 4-dword entry at `slot_idx` (for verification/tests).
    #[cfg(test)]
    pub(crate) fn read_tlas_map_slot(&self, slot_idx: usize) -> [u32; 4] {
        debug_assert!(slot_idx < TLAS_SLOTS as usize);
        let base = self.tlas_map_ptr as *const u32;
        // SAFETY: within `TLAS_MAP_BYTES`.
        unsafe {
            let src = base.add(slot_idx * 4);
            [
                src.add(0).read_volatile(),
                src.add(1).read_volatile(),
                src.add(2).read_volatile(),
                src.add(3).read_volatile(),
            ]
        }
    }

    /// Writes a `D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE` SRV
    /// descriptor at `cpu_handle` (`host_tlas_va == 0` writes a null RT-AS SRV).
    pub(crate) fn write_tlas_srv(
        device: &ID3D12Device,
        cpu_handle: D3D12_CPU_DESCRIPTOR_HANDLE,
        host_tlas_va: u64,
    ) {
        let srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_UNKNOWN,
            ViewDimension: D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                RaytracingAccelerationStructure: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_SRV {
                    Location: host_tlas_va,
                },
            },
        };
        // SAFETY: D3D12 requires `pResource = None` for RT-AS SRVs.
        unsafe { device.CreateShaderResourceView(None, Some(&srv), cpu_handle) };
    }

    /// Returns a snapshot of `(slot_offset, host_tlas_gpu_va)` for all 64 slots
    /// (`host_tlas_gpu_va == 0` for unused/evicted slots).
    pub(crate) fn snapshot_slots(&self) -> [u64; TLAS_SLOTS as usize] {
        let mut out = [0u64; TLAS_SLOTS as usize];
        for (i, slot) in self.tlas_slots.iter().enumerate() {
            if let Some(s) = slot {
                out[i] = s.alloc.gpu_va;
            }
        }
        out
    }

    /// Resolves a console GPU/CPU virtual address `(va, len)` into a host GPU VA:
    /// 1. Stage M3 `sparse_uma` reserved 2 GiB window GPU VA + offset;
    /// 2. `shadow` card-copy buffer GPU VA + offset (refreshed on `host_list`);
    /// 3. Fallback safe CPU copy (`copy_mapped`) into an aligned upload buffer.
    fn resolve_console_va(
        &mut self,
        host_list: Option<&crate::host::HostList>,
        va: u64,
        len: u64,
        retire_fence: u64,
    ) -> Option<u64> {
        if va == 0 {
            return None;
        }
        let len = len.max(1);
        // 1. Stage M3 sparse VRAM window (zero copy when inside one 2 GiB window).
        if crate::sparse_uma::active()
            && let Some((ap_base, _)) = crate::sparse_uma::aperture()
            && let Some((win, tile, off)) = crate::sparse_uma::locate(ap_base, va)
        {
            let win_off = u64::from(tile) * crate::sparse_uma::TILE_BYTES + u64::from(off);
            if win_off.checked_add(len).is_some_and(|end| end <= crate::sparse_uma::WINDOW_BYTES)
                && let Some(windows) = crate::sparse_uma::window_resources()
            {
                let res = &windows[win];
                if let Some(hl) = host_list {
                    hl.keep(res);
                }
                // SAFETY: live reserved window buffer.
                let gpu = unsafe { res.GetGPUVirtualAddress() };
                return Some(gpu + win_off);
            }
        }
        // 2. Existing `shadow` card copy / upload copy.
        if let Some(shadow) = crate::shadow::find_any(va, len)
            && shadow.offset_in(va).is_some()
            && shadow.left(va) >= len
        {
            if let Some(hl) = host_list {
                shadow.refresh(hl, va, len);
            }
            return Some(shadow.gpu(va));
        }
        // 3. Staging upload fallback (reads console unified RAM safely via `copy_mapped`).
        let aligned = align_up(len, 16);
        let (buf, gpu, ptr) = create_upload_buffer(&self.device, aligned)?;
        crate::host::copy_mapped(ptr, va as *const u8, len as usize, false);
        if let Some(hl) = host_list {
            hl.keep(&buf);
        } else {
            self.retired_uploads.push((retire_fence, buf));
        }
        Some(gpu)
    }

    /// Records a hardware DXR 1.1 BLAS build (or update) on `list` for
    /// `console_dst_va`, emitting a UAV barrier afterwards and updating
    /// `blas_map`. Returns the host BLAS GPU virtual address on success.
    pub(crate) fn build_blas(
        &mut self,
        list: &ID3D12GraphicsCommandList4,
        host_list: Option<&crate::host::HostList>,
        console_dst_va: u64,
        console_src_va: u64,
        raw_flags: i32,
        geoms: &[D3D12_RAYTRACING_GEOMETRY_DESC],
        retire_fence: u64,
    ) -> Option<u64> {
        if console_dst_va == 0 {
            return None;
        }
        let mut flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS(
            raw_flags & PC_BUILD_FLAGS_MASK,
        );
        let perform_update = (flags.0 & D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE.0) != 0;
        let existing_src = if perform_update {
            let src_key = if console_src_va != 0 {
                console_src_va
            } else {
                console_dst_va
            };
            self.blas_map
                .get(&src_key)
                .filter(|b| b.allow_update)
                .map(|b| b.alloc.clone())
        } else {
            None
        };
        if perform_update && existing_src.is_none() {
            flags.0 &= !D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE.0;
        }
        let is_update = existing_src.is_some();

        let inputs = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
            Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL,
            Flags: flags,
            NumDescs: geoms.len() as u32,
            DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
            Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
                pGeometryDescs: if geoms.is_empty() {
                    std::ptr::null()
                } else {
                    geoms.as_ptr()
                },
            },
        };
        let mut prebuild = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO::default();
        // SAFETY: valid inputs and output pointer.
        unsafe {
            self.device5
                .GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &mut prebuild);
        }
        if prebuild.ResultDataMaxSizeInBytes == 0 {
            return None;
        }
        let scratch_needed = if is_update {
            prebuild
                .UpdateScratchDataSizeInBytes
                .max(prebuild.ScratchDataSizeInBytes)
        } else {
            prebuild.ScratchDataSizeInBytes
        };

        // Reuse destination slice in-place when performing an in-place update
        // that fits; otherwise allocate a fresh slice and defer-free the old one.
        let dst_alloc = if is_update
            && console_src_va == console_dst_va
            && let Some(ref src) = existing_src
            && src.size >= prebuild.ResultDataMaxSizeInBytes
        {
            src.clone()
        } else {
            self.as_pool.alloc(&self.device, prebuild.ResultDataMaxSizeInBytes)?
        };
        let (scratch_res, scratch_gpu) = self
            .scratch
            .ensure(&self.device, scratch_needed, retire_fence)?;

        let build_desc = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
            DestAccelerationStructureData: dst_alloc.gpu_va,
            Inputs: inputs,
            SourceAccelerationStructureData: existing_src.as_ref().map_or(0, |s| s.gpu_va),
            ScratchAccelerationStructureData: scratch_gpu,
        };
        if let Some(hl) = host_list {
            hl.keep(&dst_alloc.resource);
            hl.keep(&scratch_res);
            if let Some(ref s) = existing_src {
                hl.keep(&s.resource);
            }
        }
        // SAFETY: live command list, valid AS and scratch GPU VAs in required states.
        unsafe { list.BuildRaytracingAccelerationStructure(&build_desc, None) };
        emit_uav_barrier(list);

        let host_va = dst_alloc.gpu_va;
        let allow_update =
            (flags.0 & D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE.0) != 0;
        let in_place = existing_src
            .as_ref()
            .is_some_and(|s| s.gpu_va == dst_alloc.gpu_va);
        if let Some(old) = self.blas_map.insert(
            console_dst_va,
            BlasEntry {
                alloc: dst_alloc,
                allow_update,
            },
        ) && !in_place
        {
            self.as_pool.defer_free(old.alloc, retire_fence);
        }
        STATS[0].fetch_add(1, Ordering::Relaxed);
        STATS[4].store(self.blas_map.len() as u64, Ordering::Relaxed);
        Some(host_va)
    }

    /// Records a hardware DXR 1.1 TLAS build on `list` for `console_tlas_va`,
    /// patching every `InstanceDesc::acceleration_structure` from console BLAS VA
    /// to host BLAS VA (preserving bytes `0..56` bit-for-bit), assigning a slot in
    /// `TLAS_BASE..TLAS_BASE+64`, and updating `SLOT_TLAS_MAP` (`t251`).
    ///
    /// Returns `(host_tlas_srv_slot, host_tlas_gpu_va)` on success.
    pub(crate) fn build_tlas(
        &mut self,
        list: &ID3D12GraphicsCommandList4,
        host_list: Option<&crate::host::HostList>,
        console_tlas_va: u64,
        console_src_va: u64,
        raw_flags: i32,
        console_instances: &[InstanceDesc],
        retire_fence: u64,
    ) -> Option<(u32, u64)> {
        if console_tlas_va == 0 {
            return None;
        }
        let mut flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS(
            raw_flags & PC_BUILD_FLAGS_MASK,
        );
        let perform_update = (flags.0 & D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE.0) != 0;
        let existing_src = if perform_update {
            let src_key = if console_src_va != 0 {
                console_src_va
            } else {
                console_tlas_va
            };
            self.tlas_by_va
                .get(&src_key)
                .and_then(|&idx| self.tlas_slots[idx].as_ref())
                .filter(|t| t.allow_update)
                .map(|t| t.alloc.clone())
        } else {
            None
        };
        if perform_update && existing_src.is_none() {
            flags.0 &= !D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE.0;
        }
        let is_update = existing_src.is_some();

        // Patch instance descs into an upload buffer (16-byte aligned).
        let inst_gpu = if console_instances.is_empty() {
            0
        } else {
            let bytes = (console_instances.len() * size_of::<InstanceDesc>()) as u64;
            let (buf, gpu, ptr) = create_upload_buffer(&self.device, align_up(bytes, 16))?;
            let dst = ptr.cast::<InstanceDesc>();
            for (i, src) in console_instances.iter().enumerate() {
                let mut patched = *src;
                patched.acceleration_structure = if src.acceleration_structure == 0 {
                    0
                } else {
                    self.blas_map
                        .get(&src.acceleration_structure)
                        .map_or(0, |b| b.alloc.gpu_va)
                };
                // SAFETY: `dst` has room for `console_instances.len()` aligned entries.
                unsafe { dst.add(i).write(patched) };
            }
            if let Some(hl) = host_list {
                hl.keep(&buf);
            } else {
                self.retired_uploads.push((retire_fence, buf));
            }
            gpu
        };

        let inputs = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
            Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL,
            Flags: flags,
            NumDescs: console_instances.len() as u32,
            DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
            Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
                InstanceDescs: inst_gpu,
            },
        };
        let mut prebuild = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO::default();
        // SAFETY: valid TLAS prebuild query.
        unsafe {
            self.device5
                .GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &mut prebuild);
        }
        if prebuild.ResultDataMaxSizeInBytes == 0 {
            return None;
        }
        let scratch_needed = if is_update {
            prebuild
                .UpdateScratchDataSizeInBytes
                .max(prebuild.ScratchDataSizeInBytes)
        } else {
            prebuild.ScratchDataSizeInBytes
        };

        let dst_alloc = if is_update
            && console_src_va == console_tlas_va
            && let Some(ref src) = existing_src
            && src.size >= prebuild.ResultDataMaxSizeInBytes
        {
            src.clone()
        } else {
            self.as_pool.alloc(&self.device, prebuild.ResultDataMaxSizeInBytes)?
        };
        let (scratch_res, scratch_gpu) = self
            .scratch
            .ensure(&self.device, scratch_needed, retire_fence)?;

        let build_desc = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
            DestAccelerationStructureData: dst_alloc.gpu_va,
            Inputs: inputs,
            SourceAccelerationStructureData: existing_src.as_ref().map_or(0, |s| s.gpu_va),
            ScratchAccelerationStructureData: scratch_gpu,
        };
        if let Some(hl) = host_list {
            hl.keep(&dst_alloc.resource);
            hl.keep(&scratch_res);
            if let Some(ref s) = existing_src {
                hl.keep(&s.resource);
            }
        }
        // SAFETY: live command list and valid TLAS/instance/scratch GPU VAs.
        unsafe { list.BuildRaytracingAccelerationStructure(&build_desc, None) };
        emit_uav_barrier(list);

        let host_va = dst_alloc.gpu_va;
        let allow_update =
            (flags.0 & D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE.0) != 0;
        let slot_idx = self.register_tlas_slot(console_tlas_va, dst_alloc, allow_update, retire_fence);
        let abs_slot = self.tlas_base + slot_idx as u32;
        STATS[1].fetch_add(1, Ordering::Relaxed);
        STATS[5].store(self.tlas_by_va.len() as u64, Ordering::Relaxed);
        Some((abs_slot, host_va))
    }

    fn register_tlas_slot(
        &mut self,
        console_tlas_va: u64,
        alloc: AsAlloc,
        allow_update: bool,
        retire_fence: u64,
    ) -> usize {
        self.touch_clock = self.touch_clock.wrapping_add(1);
        let slot_idx = if let Some(&existing) = self.tlas_by_va.get(&console_tlas_va) {
            existing
        } else if let Some(free) = self.tlas_slots.iter().position(Option::is_none) {
            free
        } else {
            // LRU evict oldest TLAS slot.
            self.tlas_slots
                .iter()
                .enumerate()
                .min_by_key(|(_, s)| s.as_ref().map_or(0, |t| t.last_touch))
                .map_or(0, |(i, _)| i)
        };
        let in_place = self.tlas_slots[slot_idx]
            .as_ref()
            .is_some_and(|old| old.alloc.gpu_va == alloc.gpu_va);
        if let Some(old) = self.tlas_slots[slot_idx].replace(TlasSlot {
            console_tlas_va,
            alloc,
            allow_update,
            last_touch: self.touch_clock,
        }) {
            if old.console_tlas_va != console_tlas_va {
                self.tlas_by_va.remove(&old.console_tlas_va);
            }
            if !in_place {
                self.as_pool.defer_free(old.alloc, retire_fence);
            }
        }
        self.tlas_by_va.insert(console_tlas_va, slot_idx);
        let abs_slot = self.tlas_base + slot_idx as u32;
        self.write_tlas_map_slot(slot_idx, encode_tlas_map_entry(console_tlas_va, abs_slot));
        TLAS_GEN.fetch_add(1, Ordering::Release);
        slot_idx
    }

    /// Copies/clones a host BLAS or TLAS from `console_src_va` to
    /// `console_dst_va` on `list`. Compact modes execute as `CLONE` into the
    /// host AS pool so PC Ampere internal layout sizes never truncate.
    pub(crate) fn copy_as(
        &mut self,
        list: &ID3D12GraphicsCommandList4,
        host_list: Option<&crate::host::HostList>,
        console_dst_va: u64,
        console_src_va: u64,
        mode: u32,
        retire_fence: u64,
    ) -> Option<u64> {
        if console_dst_va == 0 || console_src_va == 0 {
            return None;
        }
        // Only CLONE (0) and COMPACT (1) represent valid GPU-to-GPU AS copies.
        if mode != D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_CLONE.0 as u32
            && mode != D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_COMPACT.0 as u32
        {
            return None;
        }
        if let Some(src) = self.blas_map.get(&console_src_va) {
            let src_alloc = src.alloc.clone();
            let allow_update = src.allow_update;
            let dst_alloc = self.as_pool.alloc(&self.device, src_alloc.size)?;
            if let Some(hl) = host_list {
                hl.keep(&src_alloc.resource);
                hl.keep(&dst_alloc.resource);
            }
            // SAFETY: both VAs point to valid `RAYTRACING_ACCELERATION_STRUCTURE` slices.
            unsafe {
                list.CopyRaytracingAccelerationStructure(
                    dst_alloc.gpu_va,
                    src_alloc.gpu_va,
                    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_CLONE,
                );
            }
            emit_uav_barrier(list);
            let host_va = dst_alloc.gpu_va;
            if let Some(old) = self.blas_map.insert(
                console_dst_va,
                BlasEntry {
                    alloc: dst_alloc,
                    allow_update,
                },
            ) {
                self.as_pool.defer_free(old.alloc, retire_fence);
            }
            STATS[2].fetch_add(1, Ordering::Relaxed);
            STATS[4].store(self.blas_map.len() as u64, Ordering::Relaxed);
            return Some(host_va);
        }
        if let Some(&src_slot) = self.tlas_by_va.get(&console_src_va)
            && let Some(src) = self.tlas_slots[src_slot].as_ref()
        {
            let src_alloc = src.alloc.clone();
            let allow_update = src.allow_update;
            let dst_alloc = self.as_pool.alloc(&self.device, src_alloc.size)?;
            if let Some(hl) = host_list {
                hl.keep(&src_alloc.resource);
                hl.keep(&dst_alloc.resource);
            }
            // SAFETY: valid TLAS slices.
            unsafe {
                list.CopyRaytracingAccelerationStructure(
                    dst_alloc.gpu_va,
                    src_alloc.gpu_va,
                    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_CLONE,
                );
            }
            emit_uav_barrier(list);
            let host_va = dst_alloc.gpu_va;
            self.register_tlas_slot(console_dst_va, dst_alloc, allow_update, retire_fence);
            STATS[2].fetch_add(1, Ordering::Relaxed);
            STATS[5].store(self.tlas_by_va.len() as u64, Ordering::Relaxed);
            return Some(host_va);
        }
        None
    }

    /// Evicts any host BLAS/TLAS entries whose console VA lies in
    /// `[addr, addr + len)`, clears invalidated `SLOT_TLAS_MAP` entries, and
    /// queues their VRAM slices for reclamation after `retire_fence`.
    pub(crate) fn evict_range(&mut self, addr: u64, len: u64, retire_fence: u64) -> usize {
        if len == 0 {
            return 0;
        }
        let end = addr.saturating_add(len);
        let blas_keys: Vec<u64> = self
            .blas_map
            .range(addr..end)
            .map(|(&k, _)| k)
            .collect();
        let mut evicted = 0usize;
        for k in blas_keys {
            if let Some(old) = self.blas_map.remove(&k) {
                self.as_pool.defer_free(old.alloc, retire_fence);
                evicted += 1;
            }
        }
        let tlas_keys: Vec<(u64, usize)> = self
            .tlas_by_va
            .range(addr..end)
            .map(|(&k, &slot)| (k, slot))
            .collect();
        let mut tlas_changed = false;
        for (k, slot_idx) in tlas_keys {
            self.tlas_by_va.remove(&k);
            if let Some(old) = self.tlas_slots[slot_idx].take() {
                self.as_pool.defer_free(old.alloc, retire_fence);
                self.write_tlas_map_slot(slot_idx, [0, 0, 0, 0]);
                tlas_changed = true;
                evicted += 1;
            }
        }
        if tlas_changed {
            TLAS_GEN.fetch_add(1, Ordering::Release);
        }
        if evicted > 0 {
            STATS[3].fetch_add(evicted as u64, Ordering::Relaxed);
            STATS[4].store(self.blas_map.len() as u64, Ordering::Relaxed);
            STATS[5].store(self.tlas_by_va.len() as u64, Ordering::Relaxed);
        }
        evicted
    }

    /// Translates and executes one console `ConsoleBuildDesc` (BLAS or TLAS).
    fn shadow_build_desc(
        &mut self,
        list: &ID3D12GraphicsCommandList4,
        host_list: &crate::host::HostList,
        desc: &ConsoleBuildDesc,
        retire_fence: u64,
    ) {
        if desc.dest_as_va == 0 {
            return;
        }
        if desc.inputs_type == D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL.0 {
            let Some(geoms) = self.translate_console_geoms(host_list, desc, retire_fence) else {
                return;
            };
            let _ = self.build_blas(
                list,
                Some(host_list),
                desc.dest_as_va,
                desc.source_as_va,
                desc.inputs_flags,
                &geoms,
                retire_fence,
            );
        } else if desc.inputs_type == D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL.0 {
            let instances = read_console_instances(desc);
            let _ = self.build_tlas(
                list,
                Some(host_list),
                desc.dest_as_va,
                desc.source_as_va,
                desc.inputs_flags,
                &instances,
                retire_fence,
            );
        }
    }

    /// Translates console `D3D12_RAYTRACING_GEOMETRY_DESC` array (or array of
    /// pointers) into PC host GPU VAs via `resolve_console_va`.
    fn translate_console_geoms(
        &mut self,
        host_list: &crate::host::HostList,
        desc: &ConsoleBuildDesc,
        retire_fence: u64,
    ) -> Option<Vec<D3D12_RAYTRACING_GEOMETRY_DESC>> {
        let count = desc.num_descs as usize;
        if count == 0 {
            return Some(Vec::new());
        }
        if desc.descs_ptr_or_va == 0 || count > 65_536 {
            return None;
        }
        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            let mut raw = D3D12_RAYTRACING_GEOMETRY_DESC::default();
            let src_ptr = if desc.descs_layout == D3D12_ELEMENTS_LAYOUT_ARRAY_OF_POINTERS.0 {
                let mut entry_ptr = 0u64;
                crate::host::copy_mapped(
                    (&raw mut entry_ptr).cast::<u8>(),
                    (desc.descs_ptr_or_va + (i * size_of::<u64>()) as u64) as *const u8,
                    size_of::<u64>(),
                    false,
                );
                entry_ptr
            } else {
                desc.descs_ptr_or_va + (i * size_of::<D3D12_RAYTRACING_GEOMETRY_DESC>()) as u64
            };
            if src_ptr == 0 {
                return None;
            }
            crate::host::copy_mapped(
                (&raw mut raw).cast::<u8>(),
                src_ptr as *const u8,
                size_of::<D3D12_RAYTRACING_GEOMETRY_DESC>(),
                false,
            );

            if raw.Type == D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES {
                // SAFETY: `Type == TRIANGLES`.
                let mut tri = unsafe { raw.Anonymous.Triangles };
                if tri.VertexCount > 0 {
                    let elem = dxgi_format_bytes(tri.VertexFormat);
                    let stride = tri.VertexBuffer.StrideInBytes.max(elem);
                    let vb_len = u64::from(tri.VertexCount - 1)
                        .saturating_mul(stride)
                        .saturating_add(elem);
                    tri.VertexBuffer.StartAddress = self.resolve_console_va(
                        Some(host_list),
                        tri.VertexBuffer.StartAddress,
                        vb_len,
                        retire_fence,
                    )?;
                } else {
                    tri.VertexBuffer.StartAddress = 0;
                }
                if tri.IndexCount > 0 && tri.IndexBuffer != 0 {
                    let idx_bytes = u64::from(tri.IndexCount)
                        .saturating_mul(dxgi_format_bytes(tri.IndexFormat));
                    tri.IndexBuffer = self.resolve_console_va(
                        Some(host_list),
                        tri.IndexBuffer,
                        idx_bytes,
                        retire_fence,
                    )?;
                } else {
                    tri.IndexBuffer = 0;
                }
                if tri.Transform3x4 != 0 {
                    tri.Transform3x4 = self.resolve_console_va(
                        Some(host_list),
                        tri.Transform3x4,
                        48,
                        retire_fence,
                    )?;
                }
                raw.Anonymous.Triangles = tri;
            } else if raw.Type == D3D12_RAYTRACING_GEOMETRY_TYPE_PROCEDURAL_PRIMITIVE_AABBS {
                // SAFETY: `Type == PROCEDURAL_PRIMITIVE_AABBS`.
                let mut aabbs = unsafe { raw.Anonymous.AABBs };
                if aabbs.AABBCount > 0 {
                    let stride = aabbs.AABBs.StrideInBytes.max(24);
                    let len = (aabbs.AABBCount - 1)
                        .saturating_mul(stride)
                        .saturating_add(24);
                    aabbs.AABBs.StartAddress = self.resolve_console_va(
                        Some(host_list),
                        aabbs.AABBs.StartAddress,
                        len,
                        retire_fence,
                    )?;
                } else {
                    aabbs.AABBs.StartAddress = 0;
                }
                raw.Anonymous.AABBs = aabbs;
            }
            out.push(raw);
        }
        Some(out)
    }

    pub(crate) fn reclaim(&mut self, completed_fence: u64) -> usize {
        self.scratch.reclaim(completed_fence);
        self.retired_uploads.retain(|(f, _)| *f > completed_fence);
        self.as_pool.reclaim(completed_fence)
    }
}

/// Reads console `D3D12_RAYTRACING_INSTANCE_DESC` entries (handling both
/// `ARRAY` and `ARRAY_OF_POINTERS`) via safe `copy_mapped`.
fn read_console_instances(desc: &ConsoleBuildDesc) -> Vec<InstanceDesc> {
    let count = desc.num_descs as usize;
    if count == 0 || desc.descs_ptr_or_va == 0 || count > 262_144 {
        return Vec::new();
    }
    let mut out = vec![
        InstanceDesc {
            transform: [[0.0; 4]; 3],
            instance_id_and_mask: 0,
            contribution_and_flags: 0,
            acceleration_structure: 0,
        };
        count
    ];
    if desc.descs_layout == D3D12_ELEMENTS_LAYOUT_ARRAY_OF_POINTERS.0 {
        for (i, slot) in out.iter_mut().enumerate() {
            let mut ptr = 0u64;
            crate::host::copy_mapped(
                (&raw mut ptr).cast::<u8>(),
                (desc.descs_ptr_or_va + (i * size_of::<u64>()) as u64) as *const u8,
                size_of::<u64>(),
                false,
            );
            if ptr != 0 {
                crate::host::copy_mapped(
                    (slot as *mut InstanceDesc).cast::<u8>(),
                    ptr as *const u8,
                    size_of::<InstanceDesc>(),
                    false,
                );
            }
        }
    } else {
        crate::host::copy_mapped(
            out.as_mut_ptr().cast::<u8>(),
            desc.descs_ptr_or_va as *const u8,
            count * size_of::<InstanceDesc>(),
            false,
        );
    }
    out
}

// ---------------------------------------------------------------------------
// Process-wide singleton & victory-loader bridge
// ---------------------------------------------------------------------------

static ENGINE: Mutex<Option<DxrEngine>> = Mutex::new(None);

fn with_engine<R>(f: impl FnOnce(&mut DxrEngine) -> R) -> Option<R> {
    if !active() || !raytracing_tier_1_1() {
        return None;
    }
    let host = crate::host::host()?;
    let tlas_base = tlas_base_slot()?;
    let mut guard = ENGINE.lock().unwrap_or_else(|e| e.into_inner());
    if guard.is_none() {
        *guard = DxrEngine::new(&host.device, tlas_base);
    }
    let engine = guard.as_mut()?;
    Some(f(engine))
}

/// Returns `(gpu_va, upload_buffer)` for the persistent 1,024-byte
/// `SLOT_TLAS_MAP` (`t251`) table. When DXR is inactive or no TLAS exists yet,
/// lazily returns a valid zero-filled 1,024-byte buffer so shaders reading
/// `t251` cleanly see `valid == 0` and fall back safely.
pub(crate) fn tlas_map_gpu() -> Option<(u64, ID3D12Resource)> {
    if let Some(pair) = with_engine(|e| e.tlas_map_buffer()) {
        return Some(pair);
    }
    static ZERO_MAP: OnceLock<Option<(u64, ID3D12Resource)>> = OnceLock::new();
    ZERO_MAP
        .get_or_init(|| {
            let host = crate::host::host()?;
            let (buf, gpu, ptr) = create_upload_buffer(&host.device, TLAS_MAP_BYTES as u64)?;
            // SAFETY: `ptr` maps `TLAS_MAP_BYTES` bytes.
            unsafe { std::ptr::write_bytes(ptr, 0, TLAS_MAP_BYTES) };
            crate::host::name(&buf, || "DXR 1.1 empty SLOT_TLAS_MAP (t251)".to_owned());
            Some((gpu, buf))
        })
        .clone()
}

/// Returns `(current_gen, [host_tlas_gpu_va; 64])` for `HostHeap::sync_tlas_views`.
pub(crate) fn snapshot_tlas_slots() -> (u64, [u64; TLAS_SLOTS as usize]) {
    let g = tlas_generation();
    let slots = with_engine(|e| e.snapshot_slots()).unwrap_or([0u64; TLAS_SLOTS as usize]);
    (g, slots)
}

/// Public eviction hook: drops any shadow BLAS/TLAS entries covering
/// `[addr, addr + len)` and invalidates affected `SLOT_TLAS_MAP` slots.
pub fn evict_range(addr: u64, len: u64) -> usize {
    let Some(host) = crate::host::host() else {
        return 0;
    };
    let retire = host.graphics_retire_fence_hint();
    with_engine(|e| e.evict_range(addr, len, retire)).unwrap_or(0)
}

/// Reclaims deferred DXR AS pool slices and scratch buffers whose graphics
/// fence has completed (invoked periodically from `host::sweep`).
pub(crate) fn sweep() {
    if !active() {
        return;
    }
    let Some(host) = crate::host::host() else {
        return;
    };
    let completed = host.graphics_completed_value();
    if let Ok(mut guard) = ENGINE.lock()
        && let Some(engine) = guard.as_mut()
    {
        let _ = engine.reclaim(completed);
    }
}

/// Logs periodic DXR 1.1 shadow-builder telemetry (invoked from `Meter::frame`).
pub(crate) fn status() {
    if !active() {
        return;
    }
    let blas_builds = STATS[0].load(Ordering::Relaxed);
    let tlas_builds = STATS[1].load(Ordering::Relaxed);
    let copies = STATS[2].load(Ordering::Relaxed);
    let evictions = STATS[3].load(Ordering::Relaxed);
    let live_blas = STATS[4].load(Ordering::Relaxed);
    let live_tlas = STATS[5].load(Ordering::Relaxed);
    let pool_mb = STATS[6].load(Ordering::Relaxed) >> 20;
    if blas_builds > 0 || tlas_builds > 0 {
        tracing::info!(
            blas_builds,
            tlas_builds,
            copies,
            evictions,
            live_blas,
            live_tlas,
            pool_mb,
            "DXR 1.1 shadow AS builder"
        );
    }
}

/// Shadow-records a single `D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC`
/// from `XboxBvhBuilder::BuildRaytracingAccelerationStructure`.
pub extern "system" fn shadow_build_single(console_list: *mut c_void, desc: *const c_void) {
    if console_list.is_null() || desc.is_null() {
        return;
    }
    let Some(hl) = crate::list::host_list(console_list.cast()) else {
        return;
    };
    let Ok(list4) = hl.list.cast::<ID3D12GraphicsCommandList4>() else {
        return;
    };
    let Some(host) = crate::host::host() else {
        return;
    };
    let retire = host.graphics_retire_fence_hint();
    // SAFETY: title passed a valid `D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC`.
    let d = unsafe { *desc.cast::<ConsoleBuildDesc>() };
    let _ = with_engine(|e| e.shadow_build_desc(&list4, hl, &d, retire));
}

/// Shadow-records a `BuildRaytracingAccelerationStructureBatched2X` batch
/// (`count` × 80-byte `ConsoleBatchedDesc2` entries).
pub extern "system" fn shadow_build_batched_2x(
    console_list: *mut c_void,
    count: u32,
    descs: *const c_void,
) {
    if console_list.is_null() || descs.is_null() || count == 0 {
        return;
    }
    let Some(hl) = crate::list::host_list(console_list.cast()) else {
        return;
    };
    let Ok(list4) = hl.list.cast::<ID3D12GraphicsCommandList4>() else {
        return;
    };
    let Some(host) = crate::host::host() else {
        return;
    };
    let retire = host.graphics_retire_fence_hint();
    let ptr = descs.cast::<ConsoleBatchedDesc2>();
    let _ = with_engine(|e| {
        for i in 0..count as usize {
            // SAFETY: caller passed `count` valid 80-byte `ConsoleBatchedDesc2` entries.
            let entry = unsafe { *ptr.add(i) };
            e.shadow_build_desc(&list4, hl, &entry.desc, retire);
        }
    });
}

/// Shadow-records `RehydrateRaytracingAccelerationStructureBatched2X` /
/// `RehydrateRaytracingAccelerationStructureBatchedX` (`count` × 48-byte
/// `ConsoleBuildDesc` entries).
pub extern "system" fn shadow_rehydrate_batched(
    console_list: *mut c_void,
    count: u32,
    descs: *const c_void,
    _src_vas: *const u64,
) {
    if console_list.is_null() || descs.is_null() || count == 0 {
        return;
    }
    let Some(hl) = crate::list::host_list(console_list.cast()) else {
        return;
    };
    let Ok(list4) = hl.list.cast::<ID3D12GraphicsCommandList4>() else {
        return;
    };
    let Some(host) = crate::host::host() else {
        return;
    };
    let retire = host.graphics_retire_fence_hint();
    let ptr = descs.cast::<ConsoleBuildDesc>();
    let _ = with_engine(|e| {
        for i in 0..count as usize {
            // SAFETY: caller passed `count` valid 48-byte `ConsoleBuildDesc` entries.
            let mut d = unsafe { *ptr.add(i) };
            // Rehydrated BLASes are streamed in fresh on PC DXR 1.1.
            d.inputs_flags &= !D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE.0;
            d.source_as_va = 0;
            e.shadow_build_desc(&list4, hl, &d, retire);
        }
    });
}

/// Shadow-records `XboxBvhBuilder::CopyRaytracingAccelerationStructure`.
pub extern "system" fn shadow_copy_as(
    console_list: *mut c_void,
    dst_va: u64,
    src_va: u64,
    mode: u32,
) {
    if console_list.is_null() {
        return;
    }
    let Some(hl) = crate::list::host_list(console_list.cast()) else {
        return;
    };
    let Ok(list4) = hl.list.cast::<ID3D12GraphicsCommandList4>() else {
        return;
    };
    let Some(host) = crate::host::host() else {
        return;
    };
    let retire = host.graphics_retire_fence_hint();
    let _ = with_engine(|e| e.copy_as(&list4, Some(hl), dst_va, src_va, mode, retire));
}

/// Shared ABI block published to `victory-loader::bvh_hook` via per-PID
/// `Local\victory_dxr_bridge_<pid>` anonymous file mapping (zero crate-graph
/// coupling between `victory-loader` and `victory-gpu`).
#[repr(C)]
struct LoaderBridgeBlock {
    magic: u64,
    version: u32,
    pid: u32,
    on_build_single: extern "system" fn(*mut c_void, *const c_void),
    on_build_batched_2x: extern "system" fn(*mut c_void, u32, *const c_void),
    on_rehydrate_batched: extern "system" fn(*mut c_void, u32, *const c_void, *const u64),
    on_copy_as: extern "system" fn(*mut c_void, u64, u64, u32),
}

struct BridgeHolder {
    _handle: isize,
    _view: usize,
}

// SAFETY: process-lifetime read-only shared section published once.
unsafe impl Send for BridgeHolder {}
unsafe impl Sync for BridgeHolder {}

/// Publishes the `victory-gpu::dxr` shadow callbacks into the per-PID named
/// section `Local\victory_dxr_bridge_<pid>` so `victory-loader::bvh_hook`
/// discovers and invokes them automatically when `VICTORY_DXR_RAYQUERY=1`.
pub(crate) fn publish_loader_hooks() {
    use windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
    use windows_sys::Win32::System::Memory::{
        CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, PAGE_READWRITE,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcessId;

    static PUBLISHED: OnceLock<Option<BridgeHolder>> = OnceLock::new();
    let _ = PUBLISHED.get_or_init(|| {
        // SAFETY: creating a tiny process-scoped anonymous file mapping.
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
                size_of::<LoaderBridgeBlock>() as u32,
                name.as_ptr(),
            );
            if handle.is_null() {
                tracing::warn!("DXR 1.1: CreateFileMappingW for loader bridge failed");
                return None;
            }
            let view = MapViewOfFile(handle, FILE_MAP_WRITE, 0, 0, size_of::<LoaderBridgeBlock>());
            if view.Value.is_null() {
                tracing::warn!("DXR 1.1: MapViewOfFile for loader bridge failed");
                return None;
            }
            let block = LoaderBridgeBlock {
                magic: BRIDGE_MAGIC,
                version: BRIDGE_VERSION,
                pid,
                on_build_single: shadow_build_single,
                on_build_batched_2x: shadow_build_batched_2x,
                on_rehydrate_batched: shadow_rehydrate_batched,
                on_copy_as: shadow_copy_as,
            };
            view.Value.cast::<LoaderBridgeBlock>().write(block);
            tracing::info!(pid, "DXR 1.1 shadow builder hooks published for loader bridge");
            Some(BridgeHolder {
                _handle: handle as isize,
                _view: view.Value as usize,
            })
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dxr_env_constants_and_tlas_map_encoding() {
        assert!(!resolve_dxr_rayquery(None));
        assert!(!resolve_dxr_rayquery(Some("0")));
        assert!(!resolve_dxr_rayquery(Some("true")));
        assert!(resolve_dxr_rayquery(Some("1")));
        assert!(resolve_dxr_rayquery(Some(" 1 ")));

        assert_eq!(SLOT_TLAS_MAP, 251);
        assert_eq!(TLAS_SLOTS, 64);
        assert_eq!(TLAS_MAP_ENTRY_BYTES, 16);
        assert_eq!(TLAS_MAP_BYTES, 1024);

        let console_tlas_va = 0x12_3456_7800u64;
        let slot = 999_920u32;
        let enc = encode_tlas_map_entry(console_tlas_va, slot);
        assert_eq!(enc[0], ((console_tlas_va >> 8) & 0xFFFF_FFFF) as u32);
        assert_eq!(enc[1], (console_tlas_va >> 40) as u32);
        assert_eq!(enc[2], slot);
        assert_eq!(enc[3], 1);

        let id_mask = InstanceDesc::pack_id_mask(0x0012_3456, 0xD5);
        assert_eq!(id_mask & 0x00FF_FFFF, 0x0012_3456);
        assert_eq!(id_mask >> 24, 0xD5);
        let contrib_flags = InstanceDesc::pack_contrib_flags(0x0000_0007, 0x05);
        assert_eq!(contrib_flags & 0x00FF_FFFF, 7);
        assert_eq!(contrib_flags >> 24, 0x05);
    }

    /// Clean-room SM 6.6 `RayQuery` compute shader that looks up a console TLAS
    /// VA (`va >> 8`) in `SLOT_TLAS_MAP` (`t251`, `space1`), loads the host
    /// `RaytracingAccelerationStructure` from `ResourceDescriptorHeap[slot]`,
    /// traces one opaque ray, and writes `[status, t, bary_x, bary_y, inst_id,
    /// geom_idx, prim_idx, inst_contrib]` to `u0`.
    const RAYQUERY_TEST_HLSL: &str = r#"
ByteAddressBuffer tlas_map : register(t251, space1);
RWByteAddressBuffer out_buf : register(u0, space0);
cbuffer Params : register(b0, space0) {
    uint target_lo;
    uint target_hi;
    uint _pad0;
    uint _pad1;
};

[numthreads(1, 1, 1)]
void main() {
    uint found_slot = 0xFFFFFFFFu;
    for (uint i = 0; i < 64u; ++i) {
        uint4 e = tlas_map.Load4(i * 16u);
        if (e.w != 0u && e.x == target_lo && e.y == target_hi) {
            found_slot = e.z;
            break;
        }
    }
    if (found_slot == 0xFFFFFFFFu) {
        out_buf.Store4(0, uint4(0u, 0u, 0u, 0u));
        out_buf.Store4(16, uint4(0u, 0u, 0u, 0u));
        return;
    }
    RaytracingAccelerationStructure tlas = ResourceDescriptorHeap[found_slot];
    RayDesc ray;
    ray.Origin = float3(2.5f, 1.5f, 0.0f);
    ray.Direction = float3(0.0f, 0.0f, 1.0f);
    ray.TMin = 0.01f;
    ray.TMax = 100.0f;

    RayQuery<RAY_FLAG_FORCE_OPAQUE> q;
    q.TraceRayInline(tlas, 0u, 0xFFu, ray);
    while (q.Proceed()) {}

    uint status = q.CommittedStatus();
    float t = q.CommittedRayT();
    float2 bary = q.CommittedTriangleBarycentrics();
    uint inst_id = q.CommittedInstanceID();
    uint geom_idx = q.CommittedGeometryIndex();
    uint prim_idx = q.CommittedPrimitiveIndex();
    uint inst_contrib = q.CommittedInstanceContributionToHitGroupIndex();

    out_buf.Store4(0, uint4(status, asuint(t), asuint(bary.x), asuint(bary.y)));
    out_buf.Store4(16, uint4(inst_id, geom_idx, prim_idx, inst_contrib));
}
"#;

    /// Locates the Windows 10/11 SDK `dxc.exe` and compiles `RAYQUERY_TEST_HLSL`
    /// to validated SM 6.6 DXIL.
    fn compile_rayquery_cs_6_6() -> Vec<u8> {
        let kits = std::path::Path::new(r"C:\Program Files (x86)\Windows Kits\10\bin");
        let mut candidates: Vec<std::path::PathBuf> = std::fs::read_dir(kits)
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| e.path().join("x64").join("dxc.exe"))
            .filter(|p| p.is_file())
            .collect();
        candidates.sort();
        let dxc = candidates
            .pop()
            .expect("Windows SDK x64/dxc.exe must be installed to compile SM 6.6 RayQuery test");
        let dir = tempfile::tempdir().expect("tempdir for dxc");
        let src = dir.path().join("rayquery_test.hlsl");
        let out = dir.path().join("rayquery_test.dxil");
        std::fs::write(&src, RAYQUERY_TEST_HLSL).expect("write hlsl");
        let status = std::process::Command::new(&dxc)
            .args([
                "-T",
                "cs_6_6",
                "-E",
                "main",
                "-Fo",
                out.to_str().unwrap(),
                src.to_str().unwrap(),
            ])
            .output()
            .expect("invoke dxc.exe");
        assert!(
            status.status.success(),
            "dxc.exe failed: {}",
            String::from_utf8_lossy(&status.stderr)
        );
        std::fs::read(&out).expect("read compiled dxil")
    }

    /// Runs the full live RTX 3090 DXR 1.1 end-to-end test on `device`:
    /// 1. Verifies `RaytracingTier >= Tier 1.1`
    /// 2. Builds a real DXR 1.1 BLAS from 2 indexed triangles (VB + `R16_UINT` IB)
    /// 3. Builds a real DXR 1.1 TLAS from a 64-byte `InstanceDesc` with
    ///    `InstanceID = 0x1234`, `InstanceMask = 0xFF`, `Contribution = 7`,
    ///    translation `(+1, 0, 0)`, patching `console_blas_va -> host_blas_va`
    /// 4. Populates `SLOT_TLAS_MAP` (`t251`) and `TLAS_BASE` RT-AS SRV in a
    ///    shader-visible `CBV_SRV_UAV` heap
    /// 5. Dispatches a real SM 6.6 `RayQuery` compute shader on Ampere RT Cores
    ///    and verifies exact `status == 1`, `t == 2.0`, `bary == (0.50, 0.25)`,
    ///    `InstanceID == 0x1234`, `GeometryIndex == 0`, `PrimitiveIndex == 1`,
    ///    `InstanceContributionToHitGroupIndex == 7`
    /// 6. Exercises `copy_as` (CLONE/COMPACT) + `evict_range` + pool reclamation
    /// 7. Asserts 0 D3D12 Debug Layer errors when `info_queue` is attached.
    fn run_live_dxr11_test(device: &ID3D12Device, info_queue: Option<&ID3D12InfoQueue>) {
        use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};

        let mut o5 = D3D12_FEATURE_DATA_D3D12_OPTIONS5::default();
        // SAFETY: plain capability query.
        unsafe {
            device
                .CheckFeatureSupport(
                    D3D12_FEATURE_D3D12_OPTIONS5,
                    (&raw mut o5).cast(),
                    size_of_val(&o5) as u32,
                )
                .expect("CheckFeatureSupport OPTIONS5");
        }
        assert!(
            o5.RaytracingTier.0 >= D3D12_RAYTRACING_TIER_1_1.0,
            "expected DXR Tier 1.1 on RTX 3090, got {}",
            o5.RaytracingTier.0
        );

        let dxil = compile_rayquery_cs_6_6();
        let tlas_base = 16u32;
        let mut engine = DxrEngine::new(device, tlas_base).expect("DxrEngine::new");

        // SAFETY: live D3D12 device and valid descriptors for queue, allocator, list, heap, buffers.
        unsafe {
            let queue: ID3D12CommandQueue = device
                .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                    ..Default::default()
                })
                .expect("CreateCommandQueue");
            let alloc: ID3D12CommandAllocator = device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                .expect("CreateCommandAllocator");
            let list: ID3D12GraphicsCommandList4 = device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)
                .expect("CreateCommandList");
            let fence: ID3D12Fence = device
                .CreateFence(0, D3D12_FENCE_FLAG_NONE)
                .expect("CreateFence");
            let event = CreateEventW(None, false, false, None).expect("CreateEventW");

            // Quad at z = 2.0: 4 vertices, 2 triangles (6 u16 indices).
            // Tri 0: (0,0,2),(2,0,2),(0,2,2); Tri 1: (2,0,2),(2,2,2),(0,2,2).
            let verts: [[f32; 3]; 4] = [
                [0.0, 0.0, 2.0],
                [2.0, 0.0, 2.0],
                [0.0, 2.0, 2.0],
                [2.0, 2.0, 2.0],
            ];
            let indices: [u16; 6] = [0, 1, 2, 1, 3, 2];
            let (vb_buf, vb_gpu, vb_ptr) = create_upload_buffer(device, 256).expect("vb");
            std::ptr::copy_nonoverlapping(
                verts.as_ptr().cast::<u8>(),
                vb_ptr,
                size_of_val(&verts),
            );
            let (ib_buf, ib_gpu, ib_ptr) = create_upload_buffer(device, 256).expect("ib");
            std::ptr::copy_nonoverlapping(
                indices.as_ptr().cast::<u8>(),
                ib_ptr,
                size_of_val(&indices),
            );

            let geom = D3D12_RAYTRACING_GEOMETRY_DESC {
                Type: D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES,
                Flags: D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE,
                Anonymous: D3D12_RAYTRACING_GEOMETRY_DESC_0 {
                    Triangles: D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC {
                        Transform3x4: 0,
                        IndexFormat: DXGI_FORMAT_R16_UINT,
                        VertexFormat: DXGI_FORMAT_R32G32B32_FLOAT,
                        IndexCount: 6,
                        VertexCount: 4,
                        IndexBuffer: ib_gpu,
                        VertexBuffer: D3D12_GPU_VIRTUAL_ADDRESS_AND_STRIDE {
                            StartAddress: vb_gpu,
                            StrideInBytes: 12,
                        },
                    },
                },
            };

            let console_blas_va = 0x10_0001_0000u64;
            let console_tlas_va = 0x10_0002_0000u64;
            let console_copy_va = 0x10_0003_0000u64;

            let host_blas_va = engine
                .build_blas(
                    &list,
                    None,
                    console_blas_va,
                    0,
                    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE.0,
                    &[geom],
                    1,
                )
                .expect("build_blas");
            assert_ne!(host_blas_va, 0);
            assert_eq!(host_blas_va % AS_ALIGNMENT, 0);

            // Instance translated by (+1, 0, 0): Tri 1 world vertices = (3,0,2),(3,2,2),(1,2,2).
            // Ray origin (2.5, 1.5, 0) dir (0, 0, 1) hits Tri 1 at t = 2.0, bary = (0.50, 0.25).
            let inst = InstanceDesc {
                transform: [
                    [1.0, 0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                ],
                instance_id_and_mask: InstanceDesc::pack_id_mask(0x1234, 0xFF),
                contribution_and_flags: InstanceDesc::pack_contrib_flags(7, 0),
                acceleration_structure: console_blas_va,
            };
            let (tlas_slot, host_tlas_va) = engine
                .build_tlas(
                    &list,
                    None,
                    console_tlas_va,
                    0,
                    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE.0,
                    &[inst],
                    1,
                )
                .expect("build_tlas");
            assert_eq!(tlas_slot, tlas_base);
            assert_ne!(host_tlas_va, 0);

            // Verify SLOT_TLAS_MAP entry populated accurately.
            let expected_entry = encode_tlas_map_entry(console_tlas_va, tlas_slot);
            assert_eq!(engine.read_tlas_map_slot(0), expected_entry);

            // Also verify `copy_as` (COMPACT -> CLONE) produces a valid second TLAS slot.
            let copied_tlas_va = engine
                .copy_as(
                    &list,
                    None,
                    console_copy_va,
                    console_tlas_va,
                    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_COMPACT.0 as u32,
                    1,
                )
                .expect("copy_as");
            assert_ne!(copied_tlas_va, 0);
            assert_ne!(copied_tlas_va, host_tlas_va);

            // Create shader-visible CBV_SRV_UAV descriptor heap and install host TLAS SRV at `tlas_slot`.
            let heap: ID3D12DescriptorHeap = device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: tlas_base + TLAS_SLOTS,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    NodeMask: 0,
                })
                .expect("CreateDescriptorHeap");
            let inc = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            let mut cpu_h = heap.GetCPUDescriptorHandleForHeapStart();
            cpu_h.ptr += (tlas_slot as usize) * (inc as usize);
            DxrEngine::write_tlas_srv(device, cpu_h, host_tlas_va);
            // Also verify null RT-AS SRV (Location = 0) for unused slots passes Debug Layer.
            let null_h = D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: cpu_h.ptr + (inc as usize),
            };
            DxrEngine::write_tlas_srv(device, null_h, 0);

            // Create root signature: param 0 = root SRV t251 space1 (SLOT_TLAS_MAP),
            // param 1 = root UAV u0 space0 (output), param 2 = 4 root constants b0 space0.
            let params = [
                D3D12_ROOT_PARAMETER1 {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
                    ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
                    Anonymous: D3D12_ROOT_PARAMETER1_0 {
                        Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                            ShaderRegister: u32::from(SLOT_TLAS_MAP),
                            RegisterSpace: victory_shader::emit::ROOT_CBV_SPACE,
                            Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE,
                        },
                    },
                },
                D3D12_ROOT_PARAMETER1 {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
                    ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
                    Anonymous: D3D12_ROOT_PARAMETER1_0 {
                        Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                            ShaderRegister: 0,
                            RegisterSpace: 0,
                            Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE,
                        },
                    },
                },
                D3D12_ROOT_PARAMETER1 {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                    ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
                    Anonymous: D3D12_ROOT_PARAMETER1_0 {
                        Constants: D3D12_ROOT_CONSTANTS {
                            ShaderRegister: 0,
                            RegisterSpace: 0,
                            Num32BitValues: 4,
                        },
                    },
                },
            ];
            let rs_desc = D3D12_VERSIONED_ROOT_SIGNATURE_DESC {
                Version: D3D_ROOT_SIGNATURE_VERSION_1_1,
                Anonymous: D3D12_VERSIONED_ROOT_SIGNATURE_DESC_0 {
                    Desc_1_1: D3D12_ROOT_SIGNATURE_DESC1 {
                        NumParameters: params.len() as u32,
                        pParameters: params.as_ptr(),
                        NumStaticSamplers: 0,
                        pStaticSamplers: std::ptr::null(),
                        Flags: D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED,
                    },
                },
            };
            let mut blob = None;
            let mut err_blob = None;
            D3D12SerializeVersionedRootSignature(&rs_desc, &mut blob, Some(&mut err_blob))
                .expect("D3D12SerializeVersionedRootSignature");
            let blob = blob.expect("root sig blob");
            let rs_bytes = std::slice::from_raw_parts(
                blob.GetBufferPointer().cast::<u8>(),
                blob.GetBufferSize(),
            );
            let root_sig: ID3D12RootSignature = device
                .CreateRootSignature(0, rs_bytes)
                .expect("CreateRootSignature");

            let pso: ID3D12PipelineState = device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::ManuallyDrop::new(Some(root_sig.clone())),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: dxil.as_ptr().cast(),
                        BytecodeLength: dxil.len(),
                    },
                    ..Default::default()
                })
                .expect("CreateComputePipelineState SM 6.6 RayQuery");

            let out_uav = create_default_buffer(
                device,
                256,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COMMON,
            )
            .expect("out_uav");
            let readback: ID3D12Resource = {
                let mut r: Option<ID3D12Resource> = None;
                device
                    .CreateCommittedResource(
                        &D3D12_HEAP_PROPERTIES {
                            Type: D3D12_HEAP_TYPE_READBACK,
                            ..Default::default()
                        },
                        D3D12_HEAP_FLAG_NONE,
                        &D3D12_RESOURCE_DESC {
                            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                            Width: 256,
                            Height: 1,
                            DepthOrArraySize: 1,
                            MipLevels: 1,
                            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                            ..Default::default()
                        },
                        D3D12_RESOURCE_STATE_COPY_DEST,
                        None,
                        &mut r,
                    )
                    .expect("readback");
                r.unwrap()
            };

            let (tlas_map_gpu, _tlas_map_buf) = engine.tlas_map_buffer();
            let target_shr8 = console_tlas_va >> 8;
            let consts = [target_shr8 as u32, (target_shr8 >> 32) as u32, 0u32, 0u32];

            list.SetDescriptorHeaps(&[Some(heap.clone())]);
            list.SetComputeRootSignature(&root_sig);
            list.SetPipelineState(&pso);
            list.SetComputeRootShaderResourceView(0, tlas_map_gpu);
            list.SetComputeRootUnorderedAccessView(1, out_uav.GetGPUVirtualAddress());
            list.SetComputeRoot32BitConstants(2, 4, consts.as_ptr().cast(), 0);
            list.Dispatch(1, 1, 1);

            // Barrier UAV -> COPY_SOURCE and copy to readback.
            let to_copy = D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                Anonymous: D3D12_RESOURCE_BARRIER_0 {
                    Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                        pResource: std::mem::ManuallyDrop::new(Some(out_uav.clone())),
                        Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                        StateBefore: D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        StateAfter: D3D12_RESOURCE_STATE_COPY_SOURCE,
                    }),
                },
            };
            list.ResourceBarrier(&[to_copy]);
            list.CopyBufferRegion(&readback, 0, &out_uav, 0, 32);
            list.Close().expect("Close");

            let cmd: ID3D12CommandList = list.cast().unwrap();
            queue.ExecuteCommandLists(&[Some(cmd)]);
            queue.Signal(&fence, 1).expect("Signal");
            fence.SetEventOnCompletion(1, event).expect("SetEventOnCompletion");
            WaitForSingleObject(event, INFINITE);
            let _ = windows::Win32::Foundation::CloseHandle(event);

            let mut mapped = std::ptr::null_mut();
            readback.Map(0, None, Some(&mut mapped)).expect("Map readback");
            let words = std::slice::from_raw_parts(mapped.cast::<u32>(), 8);
            let status = words[0];
            let hit_t = f32::from_bits(words[1]);
            let bary_u = f32::from_bits(words[2]);
            let bary_v = f32::from_bits(words[3]);
            let inst_id = words[4];
            let geom_idx = words[5];
            let prim_idx = words[6];
            let inst_contrib = words[7];
            readback.Unmap(0, None);

            assert_eq!(status, 1, "expected COMMITTED_TRIANGLE_HIT (1)");
            assert!((hit_t - 2.0).abs() < 1e-4, "expected t=2.0, got {hit_t}");
            assert!((bary_u - 0.50).abs() < 1e-4, "expected bary.x=0.50, got {bary_u}");
            assert!((bary_v - 0.25).abs() < 1e-4, "expected bary.y=0.25, got {bary_v}");
            assert_eq!(inst_id, 0x1234, "expected InstanceID 0x1234");
            assert_eq!(geom_idx, 0, "expected GeometryIndex 0");
            assert_eq!(prim_idx, 1, "expected PrimitiveIndex 1");
            assert_eq!(inst_contrib, 7, "expected InstanceContributionToHitGroupIndex 7");

            // Keep VB/IB alive until after ExecuteCommandLists completed.
            drop((vb_buf, ib_buf));

            // Verify eviction clears SLOT_TLAS_MAP and retires AS pool allocations cleanly.
            let evicted = engine.evict_range(0x10_0000_0000, 0x10_0000, 1);
            assert_eq!(evicted, 3, "expected BLAS + TLAS + copied TLAS evicted");
            assert_eq!(engine.read_tlas_map_slot(0), [0, 0, 0, 0]);
            assert_eq!(engine.read_tlas_map_slot(1), [0, 0, 0, 0]);
            let reclaimed = engine.reclaim(1);
            assert_eq!(reclaimed, 3);

            if let Some(q) = info_queue {
                let n = q.GetNumStoredMessagesAllowedByRetrievalFilter();
                let mut errors = Vec::new();
                for i in 0..n {
                    let mut len = 0usize;
                    let _ = q.GetMessage(i, None, &mut len);
                    if len == 0 {
                        continue;
                    }
                    let mut buf = vec![0u8; len];
                    let msg = buf.as_mut_ptr().cast::<D3D12_MESSAGE>();
                    if q.GetMessage(i, Some(msg), &mut len).is_ok() {
                        let sev = (*msg).Severity;
                        if sev == D3D12_MESSAGE_SEVERITY_CORRUPTION
                            || sev == D3D12_MESSAGE_SEVERITY_ERROR
                        {
                            let desc = std::slice::from_raw_parts(
                                (*msg).pDescription,
                                (*msg).DescriptionByteLength.saturating_sub(1),
                            );
                            errors.push(String::from_utf8_lossy(desc).into_owned());
                        }
                    }
                }
                assert!(
                    errors.is_empty(),
                    "D3D12 Debug Layer reported {} error(s):\n{}",
                    errors.len(),
                    errors.join("\n")
                );
            }
        }
    }

    /// Runs live on the active RTX 3090 hardware device as part of normal
    /// `cargo test -p victory-gpu --lib` (without calling `EnableDebugLayer()`
    /// so `host::host()` is never disturbed by parallel tests).
    #[test]
    fn live_rtx3090_dxr11_rayquery_round_trip() {
        let Some(host) = crate::host::host() else {
            return;
        };
        assert!(
            crate::host::raytracing_tier_1_1(),
            "expected host::raytracing_tier_1_1() == true on RTX 3090"
        );
        run_live_dxr11_test(&host.device, None);
    }

    /// Standalone live RTX 3090 DXR 1.1 hardware test with the D3D12 Debug Layer
    /// enabled (`--ignored --nocapture`), asserting 0 Debug Layer errors.
    #[test]
    #[ignore = "enables D3D12 Debug Layer (must run alone, not alongside shared host::host() tests)"]
    fn live_rtx3090_dxr11_blas_tlas_and_tlas_map() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
        use windows::Win32::Graphics::Dxgi::*;

        // SAFETY: standalone hardware D3D12 device with Debug Layer ON.
        unsafe {
            let mut debug: Option<ID3D12Debug> = None;
            D3D12GetDebugInterface(&mut debug).expect("D3D12GetDebugInterface");
            debug.expect("ID3D12Debug").EnableDebugLayer();

            let factory: IDXGIFactory6 =
                CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).expect("CreateDXGIFactory2");
            let adapter: IDXGIAdapter1 = factory
                .EnumAdapterByGpuPreference(0, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE)
                .expect("EnumAdapterByGpuPreference");
            let mut device: Option<ID3D12Device> = None;
            D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device)
                .expect("D3D12CreateDevice");
            let device = device.expect("ID3D12Device");
            let info_queue: ID3D12InfoQueue = device.cast().expect("ID3D12InfoQueue");
            info_queue.ClearStoredMessages();

            run_live_dxr11_test(&device, Some(&info_queue));
        }
    }
}
