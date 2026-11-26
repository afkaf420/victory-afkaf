//! Forward (`twin.plane -> depth`) and reverse (`depth -> twin.plane`) depth
//! plane twin synchronization passes, stencil twin copies, and live RTX 3090
//! D3D12 Debug Layer regression tests for depth-plane twins and cross-list
//! `Shadow::buffer` `id=538` state hygiene.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, OnceLock};

use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use super::{compile, host, transition};

/// The pass that draws the twin of a depth plane into the plane: one
/// triangle over each mip of each slice, its pixel shader writing the
/// twin's texel as the depth, with the depth test "always" and no stencil
/// (the stencil plane is copied through a buffer and not touched here).
///
/// A 16-bit twin holds whole steps of 1/65535, and the depth written is
/// the step times the float nearest 1/65535 (0x37800080). That is how
/// tiled lighting computes the depth of rectangles it draws with test
/// "equal", one class of materials per draw, so the pass and the draw give
/// the host the same float and the host turns it into the same 24 bits.
const PLANE_CORNERS: &str = r#"
float4 corner(uint i : SV_VertexID) : SV_Position {
    return float4(float(i & 1) * 4.0 - 1.0, float(i >> 1) * 4.0 - 1.0, 0.0, 1.0);
}
"#;
const PLANE_STEPS: &str = r#"
#define RS "DescriptorTable(SRV(t0), visibility = SHADER_VISIBILITY_PIXEL)"
Texture2DArray<uint> twin : register(t0);
[RootSignature(RS)]
float plane(float4 p : SV_Position) : SV_Depth {
    return float(twin.Load(int4(int2(p.xy), 0, 0))) * asfloat(0x37800080);
}
"#;
const PLANE_FLOATS: &str = r#"
#define RS "DescriptorTable(SRV(t0), visibility = SHADER_VISIBILITY_PIXEL)"
Texture2DArray<float> twin : register(t0);
[RootSignature(RS)]
float plane(float4 p : SV_Position) : SV_Depth {
    return twin.Load(int4(int2(p.xy), 0, 0));
}
"#;

/// Reverse depth-plane copy shaders (`depth SRV -> twin.plane UAV`): sync what
/// graphics draws and `ClearDepthStencilView` wrote into a depth-stencil
/// texture's depth plane back into its compute-writable `twin.plane` (16-bit
/// `R16_UINT` steps or 32-bit `R32_FLOAT`), preserving graphics RTV/DSV/PSO
/// state.
const REVERSE_PLANE_STEPS: &str = r#"
#define RS "RootConstants(num32BitConstants = 2, b0), DescriptorTable(SRV(t0), UAV(u0))"
cbuffer Size : register(b0) { uint2 dims; };
Texture2DArray<float> depth : register(t0);
RWTexture2DArray<uint> twin : register(u0);
[RootSignature(RS)]
[numthreads(8, 8, 1)]
void plane(uint3 id : SV_DispatchThreadID) {
    if (id.x < dims.x && id.y < dims.y) {
        float d = saturate(depth.Load(int4(int2(id.xy), 0, 0)));
        twin[uint3(id.xy, 0)] = (uint)(d * 65535.0 + 0.5);
    }
}
"#;
const REVERSE_PLANE_FLOATS: &str = r#"
#define RS "RootConstants(num32BitConstants = 2, b0), DescriptorTable(SRV(t0), UAV(u0))"
cbuffer Size : register(b0) { uint2 dims; };
Texture2DArray<float> depth : register(t0);
RWTexture2DArray<float> twin : register(u0);
[RootSignature(RS)]
[numthreads(8, 8, 1)]
void plane(uint3 id : SV_DispatchThreadID) {
    if (id.x < dims.x && id.y < dims.y) {
        twin[uint3(id.xy, 0)] = depth.Load(int4(int2(id.xy), 0, 0));
    }
}
"#;

/// Views of twins and depth textures used by the forward and reverse plane
/// passes, in a shader-visible ring heap. Power of 2 and always allocated in
/// aligned pairs (`fetch_add(2)`) so 2-descriptor SRV+UAV tables never straddle
/// wrap-around, sized for thousands of inline + prelude plane syncs across
/// frames in flight.
const PLANE_VIEWS: u32 = 8192;

/// The pass's root signature and pipeline for one kind of twin and one depth
/// format.
type PlanePipeline = (ID3D12RootSignature, ID3D12PipelineState);

/// What the forward (`twin.plane -> depth`) and reverse (`depth -> twin.plane`)
/// passes run with: the ring heap of views, and lazily compiled pipelines per
/// twin kind (16-bit steps vs 32-bit floats) and depth format (`None`: refused).
struct PlanePass {
    views: ID3D12DescriptorHeap,
    cpu: usize,
    gpu: u64,
    step: u32,
    next: AtomicU32,
    made: Mutex<Vec<(bool, i32, Option<PlanePipeline>)>>,
    reverse: Mutex<Vec<(bool, Option<PlanePipeline>)>>,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for PlanePass {}
// SAFETY: as above.
unsafe impl Sync for PlanePass {}

fn plane_pass() -> Option<&'static PlanePass> {
    static PASS: OnceLock<Option<PlanePass>> = OnceLock::new();
    PASS.get_or_init(|| PlanePass::new(&host()?.device))
        .as_ref()
}

/// Shader-readable DXGI format for plane 0 (depth) of a depth or depth-stencil
/// texture.
fn depth_srv_format(format: DXGI_FORMAT) -> Option<DXGI_FORMAT> {
    match format.0 {
        44 | 45 => Some(DXGI_FORMAT(46)), // R24G8_TYPELESS / D24_UNORM_S8_UINT -> R24_UNORM_X8_TYPELESS
        19 | 20 => Some(DXGI_FORMAT(21)), // R32G8X24_TYPELESS / D32_FLOAT_S8X24_UINT -> R32_FLOAT_X8X24_TYPELESS
        53 | 55 => Some(DXGI_FORMAT(56)), // R16_TYPELESS / D16_UNORM -> R16_UNORM
        39 | 40 => Some(DXGI_FORMAT(41)), // R32_TYPELESS / D32_FLOAT -> R32_FLOAT
        _ => None,
    }
}

fn uav_barrier(resource: &ID3D12Resource) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                // SAFETY: a borrowed pointer; the barrier never releases it.
                pResource: unsafe { std::mem::transmute_copy(resource) },
            }),
        },
    }
}

impl PlanePass {
    fn new(device: &ID3D12Device) -> Option<PlanePass> {
        let kind = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        // SAFETY: plain heap creation and queries.
        unsafe {
            let views: ID3D12DescriptorHeap = device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: kind,
                    NumDescriptors: PLANE_VIEWS,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    NodeMask: 0,
                })
                .ok()?;
            Some(PlanePass {
                cpu: views.GetCPUDescriptorHandleForHeapStart().ptr,
                gpu: views.GetGPUDescriptorHandleForHeapStart().ptr,
                step: device.GetDescriptorHandleIncrementSize(kind),
                views,
                next: AtomicU32::new(0),
                made: Mutex::default(),
                reverse: Mutex::default(),
            })
        }
    }

    fn pipeline_on(
        &self,
        device: &ID3D12Device,
        steps: bool,
        format: DXGI_FORMAT,
    ) -> Option<PlanePipeline> {
        let mut made = self.made.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((_, _, pipeline)) = made.iter().find(|m| (m.0, m.1) == (steps, format.0)) {
            return pipeline.clone();
        }
        let pipeline = match Self::make_on(device, steps, format) {
            Ok(pipeline) => Some(pipeline),
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    steps,
                    format = format.0,
                    "no pipeline to draw a depth plane from its twin: such planes stay as they are"
                );
                None
            }
        };
        made.push((steps, format.0, pipeline.clone()));
        pipeline
    }

    fn reverse_pipeline_on(&self, device: &ID3D12Device, steps: bool) -> Option<PlanePipeline> {
        let mut reverse = self.reverse.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((_, pipeline)) = reverse.iter().find(|m| m.0 == steps) {
            return pipeline.clone();
        }
        let pipeline = match Self::make_reverse_on(device, steps) {
            Ok(pipeline) => Some(pipeline),
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    steps,
                    "no compute pipeline to copy a depth plane into its twin"
                );
                None
            }
        };
        reverse.push((steps, pipeline.clone()));
        pipeline
    }

    fn make_on(
        device: &ID3D12Device,
        steps: bool,
        format: DXGI_FORMAT,
    ) -> windows::core::Result<PlanePipeline> {
        use windows::core::s;
        let vs = compile(PLANE_CORNERS, s!("corner"), s!("vs_5_1"))?;
        let source = if steps { PLANE_STEPS } else { PLANE_FLOATS };
        let ps = compile(source, s!("plane"), s!("ps_5_1"))?;
        let code = |bytes: &[u8]| D3D12_SHADER_BYTECODE {
            pShaderBytecode: bytes.as_ptr().cast(),
            BytecodeLength: bytes.len(),
        };
        // SAFETY: root signature is embedded in the pixel shader; bytecode lives through call.
        unsafe {
            let root: ID3D12RootSignature = device.CreateRootSignature(0, &ps[..])?;
            let desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                VS: code(&vs[..]),
                PS: code(&ps[..]),
                SampleMask: u32::MAX,
                RasterizerState: D3D12_RASTERIZER_DESC {
                    FillMode: D3D12_FILL_MODE_SOLID,
                    CullMode: D3D12_CULL_MODE_NONE,
                    DepthClipEnable: true.into(),
                    ..Default::default()
                },
                DepthStencilState: D3D12_DEPTH_STENCIL_DESC {
                    DepthEnable: true.into(),
                    DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ALL,
                    DepthFunc: D3D12_COMPARISON_FUNC_ALWAYS,
                    ..Default::default()
                },
                PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                DSVFormat: format,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                ..Default::default()
            };
            let pipeline: ID3D12PipelineState = device.CreateGraphicsPipelineState(&desc)?;
            Ok((root, pipeline))
        }
    }

    fn make_reverse_on(
        device: &ID3D12Device,
        steps: bool,
    ) -> windows::core::Result<PlanePipeline> {
        use windows::core::s;
        let source = if steps { REVERSE_PLANE_STEPS } else { REVERSE_PLANE_FLOATS };
        let cs = compile(source, s!("plane"), s!("cs_5_1"))?;
        // SAFETY: root signature is embedded in the compute shader; bytecode lives through call.
        unsafe {
            let root: ID3D12RootSignature = device.CreateRootSignature(0, &cs[..])?;
            let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: cs.as_ptr().cast(),
                    BytecodeLength: cs.len(),
                },
                ..Default::default()
            };
            let pipeline: ID3D12PipelineState = device.CreateComputePipelineState(&desc)?;
            Ok((root, pipeline))
        }
    }
}

/// Core implementation of the forward depth-plane pass (`twin.plane -> depth`).
fn record_plane_pass_with(
    device: &ID3D12Device,
    mut make_dsv: impl FnMut(&ID3D12Resource, &D3D12_DEPTH_STENCIL_VIEW_DESC) -> D3D12_CPU_DESCRIPTOR_HANDLE,
    pass: &PlanePass,
    list: &ID3D12GraphicsCommandList,
    depth: &ID3D12Resource,
    plane: &ID3D12Resource,
) -> bool {
    // SAFETY: live resources.
    let (of, twin) = unsafe { (depth.GetDesc(), plane.GetDesc()) };
    let steps = twin.Format.0 == 53;
    let format = DXGI_FORMAT(match of.Format.0 {
        44 => 45,
        19 => 20,
        typed => typed,
    });
    let Some((root, pipeline)) = pass.pipeline_on(device, steps, format) else {
        return false;
    };
    for slice in 0..u32::from(of.DepthOrArraySize) {
        for mip in 0..u32::from(of.MipLevels) {
            let slot = pass.next.fetch_add(2, Ordering::Relaxed) % PLANE_VIEWS;
            let view = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT(if steps { 57 } else { 41 }),
                ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2DARRAY,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture2DArray: D3D12_TEX2D_ARRAY_SRV {
                        MostDetailedMip: mip,
                        MipLevels: 1,
                        FirstArraySlice: slice,
                        ArraySize: 1,
                        ..Default::default()
                    },
                },
            };
            let target = D3D12_DEPTH_STENCIL_VIEW_DESC {
                Format: format,
                ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2DARRAY,
                Anonymous: D3D12_DEPTH_STENCIL_VIEW_DESC_0 {
                    Texture2DArray: D3D12_TEX2D_ARRAY_DSV {
                        MipSlice: mip,
                        FirstArraySlice: slice,
                        ArraySize: 1,
                    },
                },
                ..Default::default()
            };
            let target = make_dsv(depth, &target);
            let (width, height) = ((of.Width >> mip).max(1) as u32, (of.Height >> mip).max(1));
            let cpu = D3D12_CPU_DESCRIPTOR_HANDLE { ptr: pass.cpu + (slot * pass.step) as usize };
            let gpu = D3D12_GPU_DESCRIPTOR_HANDLE { ptr: pass.gpu + u64::from(slot * pass.step) };
            // SAFETY: an open list; valid slot in our ring heap; live resources and PSO.
            unsafe {
                device.CreateShaderResourceView(plane, Some(&view), cpu);
                list.SetDescriptorHeaps(&[Some(pass.views.clone())]);
                list.SetGraphicsRootSignature(&root);
                list.SetPipelineState(&pipeline);
                list.SetGraphicsRootDescriptorTable(0, gpu);
                list.OMSetRenderTargets(0, None, false, Some(&target));
                list.RSSetViewports(&[D3D12_VIEWPORT {
                    TopLeftX: 0.0,
                    TopLeftY: 0.0,
                    Width: width as f32,
                    Height: height as f32,
                    MinDepth: 0.0,
                    MaxDepth: 1.0,
                }]);
                list.RSSetScissorRects(&[RECT {
                    left: 0,
                    top: 0,
                    right: width as i32,
                    bottom: height as i32,
                }]);
                list.IASetPrimitiveTopology(
                    windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY(4),
                );
                list.DrawInstanced(3, 1, 0, 0);
            }
            crate::resource::PLANES[0].fetch_add(1, Ordering::Relaxed);
        }
    }
    true
}

/// Records into `list` the pass that draws `plane`, the twin of the depth
/// plane of `depth`, into that plane (`depth` in `DEPTH_WRITE`, `plane` in
/// `PIXEL_SHADER_RESOURCE`).
pub(super) fn record_plane_pass(
    list: &ID3D12GraphicsCommandList,
    depth: &ID3D12Resource,
    plane: &ID3D12Resource,
) {
    let (Some(host), Some(pass)) = (host(), plane_pass()) else {
        return;
    };
    let _ = record_plane_pass_with(
        &host.device,
        |res, desc| host.dsv(res, Some(desc)),
        pass,
        list,
        depth,
        plane,
    );
}

/// Core implementation of the reverse depth-plane copy pass
/// (`depth SRV -> twin.plane UAV`). Expects `depth` in
/// `NON_PIXEL_SHADER_RESOURCE` and `plane` in `UNORDERED_ACCESS`, and ends
/// with a UAV barrier on `plane`.
fn record_reverse_plane_pass_with(
    device: &ID3D12Device,
    pass: &PlanePass,
    list: &ID3D12GraphicsCommandList,
    depth: &ID3D12Resource,
    plane: &ID3D12Resource,
) -> bool {
    // SAFETY: live resources.
    let (of, twin) = unsafe { (depth.GetDesc(), plane.GetDesc()) };
    let Some(srv_format) = depth_srv_format(of.Format) else {
        return false;
    };
    let steps = twin.Format.0 == 53;
    let uav_format = DXGI_FORMAT(if steps { 57 } else { 41 });
    let Some((root, pipeline)) = pass.reverse_pipeline_on(device, steps) else {
        return false;
    };
    // SAFETY: an open list; live root signature, compute PSO, and heap.
    unsafe {
        list.SetDescriptorHeaps(&[Some(pass.views.clone())]);
        list.SetComputeRootSignature(&root);
        list.SetPipelineState(&pipeline);
    }
    for slice in 0..u32::from(of.DepthOrArraySize) {
        for mip in 0..u32::from(of.MipLevels) {
            let slot = pass.next.fetch_add(2, Ordering::Relaxed) % PLANE_VIEWS;
            let srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: srv_format,
                ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2DARRAY,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture2DArray: D3D12_TEX2D_ARRAY_SRV {
                        MostDetailedMip: mip,
                        MipLevels: 1,
                        FirstArraySlice: slice,
                        ArraySize: 1,
                        PlaneSlice: 0,
                        ..Default::default()
                    },
                },
            };
            let uav = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: uav_format,
                ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2DARRAY,
                Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                    Texture2DArray: D3D12_TEX2D_ARRAY_UAV {
                        MipSlice: mip,
                        FirstArraySlice: slice,
                        ArraySize: 1,
                        PlaneSlice: 0,
                    },
                },
            };
            let (width, height) = ((of.Width >> mip).max(1) as u32, (of.Height >> mip).max(1));
            let dims = [width, height];
            let cpu_srv = D3D12_CPU_DESCRIPTOR_HANDLE { ptr: pass.cpu + (slot * pass.step) as usize };
            let cpu_uav = D3D12_CPU_DESCRIPTOR_HANDLE { ptr: pass.cpu + ((slot + 1) * pass.step) as usize };
            let gpu = D3D12_GPU_DESCRIPTOR_HANDLE { ptr: pass.gpu + u64::from(slot * pass.step) };
            // SAFETY: an open list; valid contiguous descriptor pair; live textures.
            unsafe {
                device.CreateShaderResourceView(depth, Some(&srv), cpu_srv);
                device.CreateUnorderedAccessView(plane, None, Some(&uav), cpu_uav);
                list.SetComputeRoot32BitConstants(0, 2, dims.as_ptr().cast(), 0);
                list.SetComputeRootDescriptorTable(1, gpu);
                list.Dispatch(width.div_ceil(8), height.div_ceil(8), 1);
            }
        }
    }
    // SAFETY: an open list and live texture.
    unsafe { list.ResourceBarrier(&[uav_barrier(plane)]) };
    true
}

/// Records into `list` the compute pass that copies the depth plane of `depth`
/// (`NON_PIXEL_SHADER_RESOURCE`) into `plane` (`UNORDERED_ACCESS`), preserving
/// graphics RTV/DSV/PSO/viewport/scissor state.
pub(super) fn record_reverse_plane_pass(
    list: &ID3D12GraphicsCommandList,
    depth: &ID3D12Resource,
    plane: &ID3D12Resource,
) -> bool {
    let (Some(host), Some(pass)) = (host(), plane_pass()) else {
        return false;
    };
    record_reverse_plane_pass_with(&host.device, pass, list, depth, plane)
}

/// Records into `list` the copy of a depth texture into its colour twin
/// (`colour`) or of the twin into the depth texture, both already in their
/// copy states: whole textures of one size in one format family, or for a
/// stencil-plane twin, that plane alone through `twin.through`. Separate
/// depth-plane twins (`twin.plane`) are synchronized alongside this copy by
/// [`record_reverse_plane_pass`] and [`record_plane_pass`].
pub(super) fn record_twin_copy(
    list: &ID3D12GraphicsCommandList,
    depth: &ID3D12Resource,
    twin: &crate::resource::Twin,
    colour: bool,
) {
    let (from, to) = if colour {
        (depth, &twin.colour.texture)
    } else {
        (&twin.colour.texture, depth)
    };
    crate::resource::TWINS[2 + usize::from(colour)].fetch_add(1, Ordering::Relaxed);
    let Some(through) = &twin.through else {
        // SAFETY: an open list and two live textures.
        unsafe { list.CopyResource(to, from) };
        return;
    };
    let Some(host) = host() else { return };
    // SAFETY: a live texture.
    let of = unsafe { depth.GetDesc() };
    let planes = u32::from(of.MipLevels) * u32::from(of.DepthOrArraySize);
    let mut rows = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); planes as usize];
    // SAFETY: a plain query writing `planes` footprints.
    unsafe {
        host.device.GetCopyableFootprints(
            &of,
            planes,
            planes,
            0,
            Some(rows.as_mut_ptr()),
            None,
            None,
            None,
        );
    }
    let (first_from, first_to) = if colour { (planes, 0) } else { (0, planes) };
    let subresource = |texture: &ID3D12Resource, index: u32| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: a borrowed pointer the location never releases.
        pResource: unsafe { std::mem::transmute_copy(texture) },
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: index },
    };
    let placed = |row: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: as above.
        pResource: unsafe { std::mem::transmute_copy(through) },
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: *row },
    };
    let (common, source, dest) = (
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
        D3D12_RESOURCE_STATE_COPY_DEST,
    );
    // SAFETY: an open list and live resources; `through` holds all footprints.
    unsafe {
        list.ResourceBarrier(&[transition(through, common, dest)]);
        for (i, row) in rows.iter().enumerate() {
            let index = first_from + i as u32;
            list.CopyTextureRegion(&placed(row), 0, 0, 0, &subresource(from, index), None);
        }
        list.ResourceBarrier(&[transition(through, dest, source)]);
        for (i, row) in rows.iter().enumerate() {
            let index = first_to + i as u32;
            list.CopyTextureRegion(&subresource(to, index), 0, 0, 0, &placed(row), None);
        }
        list.ResourceBarrier(&[transition(through, source, common)]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host::execute_command_list_groups;
    use windows::Win32::Foundation::{CloseHandle, HANDLE};
    use windows::Win32::Graphics::Direct3D::{D3D_FEATURE_LEVEL_12_0, D3D_PRIMITIVE_TOPOLOGY};
    use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
    use windows::core::{Interface, s};

    fn make_standalone_debug_device() -> (
        ID3D12Device,
        ID3D12InfoQueue,
        ID3D12CommandQueue,
        ID3D12Fence,
        HANDLE,
    ) {
        // SAFETY: standalone D3D12 Debug Layer device for live hardware tests.
        unsafe {
            let mut dbg: Option<ID3D12Debug> = None;
            D3D12GetDebugInterface(&mut dbg).expect("D3D12GetDebugInterface");
            dbg.expect("ID3D12Debug").EnableDebugLayer();
            let mut dev: Option<ID3D12Device> = None;
            D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut dev).expect("D3D12CreateDevice");
            let dev = dev.unwrap();
            let iq: ID3D12InfoQueue = dev.cast().expect("ID3D12InfoQueue");
            iq.ClearStoredMessages();
            let queue: ID3D12CommandQueue = dev
                .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                    ..Default::default()
                })
                .unwrap();
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let event = CreateEventW(None, false, false, None).unwrap();
            (dev, iq, queue, fence, event)
        }
    }

    fn drain_debug_errors(iq: &ID3D12InfoQueue) -> Vec<(i32, i32, String)> {
        let mut errors = Vec::new();
        // SAFETY: reading stored D3D12 InfoQueue messages into sized buffers.
        unsafe {
            for i in 0..iq.GetNumStoredMessages() {
                let mut size = 0usize;
                if iq.GetMessage(i, None, &mut size).is_err() || size == 0 {
                    continue;
                }
                let mut buf = vec![0u64; size.div_ceil(8)];
                let ptr = buf.as_mut_ptr().cast::<D3D12_MESSAGE>();
                if iq.GetMessage(i, Some(ptr), &mut size).is_ok() {
                    let m = &*ptr;
                    let text = String::from_utf8_lossy(std::slice::from_raw_parts(
                        m.pDescription,
                        m.DescriptionByteLength,
                    ))
                    .trim_end_matches('\0')
                    .to_owned();
                    if m.Severity.0 <= D3D12_MESSAGE_SEVERITY_ERROR.0 {
                        errors.push((m.ID.0, m.Severity.0, text));
                    }
                }
            }
            iq.ClearStoredMessages();
        }
        errors
    }

    unsafe fn test_buf(
        dev: &ID3D12Device,
        heap: D3D12_HEAP_TYPE,
        size: u64,
        flags: D3D12_RESOURCE_FLAGS,
        state: D3D12_RESOURCE_STATES,
    ) -> ID3D12Resource {
        let mut res: Option<ID3D12Resource> = None;
        // SAFETY: plain committed buffer creation.
        unsafe {
            dev.CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES { Type: heap, ..Default::default() },
                D3D12_HEAP_FLAG_NONE,
                &D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: size,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    Flags: flags,
                    ..Default::default()
                },
                state,
                None,
                &mut res,
            )
            .unwrap();
        }
        res.unwrap()
    }

    unsafe fn test_tex2d(
        dev: &ID3D12Device,
        w: u64,
        h: u32,
        slices: u16,
        mips: u16,
        fmt: DXGI_FORMAT,
        flags: D3D12_RESOURCE_FLAGS,
        state: D3D12_RESOURCE_STATES,
    ) -> ID3D12Resource {
        let mut res: Option<ID3D12Resource> = None;
        // SAFETY: plain committed 2D texture creation.
        unsafe {
            dev.CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
                D3D12_HEAP_FLAG_NONE,
                &D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                    Width: w,
                    Height: h,
                    DepthOrArraySize: slices,
                    MipLevels: mips,
                    Format: fmt,
                    SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    Flags: flags,
                    ..Default::default()
                },
                state,
                None,
                &mut res,
            )
            .unwrap();
        }
        res.unwrap()
    }

    /// Live RTX 3090 D3D12 Debug Layer test: verifies BOTH forward
    /// (`twin.plane -> depth`) AND reverse (`depth -> twin.plane`) depth-plane
    /// copies across multiple mips/slices for 16-bit (`R24G8` <-> `R16_UINT`)
    /// + `ClearDepthStencilView` and 32-bit (`R32G8X24` <-> `R32_FLOAT`),
    /// asserting bit-exactness and `0` D3D12 Debug Layer errors.
    #[test]
    #[ignore = "requires live D3D12 GPU with Debug Layer"]
    fn live_rtx3090_depth_plane_twin_roundtrip_debug_layer() {
        let (dev, iq, queue, fence, event) = make_standalone_debug_device();
        let pass = PlanePass::new(&dev).expect("PlanePass::new");
        // SAFETY: standalone D3D12 Debug Layer objects owned by this test.
        unsafe {
            let dsv_heap: ID3D12DescriptorHeap = dev
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
                    NumDescriptors: 64,
                    ..Default::default()
                })
                .expect("DSV heap");
            let dsv_base = dsv_heap.GetCPUDescriptorHandleForHeapStart().ptr;
            let dsv_step = dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
            let mut dsv_next = 0u32;
            let mut alloc_dsv = |res: &ID3D12Resource, desc: &D3D12_DEPTH_STENCIL_VIEW_DESC| {
                let h = D3D12_CPU_DESCRIPTOR_HANDLE {
                    ptr: dsv_base + (dsv_next * dsv_step) as usize,
                };
                dsv_next = (dsv_next + 1) % 64;
                dev.CreateDepthStencilView(res, Some(desc), h);
                h
            };
            let alloc: ID3D12CommandAllocator =
                dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let list: ID3D12GraphicsCommandList =
                dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
            list.Close().unwrap();
            let mut fval = 0u64;
            let mut run = |rec: &mut dyn FnMut(&ID3D12GraphicsCommandList)| {
                alloc.Reset().unwrap();
                list.Reset(&alloc, None).unwrap();
                rec(&list);
                list.Close().unwrap();
                queue.ExecuteCommandLists(&[Some(list.cast::<ID3D12CommandList>().unwrap())]);
                fval += 1;
                queue.Signal(&fence, fval).unwrap();
                fence.SetEventOnCompletion(fval, event).unwrap();
                WaitForSingleObject(event, INFINITE);
            };
            let sub_loc = |tex: &ID3D12Resource, idx: u32| D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(tex),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: idx },
            };
            let fp_loc = |buf: &ID3D12Resource, fp: D3D12_PLACED_SUBRESOURCE_FOOTPRINT| {
                D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::transmute_copy(buf),
                    Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: fp },
                }
            };
            let mut verify = |w: u64,
                              h: u32,
                              slices: u16,
                              mips: u16,
                              d_fmt: i32,
                              p_fmt: i32,
                              view_fmt: i32,
                              clob: u8,
                              fill: &dyn Fn(*mut u8, usize, usize, usize, usize, usize),
                              check: &dyn Fn(*const u8, usize, usize, usize, usize, usize),
                              clear_test: bool| {
                let depth = test_tex2d(
                    &dev,
                    w,
                    h,
                    slices,
                    mips,
                    DXGI_FORMAT(d_fmt),
                    D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
                    D3D12_RESOURCE_STATE_DEPTH_WRITE,
                );
                let plane = test_tex2d(
                    &dev,
                    w,
                    h,
                    slices,
                    mips,
                    DXGI_FORMAT(p_fmt),
                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                );
                let mut pd = plane.GetDesc();
                pd.Format = DXGI_FORMAT(view_fmt);
                let count = usize::from(slices) * usize::from(mips);
                let mut fps = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); count];
                let mut total = 0u64;
                dev.GetCopyableFootprints(
                    &pd,
                    0,
                    count as u32,
                    0,
                    Some(fps.as_mut_ptr()),
                    None,
                    None,
                    Some(&mut total),
                );
                let up_src = test_buf(
                    &dev,
                    D3D12_HEAP_TYPE_UPLOAD,
                    total,
                    D3D12_RESOURCE_FLAG_NONE,
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                );
                let up_clob = test_buf(
                    &dev,
                    D3D12_HEAP_TYPE_UPLOAD,
                    total,
                    D3D12_RESOURCE_FLAG_NONE,
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                );
                let rb = test_buf(
                    &dev,
                    D3D12_HEAP_TYPE_READBACK,
                    total,
                    D3D12_RESOURCE_FLAG_NONE,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                );
                let (mut ps, mut pc) = (std::ptr::null_mut(), std::ptr::null_mut());
                up_src.Map(0, None, Some(&mut ps)).unwrap();
                up_clob.Map(0, None, Some(&mut pc)).unwrap();
                std::ptr::write_bytes(pc.cast::<u8>(), clob, total as usize);
                for s in 0..usize::from(slices) {
                    for m in 0..usize::from(mips) {
                        let fp = fps[s * usize::from(mips) + m];
                        fill(
                            ps.cast::<u8>().add(fp.Offset as usize),
                            fp.Footprint.RowPitch as usize,
                            s,
                            m,
                            (w as usize >> m).max(1),
                            (h as usize >> m).max(1),
                        );
                    }
                }
                up_src.Unmap(0, None);
                up_clob.Unmap(0, None);
                run(&mut |l| {
                    for (sub, &fp) in fps.iter().enumerate() {
                        l.CopyTextureRegion(
                            &sub_loc(&plane, sub as u32),
                            0,
                            0,
                            0,
                            &fp_loc(&up_src, fp),
                            None,
                        );
                    }
                    l.ResourceBarrier(&[transition(
                        &plane,
                        D3D12_RESOURCE_STATE_COPY_DEST,
                        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                    )]);
                    assert!(record_plane_pass_with(&dev, &mut alloc_dsv, &pass, l, &depth, &plane));
                    l.ResourceBarrier(&[transition(
                        &plane,
                        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                        D3D12_RESOURCE_STATE_COPY_DEST,
                    )]);
                    for (sub, &fp) in fps.iter().enumerate() {
                        l.CopyTextureRegion(
                            &sub_loc(&plane, sub as u32),
                            0,
                            0,
                            0,
                            &fp_loc(&up_clob, fp),
                            None,
                        );
                    }
                    l.ResourceBarrier(&[
                        transition(
                            &depth,
                            D3D12_RESOURCE_STATE_DEPTH_WRITE,
                            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                        ),
                        transition(
                            &plane,
                            D3D12_RESOURCE_STATE_COPY_DEST,
                            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        ),
                    ]);
                    assert!(record_reverse_plane_pass_with(&dev, &pass, l, &depth, &plane));
                    l.ResourceBarrier(&[transition(
                        &plane,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                    )]);
                    for (sub, &fp) in fps.iter().enumerate() {
                        l.CopyTextureRegion(
                            &fp_loc(&rb, fp),
                            0,
                            0,
                            0,
                            &sub_loc(&plane, sub as u32),
                            None,
                        );
                    }
                });
                let mut pr = std::ptr::null_mut();
                rb.Map(0, None, Some(&mut pr)).unwrap();
                for s in 0..usize::from(slices) {
                    for m in 0..usize::from(mips) {
                        let fp = fps[s * usize::from(mips) + m];
                        check(
                            pr.cast::<u8>().add(fp.Offset as usize),
                            fp.Footprint.RowPitch as usize,
                            s,
                            m,
                            (w as usize >> m).max(1),
                            (h as usize >> m).max(1),
                        );
                    }
                }
                rb.Unmap(0, None);
                if clear_test {
                    run(&mut |l| {
                        l.ResourceBarrier(&[
                            transition(
                                &depth,
                                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                D3D12_RESOURCE_STATE_DEPTH_WRITE,
                            ),
                            transition(
                                &plane,
                                D3D12_RESOURCE_STATE_COPY_SOURCE,
                                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                            ),
                        ]);
                        for (slice, val) in [(0u32, 0.25f32), (1u32, 0.75f32)] {
                            let dsv = alloc_dsv(
                                &depth,
                                &D3D12_DEPTH_STENCIL_VIEW_DESC {
                                    Format: DXGI_FORMAT(45),
                                    ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2DARRAY,
                                    Anonymous: D3D12_DEPTH_STENCIL_VIEW_DESC_0 {
                                        Texture2DArray: D3D12_TEX2D_ARRAY_DSV {
                                            MipSlice: 0,
                                            FirstArraySlice: slice,
                                            ArraySize: 1,
                                        },
                                    },
                                    ..Default::default()
                                },
                            );
                            l.ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, val, 0, None);
                        }
                        l.ResourceBarrier(&[transition(
                            &depth,
                            D3D12_RESOURCE_STATE_DEPTH_WRITE,
                            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                        )]);
                        assert!(record_reverse_plane_pass_with(&dev, &pass, l, &depth, &plane));
                        l.ResourceBarrier(&[transition(
                            &plane,
                            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                            D3D12_RESOURCE_STATE_COPY_SOURCE,
                        )]);
                        for (sub, &fp) in fps.iter().enumerate() {
                            l.CopyTextureRegion(
                                &fp_loc(&rb, fp),
                                0,
                                0,
                                0,
                                &sub_loc(&plane, sub as u32),
                                None,
                            );
                        }
                    });
                    rb.Map(0, None, Some(&mut pr)).unwrap();
                    for (s, want) in [(0usize, 16384u16), (1usize, 49151u16)] {
                        let got =
                            *pr.cast::<u8>().add(fps[s * usize::from(mips)].Offset as usize).cast::<u16>();
                        assert_eq!(got, want, "ClearDSV reverse CS mismatch slice={s}");
                    }
                    rb.Unmap(0, None);
                }
            };

            let crit: [u16; 12] = [0, 1, 8, 65, 128, 129, 255, 1024, 16384, 32768, 65534, 65535];
            let exp16 = |s: usize, m: usize, x: usize, y: usize| -> u16 {
                let i = s * 1009 + m * 257 + y * 64 + x;
                if i < crit.len() { crit[i] } else { ((i * 131 + 17) & 0xFFFF) as u16 }
            };
            verify(
                64,
                64,
                2,
                2,
                44,
                53,
                57,
                0xEE,
                &|dst, pitch, s, m, mw, mh| {
                    for y in 0..mh {
                        for x in 0..mw {
                            *dst.add(y * pitch).cast::<u16>().add(x) = exp16(s, m, x, y);
                        }
                    }
                },
                &|src, pitch, s, m, mw, mh| {
                    for y in 0..mh {
                        for x in 0..mw {
                            assert_eq!(
                                *src.add(y * pitch).cast::<u16>().add(x),
                                exp16(s, m, x, y),
                                "16-bit s={s} m={m} ({x},{y})"
                            );
                        }
                    }
                },
                true,
            );

            let samp: [f32; 8] = [0.0, 0.0001, 0.125, 1.0 / 3.0, 0.5, 0.75, 0.99999994, 1.0];
            let exp32 = |s: usize, m: usize, x: usize, y: usize| {
                samp[(s * 3 + m * 5 + y * 32 + x) % samp.len()]
            };
            verify(
                32,
                32,
                2,
                2,
                19,
                39,
                41,
                0x00,
                &|dst, pitch, s, m, mw, mh| {
                    for y in 0..mh {
                        for x in 0..mw {
                            *dst.add(y * pitch).cast::<f32>().add(x) = exp32(s, m, x, y);
                        }
                    }
                },
                &|src, pitch, s, m, mw, mh| {
                    for y in 0..mh {
                        for x in 0..mw {
                            assert_eq!(
                                (*src.add(y * pitch).cast::<f32>().add(x)).to_bits(),
                                exp32(s, m, x, y).to_bits(),
                                "32-bit s={s} m={m} ({x},{y})"
                            );
                        }
                    }
                },
                false,
            );

            let errors = drain_debug_errors(&iq);
            let _ = CloseHandle(event);
            assert!(errors.is_empty(), "expected 0 Debug Layer errors, got: {errors:#?}");
            println!("live_rtx3090_depth_plane_twin_roundtrip_debug_layer: 16-bit + 32-bit multi-mip/slice + ClearDSV bit-exact, 0 Debug Layer errors");
        }
    }

    /// Live RTX 3090 D3D12 Debug Layer test for `247x id=538` cross-list
    /// `Shadow::buffer` state mismatches (`0x40` <-> `0x1`): reproduces exact
    /// `id=538` in both directions when batched in one `ExecuteCommandLists`,
    /// then verifies `sync` (`-> COMMON`) + `execute_command_list_groups`
    /// yields `0` Debug Layer errors.
    #[test]
    #[ignore = "requires live D3D12 GPU with Debug Layer"]
    fn live_rtx3090_shadow_buffer_cross_list_id538_debug_layer() {
        let (dev, iq, queue, fence, event) = make_standalone_debug_device();
        const CS: &str = r#"#define RS "SRV(t0), UAV(u0)"
ByteAddressBuffer s : register(t0); RWByteAddressBuffer d : register(u0);
[RootSignature(RS)][numthreads(1,1,1)] void main() { d.Store(0, s.Load(0)); }"#;
        const VS: &str = r#"#define RS "RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT)"
[RootSignature(RS)] float4 vs(float4 p : POSITION) : SV_Position { return p; }"#;
        const PS: &str = "float4 ps() : SV_Target { return float4(1,0,0,1); }";

        // SAFETY: standalone D3D12 Debug Layer resources owned by this test.
        unsafe {
            let (cs_bc, vs_bc, ps_bc) = (
                compile(CS, s!("main"), s!("cs_5_1")).unwrap(),
                compile(VS, s!("vs"), s!("vs_5_1")).unwrap(),
                compile(PS, s!("ps"), s!("ps_5_1")).unwrap(),
            );
            let cs_root: ID3D12RootSignature = dev.CreateRootSignature(0, &cs_bc).unwrap();
            let cs_pso: ID3D12PipelineState = dev
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&cs_root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: cs_bc.as_ptr().cast(),
                        BytecodeLength: cs_bc.len(),
                    },
                    ..Default::default()
                })
                .unwrap();
            let gfx_root: ID3D12RootSignature = dev.CreateRootSignature(0, &vs_bc).unwrap();
            let elem = D3D12_INPUT_ELEMENT_DESC {
                SemanticName: s!("POSITION"),
                Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
                InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
                ..Default::default()
            };
            let mut gd = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&gfx_root),
                VS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: vs_bc.as_ptr().cast(),
                    BytecodeLength: vs_bc.len(),
                },
                PS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: ps_bc.as_ptr().cast(),
                    BytecodeLength: ps_bc.len(),
                },
                InputLayout: D3D12_INPUT_LAYOUT_DESC { pInputElementDescs: &elem, NumElements: 1 },
                SampleMask: u32::MAX,
                RasterizerState: D3D12_RASTERIZER_DESC {
                    FillMode: D3D12_FILL_MODE_SOLID,
                    CullMode: D3D12_CULL_MODE_NONE,
                    ..Default::default()
                },
                PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                NumRenderTargets: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                ..Default::default()
            };
            gd.BlendState.RenderTarget[0].RenderTargetWriteMask = 0xF;
            gd.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
            let gfx_pso: ID3D12PipelineState = dev.CreateGraphicsPipelineState(&gd).unwrap();

            let shadow_buf = test_buf(
                &dev,
                D3D12_HEAP_TYPE_DEFAULT,
                65536,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COMMON,
            );
            let upload_buf = test_buf(
                &dev,
                D3D12_HEAP_TYPE_UPLOAD,
                4096,
                D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            );
            let out_uav = test_buf(
                &dev,
                D3D12_HEAP_TYPE_DEFAULT,
                256,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COMMON,
            );
            let rtv_tex = test_tex2d(
                &dev,
                16,
                16,
                1,
                1,
                DXGI_FORMAT_R8G8B8A8_UNORM,
                D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
            );
            let rtv_heap: ID3D12DescriptorHeap = dev
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                    NumDescriptors: 1,
                    ..Default::default()
                })
                .unwrap();
            let rtv_h = rtv_heap.GetCPUDescriptorHandleForHeapStart();
            dev.CreateRenderTargetView(&rtv_tex, None, rtv_h);

            let mk = || {
                let a: ID3D12CommandAllocator =
                    dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
                let l: ID3D12GraphicsCommandList =
                    dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &a, None).unwrap();
                l.Close().unwrap();
                (a, l)
            };
            let ((a_sync, l_sync), (a1, l1), (a2, l2), (a3, l3)) = (mk(), mk(), mk(), mk());
            let rec_sync = |buf: &ID3D12Resource| {
                a_sync.Reset().unwrap();
                l_sync.Reset(&a_sync, None).unwrap();
                l_sync.CopyBufferRegion(buf, 0, &upload_buf, 0, 4096);
                l_sync.ResourceBarrier(&[transition(
                    buf,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                )]);
                l_sync.SetComputeRootSignature(&cs_root);
                l_sync.SetPipelineState(&cs_pso);
                l_sync.SetComputeRootShaderResourceView(0, buf.GetGPUVirtualAddress());
                l_sync.SetComputeRootUnorderedAccessView(1, out_uav.GetGPUVirtualAddress());
                l_sync.Dispatch(1, 1, 1);
                l_sync.ResourceBarrier(&[transition(
                    buf,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                    D3D12_RESOURCE_STATE_COMMON,
                )]);
                l_sync.Close().unwrap();
                l_sync.cast::<ID3D12CommandList>().unwrap()
            };
            let rec_cs = |a: &ID3D12CommandAllocator,
                          l: &ID3D12GraphicsCommandList,
                          buf: &ID3D12Resource,
                          explicit: bool| {
                a.Reset().unwrap();
                l.Reset(a, None).unwrap();
                if explicit {
                    l.ResourceBarrier(&[transition(
                        buf,
                        D3D12_RESOURCE_STATE_COMMON,
                        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                    )]);
                }
                l.SetComputeRootSignature(&cs_root);
                l.SetPipelineState(&cs_pso);
                l.SetComputeRootShaderResourceView(0, buf.GetGPUVirtualAddress());
                l.SetComputeRootUnorderedAccessView(1, out_uav.GetGPUVirtualAddress());
                l.Dispatch(1, 1, 1);
                l.Close().unwrap();
                l.cast::<ID3D12CommandList>().unwrap()
            };
            let rec_vb = |a: &ID3D12CommandAllocator,
                          l: &ID3D12GraphicsCommandList,
                          buf: &ID3D12Resource,
                          explicit: bool| {
                let vbv = D3D12_VERTEX_BUFFER_VIEW {
                    BufferLocation: buf.GetGPUVirtualAddress(),
                    SizeInBytes: 48,
                    StrideInBytes: 16,
                };
                a.Reset().unwrap();
                l.Reset(a, None).unwrap();
                if explicit {
                    l.ResourceBarrier(&[transition(
                        buf,
                        D3D12_RESOURCE_STATE_COMMON,
                        D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
                    )]);
                }
                l.SetGraphicsRootSignature(&gfx_root);
                l.SetPipelineState(&gfx_pso);
                l.OMSetRenderTargets(1, Some(&rtv_h), false, None);
                l.RSSetViewports(&[D3D12_VIEWPORT {
                    Width: 16.0,
                    Height: 16.0,
                    MaxDepth: 1.0,
                    ..Default::default()
                }]);
                l.RSSetScissorRects(&[RECT { right: 16, bottom: 16, ..Default::default() }]);
                l.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY(4));
                l.IASetVertexBuffers(0, Some(&[vbv]));
                l.DrawInstanced(3, 1, 0, 0);
                l.Close().unwrap();
                l.cast::<ID3D12CommandList>().unwrap()
            };
            let mut fval = 0u64;
            let mut wait_gpu = || {
                fval += 1;
                queue.Signal(&fence, fval).unwrap();
                fence.SetEventOnCompletion(fval, event).unwrap();
                WaitForSingleObject(event, INFINITE);
            };

            // Negative control: batched ExecuteCommandLists hits id=538 on both 0x40->0x1 and 0x1->0x40.
            queue.ExecuteCommandLists(&[
                Some(rec_cs(&a1, &l1, &shadow_buf, true)),
                Some(rec_vb(&a2, &l2, &shadow_buf, false)),
            ]);
            wait_gpu();
            let raw1 = drain_debug_errors(&iq);
            queue.ExecuteCommandLists(&[
                Some(rec_vb(&a1, &l1, &shadow_buf, true)),
                Some(rec_cs(&a2, &l2, &shadow_buf, false)),
            ]);
            wait_gpu();
            let raw2 = drain_debug_errors(&iq);
            let raw_538 = raw1.iter().chain(&raw2).filter(|(id, _, _)| *id == 538).count();
            assert!(
                raw1.iter().any(|(id, _, _)| *id == 538) && raw2.iter().any(|(id, _, _)| *id == 538),
                "expected id=538, got {raw1:#?} {raw2:#?}"
            );

            // Fix verification: `sync` (-> COMMON) + `execute_command_list_groups` on alternating [CS, VB, CS].
            queue.ExecuteCommandLists(&[Some(rec_sync(&shadow_buf))]);
            execute_command_list_groups(
                &queue,
                &[
                    Some(rec_cs(&a1, &l1, &shadow_buf, true)),
                    Some(rec_vb(&a2, &l2, &shadow_buf, true)),
                    Some(rec_cs(&a3, &l3, &shadow_buf, false)),
                ],
                &[0, 1, 2],
            );
            wait_gpu();

            let fixed_errors = drain_debug_errors(&iq);
            let _ = CloseHandle(event);
            assert!(fixed_errors.is_empty(), "expected 0 Debug Layer errors, got: {fixed_errors:#?}");
            println!(
                "live_rtx3090_shadow_buffer_cross_list_id538_debug_layer: negative control caught {raw_538}x id=538 (0x40->0x1 + 0x1->0x40); fixed sync + execute_command_list_groups -> 0 Debug Layer errors"
            );
        }
    }
}
