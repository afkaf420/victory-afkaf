//! Hardware DXR 1.1 `RayQuery` tests (`VICTORY_DXR_RAYQUERY=1`), including a
//! live RTX 3090 differential GPU test comparing software RDNA2 BVH4 traversal
//! (`emit/intersect.rs`) against hardware SM 6.6 `RayQuery` (`emit/rayquery.rs`)
//! on a multi-instance, multi-geometry scene, plus regression tests for
//! non-opaque fallback and numeric SOP mask-PHI tainting.

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::Interface;

use super::host::{Dispatch, Host, Root, View};
use super::{HostShader, SLOT_TLAS_MAP, TLAS_MAP_ENTRIES, lower, validator};
use crate::bvh;
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

struct DxrRayQueryGuard;

impl DxrRayQueryGuard {
    fn set(on: bool) -> Self {
        lower::TEST_DXR_RAYQUERY.set(Some(on));
        Self
    }
}

impl Drop for DxrRayQueryGuard {
    fn drop(&mut self) {
        lower::TEST_DXR_RAYQUERY.set(None);
    }
}

struct SparseUmaGuard;

impl SparseUmaGuard {
    fn set(on: bool) -> Self {
        lower::TEST_SPARSE_UMA.set(Some(on));
        Self
    }
}

impl Drop for SparseUmaGuard {
    fn drop(&mut self) {
        lower::TEST_SPARSE_UMA.set(None);
    }
}

fn lower_and_validate_cs(native: &NativeProgram) -> HostShader {
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource");
    let io = io::raise(&p, native).expect("io::raise");
    let lowered = lower::lower(&lower::Parts {
        native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    })
    .expect("lower");
    let mut shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        root_indexed_views: Vec::new(),
    };
    if let Ok(mut v) = validator::Validator::load(None) {
        v.validate(&mut shader.dxil).expect("DXIL validation failed");
    }
    shader
}

#[test]
fn slot_tlas_map_contract_and_fingerprint() {
    assert_eq!(SLOT_TLAS_MAP, 251);
    assert_eq!(TLAS_MAP_ENTRIES, 64);

    let fp_off = {
        let _g = DxrRayQueryGuard::set(false);
        super::translation_fingerprint()
    };
    let fp_on = {
        let _g = DxrRayQueryGuard::set(true);
        super::translation_fingerprint()
    };
    assert_ne!(
        fp_off, fp_on,
        "translation_fingerprint must track VICTORY_DXR_RAYQUERY"
    );
}

// ── GFX10 encoding helpers (minimal clean-room instruction builders) ────────

const S_ENDPGM: u32 = 0xBF81_0000;
const S_WAITCNT_LGKM0: u32 = 0xBF8C_C07F;
const S_WAITCNT_VM0: u32 = 0xBF8C_3F70;

const fn v_mov_imm(vdst: u8, lit: u32) -> [u32; 2] {
    [0x7E00_02FF | ((vdst as u32) << 17), lit]
}
const fn v_mov_v(vdst: u8, vsrc: u8) -> u32 {
    0x7E00_0300 | ((vdst as u32) << 17) | (vsrc as u32)
}
const fn v_mov_s(vdst: u8, ssrc: u8) -> u32 {
    0x7E00_0200 | ((vdst as u32) << 17) | (ssrc as u32)
}
const fn v_rcp_f32(vdst: u8, vsrc: u8) -> u32 {
    0x7E00_5500 | ((vdst as u32) << 17) | (vsrc as u32)
}
const fn v_add_f32(vdst: u8, vsrc0: u8, vsrc1: u8) -> u32 {
    0x0600_0100 | ((vdst as u32) << 17) | ((vsrc1 as u32) << 9) | (vsrc0 as u32)
}
const fn v_mul_f32(vdst: u8, vsrc0: u8, vsrc1: u8) -> u32 {
    0x1000_0100 | ((vdst as u32) << 17) | ((vsrc1 as u32) << 9) | (vsrc0 as u32)
}
const fn v_add_nc_u32_v(vdst: u8, vsrc0: u8, vsrc1: u8) -> u32 {
    0x4A00_0100 | ((vdst as u32) << 17) | ((vsrc1 as u32) << 9) | (vsrc0 as u32)
}
const fn v_add_nc_u32_s(vdst: u8, ssrc0: u8, vsrc1: u8) -> u32 {
    0x4A00_0000 | ((vdst as u32) << 17) | ((vsrc1 as u32) << 9) | (ssrc0 as u32)
}
const fn v_lshlrev_b32(vdst: u8, shift: u8, vsrc1: u8) -> u32 {
    0x3400_0080 | ((vdst as u32) << 17) | ((vsrc1 as u32) << 9) | (shift as u32)
}
const fn v_lshrrev_b32(vdst: u8, shift: u8, vsrc1: u8) -> u32 {
    0x2C00_0080 | ((vdst as u32) << 17) | ((vsrc1 as u32) << 9) | (shift as u32)
}
const fn v_and_b32_lit(vdst: u8, lit: u32, vsrc1: u8) -> [u32; 2] {
    [
        0x3600_00FF | ((vdst as u32) << 17) | ((vsrc1 as u32) << 9),
        lit,
    ]
}
const fn v_cndmask_v_v(vdst: u8, vsrc0: u8, vsrc1: u8, smask: u8) -> [u32; 2] {
    // VOP3 v_cndmask_b32 (opcode 0x101): dst, src0 (false), src1 (true), src2 (mask)
    [
        0xD501_0000 | (vdst as u32),
        ((smask as u32) << 18) | (((vsrc1 as u32) | 0x100) << 9) | ((vsrc0 as u32) | 0x100),
    ]
}
const fn v_cndmask_0_1(vdst: u8, smask: u8) -> [u32; 2] {
    // VOP3 v_cndmask_b32 vdst, 0 (0x80), 1 (0x81), smask
    [
        0xD501_0000 | (vdst as u32),
        ((smask as u32) << 18) | (0x81u32 << 9) | 0x80u32,
    ]
}
const fn v_cndmask_v_1(vdst: u8, vsrc0: u8, smask: u8) -> [u32; 2] {
    [
        0xD501_0000 | (vdst as u32),
        ((smask as u32) << 18) | (0x81u32 << 9) | ((vsrc0 as u32) | 0x100),
    ]
}
const fn v_cmp_le_f32_0_v(sdst: u8, vsrc1: u8) -> [u32; 2] {
    // VOP3B v_cmp_le_f32 (opcode 0x003): sdst, 0 (0x80), vsrc1
    [
        0xD403_0000 | (sdst as u32),
        (((vsrc1 as u32) | 0x100) << 9) | 0x80u32,
    ]
}
const fn v_cmp_ge_f32_v_0(sdst: u8, vsrc0: u8) -> [u32; 2] {
    // VOP3B v_cmp_ge_f32 (opcode 0x006): sdst, vsrc0, 0 (0x80)
    [
        0xD406_0000 | (sdst as u32),
        (0x80u32 << 9) | ((vsrc0 as u32) | 0x100),
    ]
}
const fn v_cmp_lt_f32_v_v(sdst: u8, vsrc0: u8, vsrc1: u8) -> [u32; 2] {
    // VOP3B v_cmp_lt_f32 (opcode 0x001): sdst, vsrc0, vsrc1
    [
        0xD401_0000 | (sdst as u32),
        (((vsrc1 as u32) | 0x100) << 9) | ((vsrc0 as u32) | 0x100),
    ]
}
const fn v_cmp_ne_u32_lit_v(sdst: u8, lit: u32, vsrc1: u8) -> [u32; 3] {
    // VOP3 v_cmp_ne_u32 (opcode 0x0c5)
    [
        0xD4C5_0000 | (sdst as u32),
        (((vsrc1 as u32) | 0x100) << 9) | 0xFFu32,
        lit,
    ]
}
const fn global_load_dword(vdst: u8, vaddr: u8, offset: u16) -> [u32; 2] {
    [
        0xDC30_8000 | ((offset & 0x0FFF) as u32),
        ((vdst as u32) << 24) | (0x7D << 16) | (vaddr as u32),
    ]
}
const fn s_and_b32(sdst: u8, s0: u8, s1: u8) -> u32 {
    0x8700_0000 | ((sdst as u32) << 16) | ((s1 as u32) << 8) | (s0 as u32)
}
const fn s_mov_b32_s(sdst: u8, ssrc: u8) -> u32 {
    0xBE80_0300 | ((sdst as u32) << 16) | (ssrc as u32)
}
const fn s_mov_b32_imm(sdst: u8, c: u8) -> u32 {
    0xBE80_0300 | ((sdst as u32) << 16) | (c as u32)
}
const fn s_add_i32_1(sdst: u8, ssrc: u8) -> u32 {
    0x8100_8100 | ((sdst as u32) << 16) | (ssrc as u32)
}
const fn s_cmp_lt_u32_imm(s0: u8, imm_inline: u8) -> u32 {
    0xBF0A_0000 | ((imm_inline as u32) << 8) | (s0 as u32)
}
const fn s_cbranch_scc1(rel_dwords: i16) -> u32 {
    0xBF85_0000 | ((rel_dwords as u16) as u32)
}
const fn s_cbranch_execz(rel_dwords: i16) -> u32 {
    0xBF88_0000 | ((rel_dwords as u16) as u32)
}
const fn s_and_saveexec_b32(sdst: u8, ssrc: u8) -> u32 {
    0xBE80_3C00 | ((sdst as u32) << 16) | (ssrc as u32)
}
const fn image_bvh_intersect_ray_nsa(vdata: u8, srsrc: u8, regs: [u8; 11]) -> [u32; 5] {
    // MIMG NSA `image_bvh_intersect_ray v[vdata..+3], [regs[0..11]], s[srsrc..+3] dmask:0xf dim:1D r128 unorm`
    // Matches the exact encoding verified in `intersect_tests.rs`.
    let w0 = 0xF198_9F07;
    let w1 = (((srsrc >> 2) as u32) << 16) | ((vdata as u32) << 8) | (regs[0] as u32);
    let w2 = (regs[1] as u32)
        | ((regs[2] as u32) << 8)
        | ((regs[3] as u32) << 16)
        | ((regs[4] as u32) << 24);
    let w3 = (regs[5] as u32)
        | ((regs[6] as u32) << 8)
        | ((regs[7] as u32) << 16)
        | ((regs[8] as u32) << 24);
    let w4 = (regs[9] as u32) | ((regs[10] as u32) << 8);
    [w0, w1, w2, w3, w4]
}

// ── Multi-instance scene shared by software BVH4 and hardware DXR 1.1 ───────

/// Geometry 0: 2 triangles (prim 0 front-facing CW at z=2, prim 1 back-facing CCW at z=3).
const GEO0_TRIS: [[[f32; 3]; 3]; 2] = [
    // Prim 0: CW when viewed along +Z => frontFace == 1
    [[0.0, 0.0, 2.0], [0.0, 2.0, 2.0], [2.0, 0.0, 2.0]],
    // Prim 1: CCW when viewed along +Z => frontFace == 0
    [[2.5, 0.0, 3.0], [4.5, 0.0, 3.0], [2.5, 2.0, 3.0]],
];
/// Geometry 1: 1 triangle (prim 0 front-facing CW at z=4).
const GEO1_TRIS: [[[f32; 3]; 3]; 1] = [[[-2.5, 0.0, 4.0], [-2.5, 2.0, 4.0], [-0.5, 0.0, 4.0]]];

const INST0_ID: u32 = 0x123;
const INST1_ID: u32 = 0x456;
/// Instance 1 object-to-world translation `(0, +4, +1.5)` (world-to-object = `(0, -4, -1.5)`).
const INST1_OBJ_TO_WORLD: [f32; 3] = [0.0, 4.0, 1.5];

/// 8 test rays `[ox, oy, oz, tMax, dx, dy, dz, 0]` exercising both instances,
/// both geometries, both primitives, front/back faces, misses, tMax clipping,
/// and non-zero ray origins.
fn test_rays() -> [[f32; 8]; 8] {
    [
        // Ray 0: hits Inst 0 (0x123), Geo 0, Prim 0 (CW frontFace=1, z=2.0, bary=(0.35, 0.25))
        [0.5, 0.7, 0.0, 20.0, 0.0, 0.0, 1.0, 0.0],
        // Ray 1: hits Inst 0 (0x123), Geo 0, Prim 1 (CCW frontFace=0, z=3.0)
        [3.0, 0.6, 0.0, 20.0, 0.0, 0.0, 1.0, 0.0],
        // Ray 2: hits Inst 0 (0x123), Geo 1, Prim 0 (CW frontFace=1, z=4.0)
        [-2.0, 0.5, 0.0, 20.0, 0.0, 0.0, 1.0, 0.0],
        // Ray 3: hits Inst 1 (0x456, +4y +1.5z), Geo 0, Prim 0 (CW frontFace=1, z=3.5)
        [0.4, 4.6, 0.0, 20.0, 0.0, 0.0, 1.0, 0.0],
        // Ray 4: hits Inst 1 (0x456, +4y +1.5z), Geo 1, Prim 0 (CW frontFace=1, z=5.5)
        [-1.9, 4.4, 0.0, 20.0, 0.0, 0.0, 1.0, 0.0],
        // Ray 5: misses everything
        [10.0, 10.0, 0.0, 20.0, 0.0, 0.0, 1.0, 0.0],
        // Ray 6: aimed at Inst 0 Geo 0 Prim 0 (z=2.0) but clipped by tMax=1.5 => miss
        [0.5, 0.7, 0.0, 1.5, 0.0, 0.0, 1.0, 0.0],
        // Ray 7: non-zero origin.z=0.5 hitting Inst 0 Geo 0 Prim 0 => t=1.5
        [0.5, 0.7, 0.5, 20.0, 0.0, 0.0, 1.0, 0.0],
    ]
}

fn tri_aabb(tri: &[[f32; 3]; 3], offset: [f32; 3]) -> [f32; 6] {
    let mut mn = [f32::INFINITY; 3];
    let mut mx = [f32::NEG_INFINITY; 3];
    for p in tri {
        for a in 0..3 {
            let c = p[a] + offset[a];
            mn[a] = mn[a].min(c);
            mx[a] = mx[a].max(c);
        }
    }
    [mn[0] - 0.01, mn[1] - 0.01, mn[2] - 0.01, mx[0] + 0.01, mx[1] + 0.01, mx[2] + 0.01]
}

fn union_aabb(a: [f32; 6], b: [f32; 6]) -> [f32; 6] {
    [
        a[0].min(b[0]),
        a[1].min(b[1]),
        a[2].min(b[2]),
        a[3].max(b[3]),
        a[4].max(b[4]),
        a[5].max(b[5]),
    ]
}

/// Builds the RDNA2 BVH4 binary buffer placed at `tlas_va` (256-byte aligned).
/// Layout (64-byte node units, pointer = `(unit << 3) | kind`):
/// - Unit 0..2   (bytes 0x000..0x07F): TLAS Box32 (kind 5, ptr = 5) with 2 children:
///   - child 0 = Instance 0 (`unit 4`, ptr `(4 << 3) | 6 = 0x26`)
///   - child 1 = Instance 1 (`unit 6`, ptr `(6 << 3) | 6 = 0x36`)
/// - Unit 4..6   (bytes 0x100..0x17F): Instance 0 node (`InstanceID = 0x123`, identity)
/// - Unit 6..8   (bytes 0x180..0x1FF): Instance 1 node (`InstanceID = 0x456`, `+4y +1.5z`)
/// - Unit 8..10  (bytes 0x200..0x27F): BLAS Box32 (at `blas_va = tlas_va + 0x200`, root ptr = 5)
///   with 3 children (relative to `blas_va`):
///   - child 0 = TriNode 0 (Geo 0, Prim 0, unit 2 relative to `blas_va` -> ptr `(2 << 3) | 0 = 0x10`)
///   - child 1 = TriNode 1 (Geo 0, Prim 1, unit 3 relative to `blas_va` -> ptr `(3 << 3) | 0 = 0x18`)
///   - child 2 = TriNode 2 (Geo 1, Prim 0, unit 4 relative to `blas_va` -> ptr `(4 << 3) | 0 = 0x20`)
/// - Unit 10     (bytes 0x280..0x2BF): TriNode 0 (Geo 0, Prim 0)
/// - Unit 11     (bytes 0x2C0..0x2FF): TriNode 1 (Geo 0, Prim 1)
/// - Unit 12     (bytes 0x300..0x33F): TriNode 2 (Geo 1, Prim 0)
fn build_rdna2_bvh4_buffer(tlas_va: u64) -> Vec<u32> {
    let mut buf = vec![0u32; 16 * 16]; // 16 units * 64 B = 1024 B = 256 dwords
    let blas_va = tlas_va + 0x200;
    let blas_va_shr8 = (blas_va >> 8) as u32;

    let write_box32 = |buf: &mut [u32], unit: usize, children: [u32; 4], boxes: [[f32; 6]; 4]| {
        let base = unit * 16;
        buf[base..base + 4].copy_from_slice(&children);
        for (i, bx) in boxes.iter().enumerate() {
            for (k, &f) in bx.iter().enumerate() {
                buf[base + 4 + i * 6 + k] = f.to_bits();
            }
        }
    };

    let write_instance = |buf: &mut [u32], unit: usize, inst_id: u32, w2o_trans: [f32; 3]| {
        let base = unit * 16;
        buf[base + 12] = (0xFFu32 << 24) | (inst_id & 0x00FF_FFFF); // +0x30: mask | InstanceID
        buf[base + 13] = 0; // +0x34: flags | InstanceContribution
        buf[base + 14] = 5; // +0x38: BLAS root ptr (unit 0 relative to blas_va, kind 5)
        buf[base + 15] = blas_va_shr8; // +0x3C: blas_va >> 8
        buf[base + 16] = w2o_trans[0].to_bits(); // +0x40: tx
        buf[base + 17] = w2o_trans[1].to_bits(); // +0x44: ty
        buf[base + 18] = w2o_trans[2].to_bits(); // +0x48: tz
    };

    let write_tri = |buf: &mut [u32], unit: usize, tri: &[[f32; 3]; 3], geo: u32, prim: u32| {
        let base = unit * 16;
        for (vi, v) in tri.iter().enumerate() {
            buf[base + vi * 3] = v[0].to_bits();
            buf[base + vi * 3 + 1] = v[1].to_bits();
            buf[base + vi * 3 + 2] = v[2].to_bits();
        }
        buf[base + 12] = 1 | (geo << 8); // +0x30: is_opaque=1 | (GeometryIndex << 8)
        buf[base + 13] = prim; // +0x34: PrimitiveIndex
        // +0x3C: triangle 0 vertex mapping: I=vertex 1 (1), J=vertex 2 (2<<2=8) => 0x09.
        buf[base + 15] = 0x09;
    };

    let b_t0 = tri_aabb(&GEO0_TRIS[0], [0.0; 3]);
    let b_t1 = tri_aabb(&GEO0_TRIS[1], [0.0; 3]);
    let b_t2 = tri_aabb(&GEO1_TRIS[0], [0.0; 3]);
    let blas_box0 = union_aabb(union_aabb(b_t0, b_t1), b_t2);
    let blas_box1 = [
        blas_box0[0] + INST1_OBJ_TO_WORLD[0],
        blas_box0[1] + INST1_OBJ_TO_WORLD[1],
        blas_box0[2] + INST1_OBJ_TO_WORLD[2],
        blas_box0[3] + INST1_OBJ_TO_WORLD[0],
        blas_box0[4] + INST1_OBJ_TO_WORLD[1],
        blas_box0[5] + INST1_OBJ_TO_WORLD[2],
    ];
    let empty = [0.0f32; 6];

    // TLAS Box32 at unit 0 (ptr = 5)
    write_box32(
        &mut buf,
        0,
        [(4 << 3) | 6, (6 << 3) | 6, bvh::NONE, bvh::NONE],
        [blas_box0, blas_box1, empty, empty],
    );
    // Instance nodes at units 4 and 6
    write_instance(&mut buf, 4, INST0_ID, [0.0, 0.0, 0.0]);
    write_instance(
        &mut buf,
        6,
        INST1_ID,
        [
            -INST1_OBJ_TO_WORLD[0],
            -INST1_OBJ_TO_WORLD[1],
            -INST1_OBJ_TO_WORLD[2],
        ],
    );
    // BLAS Box32 at unit 8 (relative unit 0)
    write_box32(
        &mut buf,
        8,
        [(2 << 3), (3 << 3), (4 << 3), bvh::NONE],
        [b_t0, b_t1, b_t2, empty],
    );
    // TriNodes at units 10, 11, 12 (relative units 2, 3, 4)
    write_tri(&mut buf, 10, &GEO0_TRIS[0], 0, 0);
    write_tri(&mut buf, 11, &GEO0_TRIS[1], 0, 1);
    write_tri(&mut buf, 12, &GEO1_TRIS[0], 1, 0);
    buf
}

/// Synthetic Wave32 GPURT Template A compute shader traversing our 2-instance
/// BVH4 scene and writing `[hit_flag, t, bary_u, bary_v, inst_id, geo_idx,
/// prim_idx, front_face]` per thread.
///
/// Under `VICTORY_DXR_RAYQUERY=0`: executes software BVH4 (`emit/intersect.rs`).
/// Under `VICTORY_DXR_RAYQUERY=1`: lifts the entire outer+inner loop to SM 6.6
/// `RayQuery` (`emit/rayquery.rs`).
fn build_template_a_diff_shader() -> NativeProgram {
    let mut w: Vec<u32> = Vec::new();
    // Helper to record branch patch sites.
    let push_cbranch_execz = |w: &mut Vec<u32>| -> usize {
        let idx = w.len();
        w.push(s_cbranch_execz(0));
        idx
    };
    let patch_branch = |w: &mut [u32], idx: usize, target_idx: usize| {
        let rel = (target_idx as isize - idx as isize - 1) as i16;
        w[idx] = (w[idx] & 0xFFFF_0000) | u32::from(rel as u16);
    };

    // Entry: load descriptors from Table s[4:5]
    //   offset 0x00: ray input SRV  -> s[8:11]
    //   offset 0x20: result UAV     -> s[12:15]
    //   offset 0x40: TLAS buffer SRV -> s[16:19] (read only as metadata -> USER_VIEW_ADDRESS)
    w.extend([
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0202,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0x00
        0xF408_0302,
        0xFA00_0020, // s_load_dwordx4 s[12:15], s[4:5], 0x20
        0xF408_0402,
        0xFA00_0040, // s_load_dwordx4 s[16:19], s[4:5], 0x40
        S_WAITCNT_LGKM0,
        0x9094_8810, // s_lshr_b64 s[20:21], s[16:17], 8  (tlas_va >> 8)
        // Load ray: v[1:4] = [ox, oy, oz, tMax], v[5:8] = [dx, dy, dz, 0]
        v_lshlrev_b32(40, 5, 0), // v40 = v0 * 32
        0xE038_1000,
        0x8002_0128, // buffer_load_dwordx4 v[1:4], v40, s[8:11], 0 offen
        0xE038_1010,
        0x8002_0528, // buffer_load_dwordx4 v[5:8], v40, s[8:11], 0 offen offset:16
        S_WAITCNT_VM0,
        v_rcp_f32(9, 5),  // v9  = 1 / dx
        v_rcp_f32(10, 6), // v10 = 1 / dy
        v_rcp_f32(11, 7), // v11 = 1 / dz
    ]);
    // Initialize hit outputs (all 0 on miss) & outer instance cursor s22 = 0.
    for vreg in 12..=18u8 {
        w.extend(v_mov_imm(vreg, 0));
    }
    w.push(s_mov_b32_imm(22, 0x80)); // s22 = 0 (outer instance index: 0, 1)

    // ── Outer TLAS loop header ──
    let outer_hdr = w.len();
    // Test TLAS root Box32 (ptr = 5) with world-space ray (v1..v3, v4, v5..v7, v9..v11)
    w.extend(v_mov_imm(19, 5));
    w.extend([
        s_mov_b32_s(24, 20), // s24 = tlas_va_shr8_lo
        0xBE99_0B81,         // s_brev_b32 s25, 1 (= 0x8000_0000)
        0xBE9A_03C1,         // s_mov_b32 s26, -1
        0xBE9B_03FF,
        0x8900_03FF, // s_mov_b32 s27, 0x890003ff
    ]);
    w.extend(image_bvh_intersect_ray_nsa(
        20,
        24,
        [19, 4, 1, 2, 3, 5, 6, 7, 9, 10, 11],
    ));
    w.push(S_WAITCNT_VM0);

    // Pick child s22 (0 -> v20, 1 -> v21) into v24.
    w.extend([
        0xBF06_8016, // s_cmp_eq_u32 s22, 0
        0x8517_807E, // s_cselect_b32 s23, exec_lo, 0
    ]);
    w.extend(v_cndmask_v_v(24, 21, 20, 23));

    // Skip if v24 == 0xFFFF_FFFF (missed this TLAS instance box).
    w.extend(v_cmp_ne_u32_lit_v(23, bvh::NONE, 24));
    w.push(s_and_saveexec_b32(36, 23));
    let br_skip_inst = push_cbranch_execz(&mut w);

    // Compute InstanceNode 64-bit address: tlas_va + ((v24 >> 3) << 6)
    // Since tlas_va = (s16, s17) and our buffer is < 4 KiB at 256 B alignment:
    w.extend([
        v_lshrrev_b32(25, 3, 24),
        v_lshlrev_b32(25, 6, 25),
        v_add_nc_u32_s(25, 16, 25), // v25 = addr_lo
        v_mov_s(26, 17),            // v26 = addr_hi
    ]);
    w.extend(global_load_dword(27, 25, 0x30)); // v27 = InstanceNode+0x30 (mask | InstanceID)
    w.extend(global_load_dword(28, 25, 0x3C)); // v28 = InstanceNode+0x3C (blas_va >> 8)
    w.extend(global_load_dword(29, 25, 0x40)); // v29 = tx
    w.extend(global_load_dword(30, 25, 0x44)); // v30 = ty
    w.extend(global_load_dword(31, 25, 0x48)); // v31 = tz
    w.push(S_WAITCNT_VM0);

    w.extend(v_and_b32_lit(27, 0x00FF_FFFF, 27)); // v27 = InstanceID
    w.push(0x7E38_051C); // v_readfirstlane_b32 s28, v28 (blas_va >> 8)
    // Transform ray origin to object space: v41..v43 = orig + (tx, ty, tz)
    w.extend([
        v_add_f32(41, 1, 29),
        v_add_f32(42, 2, 30),
        v_add_f32(43, 3, 31),
        // Build BLAS descriptor s[28:31]
        0xBE9D_0B81, // s_brev_b32 s29, 1
        0xBE9E_03C1, // s_mov_b32 s30, -1
        0xBE9F_03FF,
        0x8900_03FF, // s_mov_b32 s31, 0x890003ff
    ]);
    // Intersect BLAS Box32 (ptr = 5) -> returns up to 3 triangle node ptrs in v[44:47]
    w.extend(v_mov_imm(48, 5));
    w.extend(image_bvh_intersect_ray_nsa(
        44,
        28,
        [48, 4, 41, 42, 43, 5, 6, 7, 9, 10, 11],
    ));
    w.push(S_WAITCNT_VM0);

    // ── Inner BLAS triangle loop (s33 = 0, 1, 2) ──
    w.push(s_mov_b32_imm(33, 0x80)); // s33 = 0
    let inner_hdr = w.len();
    // Select tri_ptr into v49: s33==0 -> v44, s33==1 -> v45, s33==2 -> v46
    w.extend([
        0xBF06_8021, // s_cmp_eq_u32 s33, 0
        0x8522_807E, // s_cselect_b32 s34, exec_lo, 0
    ]);
    w.extend(v_cndmask_v_v(49, 46, 44, 34));
    w.extend([
        0xBF06_8121, // s_cmp_eq_u32 s33, 1
        0x8522_807E, // s_cselect_b32 s34, exec_lo, 0
    ]);
    w.extend(v_cndmask_v_v(49, 49, 45, 34));

    w.extend(v_cmp_ne_u32_lit_v(34, bvh::NONE, 49));
    w.push(s_and_saveexec_b32(37, 34));
    let br_skip_tri = push_cbranch_execz(&mut w);

    // Intersect TriNode v49 -> v[32:35] = [t_num, det, bary_u_num, bary_v_num]
    w.extend(image_bvh_intersect_ray_nsa(
        32,
        28,
        [49, 4, 41, 42, 43, 5, 6, 7, 9, 10, 11],
    ));
    w.push(S_WAITCNT_VM0);

    // Compute inv_det, t, u, v, front_face, hit = (0 <= t) & (t < v4)
    w.extend([
        v_rcp_f32(36, 33),     // v36 = 1 / det
        v_mul_f32(37, 32, 36), // v37 = t
        v_mul_f32(38, 34, 36), // v38 = bary_u
        v_mul_f32(39, 35, 36), // v39 = bary_v
    ]);
    w.extend(v_cmp_ge_f32_v_0(34, 33)); // s34 = (det >= 0.0) front_face
    w.extend(v_cndmask_0_1(50, 34)); // v50 = front_face ? 1 : 0
    w.extend(v_cmp_le_f32_0_v(34, 37)); // s34 = (0.0 <= t)
    w.extend(v_cmp_lt_f32_v_v(35, 37, 4)); // s35 = (t < tMax)
    w.push(s_and_b32(35, 34, 35)); // s35 = hit commit mask

    // Compute TriNode 64-bit address: blas_va + ((v49 >> 3) << 6)
    // blas_va_lo = (s28 << 8), blas_va_hi = s17 (same 4 GiB page)
    w.extend([
        0x8F20_881C, // s_lshl_b32 s32, s28, 8
        v_lshrrev_b32(51, 3, 49),
        v_lshlrev_b32(51, 6, 51),
        v_add_nc_u32_s(51, 32, 51),
        v_mov_s(52, 17),
    ]);
    w.extend(global_load_dword(53, 51, 0x30)); // v53 = TriNode+0x30 (opaque | geo << 8)
    w.extend(global_load_dword(54, 51, 0x34)); // v54 = TriNode+0x34 (prim_idx)
    w.push(S_WAITCNT_VM0);
    w.push(v_lshrrev_b32(53, 8, 53)); // v53 = GeometryIndex

    // Commit all 8 hit outputs under commit mask s35
    w.extend(v_cndmask_v_v(4, 4, 37, 35));   // tMax = t
    w.extend(v_cndmask_v_1(12, 12, 35));     // hit_flag = 1
    w.extend(v_cndmask_v_v(13, 13, 38, 35)); // bary_u
    w.extend(v_cndmask_v_v(14, 14, 39, 35)); // bary_v
    w.extend(v_cndmask_v_v(15, 15, 27, 35)); // InstanceID
    w.extend(v_cndmask_v_v(16, 16, 53, 35)); // GeometryIndex
    w.extend(v_cndmask_v_v(17, 17, 54, 35)); // PrimitiveIndex
    w.extend(v_cndmask_v_v(18, 18, 50, 35)); // FrontFace

    let lbl_skip_tri = w.len();
    patch_branch(&mut w, br_skip_tri, lbl_skip_tri);
    w.push(0xBEFE_0325); // s_mov_b32 exec_lo, s37

    w.push(s_add_i32_1(33, 33)); // s33 += 1
    w.push(s_cmp_lt_u32_imm(33, 0x83)); // s33 < 3
    let rel_inner = (inner_hdr as isize - w.len() as isize - 1) as i16;
    w.push(s_cbranch_scc1(rel_inner));

    let lbl_skip_inst = w.len();
    patch_branch(&mut w, br_skip_inst, lbl_skip_inst);
    w.push(0xBEFE_0324); // s_mov_b32 exec_lo, s36

    w.push(s_add_i32_1(22, 22)); // s22 += 1
    w.push(s_cmp_lt_u32_imm(22, 0x82)); // s22 < 2
    let rel_outer = (outer_hdr as isize - w.len() as isize - 1) as i16;
    w.push(s_cbranch_scc1(rel_outer));

    // ── Single exit block: store 8 dwords per ray ──
    // Pack [hit_flag(v12), t(v4), bary_u(v13), bary_v(v14)] into v[56:59]
    w.extend([
        v_mov_v(56, 12),
        v_mov_v(57, 4),
        v_mov_v(58, 13),
        v_mov_v(59, 14),
        0xE078_1000,
        0x8003_3828, // buffer_store_dwordx4 v[56:59], v40, s[12:15], 0 offen
        0xE078_1010,
        0x8003_0F28, // buffer_store_dwordx4 v[15:18], v40, s[12:15], 0 offen offset:16
        S_ENDPGM,
    ]);

    NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        code: w.iter().flat_map(|x| x.to_le_bytes()).collect(),
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 4,
            slot: 0,
        }],
        used: vec![
            UsedResource {
                kind: ResourceType::Srv,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 4,
                dwords: 4,
                user_data: 4,
                slot: 0,
            },
            UsedResource {
                kind: ResourceType::Uav,
                root: RootType::Table,
                table_offset_dwords: 8,
                lower_bound: 1,
                stride: 4,
                dwords: 4,
                user_data: 4,
                slot: 0,
            },
            UsedResource {
                kind: ResourceType::Srv,
                root: RootType::Table,
                table_offset_dwords: 16,
                lower_bound: 2,
                stride: 4,
                dwords: 4,
                user_data: 4,
                slot: 0,
            },
        ],
        io: StageIo::Cs(ComputeIo {
            threads: [8, 1, 1],
            rsrc2: 0,
        }),
        vgprs: 64,
    }
}

/// Executes a lifted SM 6.6 `RayQuery` shader against a real D3D12 DXR 1.1
/// BLAS + TLAS built on the RTX 3090 (`ID3D12Device5` / `ID3D12GraphicsCommandList4`)
/// and bound via `SLOT_TLAS_MAP` (`t251`).
fn dispatch_live_dxr11_rayquery(
    host: &Host,
    shader: &HostShader,
    tlas_va: u64,
    rays: &[u32],
) -> Result<Vec<u32>, String> {
    unsafe {
        let dev5: ID3D12Device5 = host
            .device
            .cast()
            .map_err(|e| format!("ID3D12Device5: {e}"))?;
        let mut opts5 = D3D12_FEATURE_DATA_D3D12_OPTIONS5::default();
        dev5.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS5,
            (&raw mut opts5).cast(),
            size_of_val(&opts5) as u32,
        )
        .map_err(|e| format!("OPTIONS5: {e}"))?;
        if opts5.RaytracingTier.0 < D3D12_RAYTRACING_TIER_1_1.0 {
            return Err(format!("device RaytracingTier={:?} < 1.1", opts5.RaytracingTier));
        }

        let make_buf = |size: u64,
                        heap: D3D12_HEAP_TYPE,
                        flags: D3D12_RESOURCE_FLAGS,
                        state: D3D12_RESOURCE_STATES|
         -> Result<ID3D12Resource, String> {
            let hp = D3D12_HEAP_PROPERTIES {
                Type: heap,
                ..Default::default()
            };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: size.max(256),
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: DXGI_FORMAT_UNKNOWN,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: flags,
                ..Default::default()
            };
            let mut r: Option<ID3D12Resource> = None;
            dev5.CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut r)
                .map_err(|e| format!("CreateCommittedResource: {e}"))?;
            Ok(r.unwrap())
        };

        let upload = |bytes: &[u8]| -> Result<ID3D12Resource, String> {
            let r = make_buf(
                bytes.len() as u64,
                D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            )?;
            let mut ptr = std::ptr::null_mut();
            r.Map(0, None, Some(&mut ptr))
                .map_err(|e| format!("Map: {e}"))?;
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.cast::<u8>(), bytes.len());
            r.Unmap(0, None);
            Ok(r)
        };

        // 1. Upload vertex buffers for Geometry 0 (2 tris = 6 verts) and Geometry 1 (1 tri = 3 verts).
        let geo0_verts: Vec<f32> = GEO0_TRIS.iter().flatten().flatten().copied().collect();
        let geo1_verts: Vec<f32> = GEO1_TRIS.iter().flatten().flatten().copied().collect();
        let vb0 = upload(std::slice::from_raw_parts(
            geo0_verts.as_ptr().cast::<u8>(),
            geo0_verts.len() * 4,
        ))?;
        let vb1 = upload(std::slice::from_raw_parts(
            geo1_verts.as_ptr().cast::<u8>(),
            geo1_verts.len() * 4,
        ))?;

        let geom_descs = [
            D3D12_RAYTRACING_GEOMETRY_DESC {
                Type: D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES,
                Flags: D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE,
                Anonymous: D3D12_RAYTRACING_GEOMETRY_DESC_0 {
                    Triangles: D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC {
                        Transform3x4: 0,
                        IndexFormat: DXGI_FORMAT_UNKNOWN,
                        VertexFormat: DXGI_FORMAT_R32G32B32_FLOAT,
                        IndexCount: 0,
                        VertexCount: 6,
                        IndexBuffer: 0,
                        VertexBuffer: D3D12_GPU_VIRTUAL_ADDRESS_AND_STRIDE {
                            StartAddress: vb0.GetGPUVirtualAddress(),
                            StrideInBytes: 12,
                        },
                    },
                },
            },
            D3D12_RAYTRACING_GEOMETRY_DESC {
                Type: D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES,
                Flags: D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE,
                Anonymous: D3D12_RAYTRACING_GEOMETRY_DESC_0 {
                    Triangles: D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC {
                        Transform3x4: 0,
                        IndexFormat: DXGI_FORMAT_UNKNOWN,
                        VertexFormat: DXGI_FORMAT_R32G32B32_FLOAT,
                        IndexCount: 0,
                        VertexCount: 3,
                        IndexBuffer: 0,
                        VertexBuffer: D3D12_GPU_VIRTUAL_ADDRESS_AND_STRIDE {
                            StartAddress: vb1.GetGPUVirtualAddress(),
                            StrideInBytes: 12,
                        },
                    },
                },
            },
        ];

        let blas_inputs = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
            Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL,
            Flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE,
            NumDescs: geom_descs.len() as u32,
            DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
            Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
                pGeometryDescs: geom_descs.as_ptr(),
            },
        };
        let mut blas_info = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO::default();
        dev5.GetRaytracingAccelerationStructurePrebuildInfo(&blas_inputs, &mut blas_info);

        let blas_buf = make_buf(
            blas_info.ResultDataMaxSizeInBytes,
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
        )?;
        let blas_scratch = make_buf(
            blas_info.ScratchDataSizeInBytes,
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COMMON,
        )?;

        // 2. Upload 2 D3D12_RAYTRACING_INSTANCE_DESCs pointing to BLAS.
        let inst_descs = [
            D3D12_RAYTRACING_INSTANCE_DESC {
                Transform: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                _bitfield1: (0xFFu32 << 24) | (INST0_ID & 0x00FF_FFFF),
                _bitfield2: 0,
                AccelerationStructure: blas_buf.GetGPUVirtualAddress(),
            },
            D3D12_RAYTRACING_INSTANCE_DESC {
                Transform: [
                    1.0,
                    0.0,
                    0.0,
                    INST1_OBJ_TO_WORLD[0],
                    0.0,
                    1.0,
                    0.0,
                    INST1_OBJ_TO_WORLD[1],
                    0.0,
                    0.0,
                    1.0,
                    INST1_OBJ_TO_WORLD[2],
                ],
                _bitfield1: (0xFFu32 << 24) | (INST1_ID & 0x00FF_FFFF),
                _bitfield2: 0,
                AccelerationStructure: blas_buf.GetGPUVirtualAddress(),
            },
        ];
        let inst_buf = upload(std::slice::from_raw_parts(
            inst_descs.as_ptr().cast::<u8>(),
            size_of_val(&inst_descs),
        ))?;

        let tlas_inputs = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
            Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL,
            Flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE,
            NumDescs: inst_descs.len() as u32,
            DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
            Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
                InstanceDescs: inst_buf.GetGPUVirtualAddress(),
            },
        };
        let mut tlas_info = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO::default();
        dev5.GetRaytracingAccelerationStructurePrebuildInfo(&tlas_inputs, &mut tlas_info);

        let tlas_buf = make_buf(
            tlas_info.ResultDataMaxSizeInBytes,
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
        )?;
        let tlas_scratch = make_buf(
            tlas_info.ScratchDataSizeInBytes,
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COMMON,
        )?;

        // 3. Create ray input SRV, result UAV, readback buffer, and SLOT_TLAS_MAP root SRV.
        let ray_buf = upload(std::slice::from_raw_parts(
            rays.as_ptr().cast::<u8>(),
            rays.len() * 4,
        ))?;
        let out_dwords = 8 * 8usize;
        let out_bytes = (out_dwords * 4) as u64;
        let out_buf = make_buf(
            out_bytes,
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        )?;
        let readback = make_buf(
            out_bytes,
            D3D12_HEAP_TYPE_READBACK,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_COPY_DEST,
        )?;

        const HOST_TLAS_SRV_SLOT: u32 = 3;
        let mut tlas_map = vec![0u32; (TLAS_MAP_ENTRIES * 4) as usize];
        let va_shr8 = tlas_va >> 8;
        tlas_map[0] = va_shr8 as u32;
        tlas_map[1] = (va_shr8 >> 32) as u32;
        tlas_map[2] = HOST_TLAS_SRV_SLOT;
        tlas_map[3] = 1; // valid
        let tlas_map_buf = upload(std::slice::from_raw_parts(
            tlas_map.as_ptr().cast::<u8>(),
            tlas_map.len() * 4,
        ))?;

        // 4. Shader-visible descriptor heap:
        //    slot 0: ray input raw SRV
        //    slot 1: output raw UAV
        //    slot 2: dummy/unused
        //    slot 3: host DXR 1.1 TLAS SRV
        let heap: ID3D12DescriptorHeap = dev5
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 8,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            })
            .map_err(|e| format!("CreateDescriptorHeap: {e}"))?;
        let stride = dev5.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
            as usize;
        let base_cpu = heap.GetCPUDescriptorHandleForHeapStart().ptr;
        let at = |slot: usize| D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: base_cpu + slot * stride,
        };

        dev5.CreateShaderResourceView(
            &ray_buf,
            Some(&D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT_R32_TYPELESS,
                ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_SRV {
                        FirstElement: 0,
                        NumElements: rays.len() as u32,
                        StructureByteStride: 0,
                        Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                    },
                },
            }),
            at(0),
        );
        dev5.CreateUnorderedAccessView(
            &out_buf,
            None,
            Some(&D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: DXGI_FORMAT_R32_TYPELESS,
                ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_UAV {
                        FirstElement: 0,
                        NumElements: out_dwords as u32,
                        StructureByteStride: 0,
                        CounterOffsetInBytes: 0,
                        Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                    },
                },
            }),
            at(1),
        );
        dev5.CreateShaderResourceView(
            None,
            Some(&D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT_UNKNOWN,
                ViewDimension: D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    RaytracingAccelerationStructure:
                        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_SRV {
                            Location: tlas_buf.GetGPUVirtualAddress(),
                        },
                },
            }),
            at(HOST_TLAS_SRV_SLOT as usize),
        );

        // 5. Build PSO, record BLAS -> UAV barrier -> TLAS -> UAV barrier -> Dispatch -> Readback.
        let (root, pso) = host.compute_pipeline(shader)?;
        let queue: ID3D12CommandQueue = dev5
            .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                ..Default::default()
            })
            .map_err(|e| format!("CreateCommandQueue: {e}"))?;
        let alloc: ID3D12CommandAllocator = dev5
            .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
            .map_err(|e| format!("CreateCommandAllocator: {e}"))?;
        let list: ID3D12GraphicsCommandList4 = dev5
            .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)
            .map_err(|e| format!("CreateCommandList: {e}"))?;

        let uav_barrier = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                    pResource: std::mem::ManuallyDrop::new(None),
                }),
            },
        };

        list.BuildRaytracingAccelerationStructure(
            &D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
                DestAccelerationStructureData: blas_buf.GetGPUVirtualAddress(),
                Inputs: blas_inputs,
                SourceAccelerationStructureData: 0,
                ScratchAccelerationStructureData: blas_scratch.GetGPUVirtualAddress(),
            },
            None,
        );
        list.ResourceBarrier(&[uav_barrier.clone()]);
        list.BuildRaytracingAccelerationStructure(
            &D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
                DestAccelerationStructureData: tlas_buf.GetGPUVirtualAddress(),
                Inputs: tlas_inputs,
                SourceAccelerationStructureData: 0,
                ScratchAccelerationStructureData: tlas_scratch.GetGPUVirtualAddress(),
            },
            None,
        );
        list.ResourceBarrier(&[uav_barrier]);

        let heaps = [Some(heap.clone())];
        list.SetDescriptorHeaps(&heaps);
        list.SetComputeRootSignature(&root);
        list.SetPipelineState(&pso);

        // Populate user-data root constants (param 0).
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            ud[u.index as usize] = match u.root {
                RootType::Table => 0,
                RootType::Other(super::USER_VIEW_ADDRESS) => tlas_va as u32,
                RootType::Other(x) if x == super::USER_VIEW_ADDRESS + 1 => (tlas_va >> 32) as u32,
                _ => 0,
            };
        }
        list.SetComputeRoot32BitConstants(0, ud.len() as u32, ud.as_ptr().cast(), 0);
        // Root param 1: SLOT_TLAS_MAP root SRV (t251).
        list.SetComputeRootShaderResourceView(1, tlas_map_buf.GetGPUVirtualAddress());

        list.Dispatch(1, 1, 1);

        let to_copy = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                    pResource: std::mem::ManuallyDrop::new(Some(out_buf.clone())),
                    Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                    StateBefore: D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    StateAfter: D3D12_RESOURCE_STATE_COPY_SOURCE,
                }),
            },
        };
        list.ResourceBarrier(&[to_copy]);
        list.CopyBufferRegion(&readback, 0, &out_buf, 0, out_bytes);
        list.Close().map_err(|e| format!("Close: {e}"))?;

        let cmd_list: ID3D12CommandList = list.cast().unwrap();
        queue.ExecuteCommandLists(&[Some(cmd_list)]);
        let fence: ID3D12Fence = dev5
            .CreateFence(0, D3D12_FENCE_FLAG_NONE)
            .map_err(|e| format!("CreateFence: {e}"))?;
        queue.Signal(&fence, 1).map_err(|e| format!("Signal: {e}"))?;
        while fence.GetCompletedValue() < 1 {
            std::thread::yield_now();
        }

        let mut ptr = std::ptr::null_mut();
        readback
            .Map(0, None, Some(&mut ptr))
            .map_err(|e| format!("Map readback: {e}"))?;
        let mut out = vec![0u32; out_dwords];
        std::ptr::copy_nonoverlapping(ptr.cast::<u32>(), out.as_mut_ptr(), out_dwords);
        readback.Unmap(0, None);
        Ok(out)
    }
}

#[test]
fn live_rtx3090_differential_software_bvh4_vs_hardware_rayquery_multi_instance() {
    const APERTURE_HI: u32 = 0x20;
    const TLAS_VA: u64 = (APERTURE_HI as u64) << 32; // Window 0 byte 0 in sparse-UMA

    let native = build_template_a_diff_shader();
    let rays: Vec<u32> = test_rays()
        .iter()
        .flatten()
        .map(|f| f.to_bits())
        .collect();
    let bvh4_buf = build_rdna2_bvh4_buffer(TLAS_VA);

    // 1. Lower with VICTORY_DXR_RAYQUERY=0 (software BVH4 over SPARSE_UMA window 0)
    let sw_shader = {
        let _rq = DxrRayQueryGuard::set(false);
        let _sp = SparseUmaGuard::set(true);
        lower_and_validate_cs(&native)
    };
    assert!(
        !sw_shader.root_srvs.contains(&SLOT_TLAS_MAP),
        "software BVH4 shader must not bind SLOT_TLAS_MAP"
    );

    // 2. Lower the EXACT SAME shader with VICTORY_DXR_RAYQUERY=1 (hardware SM 6.6 RayQuery)
    let rq_shader = {
        let _rq = DxrRayQueryGuard::set(true);
        let _sp = SparseUmaGuard::set(true);
        lower_and_validate_cs(&native)
    };
    assert!(
        rq_shader.root_srvs.contains(&SLOT_TLAS_MAP),
        "lifted RayQuery shader must bind SLOT_TLAS_MAP (t251), got {:?}",
        rq_shader.root_srvs
    );
    assert!(
        !rq_shader.root_srvs.contains(&super::SLOT_MEMORY),
        "lifted RayQuery shader should not require SLOT_MEMORY"
    );

    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware GPU available; skipped live RT Core differential check");
        return;
    };

    // 3. Execute software BVH4 path live on RTX 3090 (using sparse-UMA window 0 = bvh4_buf).
    // Note: `Host::dispatch` View::Buffer is SRV; copy `bvh4_buf` into window 0 via a RawUav?
    // Even simpler: use `TEST_MEMORY`? No — `global_load_dword` reads `RAW_UAV_GC` at SPARSE_BASE+1!
    // So initialize SPARSE_BASE+1 by passing `bvh4_buf` via `View::Buffer` for a quick test?
    // Wait: `View::RawUav` starts zeroed in `Host::dispatch`, so instead use a tiny helper that
    // uploads `bvh4_buf` directly into a real D3D12 RWByteAddressBuffer UAV at slot SPARSE_BASE+1!
    let sw_out = dispatch_live_software_bvh4(&host, &sw_shader, TLAS_VA, APERTURE_HI, &rays, &bvh4_buf)
        .expect("live RTX 3090 software BVH4 dispatch failed");

    // 4. Execute hardware DXR 1.1 RayQuery path live on RTX 3090's 82 Ampere RT Cores!
    let rq_out = dispatch_live_dxr11_rayquery(&host, &rq_shader, TLAS_VA, &rays)
        .expect("live RTX 3090 DXR 1.1 RayQuery dispatch failed");

    // 5. Differential assertion across all 8 rays:
    //    [hit_flag, t, bary_u, bary_v, inst_id, geo_idx, prim_idx, front_face]
    let expected_hits: [(u32, u32, u32, u32, u32); 8] = [
        // (hit, inst_id, geo, prim, front_face)
        (1, INST0_ID, 0, 0, 1), // Ray 0: Inst 0 Geo 0 Prim 0 front
        (1, INST0_ID, 0, 1, 0), // Ray 1: Inst 0 Geo 0 Prim 1 back
        (1, INST0_ID, 1, 0, 1), // Ray 2: Inst 0 Geo 1 Prim 0 front
        (1, INST1_ID, 0, 0, 1), // Ray 3: Inst 1 Geo 0 Prim 0 front (translated)
        (1, INST1_ID, 1, 0, 1), // Ray 4: Inst 1 Geo 1 Prim 0 front (translated)
        (0, 0, 0, 0, 0),        // Ray 5: miss
        (0, 0, 0, 0, 0),        // Ray 6: clipped by tMax
        (1, INST0_ID, 0, 0, 1), // Ray 7: non-zero origin.z
    ];
    for r in 0..8 {
        let sw = &sw_out[r * 8..r * 8 + 8];
        let rq = &rq_out[r * 8..r * 8 + 8];
        let (ehit, eid, egeo, eprim, eff) = expected_hits[r];
        assert_eq!(sw[0], ehit, "ray {r}: sw hit mismatch");
        assert_eq!(rq[0], ehit, "ray {r}: rq hit mismatch");
        assert_eq!(sw[0], rq[0], "ray {r}: sw vs rq hit mismatch");

        let sw_t = f32::from_bits(sw[1]);
        let rq_t = f32::from_bits(rq[1]);
        assert!(
            (sw_t - rq_t).abs() < 1e-4,
            "ray {r}: t mismatch sw={sw_t} vs rq={rq_t}"
        );

        let sw_u = f32::from_bits(sw[2]);
        let rq_u = f32::from_bits(rq[2]);
        let sw_v = f32::from_bits(sw[3]);
        let rq_v = f32::from_bits(rq[3]);
        assert!(
            (sw_u - rq_u).abs() < 1e-4 && (sw_v - rq_v).abs() < 1e-4,
            "ray {r}: bary mismatch sw=({sw_u},{sw_v}) vs rq=({rq_u},{rq_v})"
        );

        assert_eq!(sw[4], eid, "ray {r}: sw InstanceID");
        assert_eq!(rq[4], eid, "ray {r}: rq InstanceID");
        assert_eq!(sw[5], egeo, "ray {r}: sw GeometryIndex");
        assert_eq!(rq[5], egeo, "ray {r}: rq GeometryIndex");
        assert_eq!(sw[6], eprim, "ray {r}: sw PrimitiveIndex");
        assert_eq!(rq[6], eprim, "ray {r}: rq PrimitiveIndex");
        assert_eq!(sw[7], eff, "ray {r}: sw frontFace");
        assert_eq!(rq[7], eff, "ray {r}: rq frontFace");
    }
}

/// Runs the software BVH4 shader live on RTX 3090 with `bvh4_buf` pre-uploaded
/// into Sparse-UMA window 0 (`SPARSE_BASE + 1`) as a raw UAV.
fn dispatch_live_software_bvh4(
    host: &Host,
    shader: &HostShader,
    tlas_va: u64,
    aperture_hi: u32,
    rays: &[u32],
    bvh4_buf: &[u32],
) -> Result<Vec<u32>, String> {
    unsafe {
        let dev = &host.device;
        let make_buf = |size: u64,
                        heap: D3D12_HEAP_TYPE,
                        flags: D3D12_RESOURCE_FLAGS,
                        state: D3D12_RESOURCE_STATES|
         -> Result<ID3D12Resource, String> {
            let hp = D3D12_HEAP_PROPERTIES {
                Type: heap,
                ..Default::default()
            };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: size.max(256),
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: DXGI_FORMAT_UNKNOWN,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: flags,
                ..Default::default()
            };
            let mut r: Option<ID3D12Resource> = None;
            dev.CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut r)
                .map_err(|e| format!("CreateCommittedResource: {e}"))?;
            Ok(r.unwrap())
        };
        let upload = |words: &[u32]| -> Result<ID3D12Resource, String> {
            let bytes = (words.len() * 4) as u64;
            let r = make_buf(
                bytes,
                D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            )?;
            let mut ptr = std::ptr::null_mut();
            r.Map(0, None, Some(&mut ptr))
                .map_err(|e| format!("Map: {e}"))?;
            std::ptr::copy_nonoverlapping(words.as_ptr(), ptr.cast::<u32>(), words.len());
            r.Unmap(0, None);
            Ok(r)
        };

        let ray_buf = upload(rays)?;
        let bvh_stage = upload(bvh4_buf)?;
        let bvh_bytes = (bvh4_buf.len() * 4) as u64;
        let bvh_uav = make_buf(
            bvh_bytes,
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_DEST,
        )?;
        let null_uav = make_buf(
            256,
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        )?;
        let out_dwords = 8 * 8usize;
        let out_bytes = (out_dwords * 4) as u64;
        let out_uav = make_buf(
            out_bytes,
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        )?;
        let readback = make_buf(
            out_bytes,
            D3D12_HEAP_TYPE_READBACK,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_COPY_DEST,
        )?;

        const SPARSE_BASE: u32 = 3; // slot 0=rays, slot 1=out, slot 3=null, slot 4=window 0 (bvh4)
        let heap: ID3D12DescriptorHeap = dev
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 20,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            })
            .map_err(|e| format!("CreateDescriptorHeap: {e}"))?;
        let stride = dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
            as usize;
        let base_cpu = heap.GetCPUDescriptorHandleForHeapStart().ptr;
        let at = |slot: usize| D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: base_cpu + slot * stride,
        };
        let make_raw_uav = |res: &ID3D12Resource, dwords: u32, slot: usize| {
            dev.CreateUnorderedAccessView(
                res,
                None,
                Some(&D3D12_UNORDERED_ACCESS_VIEW_DESC {
                    Format: DXGI_FORMAT_R32_TYPELESS,
                    ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                    Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                        Buffer: D3D12_BUFFER_UAV {
                            FirstElement: 0,
                            NumElements: dwords,
                            StructureByteStride: 0,
                            CounterOffsetInBytes: 0,
                            Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                        },
                    },
                }),
                at(slot),
            );
        };

        dev.CreateShaderResourceView(
            &ray_buf,
            Some(&D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT_R32_TYPELESS,
                ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_SRV {
                        FirstElement: 0,
                        NumElements: rays.len() as u32,
                        StructureByteStride: 0,
                        Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                    },
                },
            }),
            at(0),
        );
        make_raw_uav(&out_uav, out_dwords as u32, 1);
        make_raw_uav(&null_uav, 64, SPARSE_BASE as usize);
        make_raw_uav(&bvh_uav, bvh4_buf.len() as u32, SPARSE_BASE as usize + 1);
        for w in 1..12usize {
            make_raw_uav(&null_uav, 64, SPARSE_BASE as usize + 1 + w);
        }

        let (root, pso) = host.compute_pipeline(shader)?;
        let queue: ID3D12CommandQueue = dev
            .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                ..Default::default()
            })
            .map_err(|e| format!("CreateCommandQueue: {e}"))?;
        let alloc: ID3D12CommandAllocator = dev
            .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
            .map_err(|e| format!("CreateCommandAllocator: {e}"))?;
        let list: ID3D12GraphicsCommandList = dev
            .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)
            .map_err(|e| format!("CreateCommandList: {e}"))?;

        list.CopyBufferRegion(&bvh_uav, 0, &bvh_stage, 0, bvh_bytes);
        let barrier = |res: &ID3D12Resource, before, after| D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                    pResource: std::mem::ManuallyDrop::new(Some(res.clone())),
                    Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                    StateBefore: before,
                    StateAfter: after,
                }),
            },
        };
        list.ResourceBarrier(&[barrier(
            &bvh_uav,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        )]);

        let heaps = [Some(heap.clone())];
        list.SetDescriptorHeaps(&heaps);
        list.SetComputeRootSignature(&root);
        list.SetPipelineState(&pso);
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            ud[u.index as usize] = match u.root {
                RootType::Table => 0,
                RootType::Other(super::USER_VIEW_ADDRESS) => tlas_va as u32,
                RootType::Other(x) if x == super::USER_VIEW_ADDRESS + 1 => (tlas_va >> 32) as u32,
                RootType::Other(super::USER_MEMORY_BASE) => SPARSE_BASE,
                RootType::Other(super::USER_MEMORY_APERTURE_HI) => aperture_hi,
                _ => 0,
            };
        }
        list.SetComputeRoot32BitConstants(0, ud.len() as u32, ud.as_ptr().cast(), 0);
        list.Dispatch(1, 1, 1);

        list.ResourceBarrier(&[barrier(
            &out_uav,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        )]);
        list.CopyBufferRegion(&readback, 0, &out_uav, 0, out_bytes);
        list.Close().map_err(|e| format!("Close: {e}"))?;

        let cmd_list: ID3D12CommandList = list.cast().unwrap();
        queue.ExecuteCommandLists(&[Some(cmd_list)]);
        let fence: ID3D12Fence = dev
            .CreateFence(0, D3D12_FENCE_FLAG_NONE)
            .map_err(|e| format!("CreateFence: {e}"))?;
        queue.Signal(&fence, 1).map_err(|e| format!("Signal: {e}"))?;
        while fence.GetCompletedValue() < 1 {
            std::thread::yield_now();
        }

        let mut ptr = std::ptr::null_mut();
        readback
            .Map(0, None, Some(&mut ptr))
            .map_err(|e| format!("Map readback: {e}"))?;
        let mut out = vec![0u32; out_dwords];
        std::ptr::copy_nonoverlapping(ptr.cast::<u32>(), out.as_mut_ptr(), out_dwords);
        readback.Unmap(0, None);
        Ok(out)
    }
}

#[test]
fn non_opaque_alpha_test_loop_falls_back_losslessly_to_intersect() {
    // Take the exact Template A opaque loop and insert an `image_load` inside
    // the loop body (modelling an alpha-test texture lookup). Under
    // `VICTORY_DXR_RAYQUERY=1`, the recognizer must classify it as
    // `NonOpaqueCandidate` and fall back cleanly to `emit/intersect.rs`.
    let mut native = build_template_a_diff_shader();
    // Add an 8-dword texture SRV at table_offset_dwords: 24 (offset 0x60) and
    // insert `image_load v55, v[1:2], s[40:47] dim:2D unorm` right before `s_endpgm`'s
    // loop back-edge (inside the outer loop body).
    let mut words: Vec<u32> = native
        .code
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    // Find the outer loop's `s_add_i32 s22, s22, 1` (`0x8116_8116`) right before the back-edge.
    let pos = words
        .iter()
        .position(|&x| x == s_add_i32_1(22, 22))
        .expect("outer loop increment");
    // Patch the outer back-edge & skip-inst branch offsets cleanly by rebuilding
    // a tiny 1-loop shader with `image_bvh_intersect_ray` + `image_load`.
    let _ = pos;
    words.clear();
    words.extend([
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0202,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0x00 (ray SRV)
        0xF408_0302,
        0xFA00_0020, // s_load_dwordx4 s[12:15], s[4:5], 0x20 (out UAV)
        0xF40C_0602,
        0xFA00_0060, // s_load_dwordx8 s[24:31], s[4:5], 0x60 (alpha texture SRV)
        S_WAITCNT_LGKM0,
        v_lshlrev_b32(40, 5, 0),
        0xE038_1000,
        0x8002_0128, // buffer_load_dwordx4 v[1:4], v40, s[8:11], 0 offen
        0xE038_1010,
        0x8002_0528, // buffer_load_dwordx4 v[5:8], v40, s[8:11], 0 offen offset:16
        S_WAITCNT_VM0,
        v_rcp_f32(9, 5),
        v_rcp_f32(10, 6),
        v_rcp_f32(11, 7),
        s_mov_b32_imm(22, 0x80),
    ]);
    let hdr = words.len();
    words.extend(v_mov_imm(19, 5));
    words.extend([
        s_mov_b32_imm(16, 0x80),
        0xBE91_0B81,
        0xBE92_03C1,
        0xBE93_03FF,
        0x8900_03FF,
    ]);
    words.extend(image_bvh_intersect_ray_nsa(
        20,
        16,
        [19, 4, 1, 2, 3, 5, 6, 7, 9, 10, 11],
    ));
    // Non-opaque candidate texture fetch inside loop body:
    // `image_load v25, v[1:2], s[24:31] dim:SQ_RSRC_IMG_2D unorm`
    words.extend([0xF000_1108, (6 << 16) | (25 << 8) | 1]);
    words.push(S_WAITCNT_VM0);
    words.push(s_add_i32_1(22, 22));
    words.push(s_cmp_lt_u32_imm(22, 0x82));
    let rel = (hdr as isize - words.len() as isize - 1) as i16;
    words.push(s_cbranch_scc1(rel));
    words.extend([
        v_add_nc_u32_v(20, 20, 25),
        0xE070_1000,
        0x8003_1428, // buffer_store_dword v20, v40, s[12:15], 0 offen
        S_ENDPGM,
    ]);
    native.code = words.iter().flat_map(|x| x.to_le_bytes()).collect();
    native.used.push(UsedResource {
        kind: ResourceType::Srv,
        root: RootType::Table,
        table_offset_dwords: 24,
        lower_bound: 3,
        stride: 4,
        dwords: 8,
        user_data: 4,
        slot: 0,
    });

    let _rq = DxrRayQueryGuard::set(true);
    let _sp = SparseUmaGuard::set(true);
    let shader = lower_and_validate_cs(&native);
    assert!(
        !shader.root_srvs.contains(&SLOT_TLAS_MAP),
        "non-opaque candidate loop must fall back to emit/intersect.rs"
    );
}

#[test]
fn tainted_mask_phi_numeric_sop_lowers_exact_integer_on_the_card() {
    // Regression test for Wave32 mixed-type PHI tainting:
    // `s27` holds a dead saved `exec_lo` on an untaken uniform arm and an
    // integer constant (`0x0012_3456`) on the taken arm, then feeds
    // `s_and_b32 s31, s27, 0x001fffff` and `s_or_b32 s35, s32, s31` followed
    // by `v_cmp_ne_u32` + `s_cbranch_vccnz`. Lowering must compute the exact
    // scalar integer (`0x0052_3456`) rather than rejecting with MASK_ELSEWHERE.
    let mut w: Vec<u32> = vec![
        0xBE85_0380, // 0: s_mov_b32 s5, 0
        0xF408_0202,
        0xFA00_0000, // 1: s_load_dwordx4 s[8:11], s[4:5], 0x0 (out UAV)
        S_WAITCNT_LGKM0,
        0xBF06_8000, // s_cmp_eq_u32 s0, 0   (root constant s0 = 1 -> false)
        0xBF85_0003, // s_cbranch_scc1 +3 -> dead mask arm
        0xBE9B_03FF,
        0x0012_3456, // live arm: s27 = 0x0012_3456
        0xBF82_0001, // s_branch +1 -> join
        0xBE9B_037E, // dead arm: s_mov_b32 s27, exec_lo
        // Join:
        0x871F_FF1B,
        0x001F_FFFF, // s_and_b32 s31, s27, 0x001fffff
        0xBEA0_03FF,
        0x0040_0000, // s_mov_b32 s32, 0x0040_0000
        0x8823_1F20, // s_or_b32 s35, s32, s31  (= 0x0052_3456)
    ];
    w.extend(v_mov_imm(1, 0x0052_3456));
    w.extend([
        0x7D8A_0223, // v_cmp_ne_u32 vcc_lo, s35, v1  (false when exact!)
        0x876A_6A7E, // s_and_b32 vcc_lo, exec_lo, vcc_lo
        0xBF87_0002, // s_cbranch_vccnz +2 -> mismatch
        v_mov_s(2, 35),
        0xBF82_0002, // s_branch +2 -> store
    ]);
    w.extend(v_mov_imm(2, 0xDEAD_BEEF));
    w.extend([
        v_lshlrev_b32(3, 2, 0),
        0xE070_1000,
        0x8002_0203, // buffer_store_dword v2, v3, s[8:11], 0 offen
        S_ENDPGM,
    ]);

    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        code: w.iter().flat_map(|x| x.to_le_bytes()).collect(),
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
        ],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 4,
            dwords: 4,
            user_data: 4,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo {
            threads: [32, 1, 1],
            rsrc2: 0,
        }),
        vgprs: 8,
    };

    let shader = lower_and_validate_cs(&native);
    let mut ud = vec![0u32; shader.user_data_dwords().max(8) as usize];
    for u in &shader.user_data {
        ud[u.index as usize] = match u.root {
            RootType::Constants => 1, // take live integer arm
            RootType::Table => 0,
            _ => 0,
        };
    }

    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    // Shader declares WaveSize(32) due to `exec_lo` on the dead arm; run on SM 6.6 hardware GPU.
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware GPU available; skipped");
        return;
    };
    let (root, pso) = host
        .compute_pipeline(&shader)
        .unwrap_or_else(|e| panic!("hardware: compute_pipeline failed: {e}"));
    let out = host
        .dispatch(
            &root,
            &pso,
            &Dispatch {
                roots: vec![(0, Root::Constants(ud))],
                views: &[View::RawUav(32)],
                samplers: &[],
                groups: [1, 1, 1],
            },
        )
        .unwrap_or_else(|e| panic!("hardware: dispatch failed: {e}"));
    assert_eq!(
        out[0][0], 0x0052_3456,
        "hardware: tainted-mask-phi numeric SOP result mismatch"
    );
}
