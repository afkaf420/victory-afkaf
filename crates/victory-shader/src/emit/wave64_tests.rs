//! A console wave64 compute program on a host of 32-lane waves: a wave reduction inside a
//! per-lane branch, with that branch taken wave-wide ([`crate::structure::structure_wide`])
//! so that the two host waves can exchange their halves inside it. Lowered, validated by
//! `dxil.dll` when present, and run on the host GPU when it takes waves of 32.

use std::collections::HashSet;

use super::host::{Dispatch, Host, Root, View};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{ShaderError, io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;

/// 64 threads in one console wave. Each thread t holds the bit `1 << (t >> 1)`; the threads
/// below s0 (a root constant) OR theirs over the wave inside `if (t < s0) { ... }` and store
/// the result to dword 0 of the raw UAV.
/// `skip`: the branch past the body when no thread is below s0: `s_cbranch_execz`, or
/// `s_cbranch_scc0` on the SCC `s_and_saveexec` sets ("some lane is left").
fn program(skip: u32) -> NativeProgram {
    let mut words = vec![
        0x7E0A_0281, // v_mov_b32 v5, 1
        0x2C0C_0081, // v_lshrrev_b32 v6, 1, v0
        0x340E_0B06, // v_lshlrev_b32 v7, v6, v5
        0x7D88_0000, // v_cmp_gt_u32 vcc, s0, v0
        0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
        skip | 23,   // the branch to the restore of EXEC
    ];
    words.extend_from_slice(&super::wave::WAVE_OR_V7_TO_S7);
    words.extend_from_slice(&[
        0x7E00_0207, // v_mov_b32 v0, s7
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16, // buffer_store_dword v0, off, s[8:11], 0
        0xBEFE_040C, // s_mov_b64 exec, s[12:13]
    ]);
    native(&words)
}

/// A wave64 compute program of 64 threads (one console wave; the thread index in v0) with 4
/// root constants in s0..s3 and a raw UAV's descriptor in s[8:11], then `body`.
fn native(body: &[u32]) -> NativeProgram {
    native_of(body, true, 64)
}

/// The same for a console of 32-lane waves when `wave64` is not set, with `threads` threads.
fn native_of(body: &[u32], wave64: bool, threads: u32) -> NativeProgram {
    let mut words = vec![
        0xBE85_0380, // s_mov_b32 s5, 0
        0x3D << 26 | 0x02 << 18 | 8 << 6 | 2,
        125 << 25, // s_load_dwordx4 s[8:11], s[4:5], 0
    ];
    words.extend_from_slice(body);
    words.push(S_ENDPGM);
    NativeProgram {
        stage: HwStage::Cs,
        wave64,
        vgprs: 16,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
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
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1, // 5 user SGPRs: s0..s4
            threads: [threads, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

/// `program()` for a host of 32-lane waves, with the branches closing `wide` taken wave-wide.
fn lowered(native: &NativeProgram, p: &ir::Program, wide: &HashSet<usize>) -> Result<HostShader, ShaderError> {
    let st = structure::structure_wide(p, wide).expect("structure");
    let res = resource::track(p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(p, native).expect("io::raise");
    let was = lower::TEST_HOST_WAVE.replace(Some(32));
    let lowered = lower::lower(&lower::Parts {
        native,
        p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    });
    lower::TEST_HOST_WAVE.set(was);
    let lowered = lowered?;
    Ok(HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        root_indexed_views: res.root_indexed_views,
    })
}

#[test]
fn a_sum_of_halves_over_the_wave_is_the_consoles_tree_bit_for_bit() {
    // thread t holds the 16-bit float with bits 0x3C00 + 37 t (1.0 and up, with odd
    // mantissas, so that additions round); all of them store the wave's sum
    let mut words = vec![
        0x340C_0085, // v_lshlrev_b32 v6, 5, v0
        0x340E_0082, // v_lshlrev_b32 v7, 2, v0
        0x4A0E_0F06, // v_add_nc_u32 v7, v6, v7
        0x4A0E_0F00, // v_add_nc_u32 v7, v0, v7
        0x4A0E_0EFF,
        0x0000_3C00, // v_add_nc_u32 v7, 0x3c00, v7
    ];
    // (the wave OR's words with v_add_f16 for v_or_b32: opcode 50 for 28)
    words.extend(super::wave::WAVE_OR_V7_TO_S7.iter().map(|&w| if w >> 25 == 28 { w & 0x01FF_FFFF | 50 << 25 } else { w }));
    words.extend_from_slice(&[
        0x7E00_0207, // v_mov_b32 v0, s7
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16, // buffer_store_dword v0, off, s[8:11], 0
    ]);
    let native = native(&words);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    assert_eq!(p.blocks.iter().flat_map(|b| &b.insts).filter(|i| i.name() == "v_add_f16").count(), 6);
    let shader = lowered(&native, &p, &HashSet::new()).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    // the console's sum: lanes 2k and 2k+1, then pairs, ... each addition rounded to 16 bits
    // (to nearest, ties to even); values here are normal numbers
    let wide = |h: u16| f32::from_bits((u32::from(h >> 10) + 112) << 23 | u32::from(h & 0x3FF) << 13);
    let narrow = |f: f32| {
        let b = f.to_bits();
        let h = ((b >> 23) - 112) << 10 | (b >> 13 & 0x3FF);
        let rest = b & 0x1FFF;
        (h + u32::from(rest > 0x1000 || rest == 0x1000 && h & 1 == 1)) as u16
    };
    let add = |a: u16, b: u16| narrow(wide(a) + wide(b));
    let terms: Vec<u16> = (0..64u16).map(|t| 0x3C00 + 37 * t).collect();
    let mut tree = terms.clone();
    while tree.len() > 1 {
        tree = tree.chunks(2).map(|c| add(c[0], c[1])).collect();
    }
    let in_a_row = terms.iter().copied().reduce(add).unwrap();
    assert_ne!(tree[0], in_a_row, "these terms tell the tree from another order");
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; not run");
        return;
    };
    let (root, pso) = match host.compute_pipeline(&shader) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("the hardware device does not take this program (waves of 32?): {e}; not run");
            return;
        }
    };
    let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    let out = host
        .dispatch(
            &root,
            &pso,
            &Dispatch {
                roots: vec![(0, Root::Constants(ud))],
                views: &[View::RawUav(1)],
                samplers: &[],
                groups: [1, 1, 1],
            },
        )
        .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
    // (the high half is lane 0's term's: zero)
    assert_eq!(out[0][0], u32::from(tree[0]), "tree {:#x}, in a row {in_a_row:#x}", tree[0]);
    eprintln!("ran on the hardware device: the sum {:#x} of the console's tree", tree[0]);
}

/// One `v_add_f16` whose exact result lies between two halves: 1 + 3/1024 (0x3C03) plus 2
/// (0x4000) is 3 + 1.5 steps of that range, which the console rounds up to 0x4202. The
/// host's `f32tof16` may give 0x4201 (toward zero); with `VICTORY_F16_NEAREST=1` the
/// translation gives the console's whatever the host does.
#[test]
fn a_16_bit_sum_rounds_to_nearest_when_asked_to() {
    let words = [
        0x7E0E_02FF,
        0x0000_3C03, // v_mov_b32 v7, 0x3c03
        0x7E0C_02FF,
        0x0000_4000, // v_mov_b32 v6, 0x4000
        0x6400_0D07, // v_add_f16 v0, v7, v6
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16, // buffer_store_dword v0, off, s[8:11], 0
    ];
    let native = native(&words);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    assert!(p.blocks.iter().flat_map(|b| &b.insts).any(|i| i.name() == "v_add_f16"));
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    for nearest in [false, true] {
        let was = lower::TEST_F16_NEAREST.replace(Some(nearest));
        let shader = lowered(&native, &p, &HashSet::new());
        lower::TEST_F16_NEAREST.set(was);
        let shader = shader.expect("lower");
        if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
            let mut copy = shader.dxil.clone();
            v.validate(&mut copy).expect("DXIL validation failed");
        }
        let Ok(host) = Host::new(false) else {
            eprintln!("no SM 6.6 hardware device; not run");
            return;
        };
        let Ok((root, pso)) = host.compute_pipeline(&shader) else {
            eprintln!("the hardware device does not take this program; not run");
            return;
        };
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[View::RawUav(1)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
        let got = out[0][0] & 0xFFFF;
        eprintln!("v_add_f16 0x3c03 + 0x4000 on the hardware device, nearest {nearest}: {got:#x}");
        if nearest {
            assert_eq!(got, 0x4202);
        } else {
            assert!(got == 0x4201 || got == 0x4202, "{got:#x}");
        }
    }
}

#[test]
fn a_wave_or_inside_a_lane_branch_needs_the_branch_wave_wide() {
    for (skip, name) in [(0xBF88_0000, "s_cbranch_execz"), (0xBF84_0000, "s_cbranch_scc0")] {
        a_wave_or_inside(skip, name);
    }
}

fn a_wave_or_inside(skip: u32, name: &str) {
    let native = program(skip);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let branch = (0..p.blocks.len())
        .find(|&b| p.blocks[b].insts.last().is_some_and(|i| i.name() == name))
        .expect("the branch");
    // per lane, the two host waves cannot meet inside the branch
    lower::take_needs();
    let per_lane = lowered(&native, &p, &HashSet::new());
    assert!(
        matches!(&per_lane, Err(ShaderError::Unsupported { what, .. }) if *what == "a wave reduction where not every lane runs"),
        "{:?}",
        per_lane.err()
    );
    // ... and the lowering names the block, around which that branch is the one to widen
    let needs = lower::take_needs();
    let st = structure::structure(&p).expect("structure");
    assert_eq!(lower::branches_around(&st, &needs), Some(vec![branch]));
    // (taken, past the body, when every lane's bit is clear)
    assert_eq!(st.wave_any[branch], Some(false));
    // wave-wide, every lane runs the reduction (EXEC stays a guard on what it does there)
    let wide = HashSet::from([branch]);
    let st = structure::structure_wide(&p, &wide).expect("structure");
    let inside = match p.blocks[branch].exit {
        ir::Exit::Branch { not_taken, .. } => not_taken,
        _ => unreachable!(),
    };
    assert!(st.guards[inside].iter().any(Option::is_some));
    let shader = lowered(&native, &p, &wide).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    // On a GPU that takes waves of 32 (two host waves for the console's one): the OR is of
    // both halves, whichever half's lanes take the branch.
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; not run");
        return;
    };
    let (root, pso) = match host.compute_pipeline(&shader) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("the hardware device does not take this program (waves of 32?): {e}; not run");
            return;
        }
    };
    // (threads below s0 take the branch; thread t gives bit t >> 1)
    for (below, expected) in [(64u32, u32::MAX), (40, 0x000F_FFFF), (33, 0x0001_FFFF), (10, 0x1F), (1, 1)] {
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            if let (RootType::Constants, 0, 0) = (u.root, u.slot, u.dword) {
                ud[u.index as usize] = below;
            }
        }
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[View::RawUav(1)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
        assert_eq!(out[0][0], expected, "{name}: {below} threads in the branch");
    }
    eprintln!("ran on the hardware device ({name}): the OR of both halves in every case");
}

/// Runs `shader` on the hardware device with the root constants `s0`, `s1` and gives dword 0
/// of the UAV; None (and a line) without a device that takes it.
fn on_the_card(shader: &HostShader, constants: [u32; 2]) -> Option<u32> {
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; not run");
        return None;
    };
    let Ok((root, pso)) = host.compute_pipeline(shader) else {
        eprintln!("the hardware device does not take this program; not run");
        return None;
    };
    let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    for u in &shader.user_data {
        if let (RootType::Constants, 0, d @ 0..=1) = (u.root, u.slot, u.dword) {
            ud[u.index as usize] = constants[d as usize];
        }
    }
    let out = host
        .dispatch(
            &root,
            &pso,
            &Dispatch {
                roots: vec![(0, Root::Constants(ud))],
                views: &[View::RawUav(1)],
                samplers: &[],
                groups: [1, 1, 1],
            },
        )
        .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
    Some(out[0][0])
}

/// A 32-lane console wave. A register holds a lane mask on one path (the lanes below s0)
/// and a number on the other, and is read as a number after the join, as the trace programs
/// keep their lanes' masks and their node numbers on one stack
/// (`deferred_tiled:CS_LaserLightShadowRaytracing`, s46 at 0x104, 0x75c and 0x188). The
/// mask arrives as the ballot of its lanes.
#[test]
fn a_lane_mask_joined_with_a_number_is_read_as_the_number_it_is() {
    let words = [
        0x7D88_0000, // v_cmp_gt_u32 vcc_lo, s0, v0
        0xBE94_036A, // s_mov_b32 s20, vcc_lo
        0xBF06_8001, // s_cmp_eq_u32 s1, 0
        0xBF85_0001, // s_cbranch_scc1 over the next
        0xBE94_0385, // s_mov_b32 s20, 5
        0x7E00_0214, // v_mov_b32 v0, s20
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16, // buffer_store_dword v0, off, s[8:11], 0
    ];
    let native = native_of(&words, false, 32);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let shader = lowered(&native, &p, &HashSet::new()).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    for (below, other, expected) in [(32u32, 0u32, u32::MAX), (10, 0, 0x3FF), (0, 0, 0), (10, 1, 5)] {
        let Some(got) = on_the_card(&shader, [below, other]) else { return };
        assert_eq!(got, expected, "{below} lanes in the mask, s1 = {other}");
    }
    eprintln!("ran on the hardware device: the mask's bits, or the number");
}

/// A 32-lane console wave. `if (t < s0) { ... SCC = (s1 == 7) }` and a branch on SCC after
/// the join: the skip brings the SCC of `s_and_saveexec` ("some lane is left") along
/// (`rtdirectionalshadows:CS_RTShadowTrace32HQ_TiledFarOnly`, 0x360-0x39c). On the console
/// the skip is taken by the wave, so with some lane inside every lane sees the inner test.
/// A lane that skipped on its own would see "some lane is left" instead: refused per lane,
/// right with the branch taken wave-wide. Lane 31 stores what it saw.
#[test]
fn scc_of_a_mask_operation_joined_with_a_scalar_test_needs_the_skip_wave_wide() {
    let words = [
        0x7D88_0000, // v_cmp_gt_u32 vcc_lo, s0, v0
        0xBE8D_3C6A, // s_and_saveexec_b32 s13, vcc_lo
        0xBF88_0001, // s_cbranch_execz over the next
        0xBF06_8701, // s_cmp_eq_u32 s1, 7
        0xBEFE_030D, // s_mov_b32 exec_lo, s13
        0xBF85_0002, // s_cbranch_scc1 to "seven"
        0x7E0A_0281, // v_mov_b32 v5, 1
        0xBF82_0001, // s_branch over the next
        0x7E0A_0282, // seven: v_mov_b32 v5, 2
        0x7DA4_009F, // v_cmpx_eq_u32 exec_lo, 31, v0
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16 | 5 << 8, // buffer_store_dword v5, off, s[8:11], 0
    ];
    let native = native_of(&words, false, 32);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let branch = (0..p.blocks.len())
        .find(|&b| p.blocks[b].insts.last().is_some_and(|i| i.name() == "s_cbranch_execz"))
        .expect("the skip");
    lower::take_needs();
    let per_lane = lowered(&native, &p, &HashSet::new());
    assert!(
        matches!(&per_lane, Err(ShaderError::Unsupported { what, .. }) if *what == "a lane mask read as data"),
        "{:?}",
        per_lane.err()
    );
    let st = structure::structure(&p).expect("structure");
    assert_eq!(lower::branches_around(&st, &lower::take_needs()), Some(vec![branch]));
    let shader = lowered(&native, &p, &HashSet::from([branch])).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    // (lanes below s0 inside, s1) -> what lane 31 stores: 2 if SCC was set at the branch
    for (below, s1, expected) in [(0u32, 7u32, 1u32), (10, 7, 2), (10, 3, 1), (32, 3, 1), (32, 7, 2)] {
        let Some(got) = on_the_card(&shader, [below, s1]) else { return };
        assert_eq!(got, expected, "{below} lanes inside, s1 = {s1}");
    }
    eprintln!("ran on the hardware device: the inner test's SCC for every lane once some lane is inside");
}

/// A 128-thread wave64 compute program with LDS and cross-half exchange: inside
/// a per-lane branch (`wave_id != 0`), every arriving lane reads the same LDS
/// dword at a uniform address, adds a scalar, and broadcasts via
/// `v_readfirstlane_b32`. Proving uniform `ds_read` in `lane_uniform` lets the
/// program lower cleanly for 32-lane host waves without widening or shortcuts.
#[test]
fn uniform_lds_read_in_divergent_branch_allows_first_lane_read_on_wave32() {
    let words = [
        0xD765_0007,
        0x0001_00C1, // v_mbcnt_lo_u32_b32 v7, -1, 0 (forces exchange on wave32)
        0x7E10_0200, // v_mov_b32 v8, s0
        0x7E12_0280, // v_mov_b32 v9, 0
        0xD834_0000,
        0x0000_0809, // ds_write_b32 v9, v8
        0xBF8A_0000, // s_barrier
        0x2C0E_0086, // v_lshrrev_b32 v7, 6, v0
        0xBE8C_047E, // s_mov_b64 s[12:13], exec
        0x7DAA_0E80, // v_cmpx_ne_u32 exec, 0, v7
        0xBF88_0008, // s_cbranch_execz +8 dwords (to restore exec)
        0x7E10_0280, // v_mov_b32 v8, 0
        0xD8D8_0000,
        0x0800_0008, // ds_read_b32 v8, v8
        0x4A10_1001, // v_add_nc_u32 v8, s1, v8
        0x7E0E_0508, // v_readfirstlane_b32 s7, v8
        0x7E00_0207, // v_mov_b32 v0, s7
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16, // buffer_store_dword v0, off, s[8:11], 0
        0xBEFE_040C, // s_mov_b64 exec, s[12:13]
    ];
    let mut native = native_of(&words, true, 128);
    native.io = StageIo::Cs(ComputeIo {
        rsrc2: (5 << 1) | (1 << 15), // 5 user SGPRs + 512 B LDS
        threads: [128, 1, 1],
    });
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let shader = lowered(&native, &p, &HashSet::new()).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    for (a, b) in [(0x1234u32, 0x5678u32), (0x100, 42)] {
        let Some(got) = on_the_card(&shader, [a, b]) else { return };
        assert_eq!(got, a.wrapping_add(b), "s0={a:#x}, s1={b:#x}");
    }
}

/// Out-of-bounds LDS accesses follow RDNA2 semantics: reads return 0 and writes
/// are dropped without clobbering dword 0 or faulting host shared memory.
#[test]
fn lds_oob_reads_zero_and_drops_writes() {
    let words = [
        0x7E10_0200, // v_mov_b32 v8, s0
        0x7E12_0280, // v_mov_b32 v9, 0
        0xD834_0000,
        0x0000_0809, // ds_write_b32 v9, v8 (LDS[0] = s0)
        0x7E14_0201, // v_mov_b32 v10, s1
        0x7E16_02FF,
        0xDEAD_BEEF, // v_mov_b32 v11, 0xDEADBEEF
        0xD834_0000,
        0x0000_0B0A, // ds_write_b32 v10, v11 (dropped if s1 >= 512)
        0xD8D8_0000,
        0x0C00_000A, // ds_read_b32 v12, v10 (0 if s1 >= 512)
        0xD8D8_0000,
        0x0D00_0009, // ds_read_b32 v13, v9 (must still be s0)
        0x4A00_1B0C, // v_add_nc_u32 v0, v12, v13
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16, // buffer_store_dword v0, off, s[8:11], 0
    ];
    let mut native = native_of(&words, false, 32);
    native.io = StageIo::Cs(ComputeIo {
        rsrc2: (5 << 1) | (1 << 15), // 5 user SGPRs + 512 B (128 dwords) LDS
        threads: [32, 1, 1],
    });
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let shader = lowered(&native, &p, &HashSet::new()).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let sentinel = 0x1234_5678u32;
    for (byte_addr, expected) in [
        (512u32, sentinel),
        (0x1_0000, sentinel),
        (4, sentinel.wrapping_add(0xDEAD_BEEF)),
    ] {
        let Some(got) = on_the_card(&shader, [sentinel, byte_addr]) else { return };
        assert_eq!(got, expected, "byte_addr={byte_addr:#x}");
    }
}

/// 128-thread Wave64 CS (2 console waves = 4 host 32-lane waves on NVIDIA).
/// Console wave 0 (`t < 64`) takes a wave-wide branch while console wave 1
/// (`64 <= t < 128`) skips it. Inside the branch, console wave 0 runs TWO
/// consecutive cross-half exchanges (a 64-lane wave OR plus a cross-half
/// `v_readlane_b32(0)` + `v_readlane_b32(32)` reduction-tail pair) and stores
/// the combined result. Proves Mode B double-buffered pairwise arrival counters
/// never deadlock when console waves diverge inside a threadgroup.
#[test]
fn mode_b_128_thread_divergent_wave_pairwise_exchange() {
    let mut words = vec![
        // Each thread t computes v5 = t + 10 and v7 = 1 << (t >> 1).
        0x7E0A_028A, // v_mov_b32 v5, 10
        0x4A0A_0105, // v_add_nc_u32 v5, v5, v0 (v5 = t + 10)
        0x7E0C_0281, // v_mov_b32 v6, 1
        0x2C0E_0081, // v_lshrrev_b32 v7, 1, v0
        0x340E_0D07, // v_lshlrev_b32 v7, v7, v6 (v7 = 1 << (t >> 1))
        // Wave-wide divergence: console wave 0 (t < s0 = 64) enters, wave 1 skips!
        0x7D88_0000, // v_cmp_gt_u32 vcc, s0, v0
        0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
        0xBF88_0020, // s_cbranch_execz +32 dwords (patched below)
    ];
    let branch_idx = words.len() - 1;
    // Inside the wave-wide branch: cross-half Wave64 OR of v7 -> s7.
    words.extend_from_slice(&super::wave::WAVE_OR_V7_TO_S7);
    // Second consecutive cross-half exchange: folded v_readlane(0) + v_readlane(32) of v5!
    words.extend_from_slice(&[
        0xD760_0006,
        (128 << 9) | (256 + 5), // v_readlane_b32 s6, v5, 0  (= 10)
        0xD760_000E,
        (160 << 9) | (256 + 5), // v_readlane_b32 s14, v5, 32 (= 42)
        0x8006_0E06,            // s_add_u32 s6, s6, s14 (= 52)
        0x8006_0706,            // s_add_u32 s6, s6, s7  (= 52 + wave_or)
        0x7E02_0206,            // v_mov_b32 v1, s6
        // Only console lane 0 writes UAV[0]
        0x7DA4_0080,            // v_cmpx_eq_u32 exec, 0, v0
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16 | 1 << 8, // buffer_store_dword v1, off, s[8:11], 0
    ]);
    let target_idx = words.len();
    words[branch_idx] = 0xBF88_0000 | ((target_idx - branch_idx - 1) as u32 & 0xFFFF);
    words.push(0xBEFE_040C); // s_mov_b64 exec, s[12:13]

    let native = native_of(&words, true, 128);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let branch = (0..p.blocks.len())
        .find(|&b| p.blocks[b].insts.last().is_some_and(|i| i.name() == "s_cbranch_execz"))
        .expect("wave-wide branch");
    let shader = lowered(&native, &p, &HashSet::from([branch])).expect("lower Mode B");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("Mode B DXIL validation failed");
    }
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    // s0 = 64 -> wave 0 enters (all 64 lanes, OR = 0xFFFF_FFFF, readlane(0)+readlane(32) = 10+42 = 52),
    // while wave 1 (lanes 64..127) skips the branch completely.
    // s0 = 36 -> wave 0 enters (lanes 0..35, OR = (1<<18)-1 = 0x3_FFFF, readlane(0)+readlane(32) = 52),
    // wave 1 skips.
    for (limit, expected_or) in [(64u32, u32::MAX), (36, (1u32 << 18) - 1)] {
        let Some(got) = on_the_card(&shader, [limit, 0]) else { return };
        let expected = expected_or.wrapping_add(52);
        assert_eq!(got, expected, "Mode B 128-thread divergent wave (limit={limit})");
    }
}

/// Coherent scalar buffer load (`glc` flag) in paired Wave64-on-Wave32 mode
/// broadcasts half 0's loaded words to half 1 so both halves agree on the
/// loaded scalar value across both Mode A (64 threads) and Mode B (128 threads).
#[test]
fn glc_scalar_load_broadcasts_across_paired_halves() {
    use super::host::Stride;
    let build_prog = |threads: u32| {
        let mut w = vec![
            0xBE85_0380, // s_mov_b32 s5, 0
            // UAV descriptor s[8:11] at table offset 0
            (0x3D << 26) | (0x02 << 18) | (8 << 6) | 2,
            125 << 25,
            // SRV descriptor s[12:15] at table offset 32 bytes (slot 1 in heap)
            (0x3D << 26) | (0x02 << 18) | (12 << 6) | 2,
            (125 << 25) | 32,
            // Force paired mode: v_readlane_b32 s6, v0, 32
            0xD760_0006,
            (160 << 9) | (256 + 0),
            // Coherent `s_buffer_load_dword s18, s[12:15], 0 glc`
            (0x3D << 26) | (0x08 << 18) | (1 << 16) | (18 << 6) | (12 >> 1),
            125 << 25,
            // v1 = s18 + s6
            0x8012_0612, // s_add_u32 s18, s18, s6
            0x7E02_0212, // v_mov_b32 v1, s18
            // v2 = (v0 >> 5) * 4 (lane 0 -> 0, lane 32 -> 4)
            0x2C04_0085, // v_lshrrev_b32 v2, 5, v0
            0x3404_0482, // v_lshlrev_b32 v2, 2, v2
            // Only threads 0 and 32 write (v0 & 31 == 0 && v0 < 64)
            0x3606_009F, // v_and_b32 v3, 31, v0
            0x7DA4_0680, // v_cmpx_eq_u32 exec, 0, v3
            0x7DA8_00C0, // v_cmpx_gt_u32 exec, 64, v0
            // buffer_store_dword v1, v2, s[8:11], 0 offen
            (0x38 << 26) | (0x1C << 18) | (1 << 12),
            (128 << 24) | (2 << 16) | (1 << 8) | 2,
            S_ENDPGM,
        ];
        let code: Vec<u8> = w.drain(..).flat_map(|x: u32| x.to_le_bytes()).collect();
        NativeProgram {
            stage: HwStage::Cs,
            wave64: true,
            vgprs: 16,
            layout: vec![
                RootParam {
                    root: RootType::Constants,
                    table_offset_dwords: 0,
                    sgprs: 4,
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
            used: vec![
                UsedResource {
                    kind: ResourceType::Uav,
                    root: RootType::Table,
                    table_offset_dwords: 0,
                    lower_bound: 0,
                    stride: 0,
                    dwords: 4,
                    user_data: 4,
                    slot: 1,
                },
                UsedResource {
                    kind: ResourceType::Srv,
                    root: RootType::Table,
                    table_offset_dwords: 8,
                    lower_bound: 1,
                    stride: 0,
                    dwords: 4,
                    user_data: 4,
                    slot: 1,
                },
            ],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 5 << 1,
                threads: [threads, 1, 1],
            }),
            code,
        }
    };
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else { return };
    for threads in [64u32, 128] {
        let native = build_prog(threads);
        let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
        let shader = lowered(&native, &p, &HashSet::new()).expect("lower glc smem");
        if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
            let mut copy = shader.dxil.clone();
            v.validate(&mut copy).expect("glc smem DXIL validation failed");
        }
        let Ok((root, pso)) = host.compute_pipeline(&shader) else { return };
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let payload = 0xCAFE_1000u32;
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[
                        View::RawUav(2),
                        View::Buffer {
                            data: &[payload],
                            stride: Stride::Raw,
                        },
                    ],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("dispatch failed ({threads} threads): {e}"));
        let expected = payload.wrapping_add(32);
        assert_eq!(out[0][0], expected, "half 0 mismatch ({threads} threads)");
        assert_eq!(out[0][1], expected, "half 1 mismatch ({threads} threads)");
    }
}

/// Cross-half `ds_bpermute_b32` in both Mode A (64 threads) and Mode B
/// (128 threads): every lane reads from its opposite half (`lane ^ 32`) via
/// the double-buffered exchange bank and writes the permuted result to UAV[t].
#[test]
fn cross_half_ds_bpermute_exchanges_lanes_across_halves() {
    let words = [
        // v5 = v0 * 17 + 3
        0x340A_0084, // v_lshlrev_b32 v5, 4, v0  (t * 16)
        0x4A0A_0105, // v_add_nc_u32 v5, v5, v0  (t * 17)
        0x4A0A_0A83, // v_add_nc_u32 v5, 3, v5   (t * 17 + 3)
        // v4 = (((v0 & 63) ^ 32) << 2)
        0x3608_00BF, // v_and_b32 v4, 63, v0
        0x3A08_08A0, // v_xor_b32 v4, 32, v4
        0x3408_0882, // v_lshlrev_b32 v4, 2, v4
        // ds_bpermute_b32 v6, v4, v5
        (0x36 << 26) | (0xB3 << 18),
        (6 << 24) | (5 << 8) | 4,
        // v2 = v0 << 2 (byte offset for UAV[t])
        0x3404_0082, // v_lshlrev_b32 v2, 2, v0
        // buffer_store_dword v6, v2, s[8:11], 0 offen
        (0x38 << 26) | (0x1C << 18) | (1 << 12),
        (128 << 24) | (2 << 16) | (6 << 8) | 2,
    ];
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else { return };
    for threads in [64u32, 128] {
        let native = native_of(&words, true, threads);
        let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
        let shader = lowered(&native, &p, &HashSet::new()).expect("lower cross-half ds_bpermute");
        if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
            let mut copy = shader.dxil.clone();
            v.validate(&mut copy).expect("ds_bpermute DXIL validation failed");
        }
        let Ok((root, pso)) = host.compute_pipeline(&shader) else { return };
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[View::RawUav(threads)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("dispatch failed ({threads} threads): {e}"));
        for t in 0..threads {
            let partner = (t & !63) | ((t & 63) ^ 32);
            let expected = partner * 17 + 3;
            assert_eq!(out[0][t as usize], expected, "threads={threads}, lane={t}");
        }
    }
}

/// Full 32 KiB (8,192 dwords) console LDS + Wave64-on-Wave32 exchange tail
/// carving: total host TGSM stays <= 8,192 dwords (passing `dxil.dll`
/// validation), an OOB write to dword 8191 is clamped/dropped without
/// clobbering the carved exchange/halves region, and cross-half `v_readlane`
/// plus in-bounds LDS dword 0 both return exact results on the RTX 3090.
#[test]
fn full_32k_lds_carves_tail_for_wave64_exchange() {
    let words = [
        // Write sentinel s0 to LDS[0] and attempt OOB write to carved tail byte 32764 (dword 8191).
        0x7E10_0200, // v_mov_b32 v8, s0
        0x7E12_0280, // v_mov_b32 v9, 0
        0xD834_0000,
        0x0000_0809, // ds_write_b32 v9, v8 (LDS[0] = s0)
        0x7E14_02FF,
        32764,       // v_mov_b32 v10, 32764 (dword 8191 -> carved tail!)
        0x7E16_02FF,
        0xBAD0_C0DE, // v_mov_b32 v11, 0xBAD0C0DE
        0xD834_0000,
        0x0000_0B0A, // ds_write_b32 v10, v11 (must be dropped by console_lds_dwords clamp!)
        0xBF8A_0000, // s_barrier
        // Cross-half exchange after the OOB write (would corrupt if tail write landed!)
        0xD760_0006,
        (160 << 9) | (256 + 0), // v_readlane_b32 s6, v0, 32 (= 32)
        // Read LDS[0] (== s0) and LDS[8191] (== 0, clamped)
        0xD8D8_0000,
        0x0C00_0009, // ds_read_b32 v12, v9
        0xD8D8_0000,
        0x0D00_000A, // ds_read_b32 v13, v10
        0x4A18_1B0C, // v_add_nc_u32 v12, v12, v13
        0x4A00_1806, // v_add_nc_u32 v0, s6, v12 (= s0 + 0 + 32)
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16, // buffer_store_dword v0, off, s[8:11], 0
    ];
    let mut native = native_of(&words, true, 64);
    native.io = StageIo::Cs(ComputeIo {
        // 64 * 512 B = 32,768 B (full 8,192 dwords of console LDS!)
        rsrc2: (5 << 1) | (64 << 15),
        threads: [64, 1, 1],
    });
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let shader = lowered(&native, &p, &HashSet::new()).expect("lower full 32 KiB LDS");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy)
            .expect("32 KiB carved LDS DXIL validation failed (exceeded 32768 B?)");
    }
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let sentinel = 0x1357_9BDFu32;
    let Some(got) = on_the_card(&shader, [sentinel, 0]) else { return };
    assert_eq!(got, sentinel.wrapping_add(32));
}

/// 128-thread Wave64 Mesh Shader (`ms_6_6`) computing its console lane index
/// via `v_mbcnt_lo/hi(-1)` + a per-lane loop lowers cleanly on Wave32 with
/// ZERO Mode B spin loops, ZERO TGSM atomics, ZERO barriers, and `lds_dwords == 0`,
/// and passes `dxil.dll` validation. Genuine cross-lane `v_readlane_b32` in a
/// mesh shader is cleanly refused rather than injecting Mode B spins.
#[test]
fn wave64_128_thread_mesh_shader_v_mbcnt_emits_no_mode_b_spins() {
    use super::container::SigElement;
    use crate::native::MeshIo;
    let pos = SigElement {
        name: "SV_Position".into(),
        index: 0,
        kind: 3,
        comp: 9,
        interp: 4,
        row: Some(0),
        col: 0,
        cols: 4,
        usage: 0xF,
        half: false,
    };
    let lower_ms = |words: &[u32]| {
        let native = NativeProgram {
            stage: HwStage::Ms,
            wave64: true,
            vgprs: 8,
            layout: Vec::new(),
            used: Vec::new(),
            io: StageIo::Ms(MeshIo {
                outputs: Vec::new(),
                clip_cntl: 0,
                threads: 128,
                max_vertices: 128,
                max_primitives: 128,
                topology: 2,
                lds_bytes: 0,
            }),
            code: words.iter().flat_map(|w: &u32| w.to_le_bytes()).collect(),
        };
        let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
        let st = structure::structure(&p).expect("structure");
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
        let io = io::raise(&p, &native).expect("io::raise");
        let was = lower::TEST_HOST_WAVE.replace(Some(32));
        let out = lower::lower(&lower::Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: Vec::new(),
            outputs: vec![pos.clone()],
            prims: Vec::new(),
            mesh_stride: 0,
        });
        lower::TEST_HOST_WAVE.set(was);
        out
    };

    // Part A: `v_mbcnt_lo/hi(-1)` lane-ID idiom + per-lane self-loop + exports.
    let mbcnt_words = [
        0xD765_0001,
        0x0001_00C1, // v_mbcnt_lo_u32_b32 v1, -1, 0
        0xD766_0001,
        0x0002_02C1, // v_mbcnt_hi_u32_b32 v1, -1, v1
        0x3604_0287, // v_and_b32 v2, 7, v1
        0x4A04_0481, // v_add_nc_u32 v2, 1, v2
        0x7E06_0280, // v_mov_b32 v3, 0
        0xBE8C_047E, // s_mov_b64 s[12:13], exec
        // per-lane self-loop
        0x4A06_0681, // v_add_nc_u32 v3, 1, v3
        0x4A04_04C1, // v_add_nc_u32 v2, -1, v2
        0x7DAA_0480, // v_cmpx_ne_u32 exec, 0, v2
        0xBF89_FFFC, // s_cbranch_execnz -4
        0xBEFE_040C, // s_mov_b64 exec, s[12:13]
        0x4A02_0701, // v_add_nc_u32 v1, v1, v3
        // exp pos0, v1, v1, v1, v1
        (0x3E << 26) | (12 << 4) | 0xF,
        (1 << 24) | (1 << 16) | (1 << 8) | 1,
        // exp prim0, v0, off, off, off done
        (0x3E << 26) | (1 << 11) | (20 << 4) | 0x1,
        0,
        S_ENDPGM,
    ];
    let mut lowered_ms = lower_ms(&mbcnt_words).expect("lower 128-thread Wave64 v_mbcnt MS");
    assert_eq!(lowered_ms.wave_size, None, "ms_6_6 must not emit WaveSize metadata");
    assert_eq!(lowered_ms.lds_dwords, 0, "v_mbcnt MS must not allocate Mode B TGSM");
    assert_eq!(lowered_ms.atomic_rmws, 0, "v_mbcnt MS must emit 0 Mode B atomics/spins");
    assert_eq!(lowered_ms.barriers, 0, "v_mbcnt MS must emit 0 barriers");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut lowered_ms.dxil)
            .expect("128-thread Wave64 v_mbcnt MS DXIL validation failed");
    }

    // Part B: genuine cross-lane `v_readlane_b32` in MS is cleanly refused (no Mode B).
    let readlane_words = [
        0xD760_0006,
        (160 << 9) | (256 + 0), // v_readlane_b32 s6, v0, 32
        0x7E02_0206,            // v_mov_b32 v1, s6
        (0x3E << 26) | (12 << 4) | 0xF,
        (1 << 24) | (1 << 16) | (1 << 8) | 1,
        (0x3E << 26) | (1 << 11) | (20 << 4) | 0x1,
        0,
        S_ENDPGM,
    ];
    let refused = lower_ms(&readlane_words);
    assert!(
        matches!(
            refused,
            Err(ShaderError::Unsupported {
                what: "a cross-lane operation in a mesh shader",
                ..
            })
        ),
        "cross-lane MS should be cleanly refused without Mode B: {:?}",
        refused.err()
    );
}

/// Regression for `exchange::all_lane_blocks` self-post-dominance and paired
/// Wave64-on-Wave32 `count_iteration()`:
/// 1. A 1-block per-lane self-loop (`d -> (d, exit)`) marks `all[d] == false`.
/// 2. A multi-block per-lane loop latch (`H -> ... -> d -> (H, exit)`) marks
///    `all[H] == false` and `all[d] == false`.
/// 3. An acyclic per-lane diamond inside a uniform loop keeps `all[H] == true`
///    and `all[d] == true` (only the divergent arms are `false`).
/// 4. Paired Mode A (64 threads) and Mode B (128 threads) lowering inject ZERO
///    `barrier(9)` or Mode B spins into per-lane loops, validate with `dxil.dll`,
///    and execute without deadlocking on the RTX 3090 with exact per-lane counts.
#[test]
fn per_lane_self_loop_and_latch_are_not_all_lanes_and_emit_no_loop_sync() {
    let run_loop_case = |words: &[u32], multi_block: bool| {
        let native64 = native_of(words, true, 64);
        let p = ir::build(&native64.code, native64.wave64, native64.vgprs).expect("ir::build");
        let st = structure::structure(&p).expect("structure");
        let all = super::exchange::all_lane_blocks(&st).expect("all_lane_blocks");
        let latch = (0..p.blocks.len())
            .find(|&b| {
                p.blocks[b]
                    .insts
                    .last()
                    .is_some_and(|i| i.name() == "s_cbranch_execnz")
            })
            .expect("per-lane loop latch");
        assert!(
            !all[latch],
            "per-lane loop latch block {latch} must NOT be marked all_lanes (multi_block={multi_block})"
        );
        if multi_block {
            // Find the header targeted by the latch's back-edge.
            let crate::structure::LaneExit::Branch { taken, .. } = st.exits[latch] else {
                panic!("latch must be a branch");
            };
            assert_ne!(taken, latch, "multi-block loop must have distinct header and latch");
            assert!(
                !all[taken],
                "multi-block per-lane loop header {taken} must NOT be marked all_lanes"
            );
        }

        let _one = super::valu_tests::ONE_AT_A_TIME
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let Ok(host) = Host::new(false) else { return };
        for threads in [64u32, 128] {
            let native = native_of(words, true, threads);
            let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
            let st = structure::structure(&p).expect("structure");
            let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
            let io = io::raise(&p, &native).expect("io::raise");
            let was = lower::TEST_HOST_WAVE.replace(Some(32));
            let mut l = lower::lower(&lower::Parts {
                native: &native,
                p: &p,
                st: &st,
                io: &io,
                res: &res,
                inputs: Vec::new(),
                outputs: Vec::new(),
                prims: Vec::new(),
                mesh_stride: 0,
            })
            .expect("lower paired loop");
            lower::TEST_HOST_WAVE.set(was);
            if threads == 64 {
                // Mode A: 1 entry-block `wave_base`/exchange-header init barrier
                // + 1 barrier from the single top-level `v_readlane_b32` = 2 total;
                // 0 barriers injected into the per-lane loop!
                assert_eq!(
                    l.barriers, 2,
                    "Mode A must not inject barriers into per-lane loop (multi_block={multi_block})"
                );
            } else {
                // Mode B: 1 `wave_base` atomic + 5 atomics from the single top-level
                // `v_readlane_b32` exchange (arrival Add, volatile Or(0) loads of
                // partner/dead in spin_enter and spin_hdr, timeout Xchg) = 6 total;
                // 0 Mode B spins injected into the per-lane loop!
                assert_eq!(
                    l.atomic_rmws, 6,
                    "Mode B must not inject a spin loop into per-lane loop (multi_block={multi_block})"
                );
            }
            if let Ok(mut v) =
                validator::Validator::load(validator::Validator::default_path().as_deref())
            {
                v.validate(&mut l.dxil).expect("paired loop DXIL validation failed");
            }
            let shader = HostShader {
                stage: HwStage::Cs,
                dxil: l.dxil,
                user_data: l.user_data,
                samplers: l.samplers,
                static_samplers: Vec::new(),
                root_cbvs: l.root_cbvs,
                root_srvs: l.root_srvs,
                root_uavs: l.root_uavs,
                root_indexed_views: res.root_indexed_views,
            };
            let Ok((root, pso)) = host.compute_pipeline(&shader) else { continue };
            let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(ud))],
                        views: &[View::RawUav(threads)],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("dispatch failed (threads={threads}, multi={multi_block}): {e}"));
            for t in 0..threads {
                let partner_lane32 = (t & !63) + 32;
                let expected = ((t & 7) + 1) + partner_lane32;
                assert_eq!(
                    out[0][t as usize], expected,
                    "threads={threads}, multi={multi_block}, lane={t}"
                );
            }
        }
    };

    // Case 1: 1-block per-lane self-loop (`d -> (d, exit)`).
    let self_loop = [
        // Force paired Mode A / Mode B: s6 = lane 32's v0 (= wave_base + 32)
        0xD760_0006,
        (160 << 9) | (256 + 0), // v_readlane_b32 s6, v0, 32
        0x3602_0087,            // v_and_b32 v1, 7, v0
        0x4A02_0281,            // v_add_nc_u32 v1, 1, v1  (trip count 1..=8)
        0x7E04_0280,            // v_mov_b32 v2, 0
        0xBE8C_047E,            // s_mov_b64 s[12:13], exec
        // self-loop block `d`:
        0x4A04_0481,            // v_add_nc_u32 v2, 1, v2
        0x4A02_02C1,            // v_add_nc_u32 v1, -1, v1
        0x7DAA_0280,            // v_cmpx_ne_u32 exec, 0, v1
        0xBF89_FFFC,            // s_cbranch_execnz -4 (back to v_add_nc_u32 v2)
        // exit:
        0xBEFE_040C,            // s_mov_b64 exec, s[12:13]
        0x4A04_0406,            // v_add_nc_u32 v2, s6, v2
        0x3406_0082,            // v_lshlrev_b32 v3, 2, v0 (byte offset t * 4)
        (0x38 << 26) | (0x1C << 18) | (1 << 12),
        (128 << 24) | (2 << 16) | (2 << 8) | 3, // buffer_store_dword v2, v3, s[8:11], 0 offen
    ];
    run_loop_case(&self_loop, false);

    // Case 2: multi-block per-lane loop (`H -> B -> d -> (H, exit)`).
    let multi_block_loop = [
        0xD760_0006,
        (160 << 9) | (256 + 0), // v_readlane_b32 s6, v0, 32
        0x3602_0087,            // v_and_b32 v1, 7, v0
        0x4A02_0281,            // v_add_nc_u32 v1, 1, v1
        0x7E04_0280,            // v_mov_b32 v2, 0
        0xBE8C_047E,            // s_mov_b64 s[12:13], exec
        // header H:
        0x4A04_0481,            // v_add_nc_u32 v2, 1, v2
        0xBF82_0000,            // s_branch +0 (forces block split H -> d)
        // latch d:
        0x4A02_02C1,            // v_add_nc_u32 v1, -1, v1
        0x7DAA_0280,            // v_cmpx_ne_u32 exec, 0, v1
        0xBF89_FFFB,            // s_cbranch_execnz -5 (back to header H)
        // exit:
        0xBEFE_040C,            // s_mov_b64 exec, s[12:13]
        0x4A04_0406,            // v_add_nc_u32 v2, s6, v2
        0x3406_0082,            // v_lshlrev_b32 v3, 2, v0
        (0x38 << 26) | (0x1C << 18) | (1 << 12),
        (128 << 24) | (2 << 16) | (2 << 8) | 3,
    ];
    run_loop_case(&multi_block_loop, true);

    // Case 3: acyclic per-lane diamond INSIDE a uniform loop keeps `all[H]` & `all[d]` true.
    let diamond_in_uniform_loop = [
        0xBE8E_0380,            // s_mov_b32 s14, 0
        0x7E04_0280,            // v_mov_b32 v2, 0
        // uniform loop header H (and per-lane diamond branch d):
        0xBE8C_047E,            // s_mov_b64 s[12:13], exec
        0x7DA8_00A0,            // v_cmpx_gt_u32 exec, 32, v0 (lanes < 32)
        0xBF88_0001,            // s_cbranch_execz +1 (skip then-arm T)
        // then-arm T:
        0x4A04_0481,            // v_add_nc_u32 v2, 1, v2
        // merge J (and uniform loop latch):
        0xBEFE_040C,            // s_mov_b64 exec, s[12:13]
        0x800E_810E,            // s_add_u32 s14, s14, 1
        0xBF07_840E,            // s_cmp_lg_u32 s14, 4
        0xBF85_FFF8,            // s_cbranch_scc1 -8 (back to H)
        (0x38 << 26) | (0x1C << 18),
        (128 << 24) | (2 << 16) | (2 << 8),
    ];
    let native_d = native_of(&diamond_in_uniform_loop, true, 64);
    let p_d = ir::build(&native_d.code, native_d.wave64, native_d.vgprs).expect("ir::build");
    let st_d = structure::structure(&p_d).expect("structure");
    let all_d = super::exchange::all_lane_blocks(&st_d).expect("all_lane_blocks");
    let diamond_hdr = (0..p_d.blocks.len())
        .find(|&b| {
            p_d.blocks[b]
                .insts
                .last()
                .is_some_and(|i| i.name() == "s_cbranch_execz")
        })
        .expect("diamond branch");
    let uniform_latch = (0..p_d.blocks.len())
        .find(|&b| {
            p_d.blocks[b]
                .insts
                .last()
                .is_some_and(|i| i.name() == "s_cbranch_scc1")
        })
        .expect("uniform latch");
    let crate::structure::LaneExit::Branch { taken: skip_join, not_taken: then_arm, .. } =
        st_d.exits[diamond_hdr]
    else {
        panic!("diamond must be a branch");
    };
    assert!(all_d[diamond_hdr], "acyclic diamond header inside uniform loop must stay all_lanes");
    assert!(!all_d[then_arm], "acyclic diamond then-arm must NOT be all_lanes");
    assert!(all_d[skip_join], "acyclic diamond merge must stay all_lanes");
    assert!(all_d[uniform_latch], "uniform loop latch must stay all_lanes");
}

/// A 64-thread Wave64 CS with an LDS write and LDS read inside the same
/// per-lane branch region stays un-widened (`WaveSize=64`) by default when
/// `VICTORY_WAVE_WIDE` is unset, and widens cleanly to `WaveSize=32` when
/// explicitly opted in via `VICTORY_WAVE_WIDE=1`, passing `dxil.dll`
/// validation and executing accurately on a Wave32 RTX 3090.
#[test]
fn last_resort_widen_rescues_wave64_lds_conflict_on_wave32() {
    let words = [
        // Real `if (v0 < s0) { ds_write; s_barrier; ds_read; store } else { v5 = 99 }`
        // so `without_skips` cannot flatten the branch and `lds_meetings` sees
        // both an LDS write and read inside one per-lane branch region.
        0x7D88_0000, // v_cmp_gt_u32 vcc, s0, v0
        0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
        0xBF88_000C, // s_cbranch_execz +12 dwords (to else block)
        0x3404_0082, // v_lshlrev_b32 v2, 2, v0 (byte offset = t * 4)
        0x4A06_0001, // v_add_nc_u32 v3, s1, v0 (val = s1 + t)
        0xD834_0000,
        0x0000_0302, // ds_write_b32 v2, v3
        0xBF8A_0000, // s_barrier
        // Read partner lane (t ^ 32)'s LDS slot:
        0x3A08_00A0, // v_xor_b32 v4, 32, v0
        0x3408_0882, // v_lshlrev_b32 v4, 2, v4
        0xD8D8_0000,
        0x0500_0004, // ds_read_b32 v5, v4
        (0x38 << 26) | (0x1C << 18) | (1 << 12),
        (128 << 24) | (2 << 16) | (5 << 8) | 2, // buffer_store_dword v5, v2, s[8:11], 0 offen
        0xBF82_0002, // s_branch +2 dwords (skip 2-dword else block)
        // else block (keeps branch from being a plain skip):
        0x7E0A_02FF,
        99,          // v_mov_b32 v5, 99
        0xBEFE_040C, // s_mov_b64 exec, s[12:13]
    ];
    let mut native = native_of(&words, true, 64);
    native.io = StageIo::Cs(ComputeIo {
        rsrc2: (5 << 1) | (1 << 15), // 5 user SGPRs + 512 B LDS
        threads: [64, 1, 1],
    });
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, &native).expect("io::raise");
    let empty_dxbc = super::container::build(&[]);
    let container = crate::dxbc::Container::parse(&empty_dxbc).expect("empty container");

    // 1. Default (`VICTORY_WAVE_WIDE` off): does NOT widen (remains WaveSize=64).
    let was_wave = lower::TEST_HOST_WAVE.replace(Some(32));
    let was_wide = super::TEST_WAVE_WIDE.replace(Some(false));
    let default_shader = super::emit(&container, &native, &p, &st, &io, &res)
        .expect("default emit succeeds with WaveSize=64");
    // 2. Opt-in (`VICTORY_WAVE_WIDE=1`): widens cleanly to WaveSize=32.
    super::TEST_WAVE_WIDE.set(Some(true));
    let shader = super::emit(&container, &native, &p, &st, &io, &res)
        .expect("opt-in VICTORY_WAVE_WIDE=1 should rescue Wave64 LDS conflict on Wave32");
    super::TEST_WAVE_WIDE.set(was_wide);
    lower::TEST_HOST_WAVE.set(was_wave);

    assert_ne!(
        default_shader.dxil, shader.dxil,
        "opt-in VICTORY_WAVE_WIDE=1 must produce distinct widened Wave32 DXIL"
    );
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy)
            .expect("rescued Wave64 CS DXIL validation failed");
    }
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else { return };
    assert!(
        host.compute_pipeline(&default_shader).is_err(),
        "default (non-widened) WaveSize=64 shader must remain un-widened on Wave32 host"
    );
    let (root, pso) = host
        .compute_pipeline(&shader)
        .expect("Wave32 host must accept opt-in widened shader (WaveSize <= 32)");
    let ud = [64u32, 100, 0, 0];
    let out = host
        .dispatch(
            &root,
            &pso,
            &Dispatch {
                roots: vec![(0, Root::Constants(ud.to_vec()))],
                views: &[View::RawUav(64)],
                samplers: &[],
                groups: [1, 1, 1],
            },
        )
        .expect("dispatch rescued shader");
    for t in 0..64u32 {
        assert_eq!(out[0][t as usize], 100 + (t ^ 32), "lane {t}");
    }
}
