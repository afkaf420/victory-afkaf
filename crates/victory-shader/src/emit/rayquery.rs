//! Hardware DXR 1.1 SM 6.6 `RayQuery` whole-loop recognizer and DXIL emitter
//! (`VICTORY_DXR_RAYQUERY=1`).
//!
//! Recognizes inline ray-tracing traversal loops compiled to RDNA2
//! `image_bvh_intersect_ray` (two-level TLAS+BLAS Template A) and
//! `image_bvh64_intersect_ray` (unified single-loop Template B) and lifts
//! pure-opaque loops (Phase 5B-1) directly to DXIL SM 6.6 `RayQuery`
//! intrinsics backed by root SRV `t<SLOT_TLAS_MAP>` (`t251`, 64 × 16-byte
//! `[tlas_va_shr8_lo, tlas_va_shr8_hi, host_tlas_srv_slot, valid]`).
//!
//! Loops with non-opaque / alpha-test candidate bodies (Phase 5B-2 hook) or
//! any unrecognized side effects / live-outs fall back losslessly to the
//! software per-node lowering in `emit/intersect.rs`.

use std::collections::{HashMap, HashSet, VecDeque};

use super::*;
use crate::gfx10::{Fields, NULL, Operand};
use crate::ir::{Exit, Inst, Program, Reg, Value};
use crate::resource::BvhNodes;

// DXR 1.1 RayQuery `RAY_FLAG_*` constants (HLSL / DXIL 1.6 specification).
pub(super) const RAY_FLAG_NONE: u32 = 0x00;
pub(super) const RAY_FLAG_FORCE_OPAQUE: u32 = 0x01;
pub(super) const RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH: u32 = 0x04;
pub(super) const RAY_FLAG_CULL_BACK_FACING_TRIANGLES: u32 = 0x10;
pub(super) const RAY_FLAG_CULL_FRONT_FACING_TRIANGLES: u32 = 0x20;

/// DXIL `ResourceProperties` word 0 for `RaytracingAccelerationStructure`
/// (verified via Windows SDK 10.0.26100.0 `dxc.exe -T cs_6_6`).
const RT_AS_PROPS: (u32, u32) = (16, 0);

/// Classification of traversal-loop candidate handling.
/// Phase 5B-1 lifts `OpaqueOnly`; `NonOpaqueCandidate` cleanly falls back to
/// `emit/intersect.rs` until Phase 5B-2 candidate-body extraction is enabled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CandidateKind {
    OpaqueOnly,
    NonOpaqueCandidate,
}

/// Which GPURT inline traversal loop template was recognized.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum LoopTemplate {
    /// Two-level coherent TLAS + per-lane BLAS (`image_bvh_intersect_ray`).
    TemplateA,
    /// Unified single-loop TLAS+BLAS (`image_bvh64_intersect_ray`).
    TemplateB,
}

/// How an SSA value flowing out of a lifted traversal loop is synthesized from
/// DXR 1.1 `Committed*` intrinsics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum LiveOut {
    /// Closest hit distance (`CommittedRayT`), falling back to `t_max` on miss.
    CommittedT,
    /// Non-zero when a triangle hit was committed (`CommittedStatus == 1`), else 0.
    CommittedHitFlag,
    /// `CommittedStatus` (`0 = NOTHING`, `1 = TRIANGLE_HIT`).
    CommittedStatus,
    /// Barycentric coordinate `u` (`CommittedTriangleBarycentrics(0)`), `0.0` on miss.
    CommittedBaryU,
    /// Barycentric coordinate `v` (`CommittedTriangleBarycentrics(1)`), `0.0` on miss.
    CommittedBaryV,
    /// Front-face boolean (`CommittedTriangleFrontFace ? 1 : 0`), `0` on miss.
    CommittedFrontFace,
    /// `CommittedInstanceID`, `0` on miss.
    CommittedInstanceId,
    /// `CommittedInstanceIndex`, `0` on miss.
    #[allow(dead_code)]
    CommittedInstanceIndex,
    /// `CommittedInstanceContributionToHitGroupIndex`, `0` on miss.
    CommittedInstanceContrib,
    /// `CommittedGeometryIndex`, `0` on miss.
    CommittedGeometryIndex,
    /// `CommittedPrimitiveIndex`, `0` on miss.
    CommittedPrimitiveIndex,
    /// Loop iteration watchdog counter (always `0`, never trips `0xFFFE`).
    ConstZero,
    /// Value carried unchanged from before the loop header.
    PreLoop(Value),
}

/// A constant or pre-loop SSA scalar/vector input for `TraceRayInline`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RqInput {
    ConstU32(u32),
    Value(Value),
}

/// A recognized whole-loop inline ray-tracing region ready for SM 6.6 `RayQuery`
/// lifting.
#[derive(Debug, Clone)]
pub(super) struct RecognizedLoop {
    #[allow(dead_code)]
    pub(super) template: LoopTemplate,
    pub(super) candidate: CandidateKind,
    /// Single loop-entry IR block replaced by the `RayQuery` sequence.
    #[allow(dead_code)]
    pub(super) header: usize,
    /// All IR blocks belonging to the traversal loop (including `header`).
    pub(super) body: HashSet<usize>,
    /// Single IR block reached on all loop exits.
    pub(super) exit: usize,
    /// Low 32 bits of `tlas_va >> 8` (pre-loop SSA value).
    pub(super) tlas_va_shr8_lo: Value,
    /// High 32 bits of `tlas_va >> 8` (pre-loop SSA value, or `ConstU32(0)`).
    pub(super) tlas_va_shr8_hi: RqInput,
    /// World-space ray origin `(x, y, z)` (pre-loop SSA values, f32 bits).
    pub(super) origin: [Value; 3],
    /// World-space ray direction `(x, y, z)` (pre-loop SSA values, f32 bits).
    pub(super) dir: [Value; 3],
    /// Ray `tMin` (f32 bits).
    pub(super) t_min: RqInput,
    /// Initial ray `tMax` (pre-loop SSA value, f32 bits).
    pub(super) t_max: Value,
    /// 8-bit `InstanceInclusionMask` (`0xFF` when unfiltered).
    pub(super) instance_mask: RqInput,
    /// Compile-time `RAY_FLAG_*` bitmask passed to `allocateRayQuery`.
    pub(super) const_ray_flags: u32,
    /// Every SSA value produced inside `body` that is consumed outside `body`
    /// (directly or via an `exit`-block phi), mapped to its `RayQuery` source.
    pub(super) live_outs: HashMap<Value, LiveOut>,
}

/// Recognized whole-loop plan for one shader lowering.
#[derive(Default)]
pub(super) struct RayQueryPlan {
    /// Header IR block -> recognized loop specification.
    pub(super) by_header: HashMap<usize, RecognizedLoop>,
    /// All IR blocks skipped because they lie inside a lifted loop body.
    pub(super) skipped: HashSet<usize>,
    /// Exit IR block -> header IR block of the lifted loop exiting into it.
    pub(super) exit_to_header: HashMap<usize, usize>,
    /// Filled during lowering: header IR block -> final DXIL join block that
    /// branches to `exit`, plus the DXIL values produced for each live-out IR
    /// `Value`.
    pub(super) lowered_join: HashMap<usize, usize>,
}

impl RayQueryPlan {
    #[allow(dead_code)]
    pub(super) fn is_empty(&self) -> bool {
        self.by_header.is_empty()
    }
}

/// Recognizes pure-opaque GPURT Template A / Template B ray-tracing traversal
/// loops in `x` when `VICTORY_DXR_RAYQUERY=1`. Returns an empty plan if no
/// qualifying loops exist (preserving lossless `emit/intersect.rs` fallback).
pub(super) fn recognize(x: &Parts<'_>) -> RayQueryPlan {
    let mut plan = RayQueryPlan::default();
    if !super::super::dxr_rayquery() || x.res.bvh.is_empty() {
        return plan;
    }
    let p = x.p;
    let rpo = crate::ir::reverse_postorder(&p.blocks);
    let mut rpo_idx = vec![usize::MAX; p.blocks.len()];
    for (n, &b) in rpo.iter().enumerate() {
        rpo_idx[b] = n;
    }

    // Build quick value -> defining (block, inst_idx) and phi maps.
    let mut def_inst: HashMap<Value, (usize, usize)> = HashMap::new();
    let mut def_phi: HashMap<Value, (usize, usize)> = HashMap::new();
    for (b, blk) in p.blocks.iter().enumerate() {
        for (pi, phi) in blk.phis.iter().enumerate() {
            def_phi.insert(phi.value, (b, pi));
        }
        for (ii, inst) in blk.insts.iter().enumerate() {
            for &(_, v) in &inst.writes {
                def_inst.insert(v, (b, ii));
            }
        }
    }

    // Collect every natural loop (header -> body blocks) in the CFG.
    let loops = find_natural_loops(p, &rpo_idx);
    if loops.is_empty() {
        return plan;
    }

    // Match BVH instructions with console GPURT descriptor flags
    // (`word1 == 0x8000_0000`, `word3 == 0x8900_03FF`).
    let mut bvh_sites: Vec<(usize, usize, bool)> = Vec::new();
    for &(site, mode) in &x.res.bvh {
        if let BvhNodes::Memory { word1, word3 } = mode
            && word1 == 0x8000_0000
            && word3 == 0x8900_03FF
        {
            let inst = &p.blocks[site.0].insts[site.1];
            let is_64 = inst.name().contains("bvh64");
            bvh_sites.push((site.0, site.1, is_64));
        }
    }
    if bvh_sites.is_empty() {
        return plan;
    }

    // Inspect outermost natural loops that enclose at least one GPURT BVH site.
    // Sort by body size descending so we consider enclosing outer loops first.
    let mut candidates: Vec<(usize, HashSet<usize>)> = loops
        .into_iter()
        .filter(|(_, body)| bvh_sites.iter().any(|&(b, _, _)| body.contains(&b)))
        .collect();
    candidates.sort_by(|a, b| b.1.len().cmp(&a.1.len()));

    let rq_why = std::env::var_os("VICTORY_RQ_WHY").is_some();
    let mut claimed: HashSet<usize> = HashSet::new();
    for (header, body) in candidates {
        if body.iter().any(|b| claimed.contains(b)) {
            continue;
        }
        let Some(rec) = analyze_loop(
            p,
            &rpo_idx,
            &def_inst,
            &def_phi,
            &bvh_sites,
            header,
            &body,
            rq_why,
        ) else {
            continue;
        };
        if rec.candidate != CandidateKind::OpaqueOnly {
            continue;
        }
        // Only one lifted loop per exit block to keep PHI rewriting simple.
        if plan.exit_to_header.contains_key(&rec.exit) {
            continue;
        }
        claimed.extend(&rec.body);
        plan.skipped.extend(rec.body.iter().copied().filter(|&b| b != header));
        plan.exit_to_header.insert(rec.exit, header);
        plan.by_header.insert(header, rec);
    }

    plan
}

/// Computes natural loops keyed by loop header block using RPO back-edges.
fn find_natural_loops(p: &Program, rpo_idx: &[usize]) -> HashMap<usize, HashSet<usize>> {
    let mut loops: HashMap<usize, HashSet<usize>> = HashMap::new();
    for (b, blk) in p.blocks.iter().enumerate() {
        if rpo_idx[b] == usize::MAX {
            continue;
        }
        for succ in exit_succs(blk.exit) {
            if rpo_idx[succ] <= rpo_idx[b] {
                let entry = loops.entry(succ).or_default();
                entry.insert(succ);
                if entry.insert(b) {
                    let mut work = VecDeque::from([b]);
                    while let Some(cur) = work.pop_front() {
                        for &pred in &p.blocks[cur].preds {
                            if rpo_idx[pred] != usize::MAX && entry.insert(pred) {
                                work.push_back(pred);
                            }
                        }
                    }
                }
            }
        }
    }
    loops
}

fn exit_succs(exit: Exit) -> Vec<usize> {
    match exit {
        Exit::Jump(t) => vec![t],
        Exit::Branch { taken, not_taken } => vec![taken, not_taken],
        Exit::End | Exit::Indirect => Vec::new(),
    }
}

/// Attempts to recognize and validate a single outer traversal loop `(header, body)`.
#[allow(clippy::too_many_arguments)]
fn analyze_loop(
    p: &Program,
    rpo_idx: &[usize],
    def_inst: &HashMap<Value, (usize, usize)>,
    def_phi: &HashMap<Value, (usize, usize)>,
    bvh_sites: &[(usize, usize, bool)],
    header: usize,
    body: &HashSet<usize>,
    why: bool,
) -> Option<RecognizedLoop> {
    let hdr_off = p.blocks[header]
        .insts
        .first()
        .map_or(0, |i| i.machine.offset);
    let reject = |reason: &str| -> Option<RecognizedLoop> {
        if why {
            eprintln!("RQ-WHY header=b{header}@{hdr_off:#x} (body={}): {reason}", body.len());
        }
        None
    };
    // 1. Single entry (only `header` may have predecessors outside `body`).
    for &b in body {
        if b == header {
            continue;
        }
        if p.blocks[b]
            .preds
            .iter()
            .any(|&pred| rpo_idx[pred] != usize::MAX && !body.contains(&pred))
        {
            return reject("multiple loop entries");
        }
    }
    // Header must have at least one reachable pre-loop predecessor.
    let pre_preds: Vec<usize> = p.blocks[header]
        .preds
        .iter()
        .copied()
        .filter(|&pred| rpo_idx[pred] != usize::MAX && !body.contains(&pred))
        .collect();
    if pre_preds.is_empty() {
        return reject("no pre-loop predecessor");
    }

    // 2. Fold trivial single-jump exit trampolines (`s_branch exit`, no
    //    instructions other than `s_branch`/`s_nop`, predecessors strictly in
    //    `body`) into `body`, then require a single exit block.
    let mut body = body.clone();
    let mut folded = true;
    while folded {
        folded = false;
        let mut to_add = Vec::new();
        for &b in &body {
            for succ in exit_succs(p.blocks[b].exit) {
                if !body.contains(&succ)
                    && let Exit::Jump(_) = p.blocks[succ].exit
                    && p.blocks[succ]
                        .preds
                        .iter()
                        .all(|&pred| rpo_idx[pred] == usize::MAX || body.contains(&pred))
                    && p.blocks[succ]
                        .insts
                        .iter()
                        .all(|i| matches!(i.name(), "s_branch" | "s_nop"))
                {
                    to_add.push(succ);
                }
            }
        }
        for t in to_add {
            if body.insert(t) {
                folded = true;
            }
        }
    }
    let body = &body;

    let mut exits: HashSet<usize> = HashSet::new();
    for &b in body {
        for succ in exit_succs(p.blocks[b].exit) {
            if !body.contains(&succ) {
                exits.insert(succ);
            }
        }
        if matches!(p.blocks[b].exit, Exit::End | Exit::Indirect) {
            return reject("Exit::End or Indirect inside loop");
        }
    }
    if exits.len() != 1 {
        return reject("exits.len() != 1");
    }
    let exit = *exits.iter().next()?;

    // 3. Reject any loop containing external memory writes, texture sampling,
    //    or non-BVH buffer loads (signals an alpha-test / any-hit candidate
    //    body -> Phase 5B-2, or side effects that must not be elided).
    let mut candidate = CandidateKind::OpaqueOnly;
    for &b in body {
        for inst in &p.blocks[b].insts {
            let n = inst.name();
            if n.starts_with("image_bvh") {
                continue;
            }
            if n.starts_with("image_") {
                // Texture sampling/loading inside the traversal loop indicates
                // an inlined alpha-test / non-opaque candidate body.
                candidate = CandidateKind::NonOpaqueCandidate;
                continue;
            }
            if n.starts_with("buffer_load")
                || n.starts_with("buffer_store")
                || n.starts_with("buffer_atomic")
                || n.starts_with("global_store")
                || n.starts_with("global_atomic")
                || n.starts_with("s_store")
                || n.starts_with("s_buffer_store")
                || n.starts_with("s_atomic")
                || n.starts_with("exp")
                || n == "s_barrier"
            {
                return reject("external side effect / buffer_load inside loop");
            }
        }
    }
    if candidate != CandidateKind::OpaqueOnly {
        return reject("non-opaque candidate body (Phase 5B-2)");
    }

    // 4. Identify BVH sites inside `body` and classify Template A vs Template B.
    let inner_bvhs: Vec<(usize, usize, bool)> = bvh_sites
        .iter()
        .copied()
        .filter(|&(b, _, _)| body.contains(&b))
        .collect();
    if inner_bvhs.is_empty() {
        return reject("no BVH sites inside body");
    }
    let template = if inner_bvhs.iter().any(|&(_, _, is_64)| is_64) {
        LoopTemplate::TemplateB
    } else {
        LoopTemplate::TemplateA
    };

    // Pick the outermost BVH site (earliest in RPO) for world-space ray inputs
    // and the triangle-testing BVH site (latest in RPO; same as outer for
    // Template B, inner BLAS site for Template A).
    let outer_bvh = *inner_bvhs.iter().min_by_key(|&&(b, i, _)| (rpo_idx[b], i))?;
    let tri_bvh = *inner_bvhs.iter().max_by_key(|&&(b, i, _)| (rpo_idx[b], i))?;

    // 5. Extract world-space ray `(origin, dir, tMax)` and `tlas_va_shr8`.
    let outer_inst = &p.blocks[outer_bvh.0].insts[outer_bvh.1];
    let Some((v_tmax, v_orig, v_dir, rsrc_s0)) = extract_bvh_operands(outer_inst, outer_bvh.2)
    else {
        return reject("extract_bvh_operands failed");
    };

    let Some(t_max) = resolve_pre_loop(p, body, header, def_inst, def_phi, v_tmax, 12) else {
        return reject("resolve t_max failed");
    };
    let (Some(ox), Some(oy), Some(oz)) = (
        resolve_pre_loop(p, body, header, def_inst, def_phi, v_orig[0], 12),
        resolve_pre_loop(p, body, header, def_inst, def_phi, v_orig[1], 12),
        resolve_pre_loop(p, body, header, def_inst, def_phi, v_orig[2], 12),
    ) else {
        return reject("resolve origin failed");
    };
    let (Some(dx), Some(dy), Some(dz)) = (
        resolve_pre_loop(p, body, header, def_inst, def_phi, v_dir[0], 12),
        resolve_pre_loop(p, body, header, def_inst, def_phi, v_dir[1], 12),
        resolve_pre_loop(p, body, header, def_inst, def_phi, v_dir[2], 12),
    ) else {
        return reject("resolve dir failed");
    };

    let (tlas_va_shr8_lo, tlas_va_shr8_hi) = match template {
        LoopTemplate::TemplateA => {
            let Some(s0) = rsrc_s0 else {
                return reject("no rsrc_s0");
            };
            let Some(s0_val) = resolve_pre_loop(p, body, header, def_inst, def_phi, s0, 12) else {
                return reject("resolve tlas_va_shr8_lo failed");
            };
            find_shr8_pair(p, def_inst, s0_val)
        }
        LoopTemplate::TemplateB => {
            let Some(pair) = find_template_b_tlas(p, body, header, def_inst, def_phi, outer_inst)
            else {
                return reject("find_template_b_tlas failed");
            };
            pair
        }
    };

    // 6. Locate triangle commit computation (rcp(det), t = t_num * inv_det,
    //    tMin, cull flags, accept-first-hit, and instance mask).
    let tri_inst = &p.blocks[tri_bvh.0].insts[tri_bvh.1];
    let Some(tri_res) = extract_bvh_results(tri_inst) else {
        return reject("extract_bvh_results failed");
    };
    let Some(tri_info) = analyze_triangle_commit(p, body, header, def_inst, def_phi, tri_res)
    else {
        return reject("analyze_triangle_commit failed");
    };

    let instance_mask = detect_instance_mask(p, body, header, def_inst, def_phi);
    let mut const_ray_flags = RAY_FLAG_FORCE_OPAQUE | tri_info.cull_flags;
    if tri_info.accept_first_hit {
        const_ray_flags |= RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH;
    }

    // 7. Find EVERY value written inside `body` that is read outside `body`
    //    (either directly or via an `exit`-block PHI) and classify each one.
    let live_out_vals = collect_live_outs(p, body, exit);
    let mut live_outs = HashMap::new();
    for v in live_out_vals {
        let Some(kind) = classify_live_out(
            p,
            body,
            header,
            def_inst,
            def_phi,
            &tri_info,
            t_max,
            v,
        ) else {
            if why {
                eprintln!("RQ-WHY header=b{header}@{hdr_off:#x}: unclassified live_out {v:?}");
            }
            return None;
        };
        live_outs.insert(v, kind);
    }
    if why {
        eprintln!(
            "RQ-LIFT header=b{header}@{hdr_off:#x} exit=b{exit} body={} template={template:?} flags={const_ray_flags:#x} mask={instance_mask:?} live_outs={live_outs:?}",
            body.len()
        );
    }

    Some(RecognizedLoop {
        template,
        candidate,
        header,
        body: body.clone(),
        exit,
        tlas_va_shr8_lo,
        tlas_va_shr8_hi,
        origin: [ox, oy, oz],
        dir: [dx, dy, dz],
        t_min: tri_info.t_min,
        t_max,
        instance_mask,
        const_ray_flags,
        live_outs,
    })
}

/// Extracts `(tMax, origin[3], dir[3], rsrc_s0)` SSA values from an
/// `image_bvh_intersect_ray` (11 VGPRs) or `image_bvh64_intersect_ray` (12 VGPRs).
fn extract_bvh_operands(
    inst: &Inst,
    is_64: bool,
) -> Option<(Value, [Value; 3], [Value; 3], Option<Value>)> {
    let Fields::Image { addr, rsrc, .. } = &inst.fields else {
        return None;
    };
    let mut vgprs: Vec<Value> = Vec::new();
    for op in addr {
        if let Operand::Vector { reg, dwords } = *op {
            for k in 0..dwords {
                vgprs.push(inst.read(Reg::V(reg + k))?);
            }
        }
    }
    let off = usize::from(is_64);
    if vgprs.len() < 8 + off {
        return None;
    }
    let t_max = vgprs[1 + off];
    let orig = [vgprs[2 + off], vgprs[3 + off], vgprs[4 + off]];
    let dir = [vgprs[5 + off], vgprs[6 + off], vgprs[7 + off]];
    let s0 = match *rsrc {
        Operand::Scalar { code, .. } => inst.read(Reg::S(code)),
        _ => None,
    };
    Some((t_max, orig, dir, s0))
}

/// Extracts the 4 destination VGPRs `[t_num, det, bary_u_num, bary_v_num]` of
/// an `image_bvh*_intersect_ray` instruction.
fn extract_bvh_results(inst: &Inst) -> Option<[Value; 4]> {
    let Fields::Image { data, .. } = &inst.fields else {
        return None;
    };
    let Operand::Vector { reg, dwords } = *data else {
        return None;
    };
    if dwords < 4 {
        return None;
    }
    Some([
        inst.write(Reg::V(reg))?,
        inst.write(Reg::V(reg + 1))?,
        inst.write(Reg::V(reg + 2))?,
        inst.write(Reg::V(reg + 3))?,
    ])
}

/// Traces `v` backward through copies and loop-header PHIs to find a unique
/// pre-loop SSA value defined strictly outside `body`.
fn resolve_pre_loop(
    p: &Program,
    body: &HashSet<usize>,
    header: usize,
    def_inst: &HashMap<Value, (usize, usize)>,
    def_phi: &HashMap<Value, (usize, usize)>,
    mut v: Value,
    depth: u32,
) -> Option<Value> {
    for _ in 0..depth {
        if let Some(&(b, _)) = def_inst.get(&v) {
            if !body.contains(&b) {
                return Some(v);
            }
            let inst = &p.blocks[b].insts[def_inst[&v].1];
            match inst.name() {
                "v_mov_b32" | "s_mov_b32" | "v_readfirstlane_b32" => {
                    v = first_reg_read(inst)?;
                    continue;
                }
                _ => return None,
            }
        }
        if let Some(&(b, pi)) = def_phi.get(&v) {
            if !body.contains(&b) {
                return Some(v);
            }
            let phi = &p.blocks[b].phis[pi];
            if b == header {
                let mut pre: Option<Value> = None;
                for (&pred, &arg) in p.blocks[b].preds.iter().zip(&phi.args) {
                    if !body.contains(&pred) {
                        match pre {
                            None => pre = Some(arg),
                            Some(prev) if prev == arg => {}
                            Some(_) => return None,
                        }
                    }
                }
                v = pre?;
                continue;
            }
            // Non-header PHI inside body: accept only if all non-self args agree
            // on a common pre-loop origin.
            let mut found: Option<Value> = None;
            for &arg in &phi.args {
                if arg == v {
                    continue;
                }
                let r = resolve_pre_loop(p, body, header, def_inst, def_phi, arg, depth - 1)?;
                match found {
                    None => found = Some(r),
                    Some(prev) if prev == r => {}
                    Some(_) => return None,
                }
            }
            return found;
        }
        // Program input (no def in blocks): always pre-loop.
        return Some(v);
    }
    None
}

fn first_reg_read(inst: &Inst) -> Option<Value> {
    inst.reads
        .iter()
        .find(|&&(r, _)| !matches!(r, Reg::S(EXEC) | Reg::Scc))
        .map(|&(_, v)| v)
}

/// Given `lo` = `tlas_va_shr8_lo` (outside the loop), looks for the matching
/// high dword produced by `s_lshr_b64 s[k:k+1], ..., 8`.
fn find_shr8_pair(
    p: &Program,
    def_inst: &HashMap<Value, (usize, usize)>,
    lo: Value,
) -> (Value, RqInput) {
    let mut cur = lo;
    for _ in 0..6 {
        let Some(&(b, i)) = def_inst.get(&cur) else {
            break;
        };
        let inst = &p.blocks[b].insts[i];
        if inst.name() == "s_lshr_b64"
            && let Some(Operand::Scalar { code, .. }) = match &inst.fields {
                Fields::Sop { dst, .. } => *dst,
                _ => None,
            }
            && inst.write(Reg::S(code)) == Some(cur)
            && let Some(hi) = inst.write(Reg::S(code + 1))
        {
            return (cur, RqInput::Value(hi));
        }
        if matches!(inst.name(), "s_mov_b32" | "v_mov_b32" | "v_readfirstlane_b32")
            && let Some(next) = first_reg_read(inst)
        {
            cur = next;
            continue;
        }
        break;
    }
    (lo, RqInput::ConstU32(0))
}

/// Locates `(tlas_va_shr8_lo, tlas_va_shr8_hi)` for a Template B
/// (`image_bvh64_intersect_ray`) loop by tracing `ptr64` back to pre-loop
/// `s_lshr_b64 ..., 8` or `v_lshrrev_b64 ..., 8` / `(ptr_lo, ptr_hi) >> 5`.
fn find_template_b_tlas(
    p: &Program,
    body: &HashSet<usize>,
    header: usize,
    def_inst: &HashMap<Value, (usize, usize)>,
    def_phi: &HashMap<Value, (usize, usize)>,
    outer_inst: &Inst,
) -> Option<(Value, RqInput)> {
    let Fields::Image { addr, .. } = &outer_inst.fields else {
        return None;
    };
    let mut ptr_regs = Vec::new();
    for op in addr {
        if let Operand::Vector { reg, dwords } = *op {
            for k in 0..dwords {
                ptr_regs.push(outer_inst.read(Reg::V(reg + k))?);
            }
        }
    }
    if ptr_regs.len() < 2 {
        return None;
    }
    // Walk backward from pre-loop `ptr_lo` looking for `s_lshr_b64 ..., 8` or
    // an explicit pre-loop `tlas_va_shr8` input.
    let pre_lo = resolve_pre_loop(p, body, header, def_inst, def_phi, ptr_regs[0], 12)?;
    let mut work = VecDeque::from([pre_lo]);
    let mut seen = HashSet::from([pre_lo]);
    for _ in 0..32 {
        let Some(v) = work.pop_front() else { break };
        if let Some(&(b, i)) = def_inst.get(&v) {
            let inst = &p.blocks[b].insts[i];
            if inst.name() == "s_lshr_b64"
                && let Fields::Sop {
                    dst: Some(Operand::Scalar { code, .. }),
                    src,
                    ..
                } = &inst.fields
                && matches!(src.get(1), Some(Operand::Const(136)))
            // inline constant 8 (128 + 8)
            {
                let lo = inst.write(Reg::S(*code))?;
                let hi = inst
                    .write(Reg::S(*code + 1))
                    .map_or(RqInput::ConstU32(0), RqInput::Value);
                return Some((lo, hi));
            }
            for &(r, rv) in &inst.reads {
                if !matches!(r, Reg::S(EXEC) | Reg::Scc) && seen.insert(rv) {
                    work.push_back(rv);
                }
            }
        }
    }
    // Fallback: if `pre_lo` (and optional `pre_hi`) already came straight from
    // pre-loop `s_lshr_b64`, check all pre-loop `s_lshr_b64 ..., 8` feeding
    // `ptr_lo` or dominating `header` as a single unique TLAS descriptor shift.
    let mut shr8s = Vec::new();
    for (b, blk) in p.blocks.iter().enumerate() {
        if body.contains(&b) {
            continue;
        }
        for inst in &blk.insts {
            if inst.name() == "s_lshr_b64"
                && let Fields::Sop {
                    dst: Some(Operand::Scalar { code, .. }),
                    src,
                    ..
                } = &inst.fields
                && matches!(src.get(1), Some(Operand::Const(136)))
                && let Some(lo) = inst.write(Reg::S(*code))
            {
                let hi = inst
                    .write(Reg::S(*code + 1))
                    .map_or(RqInput::ConstU32(0), RqInput::Value);
                shr8s.push((lo, hi));
            }
        }
    }
    if shr8s.len() == 1 {
        return Some(shr8s[0]);
    }
    None
}

/// Parsed triangle-hit commit metadata inside the traversal loop.
struct TriCommitInfo {
    /// Values representing computed hit `t = t_num * rcp(det)`.
    t_vals: HashSet<Value>,
    /// Values representing computed `bary_u = bary_u_num * rcp(det)`.
    bary_u_vals: HashSet<Value>,
    /// Values representing computed `bary_v = bary_v_num * rcp(det)`.
    bary_v_vals: HashSet<Value>,
    /// Values representing `front_face` (`det >= 0` / `det > 0`).
    front_face_vals: HashSet<Value>,
    /// Hit commit mask value(s) feeding `v_cndmask_b32`.
    commit_masks: HashSet<Value>,
    /// Lower distance bound `tMin`.
    t_min: RqInput,
    /// Static cull ray flags (`0`, `CULL_BACK_FACING`, or `CULL_FRONT_FACING`).
    cull_flags: u32,
    /// Whether committing a hit immediately terminates traversal.
    accept_first_hit: bool,
}

/// Traces `tri_res = [t_num, det, bary_u_num, bary_v_num]` through the
/// triangle-commit sequence inside `body`.
fn analyze_triangle_commit(
    p: &Program,
    body: &HashSet<usize>,
    header: usize,
    def_inst: &HashMap<Value, (usize, usize)>,
    def_phi: &HashMap<Value, (usize, usize)>,
    tri_res: [Value; 4],
) -> Option<TriCommitInfo> {
    let [t_num, det, bary_u_num, bary_v_num] = tri_res;
    let mut inv_dets = HashSet::new();
    let mut t_vals = HashSet::new();
    let mut bary_u_vals = HashSet::new();
    let mut bary_v_vals = HashSet::new();
    let mut front_face_vals = HashSet::new();
    let mut t_min = RqInput::ConstU32(0);
    let mut cull_flags = RAY_FLAG_NONE;

    // Pass 1: find `inv_det = v_rcp_f32(det)` and front-face/cull checks on `det`.
    for &b in body {
        for inst in &p.blocks[b].insts {
            let n = inst.name();
            if n == "v_rcp_f32" && inst.reads.iter().any(|&(_, v)| v == det) {
                if let Some(&(_, w)) = inst.writes.first() {
                    inv_dets.insert(w);
                }
            }
            // Front-face cull check: `v_mul_i32_i24 v, sext(det), cull_sign`
            if n == "v_mul_i32_i24" && inst.reads.iter().any(|&(_, v)| v == det) {
                // Check static cull multiplier inside body (e.g. `v_mul_i32_i24 v28, 1, v28`
                // = +1 => cull back-facing; -1 => cull front-facing).
                cull_flags = detect_cull_flags(p, body, def_inst, inst);
            }
            // Direct front-face boolean from `det`: `v_cmp_ge_f32` / `v_cmp_gt_f32` /
            // `v_cmp_lt_i32(-1, det)` / `v_cndmask_b32` with 0/1.
            if (n.starts_with("v_cmp_ge_f32")
                || n.starts_with("v_cmp_gt_f32")
                || n.starts_with("v_cmp_lt_f32")
                || n.starts_with("v_cmp_le_f32")
                || n.starts_with("v_cmp_lt_i32")
                || n.starts_with("v_cmp_ge_i32"))
                && inst.reads.iter().any(|&(_, v)| v == det)
            {
                for &(_, w) in &inst.writes {
                    front_face_vals.insert(w);
                }
            }
        }
    }
    if inv_dets.is_empty() {
        return None;
    }

    // Pass 2: find `t = t_num * inv_det`, `u = bary_u_num * inv_det`,
    // `v = bary_v_num * inv_det`.
    for &b in body {
        for inst in &p.blocks[b].insts {
            let n = inst.name();
            if matches!(n, "v_mul_f32" | "v_mul_legacy_f32")
                && inst.reads.iter().any(|&(_, v)| inv_dets.contains(&v))
                && let Some(&(_, w)) = inst.writes.first()
            {
                if inst.reads.iter().any(|&(_, v)| v == t_num) {
                    t_vals.insert(w);
                }
                if inst.reads.iter().any(|&(_, v)| v == bary_u_num) {
                    bary_u_vals.insert(w);
                }
                if inst.reads.iter().any(|&(_, v)| v == bary_v_num) {
                    bary_v_vals.insert(w);
                }
            }
        }
    }
    if t_vals.is_empty() {
        return None;
    }

    // Pass 3: find `tMin <= t` and `t < tMax` comparisons and commit masks,
    // plus `v_cndmask_b32` converting `front_face` mask to 0/1 VGPR.
    let mut t_cmp_masks = HashSet::new();
    for &b in body {
        for inst in &p.blocks[b].insts {
            let n = inst.name();
            if n.starts_with("v_cmp_") && inst.reads.iter().any(|&(_, v)| t_vals.contains(&v)) {
                for &(_, w) in &inst.writes {
                    t_cmp_masks.insert(w);
                }
                // Detect `tMin`: `v_cmp_le_f32(tMin, t)` or `v_cmp_ge_f32(t, tMin)`.
                if let Fields::Valu(v) = &inst.fields
                    && v.src.len() >= 2
                {
                    let (src0, src1) = (v.src[0], v.src[1]);
                    let other = if operand_is_in(inst, &src1, &t_vals)
                        && (n.contains("_le_f32") || n.contains("_lt_f32"))
                    {
                        Some(src0)
                    } else if operand_is_in(inst, &src0, &t_vals)
                        && (n.contains("_ge_f32") || n.contains("_gt_f32"))
                    {
                        Some(src1)
                    } else {
                        None
                    };
                    if let Some(op) = other {
                        if let Some(c) = const_bits(&op) {
                            t_min = RqInput::ConstU32(c);
                        } else if let Some(val) = operand_value(inst, &op)
                            && let Some(pre) =
                                resolve_pre_loop(p, body, header, def_inst, def_phi, val, 10)
                        {
                            t_min = RqInput::Value(pre);
                        }
                    }
                }
            }
            if n == "v_cndmask_b32"
                && inst.reads.iter().any(|&(_, v)| front_face_vals.contains(&v))
                && let Some(&(_, w)) = inst.writes.first()
            {
                front_face_vals.insert(w);
            }
        }
    }

    // Propagate commit masks through `s_and_b32` / `s_and_b64` combining `t_cmp_masks`.
    let mut commit_masks = t_cmp_masks.clone();
    let mut changed = true;
    while changed {
        changed = false;
        for &b in body {
            for inst in &p.blocks[b].insts {
                if matches!(inst.name(), "s_and_b32" | "s_and_b64")
                    && inst.reads.iter().any(|&(_, v)| commit_masks.contains(&v))
                {
                    for &(r, w) in &inst.writes {
                        if matches!(r, Reg::S(_)) && commit_masks.insert(w) {
                            changed = true;
                        }
                    }
                }
            }
        }
    }

    // Detect whether `t` is committed into `tMax` (closest hit) vs whether the
    // loop sets a termination sentinel immediately on hit (`ACCEPT_FIRST_HIT`).
    let mut updates_tmax = false;
    for &b in body {
        for inst in &p.blocks[b].insts {
            if inst.name() == "v_cndmask_b32"
                && inst.reads.iter().any(|&(_, v)| t_vals.contains(&v))
                && inst.reads.iter().any(|&(_, v)| commit_masks.contains(&v))
            {
                updates_tmax = true;
            }
        }
    }

    Some(TriCommitInfo {
        t_vals,
        bary_u_vals,
        bary_v_vals,
        front_face_vals,
        commit_masks,
        t_min,
        cull_flags,
        accept_first_hit: !updates_tmax,
    })
}

/// Inspects the cull-sign computation feeding `v_mul_i32_i24 v, sext(det), cull`
/// to distinguish `CULL_BACK_FACING_TRIANGLES` (`+1`), `CULL_FRONT_FACING_TRIANGLES`
/// (`-1`), or `RAY_FLAG_NONE` (`0`).
fn detect_cull_flags(
    p: &Program,
    body: &HashSet<usize>,
    _def_inst: &HashMap<Value, (usize, usize)>,
    _mul_inst: &Inst,
) -> u32 {
    for &b in body {
        for inst in &p.blocks[b].insts {
            if inst.name() == "v_mul_i32_i24"
                && let Fields::Valu(v) = &inst.fields
                && let Some(op) = v.src.first()
            {
                match const_bits(op) {
                    Some(1) => return RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
                    Some(0xFFFF_FFFF) => return RAY_FLAG_CULL_FRONT_FACING_TRIANGLES,
                    _ => {}
                }
            }
        }
    }
    RAY_FLAG_NONE
}

/// Scans `body` for the GPURT instance-node mask check (`and ..., mask << 24`,
/// e.g. `s_and_b32 s0, s0, 0xd5000000` -> `0xD5`). Defaults to `0xFF`.
fn detect_instance_mask(
    p: &Program,
    body: &HashSet<usize>,
    header: usize,
    def_inst: &HashMap<Value, (usize, usize)>,
    def_phi: &HashMap<Value, (usize, usize)>,
) -> RqInput {
    for &b in body {
        for inst in &p.blocks[b].insts {
            let n = inst.name();
            if matches!(n, "s_and_b32" | "v_and_b32") {
                let srcs = match &inst.fields {
                    Fields::Sop { src, .. } => src.as_slice(),
                    Fields::Valu(v) => v.src.as_slice(),
                    _ => continue,
                };
                for op in srcs {
                    if let Operand::Literal(lit) = *op
                        && (lit & 0x00FF_FFFF) == 0
                        && lit != 0
                        && lit != 0x8000_0000
                        && lit != 0xFF00_0000
                    {
                        return RqInput::ConstU32(lit >> 24);
                    }
                    // Also handle dynamic `(mask << 24)` computed before the loop.
                    if let Some(val) = operand_value(inst, op)
                        && let Some(&(db, di)) = def_inst.get(&val)
                        && matches!(
                            p.blocks[db].insts[di].name(),
                            "s_lshl_b32" | "v_lshlrev_b32"
                        )
                    {
                        let sh = &p.blocks[db].insts[di];
                        let has_24 = match &sh.fields {
                            Fields::Sop { src, .. } => {
                                src.iter().any(|o| const_bits(o) == Some(24))
                            }
                            Fields::Valu(v) => v.src.iter().any(|o| const_bits(o) == Some(24)),
                            _ => false,
                        };
                        if has_24
                            && let Some(raw) = first_reg_read(sh)
                            && let Some(pre) =
                                resolve_pre_loop(p, body, header, def_inst, def_phi, raw, 8)
                        {
                            return RqInput::Value(pre);
                        }
                    }
                }
            }
        }
    }
    RqInput::ConstU32(0xFF)
}

/// Collects every SSA value defined inside `body` that is referenced by an
/// instruction or phi outside `body` (ignoring `EXEC` / `SCC` control-flow
/// masks that `Lower` does not carry as data).
fn collect_live_outs(p: &Program, body: &HashSet<usize>, exit: usize) -> Vec<Value> {
    let mut defined_in_body: HashSet<Value> = HashSet::new();
    for &b in body {
        for phi in &p.blocks[b].phis {
            if !matches!(phi.reg, Reg::S(EXEC) | Reg::Scc) && phi.reg != Reg::S(EXEC + 1) {
                defined_in_body.insert(phi.value);
            }
        }
        for inst in &p.blocks[b].insts {
            for &(r, v) in &inst.writes {
                if !matches!(r, Reg::S(EXEC) | Reg::Scc) && r != Reg::S(EXEC + 1) {
                    defined_in_body.insert(v);
                }
            }
        }
    }

    let mut live: Vec<Value> = Vec::new();
    let mut seen: HashSet<Value> = HashSet::new();
    let mut note = |v: Value| {
        if defined_in_body.contains(&v) && seen.insert(v) {
            live.push(v);
        }
    };

    // Check `exit`-block PHIs (for incoming edges from `body`) and all
    // instructions/PHIs outside `body`.
    for (b, blk) in p.blocks.iter().enumerate() {
        if body.contains(&b) {
            continue;
        }
        for phi in &blk.phis {
            if matches!(phi.reg, Reg::S(EXEC) | Reg::Scc) || phi.reg == Reg::S(EXEC + 1) {
                continue;
            }
            for (&pred, &arg) in blk.preds.iter().zip(&phi.args) {
                if b == exit || !body.contains(&pred) {
                    note(arg);
                }
            }
        }
        for inst in &blk.insts {
            for &(r, v) in &inst.reads {
                if !matches!(r, Reg::S(EXEC) | Reg::Scc) && r != Reg::S(EXEC + 1) {
                    note(v);
                }
            }
        }
    }
    live
}

/// Classifies one live-out SSA value `v` produced inside `body` into a
/// `LiveOut` commitment attribute, or returns `None` if unrecognized (causing
/// safe fallback to `emit/intersect.rs`).
#[allow(clippy::too_many_arguments)]
fn classify_live_out(
    p: &Program,
    body: &HashSet<usize>,
    header: usize,
    def_inst: &HashMap<Value, (usize, usize)>,
    def_phi: &HashMap<Value, (usize, usize)>,
    tri: &TriCommitInfo,
    t_max: Value,
    v: Value,
) -> Option<LiveOut> {
    // Collect every non-trivial value written into `v`'s loop-carried value web
    // inside `body`.
    let mut work = VecDeque::from([v]);
    let mut visited = HashSet::from([v]);
    let mut pre_values = HashSet::new();
    let mut commit_sources: Vec<Value> = Vec::new();
    let mut is_iter_counter = false;
    let mut unknown_write = false;

    while let Some(cur) = work.pop_front() {
        if let Some(&(b, pi)) = def_phi.get(&cur) {
            if !body.contains(&b) {
                pre_values.insert(cur);
                continue;
            }
            for (&pred, &arg) in p.blocks[b].preds.iter().zip(&p.blocks[b].phis[pi].args) {
                if !body.contains(&pred) {
                    pre_values.insert(arg);
                } else if visited.insert(arg) {
                    work.push_back(arg);
                }
            }
            continue;
        }
        let Some(&(b, ii)) = def_inst.get(&cur) else {
            pre_values.insert(cur);
            continue;
        };
        if !body.contains(&b) {
            pre_values.insert(cur);
            continue;
        }
        let inst = &p.blocks[b].insts[ii];
        match inst.name() {
            "v_mov_b32" | "s_mov_b32" | "v_readfirstlane_b32" => {
                if let Some(src) = first_reg_read(inst) {
                    if visited.insert(src) {
                        work.push_back(src);
                    }
                } else {
                    unknown_write = true;
                }
            }
            // Watchdog iteration counter: `v_add_nc_u32 v, 1, v`.
            "v_add_nc_u32" | "v_add_u32" | "s_add_u32" | "s_add_i32" => {
                let srcs = match &inst.fields {
                    Fields::Valu(f) => f.src.as_slice(),
                    Fields::Sop { src, .. } => src.as_slice(),
                    _ => &[],
                };
                if srcs.iter().any(|o| const_bits(o) == Some(1)) {
                    is_iter_counter = true;
                    for op in srcs {
                        if let Some(rv) = operand_value(inst, op)
                            && visited.insert(rv)
                        {
                            work.push_back(rv);
                        }
                    }
                } else {
                    unknown_write = true;
                }
            }
            // Triangle commit select: `v_cndmask_b32 dst, old, new, commit_mask`.
            "v_cndmask_b32" => {
                if let Fields::Valu(f) = &inst.fields
                    && f.src.len() >= 3
                    && let Some(mask_v) = operand_value(inst, &f.src[2])
                    && tri.commit_masks.contains(&mask_v)
                {
                    if let Some(old_v) = operand_value(inst, &f.src[0])
                        && visited.insert(old_v)
                    {
                        work.push_back(old_v);
                    }
                    if let Some(new_v) = operand_value(inst, &f.src[1]) {
                        commit_sources.push(new_v);
                    } else if let Some(c) = const_bits(&f.src[1])
                        && c != 0
                    {
                        // Constant non-zero committed status/hit flag (`v_cndmask_b32 dst, 0, 1, hit`).
                        return Some(LiveOut::CommittedHitFlag);
                    }
                } else {
                    unknown_write = true;
                }
            }
            _ => {
                unknown_write = true;
            }
        }
    }

    if unknown_write {
        return None;
    }

    // Pure iteration counter (only `+1` writes, zero commit selects, initialized to 0).
    if is_iter_counter && commit_sources.is_empty() {
        return Some(LiveOut::ConstZero);
    }

    // Unmodified pass-through from pre-loop.
    if commit_sources.is_empty() && pre_values.len() == 1 {
        let pre = *pre_values.iter().next()?;
        return Some(LiveOut::PreLoop(pre));
    }

    // Must have commit select source(s); classify all of them consistently.
    if commit_sources.is_empty() {
        return None;
    }
    let mut resolved: Option<LiveOut> = None;
    for src in commit_sources {
        let kind = classify_commit_source(p, body, header, def_inst, def_phi, tri, src)?;
        match resolved {
            None => resolved = Some(kind),
            Some(prev) if prev == kind => {}
            Some(_) => return None,
        }
    }
    let kind = resolved?;
    // Sanity check: `CommittedT` must fall back to `t_max`.
    if kind == LiveOut::CommittedT && !pre_values.is_empty() {
        let ok = pre_values.iter().all(|&pv| {
            pv == t_max
                || resolve_pre_loop(p, body, header, def_inst, def_phi, pv, 6) == Some(t_max)
        });
        if !ok {
            return None;
        }
    }
    Some(kind)
}

/// Maps the `new` operand of a triangle-commit `v_cndmask_b32` to the matching
/// DXR 1.1 `Committed*` attribute.
fn classify_commit_source(
    p: &Program,
    body: &HashSet<usize>,
    _header: usize,
    def_inst: &HashMap<Value, (usize, usize)>,
    def_phi: &HashMap<Value, (usize, usize)>,
    tri: &TriCommitInfo,
    mut v: Value,
) -> Option<LiveOut> {
    if tri.t_vals.contains(&v) {
        return Some(LiveOut::CommittedT);
    }
    if tri.bary_u_vals.contains(&v) {
        return Some(LiveOut::CommittedBaryU);
    }
    if tri.bary_v_vals.contains(&v) {
        return Some(LiveOut::CommittedBaryV);
    }
    if tri.front_face_vals.contains(&v) {
        return Some(LiveOut::CommittedFrontFace);
    }

    // Follow copies / masks / shifts back to BVH node loads or triangle node pointer.
    let mut masked_24 = false;
    let mut shifted_8 = false;
    for _ in 0..8 {
        if let Some(&(b, ii)) = def_inst.get(&v) {
            let inst = &p.blocks[b].insts[ii];
            let n = inst.name();
            match n {
                "v_mov_b32" | "s_mov_b32" | "v_readfirstlane_b32" => {
                    if let Some(next) = first_reg_read(inst) {
                        v = next;
                        continue;
                    }
                    // `v_mov_b32 v, 1` -> hit flag / status.
                    if let Fields::Valu(f) = &inst.fields
                        && f.src.first().and_then(const_bits) == Some(1)
                    {
                        return Some(LiveOut::CommittedStatus);
                    }
                    return None;
                }
                "v_and_b32" | "s_and_b32" => {
                    let srcs = match &inst.fields {
                        Fields::Valu(f) => f.src.as_slice(),
                        Fields::Sop { src, .. } => src.as_slice(),
                        _ => return None,
                    };
                    if srcs.iter().any(|o| const_bits(o) == Some(0x00FF_FFFF)) {
                        masked_24 = true;
                        v = first_reg_read(inst)?;
                        continue;
                    }
                    return None;
                }
                "v_lshrrev_b32" | "s_lshr_b32" => {
                    let srcs = match &inst.fields {
                        Fields::Valu(f) => f.src.as_slice(),
                        Fields::Sop { src, .. } => src.as_slice(),
                        _ => return None,
                    };
                    if srcs.iter().any(|o| const_bits(o) == Some(8)) {
                        shifted_8 = true;
                        v = first_reg_read(inst)?;
                        continue;
                    }
                    return None;
                }
                // BVH node metadata loads (`global_load_dword` / `s_load_dword*`).
                _ if n.starts_with("global_load") || n.starts_with("s_load") => {
                    return classify_node_meta_load(p, def_inst, inst, v, masked_24, shifted_8);
                }
                _ => {
                    // Any non-zero triangle node pointer written into the hit
                    // node VGPR (`v16 = v30` in Template A, checked only as
                    // `v7 != 0` after the loop!) maps to `CommittedHitFlag`.
                    if body.contains(&b) {
                        return Some(LiveOut::CommittedHitFlag);
                    }
                    return None;
                }
            }
        }
        if let Some(&(b, pi)) = def_phi.get(&v)
            && body.contains(&b)
        {
            // Follow a loop-invariant inner PHI (e.g. `inst_id` loaded in the
            // outer loop and unchanged across inner-loop iterations).
            let phi = &p.blocks[b].phis[pi];
            let unique: Vec<Value> = phi
                .args
                .iter()
                .copied()
                .filter(|&a| a != v)
                .collect();
            if unique.len() == 1 || (!unique.is_empty() && unique.iter().all(|&a| a == unique[0])) {
                v = unique[0];
                continue;
            }
            // Loop-carried triangle node pointer (`v30`) committed into hit flag.
            return Some(LiveOut::CommittedHitFlag);
        }
        break;
    }
    None
}

/// Identifies InstanceNode / TriangleNode metadata dword offsets (`+0x30`,
/// `+0x34`, `+0x38`) for `CommittedInstanceID`, `CommittedInstanceContrib`,
/// `CommittedGeometryIndex`, and `CommittedPrimitiveIndex`.
fn classify_node_meta_load(
    p: &Program,
    def_inst: &HashMap<Value, (usize, usize)>,
    inst: &Inst,
    loaded_val: Value,
    masked_24: bool,
    shifted_8: bool,
) -> Option<LiveOut> {
    // Check explicit offset on `global_load_*` or `s_load_*`, plus OR'd low-byte offset.
    let mut byte_off: i32 = match &inst.fields {
        Fields::Flat { offset, .. } => i32::from(*offset),
        Fields::Smem { offset, data, .. } => {
            let mut off = *offset;
            if let Some(Operand::Scalar { code, dwords }) = *data {
                for k in 0..dwords {
                    if inst.write(Reg::S(code + k)) == Some(loaded_val) {
                        off += i32::from(k) * 4;
                        break;
                    }
                }
            }
            off
        }
        _ => return None,
    };
    if byte_off == 0
        && let Some(addr_v) = first_reg_read(inst)
        && let Some(&(ab, ai)) = def_inst.get(&addr_v)
    {
        let ainst = &p.blocks[ab].insts[ai];
        if matches!(
            ainst.name(),
            "v_or_b32" | "s_or_b32" | "v_add_nc_u32" | "s_add_u32"
        ) {
            let srcs = match &ainst.fields {
                Fields::Valu(f) => f.src.as_slice(),
                Fields::Sop { src, .. } => src.as_slice(),
                _ => &[],
            };
            if let Some(c) = srcs.iter().find_map(const_bits) {
                byte_off = c as i32;
            }
        }
    }

    match (byte_off & 0x7F, masked_24, shifted_8) {
        (0x30, true, false) => Some(LiveOut::CommittedInstanceId),
        (0x34, true, false) => Some(LiveOut::CommittedInstanceContrib),
        (0x30, false, true) => Some(LiveOut::CommittedGeometryIndex),
        (0x34 | 0x38, false, false) => Some(LiveOut::CommittedPrimitiveIndex),
        _ => None,
    }
}

fn const_bits(op: &Operand) -> Option<u32> {
    match *op {
        Operand::Const(c) => crate::resource::inline(c),
        Operand::Literal(x) => Some(x),
        _ => None,
    }
}

fn operand_value(inst: &Inst, op: &Operand) -> Option<Value> {
    match *op {
        Operand::Vector { reg, .. } => inst.read(Reg::V(reg)),
        Operand::Scalar { code, .. } if code != NULL => inst.read(Reg::S(code)),
        _ => None,
    }
}

fn operand_is_in(inst: &Inst, op: &Operand, set: &HashSet<Value>) -> bool {
    operand_value(inst, op).is_some_and(|v| set.contains(&v))
}

impl<'a> Lower<'a> {
    // ── Exact DXIL SM 6.6 RayQuery Intrinsics (verified via dxc.exe 10.0.26100.0) ──

    pub(super) fn rq_allocate(&mut self, blk: usize, const_ray_flags: u32) -> V {
        self.s.flags |= flags::RAYTRACING_TIER_1_1;
        let f = self.s.m.i32(const_ray_flags);
        let i32t = self.s.t.i32;
        self.s
            .op(blk, "allocateRayQuery", Attr::NoUnwind, i32t, 178, &[f])
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn rq_trace_ray_inline(
        &mut self,
        blk: usize,
        q: V,
        tlas: V,
        dyn_flags: V,
        mask: V,
        orig: [V; 3],
        t_min: V,
        dir: [V; 3],
        t_max: V,
    ) {
        self.s.flags |= flags::RAYTRACING_TIER_1_1;
        let void = self.s.t.void;
        self.s.op(
            blk,
            "rayQuery_TraceRayInline",
            Attr::NoUnwind,
            void,
            179,
            &[
                q, tlas, dyn_flags, mask, orig[0], orig[1], orig[2], t_min, dir[0], dir[1], dir[2],
                t_max,
            ],
        );
    }

    pub(super) fn rq_proceed(&mut self, blk: usize, q: V) -> V {
        self.s.flags |= flags::RAYTRACING_TIER_1_1;
        let i1t = self.s.t.i1;
        self.s
            .op(blk, "rayQuery_Proceed.i1", Attr::NoUnwind, i1t, 180, &[q])
    }

    #[allow(dead_code)]
    pub(super) fn rq_abort(&mut self, blk: usize, q: V) {
        let void = self.s.t.void;
        self.s
            .op(blk, "rayQuery_Abort", Attr::NoUnwind, void, 181, &[q]);
    }

    #[allow(dead_code)]
    pub(super) fn rq_commit_non_opaque_triangle_hit(&mut self, blk: usize, q: V) {
        let void = self.s.t.void;
        self.s.op(
            blk,
            "rayQuery_CommitNonOpaqueTriangleHit",
            Attr::NoUnwind,
            void,
            182,
            &[q],
        );
    }

    fn rq_state_i32(&mut self, blk: usize, opcode: u32, q: V) -> V {
        let i32t = self.s.t.i32;
        self.s
            .op(blk, "rayQuery_StateScalar.i32", Attr::ReadOnly, i32t, opcode, &[q])
    }

    fn rq_state_f32(&mut self, blk: usize, opcode: u32, q: V) -> V {
        let f32t = self.s.t.f32;
        self.s
            .op(blk, "rayQuery_StateScalar.f32", Attr::ReadOnly, f32t, opcode, &[q])
    }

    fn rq_state_i1(&mut self, blk: usize, opcode: u32, q: V) -> V {
        let i1t = self.s.t.i1;
        self.s
            .op(blk, "rayQuery_StateScalar.i1", Attr::ReadOnly, i1t, opcode, &[q])
    }

    fn rq_state_vec_f32(&mut self, blk: usize, opcode: u32, q: V, comp: u8) -> V {
        let f32t = self.s.t.f32;
        let c = self.s.m.i8(comp);
        self.s.op(
            blk,
            "rayQuery_StateVector.f32",
            Attr::ReadOnly,
            f32t,
            opcode,
            &[q, c],
        )
    }

    pub(super) fn rq_committed_status(&mut self, blk: usize, q: V) -> V {
        self.rq_state_i32(blk, 184, q)
    }
    #[allow(dead_code)]
    pub(super) fn rq_candidate_type(&mut self, blk: usize, q: V) -> V {
        self.rq_state_i32(blk, 185, q)
    }
    #[allow(dead_code)]
    pub(super) fn rq_candidate_front_face(&mut self, blk: usize, q: V) -> V {
        self.rq_state_i1(blk, 191, q)
    }
    pub(super) fn rq_committed_front_face(&mut self, blk: usize, q: V) -> V {
        self.rq_state_i1(blk, 192, q)
    }
    #[allow(dead_code)]
    pub(super) fn rq_candidate_bary(&mut self, blk: usize, q: V, comp: u8) -> V {
        self.rq_state_vec_f32(blk, 193, q, comp)
    }
    pub(super) fn rq_committed_bary(&mut self, blk: usize, q: V, comp: u8) -> V {
        self.rq_state_vec_f32(blk, 194, q, comp)
    }
    #[allow(dead_code)]
    pub(super) fn rq_candidate_ray_t(&mut self, blk: usize, q: V) -> V {
        self.rq_state_f32(blk, 199, q)
    }
    pub(super) fn rq_committed_ray_t(&mut self, blk: usize, q: V) -> V {
        self.rq_state_f32(blk, 200, q)
    }
    #[allow(dead_code)]
    pub(super) fn rq_candidate_instance_index(&mut self, blk: usize, q: V) -> V {
        self.rq_state_i32(blk, 201, q)
    }
    #[allow(dead_code)]
    pub(super) fn rq_candidate_instance_id(&mut self, blk: usize, q: V) -> V {
        self.rq_state_i32(blk, 202, q)
    }
    #[allow(dead_code)]
    pub(super) fn rq_candidate_geometry_index(&mut self, blk: usize, q: V) -> V {
        self.rq_state_i32(blk, 203, q)
    }
    #[allow(dead_code)]
    pub(super) fn rq_candidate_primitive_index(&mut self, blk: usize, q: V) -> V {
        self.rq_state_i32(blk, 204, q)
    }
    pub(super) fn rq_committed_instance_index(&mut self, blk: usize, q: V) -> V {
        self.rq_state_i32(blk, 207, q)
    }
    pub(super) fn rq_committed_instance_id(&mut self, blk: usize, q: V) -> V {
        self.rq_state_i32(blk, 208, q)
    }
    pub(super) fn rq_committed_geometry_index(&mut self, blk: usize, q: V) -> V {
        self.rq_state_i32(blk, 209, q)
    }
    pub(super) fn rq_committed_primitive_index(&mut self, blk: usize, q: V) -> V {
        self.rq_state_i32(blk, 210, q)
    }
    #[allow(dead_code)]
    pub(super) fn rq_candidate_instance_contrib(&mut self, blk: usize, q: V) -> V {
        self.rq_state_i32(blk, 214, q)
    }
    pub(super) fn rq_committed_instance_contrib(&mut self, blk: usize, q: V) -> V {
        self.rq_state_i32(blk, 215, q)
    }

    /// Scans root SRV `t<SLOT_TLAS_MAP>` (`t251`, 64 × 16-byte entries
    /// `[tlas_va_shr8_lo, tlas_va_shr8_hi, host_tlas_srv_slot, valid]`) for
    /// `(va_lo, va_hi)`. Returns `(after_block, matched_slot, found_i1)`.
    fn rq_lookup_tlas_slot(&mut self, entry_blk: usize, va_lo: V, va_hi: V) -> (usize, V, V) {
        use super::super::shader::{SLOT_TLAS_MAP, TLAS_MAP_ENTRIES};

        let i32t = self.s.t.i32;
        let i1t = self.s.t.i1;
        let map_h = self.s.root_srv_handle(SLOT_TLAS_MAP);
        let zero = self.s.m.i32(0);
        let one = self.s.m.i32(1);
        let four = self.s.m.i32(4);
        let limit = self.s.m.i32(TLAS_MAP_ENTRIES);
        let false_b = self.s.m.bool(false);
        let iu = self.s.m.undef(i32t);
        let mask_f = self.s.m.i8(0xF);
        let align4 = self.s.m.i32(4);

        let loop_blk = self.s.m.new_block();
        let done_blk = self.s.m.new_block();
        self.s.m.terminate(entry_blk, Term::Br(loop_blk));

        let phi_i = self.s.m.phi(loop_blk, i32t);
        let phi_slot = self.s.m.phi(loop_blk, i32t);
        let phi_found = self.s.m.phi(loop_blk, i1t);

        let byte_off = self.s.m.bin(loop_blk, Bin::Shl, phi_i, four);
        let raw = self.s.op(
            loop_blk,
            "rawBufferLoad.f32",
            Attr::ReadOnly,
            self.s.t.res_ret,
            139,
            &[map_h, byte_off, iu, mask_f, align4],
        );
        let (f0, f1, f2, f3) = (
            self.s.m.extract(loop_blk, raw, 0),
            self.s.m.extract(loop_blk, raw, 1),
            self.s.m.extract(loop_blk, raw, 2),
            self.s.m.extract(loop_blk, raw, 3),
        );
        let e_lo = self.s.m.cast(loop_blk, Cast::Bitcast, f0, i32t);
        let e_hi = self.s.m.cast(loop_blk, Cast::Bitcast, f1, i32t);
        let e_slot = self.s.m.cast(loop_blk, Cast::Bitcast, f2, i32t);
        let e_valid = self.s.m.cast(loop_blk, Cast::Bitcast, f3, i32t);

        let ok_v = self.s.m.cmp(loop_blk, pred::NE, e_valid, zero);
        let ok_lo = self.s.m.cmp(loop_blk, pred::EQ, e_lo, va_lo);
        let ok_hi = self.s.m.cmp(loop_blk, pred::EQ, e_hi, va_hi);
        let ok_a = self.s.m.bin(loop_blk, Bin::And, ok_v, ok_lo);
        let hit = self.s.m.bin(loop_blk, Bin::And, ok_a, ok_hi);

        let next_slot = self.s.m.select(loop_blk, hit, e_slot, phi_slot);
        let next_found = self.s.m.bin(loop_blk, Bin::Or, phi_found, hit);
        let next_i = self.s.m.bin(loop_blk, Bin::Add, phi_i, one);
        let more = self.s.m.cmp(loop_blk, pred::IULT, next_i, limit);
        let true_b = self.s.m.bool(true);
        let not_found = self.s.m.bin(loop_blk, Bin::Xor, next_found, true_b);
        let cont = self.s.m.bin(loop_blk, Bin::And, more, not_found);

        self.s
            .m
            .set_phi(phi_i, vec![(zero, entry_blk), (next_i, loop_blk)]);
        self.s
            .m
            .set_phi(phi_slot, vec![(zero, entry_blk), (next_slot, loop_blk)]);
        self.s
            .m
            .set_phi(phi_found, vec![(false_b, entry_blk), (next_found, loop_blk)]);
        self.s
            .m
            .terminate(loop_blk, Term::CondBr(cont, loop_blk, done_blk));
        (done_blk, next_slot, next_found)
    }

    /// Resolves an `RqInput` to an `i32` DXIL value in `self.cur`.
    fn rq_eval_i32(&mut self, input: RqInput) -> Result<V, Stop> {
        match input {
            RqInput::ConstU32(c) => Ok(self.s.m.i32(c)),
            RqInput::Value(v) => self.i32_of(v).map_err(|w| unsupported(0, w)),
        }
    }

    /// Lowers a recognized pure-opaque traversal loop at `header` directly to
    /// `SLOT_TLAS_MAP` lookup -> `createHandleFromHeap(tlas_slot)` ->
    /// `allocateRayQuery` -> `TraceRayInline` -> `while (Proceed()) {}` ->
    /// `Committed*` live-outs, branching directly to `self.block[rec.exit]`.
    pub(super) fn emit_rayquery_loop(&mut self, header: usize) -> Result<(), Stop> {
        let rec = self
            .rq_plan
            .by_header
            .get(&header)
            .cloned()
            .expect("recognized rayquery loop");
        let i32t = self.s.t.i32;
        let f32t = self.s.t.f32;
        let zero_i = self.s.m.i32(0);
        let one_i = self.s.m.i32(1);

        // Evaluate all pre-loop ray inputs in the header entry block.
        let va_lo = self
            .i32_of(rec.tlas_va_shr8_lo)
            .map_err(|w| unsupported(0, w))?;
        let va_hi = self.rq_eval_i32(rec.tlas_va_shr8_hi)?;
        let mask = self.rq_eval_i32(rec.instance_mask)?;
        let t_min_i = self.rq_eval_i32(rec.t_min)?;
        let t_max_i = self.i32_of(rec.t_max).map_err(|w| unsupported(0, w))?;

        let mut orig_f = [zero_i; 3];
        let mut dir_f = [zero_i; 3];
        for k in 0..3 {
            let oi = self.i32_of(rec.origin[k]).map_err(|w| unsupported(0, w))?;
            let di = self.i32_of(rec.dir[k]).map_err(|w| unsupported(0, w))?;
            orig_f[k] = self.s.m.cast(self.cur, Cast::Bitcast, oi, f32t);
            dir_f[k] = self.s.m.cast(self.cur, Cast::Bitcast, di, f32t);
        }
        let t_min_f = self.s.m.cast(self.cur, Cast::Bitcast, t_min_i, f32t);
        let t_max_f = self.s.m.cast(self.cur, Cast::Bitcast, t_max_i, f32t);

        // Pre-resolve any `PreLoop(v)` pass-through values before branching.
        let mut pre_resolved: HashMap<Value, V> = HashMap::new();
        for &lo_kind in rec.live_outs.values() {
            if let LiveOut::PreLoop(pv) = lo_kind {
                let x = self.i32_of(pv).map_err(|w| unsupported(0, w))?;
                pre_resolved.insert(pv, x);
            }
        }

        // 1. Look up `(va_lo, va_hi)` in `SLOT_TLAS_MAP` (`t251`).
        let (check_blk, tlas_slot, found) = self.rq_lookup_tlas_slot(self.cur, va_lo, va_hi);
        let trace_blk = self.s.m.new_block();
        let proceed_blk = self.s.m.new_block();
        let commit_blk = self.s.m.new_block();
        let join_blk = self.s.m.new_block();

        self.s
            .m
            .terminate(check_blk, Term::CondBr(found, trace_blk, join_blk));

        // 2. Hardware DXR 1.1 `RayQuery` traversal on valid host TLAS handle.
        let tlas_h = self
            .s
            .heap_handle(trace_blk, tlas_slot, false, false, RT_AS_PROPS);
        let q = self.rq_allocate(trace_blk, rec.const_ray_flags);
        self.rq_trace_ray_inline(
            trace_blk,
            q,
            tlas_h,
            zero_i,
            mask,
            orig_f,
            t_min_f,
            dir_f,
            t_max_f,
        );
        self.s.m.terminate(trace_blk, Term::Br(proceed_blk));

        // 3. `while (q.Proceed()) {}` (Phase 5B-1 opaque-only loop; Phase 5B-2
        //    will emit the lifted non-opaque candidate body on `proceed == true`).
        let more = self.rq_proceed(proceed_blk, q);
        self.s
            .m
            .terminate(proceed_blk, Term::CondBr(more, proceed_blk, commit_blk));

        // 4. Read only the `Committed*` attributes required by `rec.live_outs`.
        let status = self.rq_committed_status(commit_blk, q);
        let hit = self.s.m.cmp(commit_blk, pred::EQ, status, one_i);

        let mut needed: Vec<(Value, LiveOut)> = rec.live_outs.iter().map(|(&v, &k)| (v, k)).collect();
        needed.sort_by_key(|&(v, _)| v.0);

        let mut produced: Vec<(Value, V, V)> = Vec::with_capacity(needed.len());
        for (val_id, kind) in needed {
            let (on_trace, on_miss) = match kind {
                LiveOut::CommittedT => {
                    let tf = self.rq_committed_ray_t(commit_blk, q);
                    let ti = self.s.m.cast(commit_blk, Cast::Bitcast, tf, i32t);
                    let sel = self.s.m.select(commit_blk, hit, ti, t_max_i);
                    (sel, t_max_i)
                }
                LiveOut::CommittedHitFlag => {
                    let sel = self.s.m.select(commit_blk, hit, one_i, zero_i);
                    (sel, zero_i)
                }
                LiveOut::CommittedStatus => (status, zero_i),
                LiveOut::CommittedBaryU => {
                    let bf = self.rq_committed_bary(commit_blk, q, 0);
                    let bi = self.s.m.cast(commit_blk, Cast::Bitcast, bf, i32t);
                    (self.s.m.select(commit_blk, hit, bi, zero_i), zero_i)
                }
                LiveOut::CommittedBaryV => {
                    let bf = self.rq_committed_bary(commit_blk, q, 1);
                    let bi = self.s.m.cast(commit_blk, Cast::Bitcast, bf, i32t);
                    (self.s.m.select(commit_blk, hit, bi, zero_i), zero_i)
                }
                LiveOut::CommittedFrontFace => {
                    let ff = self.rq_committed_front_face(commit_blk, q);
                    let ffi = self.s.m.cast(commit_blk, Cast::ZExt, ff, i32t);
                    (self.s.m.select(commit_blk, hit, ffi, zero_i), zero_i)
                }
                LiveOut::CommittedInstanceId => {
                    let x = self.rq_committed_instance_id(commit_blk, q);
                    (self.s.m.select(commit_blk, hit, x, zero_i), zero_i)
                }
                LiveOut::CommittedInstanceIndex => {
                    let x = self.rq_committed_instance_index(commit_blk, q);
                    (self.s.m.select(commit_blk, hit, x, zero_i), zero_i)
                }
                LiveOut::CommittedInstanceContrib => {
                    let x = self.rq_committed_instance_contrib(commit_blk, q);
                    (self.s.m.select(commit_blk, hit, x, zero_i), zero_i)
                }
                LiveOut::CommittedGeometryIndex => {
                    let x = self.rq_committed_geometry_index(commit_blk, q);
                    (self.s.m.select(commit_blk, hit, x, zero_i), zero_i)
                }
                LiveOut::CommittedPrimitiveIndex => {
                    let x = self.rq_committed_primitive_index(commit_blk, q);
                    (self.s.m.select(commit_blk, hit, x, zero_i), zero_i)
                }
                LiveOut::ConstZero => (zero_i, zero_i),
                LiveOut::PreLoop(pv) => {
                    let x = pre_resolved[&pv];
                    (x, x)
                }
            };
            produced.push((val_id, on_trace, on_miss));
        }
        self.s.m.terminate(commit_blk, Term::Br(join_blk));

        // 5. Merge `commit_blk` (valid TLAS) and `check_blk` (unmapped TLAS miss)
        //    in `join_blk`, then jump directly to `self.block[rec.exit]`.
        for (val_id, on_trace, on_miss) in produced {
            let merged = if on_trace == on_miss {
                on_trace
            } else {
                let phi = self.s.m.phi(join_blk, i32t);
                self.s
                    .m
                    .set_phi(phi, vec![(on_trace, commit_blk), (on_miss, check_blk)]);
                phi
            };
            self.vals.insert(val_id, Val::I32(merged));
        }

        let exit_dxil = self.block[rec.exit].expect("lifted loop exit reachable");
        self.s.m.terminate(join_blk, Term::Br(exit_dxil));
        self.cur = join_blk;
        self.tail[header] = Some(join_blk);
        self.rq_plan.lowered_join.insert(header, join_blk);

        // If `header` had an iteration-limit counter phi pre-created in `Lower::run`,
        // neutralize it cleanly so `fill_phis` does not see an empty phi.
        if let Some((ctr_phi, _)) = self.counters.remove(&header) {
            let mut inc = Vec::new();
            for &pred in &self.x.p.blocks[header].preds {
                if !rec.body.contains(&pred)
                    && let Some(pb) = self.tail[pred]
                {
                    let pb = self.edges.get(&(pred, header)).copied().unwrap_or(pb);
                    inc.push((zero_i, pb));
                }
            }
            if !inc.is_empty() {
                self.s.m.set_phi(ctr_phi, inc);
            }
        }
        Ok(())
    }
}
