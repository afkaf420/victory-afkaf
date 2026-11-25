//! Backward SGPR whole-liveness analysis for Wave64-on-Wave32 lowering.
//!
//! On RDNA2 Wave64, 64-bit lane masks live in pairs of SGPRs (`vcc`, `exec`,
//! saved masks). When a Wave64 program runs as paired 32-lane host waves,
//! materialising a mask as a 64-bit integer requires a cross-half exchange.
//! Masks that are only consumed per lane (EXEC guards, structurised branches,
//! `v_cndmask_b32` selectors, `v_*_co_ci_*` carries, mask-to-mask SOP ops, or
//! dead copies) never need cross-half materialisation.
//!
//! [`analyze`] runs a backward fixpoint over [`Program`] and [`Structured`] to
//! find every tracked lane-mask SSA value that is genuinely live as whole-wave
//! 64-bit data. Any mask outside [`WholeLive::reads_whole`] stays half-local
//! with zero cross-half exchanges.

use std::collections::{HashMap, HashSet};

use crate::gfx10::{Fields, NULL, Operand};
use crate::ir::{EXEC, Exit, Inst, M0, Program, Reg, Value};
use crate::structure::{Atom, Bool, LaneExit, Structured};

/// Result of backward SGPR whole-liveness analysis.
#[derive(Debug, Default)]
pub(super) struct WholeLive {
    /// Tracked lane-mask SSA values (both low and high dwords of each pair)
    /// that have a live whole-wave data consumer.
    pub reads_whole: HashSet<Value>,
    /// Instruction sites `(block, inst)` that perform a live whole-wave read
    /// of a tracked lane mask.
    pub mask_data_sites: HashSet<(usize, usize)>,
}

impl WholeLive {
    /// Whether instruction `i` in block `b` performs a live whole-wave read of
    /// a tracked lane mask.
    pub fn reads_mask_as_data(&self, b: usize, i: usize) -> bool {
        self.mask_data_sites.contains(&(b, i))
    }
}

#[derive(Clone, Copy)]
enum Def {
    Inst(usize, usize),
    Phi(usize, usize),
}

fn is_mask(st: &Structured, v: Value) -> bool {
    st.masks.contains_key(&v) || st.highs.contains_key(&v)
}

fn has_effect(inst: &Inst) -> bool {
    let name = inst.name();
    match &inst.fields {
        Fields::Exp { .. } => true,
        Fields::Buffer { .. } | Fields::Image { .. } | Fields::Flat { .. } => {
            name.contains("store") || name.contains("atomic") || name.starts_with("image_bvh")
        }
        Fields::Ds { .. } => {
            !(name.starts_with("ds_read")
                || name.starts_with("ds_swizzle")
                || name.contains("permute"))
        }
        Fields::Smem { .. } => {
            name.contains("store")
                || name.contains("discard")
                || matches!(name, "s_dcache_wb" | "s_dcache_inv" | "s_gl1_inv")
        }
        Fields::Sopp { .. } => matches!(name, "s_sendmsg" | "s_sendmsghalt"),
        Fields::Sop { .. } | Fields::Valu(_) | Fields::Vintrp { .. } => false,
    }
}

/// Whether `inst` writes a tracked lane mask to an SGPR (so its mask-typed
/// inputs remain per-lane masks rather than whole-wave numbers).
fn writes_tracked_mask(st: &Structured, inst: &Inst) -> bool {
    matches!(inst.fields, Fields::Sop { .. })
        && inst
            .writes
            .iter()
            .any(|&(r, w)| matches!(r, Reg::S(_)) && is_mask(st, w))
}

/// Whether `inst` is a non-kill partial SGPR write whose old destination bits
/// stay live whenever the result is live.
fn is_partial_scalar_write(inst: &Inst) -> bool {
    let name = inst.name();
    name.starts_with("s_cmov")
        || name.starts_with("s_bitset")
        || name.starts_with("s_movrel")
        || matches!(name, "s_addk_i32" | "s_mulk_i32")
}

/// Collects every tracked lane-mask SSA value that `inst` reads as whole-wave
/// 64-bit data (as opposed to per-lane boolean use).
fn whole_mask_reads(st: &Structured, inst: &Inst, out: &mut Vec<Value>) {
    let mut push_scalar = |op: &Operand| {
        if let Operand::Scalar { code, dwords } = *op
            && code != NULL
        {
            for k in 0..dwords {
                if let Some(v) = inst.read(Reg::S(code + k))
                    && is_mask(st, v)
                {
                    out.push(v);
                }
            }
        }
    };
    match &inst.fields {
        Fields::Sop { dst, src, .. } => {
            if writes_tracked_mask(st, inst) {
                return;
            }
            for op in src {
                push_scalar(op);
            }
            if is_partial_scalar_write(inst)
                && let Some(d) = dst
            {
                push_scalar(d);
            }
        }
        Fields::Valu(v) => {
            let name = inst.name();
            let lane_mask_src = if name == "v_cndmask_b32" || name.contains("_co_ci_") {
                Some(2usize)
            } else {
                None
            };
            for (idx, op) in v.src.iter().enumerate() {
                if lane_mask_src == Some(idx) {
                    continue;
                }
                push_scalar(op);
            }
        }
        Fields::Smem {
            base,
            soffset,
            data,
            ..
        } => {
            push_scalar(base);
            push_scalar(soffset);
            if inst.name().contains("store")
                && let Some(d) = data
            {
                push_scalar(d);
            }
        }
        Fields::Buffer { rsrc, soffset, .. } => {
            push_scalar(rsrc);
            push_scalar(soffset);
        }
        Fields::Image { rsrc, sampler, .. } => {
            push_scalar(rsrc);
            if let Some(s) = sampler {
                push_scalar(s);
            }
        }
        Fields::Flat { saddr, .. } => {
            if let Some(s) = saddr {
                push_scalar(s);
            }
        }
        Fields::Sopp { .. } => {
            if matches!(inst.name(), "s_sendmsg" | "s_sendmsghalt")
                && let Some(v) = inst.read(Reg::S(M0))
                && is_mask(st, v)
            {
                out.push(v);
            }
        }
        Fields::Ds { .. } | Fields::Exp { .. } | Fields::Vintrp { .. } => {}
    }
}

/// Computes backward SGPR whole-liveness over `p` and `st`.
pub(super) fn analyze(p: &Program, st: &Structured) -> WholeLive {
    // Map each mask low dword to its high dword (if wave64).
    let mut lo_to_hi: HashMap<Value, Value> = HashMap::new();
    for (&hi, &lo) in &st.highs {
        lo_to_hi.insert(lo, hi);
    }

    // Build definition lookup and high-partner lookup for non-mask pairs.
    let mut def: HashMap<Value, Def> = HashMap::new();
    for (b, block) in p.blocks.iter().enumerate() {
        for (k, phi) in block.phis.iter().enumerate() {
            def.insert(phi.value, Def::Phi(b, k));
        }
        for (i, inst) in block.insts.iter().enumerate() {
            for &(_, w) in &inst.writes {
                def.insert(w, Def::Inst(b, i));
            }
        }
    }

    // Conservative fallback when an indirect exit (`s_setpc_b64`) is present:
    // any live mask could be read past the handover.
    if p.blocks.iter().any(|b| b.exit == Exit::Indirect) {
        let mut out = WholeLive::default();
        for &lo in st.masks.keys() {
            out.reads_whole.insert(lo);
            if let Some(&hi) = lo_to_hi.get(&lo) {
                out.reads_whole.insert(hi);
            }
        }
        for (b, block) in p.blocks.iter().enumerate() {
            let mut tmp = Vec::new();
            for (i, inst) in block.insts.iter().enumerate() {
                tmp.clear();
                whole_mask_reads(st, inst, &mut tmp);
                if !tmp.is_empty() {
                    out.mask_data_sites.insert((b, i));
                }
            }
        }
        return out;
    }

    let mut live_vals: HashSet<Value> = HashSet::new();
    let mut live_insts: HashSet<(usize, usize)> = HashSet::new();
    let mut live_phis: HashSet<(usize, usize)> = HashSet::new();
    let mut reads_whole: HashSet<Value> = HashSet::new();
    let mut mask_data_sites: HashSet<(usize, usize)> = HashSet::new();
    let mut seen_bools: HashSet<Bool> = HashSet::new();
    let mut seen_phi_masks: HashSet<Value> = HashSet::new();

    let mut val_work: Vec<Value> = Vec::new();
    let mut inst_work: Vec<(usize, usize)> = Vec::new();
    let mut bool_work: Vec<Bool> = Vec::new();

    let pair_high = |v: Value| -> Option<Value> {
        if !p.wave64 {
            return None;
        }
        let next_reg = |reg: Reg| match reg {
            Reg::S(c) => Some(Reg::S(c + 1)),
            _ => None,
        };
        if let Some(&(r, _)) = p.inputs.iter().find(|x| x.1 == v) {
            let h = next_reg(r)?;
            return p.inputs.iter().find(|x| x.0 == h).map(|x| x.1);
        }
        match def.get(&v).copied()? {
            Def::Phi(b, k) => {
                let h = next_reg(p.blocks[b].phis[k].reg)?;
                p.blocks[b]
                    .phis
                    .iter()
                    .find(|q| q.reg == h)
                    .map(|q| q.value)
            }
            Def::Inst(b, i) => {
                let inst = &p.blocks[b].insts[i];
                let &(r, _) = inst.writes.iter().find(|w| w.1 == v)?;
                let h = next_reg(r)?;
                inst.write(h)
            }
        }
    };

    let mark_val = |v: Value, live: &mut HashSet<Value>, work: &mut Vec<Value>| {
        if live.insert(v) {
            work.push(v);
        }
    };

    let mark_whole = |v: Value,
                      st: &Structured,
                      lo_to_hi: &HashMap<Value, Value>,
                      reads_whole: &mut HashSet<Value>,
                      bool_work: &mut Vec<Bool>,
                      live: &mut HashSet<Value>,
                      val_work: &mut Vec<Value>| {
        let lo = st.highs.get(&v).copied().unwrap_or(v);
        let inserted = reads_whole.insert(lo);
        if let Some(&hi) = lo_to_hi.get(&lo) {
            reads_whole.insert(hi);
        }
        if inserted {
            if let Some(&m) = st.masks.get(&lo) {
                bool_work.push(m);
            }
            mark_val(lo, live, val_work);
            if let Some(&hi) = lo_to_hi.get(&lo) {
                mark_val(hi, live, val_work);
            }
        }
    };

    // Seed observable roots: branch exits and side-effecting instructions.
    for (b, block) in p.blocks.iter().enumerate() {
        if let Some(LaneExit::Branch { when, .. }) = st.exits.get(b).copied() {
            bool_work.push(when);
        }
        for (i, inst) in block.insts.iter().enumerate() {
            if has_effect(inst) && live_insts.insert((b, i)) {
                inst_work.push((b, i));
            }
        }
    }

    let mut mask_reads = Vec::new();
    while !val_work.is_empty() || !inst_work.is_empty() || !bool_work.is_empty() {
        // 1. Expand structurised booleans into their underlying SSA atoms.
        while let Some(bm) = bool_work.pop() {
            if !seen_bools.insert(bm) {
                continue;
            }
            for atom in st.bools.support(bm) {
                match atom {
                    Atom::Lane(v) | Atom::Uniform(v) | Atom::Quad(v) => {
                        mark_val(v, &mut live_vals, &mut val_work);
                    }
                    Atom::Phi(id) => {
                        if seen_phi_masks.insert(id) {
                            if let Some(edges) = st.phi_masks.get(&id) {
                                bool_work.extend(edges.iter().copied());
                            }
                        }
                        mark_val(id, &mut live_vals, &mut val_work);
                    }
                    Atom::Opaque(v) => {
                        mark_val(v, &mut live_vals, &mut val_work);
                        if let Some(hi) = pair_high(v) {
                            mark_val(hi, &mut live_vals, &mut val_work);
                        }
                        if is_mask(st, v) {
                            mark_whole(
                                v,
                                st,
                                &lo_to_hi,
                                &mut reads_whole,
                                &mut bool_work,
                                &mut live_vals,
                                &mut val_work,
                            );
                        }
                    }
                    Atom::Constant(_) => {}
                }
            }
        }

        // 2. Follow live SSA values back to their defining instruction or phi.
        while let Some(v) = val_work.pop() {
            match def.get(&v).copied() {
                Some(Def::Inst(b, i)) => {
                    if live_insts.insert((b, i)) {
                        inst_work.push((b, i));
                    }
                }
                Some(Def::Phi(b, k)) => {
                    let phi = &p.blocks[b].phis[k];
                    if live_phis.insert((b, k)) {
                        let phi_is_mask = is_mask(st, phi.value);
                        for &arg in &phi.args {
                            mark_val(arg, &mut live_vals, &mut val_work);
                            if phi.reg == Reg::Scc {
                                if let Some(&m) = st.scc_some.get(&arg) {
                                    bool_work.push(m);
                                }
                            } else if !phi_is_mask && is_mask(st, arg) {
                                // A data phi joining a lane mask with numbers
                                // materialises the mask via `mask_dword`.
                                mark_whole(
                                    arg,
                                    st,
                                    &lo_to_hi,
                                    &mut reads_whole,
                                    &mut bool_work,
                                    &mut live_vals,
                                    &mut val_work,
                                );
                            }
                        }
                        if phi_is_mask {
                            let lo = st.highs.get(&phi.value).copied().unwrap_or(phi.value);
                            if seen_phi_masks.insert(lo)
                                && let Some(edges) = st.phi_masks.get(&lo)
                            {
                                bool_work.extend(edges.iter().copied());
                            }
                        }
                    }
                }
                None => {}
            }
        }

        // 3. Process live instructions: guards, operand liveness, and whole reads.
        while let Some((b, i)) = inst_work.pop() {
            if let Some(g) = st.guards.get(b).and_then(|v| v.get(i)).copied().flatten() {
                bool_work.push(g);
            }
            let inst = &p.blocks[b].insts[i];
            mask_reads.clear();
            whole_mask_reads(st, inst, &mut mask_reads);
            if !mask_reads.is_empty() {
                mask_data_sites.insert((b, i));
                for &mv in &mask_reads {
                    mark_whole(
                        mv,
                        st,
                        &lo_to_hi,
                        &mut reads_whole,
                        &mut bool_work,
                        &mut live_vals,
                        &mut val_work,
                    );
                }
            }
            // Mark structurised mask inputs for lane-local VALU consumers
            // (`v_cndmask_b32`, `v_*_co_ci_*`) and mask-producing SOPs so their
            // underlying compares stay live.
            if let Fields::Valu(v) = &inst.fields {
                let name = inst.name();
                if (name == "v_cndmask_b32" || name.contains("_co_ci_"))
                    && let Some(&Operand::Scalar { code, .. }) = v.src.get(2)
                    && code != NULL
                    && let Some(mv) = inst.read(Reg::S(code))
                {
                    let lo = st.highs.get(&mv).copied().unwrap_or(mv);
                    if let Some(&m) = st.masks.get(&lo) {
                        bool_work.push(m);
                    }
                }
            }
            // Propagate liveness to all non-EXEC register reads (including
            // non-kill partial-write destination reads and SDWA preserved dsts).
            for &(r, rv) in &inst.reads {
                if matches!(r, Reg::S(c) if c == EXEC || c == EXEC + 1) {
                    continue;
                }
                if r == Reg::Scc
                    && let Some(&m) = st.scc_some.get(&rv)
                {
                    bool_work.push(m);
                }
                mark_val(rv, &mut live_vals, &mut val_work);
            }
        }
    }

    WholeLive {
        reads_whole,
        mask_data_sites,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ir, structure};

    const S_ENDPGM: u32 = 0xBF81_0000;

    fn build(words: &[u32]) -> (Program, Structured, WholeLive) {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let p = ir::build(&code, true, 16).expect("ir::build");
        let st = structure::structure(&p).expect("structure");
        let wl = analyze(&p, &st);
        (p, st, wl)
    }

    #[test]
    fn lane_local_mask_stays_half_local() {
        // v_cmp_gt_u32 vcc, s0, v0
        // v_cndmask_b32 v1, v2, v3, vcc
        // buffer_store_dword v1, off, s[8:11], 0
        // s_endpgm
        let words = [
            0x7D88_0000, // v_cmp_gt_u32 vcc, s0, v0
            0x0202_0702, // v_cndmask_b32 v1, v2, v3, vcc (VOP2 opcode 1: v1, v2, v3)
            0x38 << 26 | 0x1C << 18,
            0x80 << 24 | 2 << 16 | 1 << 8, // buffer_store_dword v1, off, s[8:11], 0
            S_ENDPGM,
        ];
        let (_p, st, wl) = build(&words);
        assert!(!st.masks.is_empty());
        assert!(
            wl.reads_whole.is_empty(),
            "lane-local v_cndmask selector should not require whole-wave exchange"
        );
        assert!(wl.mask_data_sites.is_empty());
    }

    #[test]
    fn dead_mask_data_read_is_not_whole_live() {
        // v_cmp_gt_u32 vcc, s0, v0
        // s_bcnt1_i32_b64 s6, vcc  (result s6 is never used -> dead!)
        // s_endpgm
        let words = [
            0x7D88_0000, // v_cmp_gt_u32 vcc, s0, v0
            0xBE86_0B6A, // s_bcnt1_i32_b64 s6, vcc (SOP1 op 11)
            S_ENDPGM,
        ];
        let (_p, _st, wl) = build(&words);
        assert!(
            wl.reads_whole.is_empty(),
            "dead s_bcnt1 of mask must not force whole-wave exchange"
        );
        assert!(wl.mask_data_sites.is_empty());
    }

    #[test]
    fn live_mask_data_read_marks_mask_whole_live() {
        // v_cmp_gt_u32 vcc, s0, v0
        // s_bcnt1_i32_b64 s6, vcc
        // v_mov_b32 v1, s6
        // buffer_store_dword v1, off, s[8:11], 0
        // s_endpgm
        let words = [
            0x7D88_0000, // v_cmp_gt_u32 vcc, s0, v0
            0xBE86_0B6A, // s_bcnt1_i32_b64 s6, vcc
            0x7E02_0206, // v_mov_b32 v1, s6
            0x38 << 26 | 0x1C << 18,
            0x80 << 24 | 2 << 16 | 1 << 8, // buffer_store_dword v1, off, s[8:11], 0
            S_ENDPGM,
        ];
        let (p, _st, wl) = build(&words);
        let vcc_lo = p.blocks[0].insts[0].write(Reg::S(106)).expect("vcc_lo");
        let vcc_hi = p.blocks[0].insts[0].write(Reg::S(107)).expect("vcc_hi");
        assert!(wl.reads_whole.contains(&vcc_lo));
        assert!(wl.reads_whole.contains(&vcc_hi));
        assert!(wl.reads_mask_as_data(0, 1));
    }

    #[test]
    fn partial_scalar_write_preserves_whole_liveness() {
        // v_cmp_gt_u32 vcc, s0, v0
        // s_mov_b32 s6, vcc_lo          (reads mask low dword as data)
        // s_bitset1_b32 s6, 3           (partial write: keeps bits of s6!)
        // s_addk_i32 s6, 1              (partial write: reads & updates s6!)
        // s_cmov_b32 s6, s1             (partial write: reads old s6!)
        // v_mov_b32 v1, s6
        // buffer_store_dword v1, off, s[8:11], 0
        // s_endpgm
        let words = [
            0x7D88_0000, // v_cmp_gt_u32 vcc, s0, v0
            0xBE86_036A, // s_mov_b32 s6, vcc_lo
            0xBE86_1D83, // s_bitset1_b32 s6, 3
            0xB786_0001, // s_addk_i32 s6, 1
            0xBE86_0501, // s_cmov_b32 s6, s1
            0x7E02_0206, // v_mov_b32 v1, s6
            0x38 << 26 | 0x1C << 18,
            0x80 << 24 | 2 << 16 | 1 << 8, // buffer_store_dword v1, off, s[8:11], 0
            S_ENDPGM,
        ];
        let (p, _st, wl) = build(&words);
        let vcc_lo = p.blocks[0].insts[0].write(Reg::S(106)).expect("vcc_lo");
        assert!(
            wl.reads_whole.contains(&vcc_lo),
            "non-kill partial writes (s_bitset1/s_addk/s_cmov) must preserve backward liveness"
        );
        assert!(wl.reads_mask_as_data(0, 1));
    }
}
