//! Wave64-on-Wave32 analysis, region classification, and TGSM exchange layout.

use std::collections::{HashMap, HashSet};

use crate::gfx10::{Fields, NULL, Operand};
use crate::ir::{EXEC, Inst, Program, Reg, Value};
use crate::structure::{Atom, Bool, LaneExit, Structured};

type Why = &'static str;

/// Maximum D3D12 thread-group shared memory size in 32-bit dwords (32 KiB).
pub(super) const MAX_LDS_DWORDS: u32 = 8192;
/// Maximum spin iterations per Mode B pairwise half-wave arrival wait before
/// tripping the per-console-wave TDR fuse (`dead`).
pub(super) const MODE_B_SPIN_LIMIT: u32 = 1 << 20;

/// Layout and synchronisation mode of the TGSM cross-half exchange region for
/// a console Wave64 program lowered to paired 32-lane host waves.
pub(super) struct Exchange {
    /// First TGSM dword of the exchange area (carved at the tail of LDS).
    pub slot: u32,
    /// Group has 32 threads or fewer: no upper half exists.
    pub small: bool,
    /// Multi-wave group (> 64 threads): pairwise per-console-wave arrival
    /// counters + TGSM-only `barrier(8)` + bounded spin + TDR fuse.
    /// Single-wave group (33..=64 threads): Mode A double-buffered 1-barrier.
    pub mode_b: bool,
    /// Total threads in the thread group.
    pub group: u32,
    /// Number of 64-lane console waves (`group.div_ceil(64).max(1)`).
    pub console_waves: u32,
    /// Control header dwords per console wave (Mode A: 2 `[parity0, parity1]`;
    /// Mode B: 3 `[cnt0, cnt1, dead]`).
    pub hdr_dwords: u32,
    /// Payload dwords per half per bank (`>= 1`, `>= 2` for packed reads/f64,
    /// `>= 4` for coherent `smem`, `>= 32` for cross-half `ds_bpermute_b32`).
    pub width: u32,
    /// Total TGSM dwords per console wave (`hdr_dwords + 4 * width`).
    pub stride: u32,
    /// Per IR block: whether every lane of a console wave runs it.
    pub all: Vec<bool>,
}

impl Exchange {
    pub fn paired(&self) -> bool {
        !self.small
    }
}

/// Per IR block: every lane of the wave that started the program runs it – no per-lane branch
/// separates it from the entry (it lies in no lane branch's region: the blocks reachable from
/// the branch that do not post-dominate it). None with an indirect exit (not followed).
pub(super) fn all_lane_blocks(st: &Structured) -> Option<Vec<bool>> {
    all_blocks_inner(st, true)
}

/// Per IR block: every console wave of the thread group runs it (excludes
/// regions of wave-wide `st.wide` branches, where distinct console waves in a
/// multi-wave group may diverge).
pub(super) fn all_group_blocks(st: &Structured) -> Option<Vec<bool>> {
    all_blocks_inner(st, false)
}

fn all_blocks_inner(st: &Structured, include_wide: bool) -> Option<Vec<bool>> {
    let n = st.exits.len();
    let mut all = vec![true; n];
    for (_, region) in lane_regions_inner(st, include_wide)? {
        for b in (0..n).filter(|&b| region[b]) {
            all[b] = false;
        }
    }
    Some(all)
}

/// Every per-lane branch (by its block) with its region: the blocks reachable from it that do
/// not post-dominate it, which only some lanes run. Not the branches the wave takes together
/// ([`Structured::wide`]), nor those on scalar conditions. None with an indirect exit.
pub(super) fn lane_regions(st: &Structured) -> Option<Vec<(usize, Vec<bool>)>> {
    lane_regions_inner(st, true)
}

fn lane_regions_inner(st: &Structured, include_wide: bool) -> Option<Vec<(usize, Vec<bool>)>> {
    let n = st.exits.len();
    let mut succ: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (b, e) in st.exits.iter().enumerate() {
        match *e {
            LaneExit::Jump(t) => succ[b].push(t),
            LaneExit::Branch { taken, not_taken, .. } => {
                succ[b].push(taken);
                succ[b].push(not_taken);
            }
            LaneExit::End => {}
            LaneExit::Indirect => return None,
        }
    }
    // post-dominators (bitsets; a virtual exit after every End)
    let words = n.div_ceil(64);
    let full = vec![u64::MAX; words];
    let mut pdom: Vec<Vec<u64>> = (0..n)
        .map(|b| {
            if succ[b].is_empty() {
                let mut v = vec![0; words];
                v[b / 64] |= 1 << (b % 64);
                v
            } else {
                full.clone()
            }
        })
        .collect();
    let mut changed = true;
    while changed {
        changed = false;
        for b in (0..n).rev() {
            if succ[b].is_empty() {
                continue;
            }
            let mut v = full.clone();
            for &t in &succ[b] {
                for w in 0..words {
                    v[w] &= pdom[t][w];
                }
            }
            v[b / 64] |= 1 << (b % 64);
            if v != pdom[b] {
                pdom[b] = v;
                changed = true;
            }
        }
    }
    let mut regions = Vec::new();
    for (d, e) in st.exits.iter().enumerate() {
        let LaneExit::Branch { when, .. } = *e else {
            continue;
        };
        if (include_wide && st.wide.contains(&d))
            || st
                .bools
                .support(when)
                .iter()
                .all(|a| matches!(a, Atom::Uniform(_)))
        {
            continue; // a wave-wide or uniform branch: all lanes go the same way
        }
        // blocks reachable from the lane branch that do not post-dominate it
        let mut seen = vec![false; n];
        let mut stack: Vec<usize> = succ[d].clone();
        while let Some(b) = stack.pop() {
            if seen[b] || pdom[d][b / 64] >> (b % 64) & 1 == 1 {
                continue;
            }
            seen[b] = true;
            stack.extend(succ[b].iter().copied());
        }
        regions.push((d, seen));
    }
    Some(regions)
}

/// The per-lane branches (by block) whose regions hold a block of `needs` (blocks where two
/// host waves must meet, [`super::lower::take_needs`]): the ones to take wave-wide
/// ([`crate::structure::structure_wide`]) so that every lane runs those blocks. None if one
/// of them is a branch whose wave-wide meaning is not known ([`Structured::wave_any`]).
pub(super) fn branches_around(st: &Structured, needs: &[usize]) -> Option<Vec<usize>> {
    let mut out = Vec::new();
    for (d, region) in lane_regions(st)? {
        if needs.iter().any(|&b| region.get(b).copied().unwrap_or(false)) {
            st.wave_any.get(d).copied().flatten()?;
            out.push(d);
        }
    }
    Some(out)
}

/// VGPR values that are the same in every lane holding them (`all` is [`all_lane_blocks`]):
/// written by an instruction every arriving lane runs (no guard), lane by lane, from scalars,
/// constants and such values; or loaded from an address made of them. The compiler keeps
/// wave-uniform data in VGPRs and fetches it with `v_readfirstlane`.
/// shortcut: a value joined by a phi counts only where every predecessor is a block all lanes
/// run (no lane branch can have split the arriving lanes); loop-carried values never do.
pub(super) fn lane_uniform(p: &Program, st: &Structured, all: &[bool]) -> HashSet<Value> {
    let is_mask = |v: Value| st.masks.contains_key(&v) || st.highs.contains_key(&v);
    let mut same: HashSet<Value> = HashSet::new();
    for b in crate::ir::reverse_postorder(&p.blocks) {
        let block = &p.blocks[b];
        let whole = block
            .preds
            .iter()
            .all(|&q| all.get(q).copied().unwrap_or(false));
        for phi in &block.phis {
            if matches!(phi.reg, Reg::V(_)) && whole && phi.args.iter().all(|a| same.contains(a)) {
                same.insert(phi.value);
            }
        }
        for (i, inst) in block.insts.iter().enumerate() {
            let name = inst.name();
            let ops: Vec<Operand> = match &inst.fields {
                Fields::Valu(v)
                    if v.dpp.is_none()
                        && v.sdwa.is_none()
                        && !name.starts_with("v_mbcnt")
                        && !name.starts_with("v_cmpx")
                        && !name.contains("lane") =>
                {
                    v.src.clone()
                }
                // A full-overwrite LDS read at a uniform address returns the
                // same value to every arriving lane (`split_lds` / `lds_meet`
                // guarantees no concurrent LDS write inside a read region).
                Fields::Ds { addr: Some(a), .. }
                    if name.starts_with("ds_read") && !name.contains("_d16") =>
                {
                    vec![*a]
                }
                _ => continue,
            };
            if st.guards[b][i].is_some() {
                continue; // lanes outside EXEC keep another value
            }
            let alike = ops.iter().all(|op| match *op {
                Operand::Vector { reg, .. } => {
                    // every dword of the operand the instruction reads from that register on
                    let read: Vec<Value> = inst
                        .reads
                        .iter()
                        .filter(|r| matches!(r.0, Reg::V(x) if x >= reg))
                        .map(|r| r.1)
                        .collect();
                    !read.is_empty() && read.iter().all(|x| same.contains(x))
                }
                // a scalar is one value; a lane mask is not, unless a compare of such values made it
                Operand::Scalar { code, .. } if code != NULL => inst
                    .read(Reg::S(code))
                    .is_none_or(|x| !is_mask(x) || same.contains(&x)),
                _ => true,
            });
            if alike {
                same.extend(inst.writes.iter().filter(|w| w.0 != Reg::Scc).map(|w| w.1));
            }
        }
    }
    same
}

/// What a register value holds, for [`wave_local`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Wave {
    /// Nothing that depends on how the threads are divided into waves.
    Plain,
    /// The number of the wave's first lane in a mask (`s_ff1`).
    First,
    /// A reduction over the wave's lanes: a count, a sum, a minimum...
    Total(super::wave::Op),
    /// A count or sum over the lanes below this one.
    Prefix,
    /// Derived from what a first-lane atomic returned: the slots the wave reserved.
    Base,
    /// A mix the idiom does not make.
    Mixed,
}

/// Whether a console wave64 program may run with each host wave as a wave of its own
/// ([`super::lower::with_local_waves`]): its wave-wide values are used only as the allocation
/// idiom uses them, which does not care how the threads are divided into waves.
pub(super) fn wave_local(p: &Program, st: &Structured) -> Result<(), (usize, &'static str)> {
    use super::wave::{Op, Shape};
    const FIRST: Why = "a wave's first-lane number used as data";
    const TOTAL: Why = "a count or sum over the wave used as data";
    const PREFIX: Why = "a count or sum over the lanes below used without a reserved base";
    const MIXED: Why = "a value that depends on the wave on some paths only";
    const EFFECT: Why = "a write by the wave's first lane other than its reservation";
    fn of(kind: &HashMap<Value, Wave>, v: Option<Value>) -> Wave {
        v.and_then(|v| kind.get(&v).copied()).unwrap_or(Wave::Plain)
    }
    fn put(kind: &mut HashMap<Value, Wave>, v: Value, k: Wave) {
        if k == Wave::Plain {
            kind.remove(&v);
        } else {
            kind.insert(v, k);
        }
    }
    let wild = |k: Wave| !matches!(k, Wave::Plain | Wave::Base);
    let why = |k: Wave| match k {
        Wave::First => FIRST,
        Wave::Total(_) => TOTAL,
        Wave::Prefix => PREFIX,
        _ => MIXED,
    };
    let first_mask = |firsts: &HashSet<Value>, m: Option<Value>| {
        m.and_then(|m| st.masks.get(&m)).is_some_and(|&b| {
            st.bools
                .support(b)
                .iter()
                .any(|a| matches!(a, Atom::Lane(v) if firsts.contains(v)))
        })
    };
    let step = |inst: &Inst, kind: &HashMap<Value, Wave>, firsts: &mut HashSet<Value>| -> Result<(Wave, bool), Why> {
        let name = inst.name();
        let val = |op: &Operand| match *op {
            Operand::Vector { reg, .. } => inst.read(Reg::V(reg)),
            Operand::Scalar { code, .. } if code != NULL => inst.read(Reg::S(code)),
            _ => None,
        };
        let ops: Vec<Operand> = match &inst.fields {
            Fields::Valu(v) => v.src.clone(),
            Fields::Sop { dst, src, .. } => {
                let mut o = src.clone();
                if matches!(name, "s_addk_i32" | "s_mulk_i32") {
                    o.extend(dst.iter().copied());
                }
                o
            }
            Fields::Buffer { data, .. } if name.contains("_atomic_") || name.contains("_store_") => {
                data.iter().copied().collect()
            }
            _ => Vec::new(),
        };
        let kinds: Vec<Wave> = ops.iter().map(|o| of(kind, val(o))).collect();
        let named: Vec<Value> = ops.iter().filter_map(&val).collect();
        for &(r, v) in &inst.reads {
            let kept = matches!(r, Reg::V(_)) && inst.writes.iter().any(|w| w.0 == r);
            if matches!(r, Reg::S(c) if c == EXEC || c == EXEC + 1) || kept || named.contains(&v) {
                continue;
            }
            let k = of(kind, Some(v));
            if wild(k) {
                return Err(why(k));
            }
        }
        let w = kinds.iter().copied().find(|&k| wild(k)).unwrap_or(Wave::Mixed);
        let sum = |k: Wave| k == Wave::Total(Op::AddU);
        let is_mask = |o: &Operand| {
            val(o).is_some_and(|v| st.masks.contains_key(&v) || st.highs.contains_key(&v))
        };
        let mut reserved = false;
        let out = match (name, &kinds[..]) {
            ("s_ff1_i32_b64" | "s_ff1_i32_b32", _) if ops.first().is_some_and(is_mask) => Wave::First,
            ("s_bcnt1_i32_b64" | "s_bcnt1_i32_b32", _) if ops.first().is_some_and(is_mask) => {
                Wave::Total(Op::AddU)
            }
            ("v_mbcnt_lo_u32_b32" | "v_mbcnt_hi_u32_b32", [_, b]) => {
                match (matches!(ops[0], Operand::Scalar { .. }), *b) {
                    (_, Wave::Prefix) | (true, Wave::Plain | Wave::Base) => Wave::Prefix,
                    (false, Wave::Plain | Wave::Base) => Wave::Plain,
                    (_, k) => return Err(why(k)),
                }
            }
            _ if !kinds.iter().any(|&k| wild(k)) => {
                let derives = matches!(inst.fields, Fields::Valu(_) | Fields::Sop { .. });
                if derives && kinds.contains(&Wave::Base) {
                    Wave::Base
                } else {
                    Wave::Plain
                }
            }
            ("v_mov_b32" | "s_mov_b32", [k]) => *k,
            ("v_cmp_eq_i32" | "v_cmp_eq_u32" | "v_cmpx_eq_i32" | "v_cmpx_eq_u32", [a, b]) => {
                let lane = match (*a, *b) {
                    (Wave::First, k) if !wild(k) => &ops[1],
                    (k, Wave::First) if !wild(k) => &ops[0],
                    _ => return Err(why(w)),
                };
                if !matches!(lane, Operand::Vector { .. }) {
                    return Err(FIRST);
                }
                let mask = match &inst.fields {
                    _ if name.starts_with("v_cmpx") => inst.write(Reg::S(EXEC)),
                    Fields::Valu(v) => match v.dst {
                        Some(Operand::Scalar { code, .. }) => inst.write(Reg::S(code)),
                        _ => None,
                    },
                    _ => None,
                };
                firsts.extend(mask);
                Wave::Plain
            }
            ("v_add_nc_u32" | "s_add_i32", [a, b]) => match (*a, *b) {
                (Wave::Prefix, Wave::Base) | (Wave::Base, Wave::Prefix) => Wave::Base,
                (Wave::Prefix, Wave::Plain | Wave::Prefix) | (Wave::Plain, Wave::Prefix) => Wave::Prefix,
                (x, y) if sum(x) && sum(y) => x,
                _ => return Err(why(w)),
            },
            ("v_sub_nc_u32" | "s_sub_i32", [a, b]) => match (*a, *b) {
                (Wave::Base, Wave::Prefix) => Wave::Base,
                (Wave::Prefix, Wave::Plain) => Wave::Prefix,
                (Wave::Plain, y) if sum(y) && matches!(ops[0], Operand::Const(128)) => y,
                _ => return Err(why(w)),
            },
            ("s_mul_i32" | "v_mul_lo_u32" | "v_mul_u32_u24", [a, b]) => match (*a, *b) {
                (x, Wave::Plain) | (Wave::Plain, x) if sum(x) || x == Wave::Prefix => x,
                _ => return Err(why(w)),
            },
            ("s_lshl_b32", [x, Wave::Plain]) | ("v_lshlrev_b32", [Wave::Plain, x])
                if sum(*x) || *x == Wave::Prefix =>
            {
                *x
            }
            ("v_lshl_add_u32", [Wave::Prefix, Wave::Plain, c]) if !wild(*c) => {
                if *c == Wave::Base { Wave::Base } else { Wave::Prefix }
            }
            (_, [Wave::Total(op)]) if name.contains("_atomic_") => {
                let want = match op {
                    Op::AddU => "buffer_atomic_add",
                    Op::MinU => "buffer_atomic_umin",
                    Op::MaxU => "buffer_atomic_umax",
                    Op::MinI => "buffer_atomic_smin",
                    Op::MaxI => "buffer_atomic_smax",
                    Op::Or => "buffer_atomic_or",
                    Op::And => "buffer_atomic_and",
                    Op::Xor => "buffer_atomic_xor",
                    _ => "",
                };
                if name != want || !first_mask(firsts, inst.read(Reg::S(EXEC))) {
                    return Err(TOTAL);
                }
                reserved = true;
                Wave::Base
            }
            _ => return Err(why(w)),
        };
        Ok((out, reserved))
    };
    let sections: Vec<_> = (0..p.blocks.len())
        .map(|b| super::wave::sections(p, b))
        .collect();
    let order = crate::ir::reverse_postorder(&p.blocks);
    let pass = |kind: &mut HashMap<Value, Wave>,
                firsts: &mut HashSet<Value>,
                check: bool|
     -> Result<(), (usize, Why)> {
        for &b in &order {
            let block = &p.blocks[b];
            for phi in &block.phis {
                let mut k = Wave::Plain;
                for (n, &a) in phi.args.iter().enumerate() {
                    let x = of(kind, Some(a));
                    k = match (k, x) {
                        _ if n == 0 => x,
                        (k, x) if k == x => k,
                        (Wave::Plain, m @ (Wave::Base | Wave::Prefix))
                        | (m @ (Wave::Base | Wave::Prefix), Wave::Plain) => m,
                        _ => Wave::Mixed,
                    };
                }
                put(kind, phi.value, k);
            }
            for (i, inst) in block.insts.iter().enumerate() {
                if let Some((_, sec)) = sections[b].iter().find(|(s, sec)| i > **s && i < sec.end) {
                    for &(_, w) in &inst.writes {
                        let k = match sec.shapes.get(&w) {
                            None => Wave::Mixed,
                            Some(&Shape::K(_)) => Wave::Plain,
                            Some(&Shape::Value(v)) => of(kind, Some(v)),
                            Some(&Shape::Masked(src, _, _)) => of(kind, Some(src)),
                            Some(&Shape::Active(op, src, mask, _)) => match of(kind, Some(src)) {
                                Wave::Plain | Wave::Base => Wave::Total(op),
                                Wave::Total(o) if o == op && first_mask(firsts, Some(mask)) => {
                                    Wave::Total(op)
                                }
                                _ => Wave::Mixed,
                            },
                            Some(&Shape::Prefix(Op::AddU, src, mask, _, _)) => {
                                match of(kind, Some(src)) {
                                    Wave::Plain | Wave::Base => Wave::Prefix,
                                    Wave::Total(Op::AddU) if first_mask(firsts, Some(mask)) => {
                                        Wave::Prefix
                                    }
                                    _ => Wave::Mixed,
                                }
                            }
                            Some(_) => Wave::Mixed,
                        };
                        put(kind, w, k);
                    }
                    continue;
                }
                let name = inst.name();
                if name.starts_with("s_cbranch") {
                    continue;
                }
                let (k, reserved) = match step(inst, kind, firsts) {
                    Ok(r) => r,
                    Err(why) if check => return Err((inst.machine.offset, why)),
                    Err(_) => (Wave::Mixed, false),
                };
                let effect = matches!(inst.fields, Fields::Exp { .. })
                    || matches!(inst.fields, Fields::Flat { .. })
                        && !name.starts_with("global_load_")
                    || name.contains("store")
                    || name.contains("atomic")
                    || matches!(inst.fields, Fields::Ds { .. })
                        && !(name.starts_with("ds_read")
                            || name.starts_with("ds_swizzle")
                            || name.contains("permute"));
                if check && effect && !reserved && first_mask(firsts, inst.read(Reg::S(EXEC))) {
                    return Err((inst.machine.offset, EFFECT));
                }
                let guarded = st.guards[b][i].is_some();
                for &(r, w) in &inst.writes {
                    let mut kw = match r {
                        Reg::Scc if !name.starts_with("s_bcnt1") => {
                            if wild(k) { Wave::Mixed } else { Wave::Plain }
                        }
                        _ => k,
                    };
                    if guarded
                        && matches!(r, Reg::V(_))
                        && kw == Wave::Plain
                        && matches!(
                            of(kind, inst.read(r)),
                            Wave::First | Wave::Total(_) | Wave::Prefix
                        )
                    {
                        kw = Wave::Mixed;
                    }
                    put(kind, w, kw);
                }
            }
            if check
                && let Some(last) = block.insts.last()
                && last.name().starts_with("s_cbranch_scc")
                && let Some(scc) = last.read(Reg::Scc)
                && wild(of(kind, Some(scc)))
            {
                let own = matches!(st.exits[b], LaneExit::Branch { when, .. }
                    if !st.bools.support(when).iter().any(|a| matches!(a, Atom::Uniform(v) if *v == scc)));
                if !own {
                    return Err((last.machine.offset, TOTAL));
                }
            }
        }
        Ok(())
    };
    let mut kind: HashMap<Value, Wave> = HashMap::new();
    let mut firsts: HashSet<Value> = HashSet::new();
    let mut settled = false;
    for _ in 0..16 {
        let before = (kind.clone(), firsts.len());
        pass(&mut kind, &mut firsts, false)?;
        if (&before.0, before.1) == (&kind, firsts.len()) {
            settled = true;
            break;
        }
    }
    if !settled {
        return Err((0, MIXED));
    }
    pass(&mut kind, &mut firsts, true)
}

/// The mode of a group barrier the translation adds (host waves meeting):
/// a wait for the group with a shared-memory fence; the fence alone with
/// `VICTORY_GROUP_SYNC=0` ([`super::group_sync`]).
pub(super) fn meeting_mode() -> u32 {
    if super::group_sync() == 0 { 8 } else { 1 | 8 }
}

/// LDS accesses of a console wave run as two host waves, where some sit in blocks that only
/// some lanes run (`all` is [`all_lane_blocks`]): per IR block, whether the host waves meet (a
/// barrier) at its start and before its exit. `None` if barriers cannot keep the console's order.
pub(super) fn lds_meetings(
    p: &Program,
    st: &Structured,
    all: &[bool],
) -> Option<Vec<(bool, bool)>> {
    let n = st.exits.len();
    let succ = |b: usize| -> Vec<usize> {
        match st.exits[b] {
            LaneExit::Jump(t) => vec![t],
            LaneExit::Branch { taken, not_taken, .. } => vec![taken, not_taken],
            LaneExit::End | LaneExit::Indirect => Vec::new(),
        }
    };
    let mut region: Vec<usize> = (0..n).collect();
    fn find(region: &mut [usize], mut b: usize) -> usize {
        while region[b] != b {
            region[b] = region[region[b]];
            b = region[b];
        }
        b
    }
    for b in 0..n {
        for t in succ(b) {
            if !all[b] && !all[t] {
                let (x, y) = (find(&mut region, b), find(&mut region, t));
                region[x] = y;
            }
        }
    }
    let mut reads = vec![false; n];
    let mut writes = vec![false; n];
    for (b, block) in p.blocks.iter().enumerate().take(n) {
        if all[b] {
            continue;
        }
        let r = find(&mut region, b);
        for i in &block.insts {
            let name = i.name();
            if !matches!(i.fields, Fields::Ds { .. })
                || name.starts_with("ds_swizzle")
                || name.contains("permute")
            {
                continue;
            }
            if name.starts_with("ds_read") {
                reads[r] = true;
            } else {
                writes[r] = true;
            }
        }
    }
    if (0..n).any(|r| reads[r] && writes[r]) {
        return None;
    }
    let mut meet = vec![(false, false); n];
    for b in 0..n {
        for t in succ(b) {
            if all[b] && !all[t] {
                let r = find(&mut region, t);
                meet[b].1 |= reads[r] || writes[r];
            } else if !all[b] && all[t] {
                let r = find(&mut region, b);
                meet[t].0 |= reads[r] || writes[r];
            }
        }
    }
    Some(meet)
}

/// Masks held as numbers or constant lane sets in guards or lane branches: the lane tests its
/// own bit by its console lane.
pub(super) fn lane_constant_masks(st: &Structured) -> bool {
    let whens = st.exits.iter().filter_map(|e| match *e {
        LaneExit::Branch { when, .. } => Some(when),
        _ => None,
    });
    let mut work: Vec<Bool> = st
        .guards
        .iter()
        .flatten()
        .flatten()
        .copied()
        .chain(whens)
        .collect();
    let (mut seen, mut joins) = (HashSet::new(), HashSet::new());
    while let Some(g) = work.pop() {
        if !seen.insert(g) {
            continue;
        }
        for a in st.bools.support(g) {
            match a {
                Atom::Opaque(_) | Atom::Constant(_) => return true,
                Atom::Phi(id) if joins.insert(id) => {
                    if let Some(edges) = st.phi_masks.get(&id) {
                        work.extend(edges.iter().copied());
                    }
                }
                _ => {}
            }
        }
    }
    false
}

pub(super) fn reads_mask_as_data(st: &Structured, i: &Inst) -> bool {
    let is_mask = |v: Value| st.masks.contains_key(&v) || st.highs.contains_key(&v);
    let ops: Vec<&Operand> = match &i.fields {
        // A mask operation (its result a mask) is no data use.
        Fields::Sop { .. }
            if i.writes
                .iter()
                .any(|&(r, w)| matches!(r, Reg::S(_)) && is_mask(w)) =>
        {
            return false;
        }
        Fields::Sop { src, .. } => src.iter().collect(),
        Fields::Valu(v)
            if !matches!(i.name(), "v_cndmask_b32") && !i.name().contains("_co_ci_") =>
        {
            v.src.iter().collect()
        }
        _ => return false,
    };
    ops.into_iter().any(|op| match *op {
        Operand::Scalar { code, .. } if code != NULL => i.read(Reg::S(code)).is_some_and(is_mask),
        _ => false,
    })
}
