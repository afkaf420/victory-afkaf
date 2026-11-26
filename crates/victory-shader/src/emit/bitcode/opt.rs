//! Clean-up of a finished module before it is written: the lowering emits
//! one instruction after another as the console code names them, and
//! nothing stands between it and the driver's compiler. Every pass here is
//! exact: the module computes the same values and has the same effects for
//! every input.
//!
//! - [`DEAD`]: instructions without an effect whose result nothing reads
//!   are left out (the lowering computes every dword of every register a
//!   console instruction writes; a guarded write reads the register's old
//!   value whether anything reads the new one or not).
//! - [`FOLD`]: integer arithmetic on constants is done here, and identities
//!   (`x + 0`, `x & -1`, a select between a value and itself, a cast there
//!   and back, a join of one value) give their operand. Float arithmetic is
//!   left alone: `x + 0.0` and `x * 1.0` are not `x` for every `x`.
//! - [`REUSE`]: an instruction without an effect whose operands are those
//!   of an earlier one that every path to it has run gives that one's
//!   result (a descriptor handle made again in each block, an index clamped
//!   again, a register read as a float again).
//! - [`FACTS`]: in the blocks only one side of a branch leads to, the
//!   branch's condition is that side's constant, and so is what the
//!   condition was made of (both terms of an `and` that held, the opposite
//!   comparison). The lowering branches on a lane's condition and then
//!   guards each instruction behind the branch with the same condition.
//! - [`LOADS`]: within a block, a read of the same place as an earlier read
//!   gives that one's result while no store, atomic, barrier or other call
//!   with an effect stands between them (another thread's unsynchronised
//!   store may fall between any two reads or before both: the title's code
//!   cannot tell which).
//!
//! Which passes run is `emit::shader_opt()` (`VICTORY_SHADER_OPT`), part of
//! the translation and so of the shader cache's key.

use std::collections::HashMap;

use super::{
    Attr, Bin, Cast, Const, Func, MdDef, Module, Op, Term, Ty, TypeDef, V, for_each_operand, pred,
    successors,
};

/// Dead-code elimination.
pub const DEAD: u32 = 1;
/// Constants folded, identities removed.
pub const FOLD: u32 = 2;
/// One instruction for one value.
pub const REUSE: u32 = 4;
/// What a branch taken says about its condition.
pub const FACTS: u32 = 8;
/// One read for one place within a block, while nothing is stored.
pub const LOADS: u32 = 16;
/// Not a pass over the module: the lowering of a first-lane read in a
/// wave64 program run as two host waves skips the exchange that asks which
/// half has a lane where every lane is in EXEC (`lower.rs`,
/// `v_readfirstlane_b32`).
pub const FIRST: u32 = 32;
/// The passes over the module.
pub const MODULE: u32 = DEAD | FOLD | REUSE | FACTS | LOADS;
/// Every pass.
pub const ALL: u32 = MODULE | FIRST;

/// What the instructions a module writes use, of the features a shader
/// declares by flag (the validator's "flags must match usage").
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Usage {
    /// A wave or quad operation.
    pub wave_ops: bool,
    /// A 64-bit integer, as a result or an operand.
    pub int64: bool,
    /// A typed load of several components from a UAV.
    pub typed_uav_loads: bool,
    /// A rasterizer ordered view.
    pub rovs: bool,
    /// A handle from the view heap.
    pub heap_views: bool,
    /// A handle from the sampler heap.
    pub heap_samplers: bool,
    /// A 64-bit atomic.
    pub atomic64: bool,
    /// Bitmask of input signature element IDs referenced by live input-read
    /// intrinsics (`u64::MAX` if an index >= 64 or non-constant is seen).
    pub live_inputs: u64,
    /// Whether `attributeAtVertex` (pull-model barycentric evaluation) is live.
    pub barycentrics_attr: bool,
}

/// What makes two instructions without an effect the same value.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Key {
    Bin(u8, V, V),
    Cast(u8, Ty, V),
    Cmp(u8, V, V),
    Select(V, V, V),
    Extract(V, u32),
    Call(u32, Vec<V>),
    Gep(Ty, V, Vec<V>),
    Load(V),
}

/// The blocks reached from the entry, each after the blocks that dominate
/// it, and each block's children in the dominator tree.
struct Tree {
    order: Vec<usize>,
    children: Vec<Vec<usize>>,
    /// Per block: the reached blocks that branch to it, once per edge.
    preds: Vec<Vec<usize>>,
}

/// The comparison that holds exactly when `p` does not (floats 0-15,
/// integers 32-41).
fn opposite(p: u8) -> u8 {
    match p {
        0..=15 => 15 - p,
        pred::EQ => pred::NE,
        pred::NE => pred::EQ,
        pred::IUGT => pred::IULE,
        pred::IULE => pred::IUGT,
        pred::IUGE => pred::IULT,
        pred::IULT => pred::IUGE,
        pred::SGT => pred::SLE,
        pred::SLE => pred::SGT,
        pred::SGE => pred::SLT,
        _ => pred::SGE,
    }
}

/// What holds in the block being walked: values that are a known bit there,
/// and comparisons (by their operands) with a known outcome.
#[derive(Default)]
struct Facts {
    bits: HashMap<V, bool>,
    compares: HashMap<(u8, V, V), bool>,
}

/// The facts a block added, to forget when its part of the tree is done.
#[derive(Default)]
struct Learned {
    bits: Vec<V>,
    compares: Vec<(u8, V, V)>,
}

fn mask(bits: u32) -> u64 {
    if bits >= 64 { !0 } else { (1 << bits) - 1 }
}

/// `x` of `bits` bits as a signed number.
fn signed(bits: u32, x: u64) -> i64 {
    if bits < 64 && x >> (bits - 1) & 1 != 0 {
        (x | !mask(bits)) as i64
    } else {
        x as i64
    }
}

/// What `v` stands for after the replacements so far.
fn resolve(repl: &[Option<V>], mut v: V) -> V {
    while let V::Inst(i) = v {
        match repl[i as usize] {
            Some(r) => v = r,
            None => break,
        }
    }
    v
}

/// Replaces every value `op` reads by `f` of it.
fn map_operands(op: &mut Op, mut f: impl FnMut(V) -> V) {
    match op {
        Op::Bin(_, a, b) | Op::Cmp(_, a, b) | Op::Store(a, b) | Op::AtomicRmw(_, a, b) => {
            *a = f(*a);
            *b = f(*b);
        }
        Op::Cast(_, v) | Op::Extract(v, _) | Op::Load(v) => *v = f(*v),
        Op::Select(a, b, c) | Op::CmpXchg(a, b, c) => {
            *a = f(*a);
            *b = f(*b);
            *c = f(*c);
        }
        Op::Call(_, args) => args.iter_mut().for_each(|a| *a = f(*a)),
        Op::Phi(inc) => inc.iter_mut().for_each(|x| x.0 = f(x.0)),
        Op::Gep(_, p, idx) => {
            *p = f(*p);
            idx.iter_mut().for_each(|i| *i = f(*i));
        }
    }
}

/// What a call does besides returning its result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Effect {
    /// Nothing, and the result depends on the arguments alone for the whole
    /// run of the shader (arithmetic, handles, inputs, constant buffers,
    /// resource sizes).
    Pure,
    /// Nothing, but the result depends on memory a store may change
    /// (buffer and texture reads).
    Reads,
    /// Nothing, but the result depends on which lanes run the instruction
    /// (wave and quad operations).
    Lanes,
    /// Stores, atomics, barriers, outputs, discards: always kept.
    Writes,
}

pub(super) fn effect(f: &Func) -> Effect {
    let name = f.name.strip_prefix("dx.op.").unwrap_or(&f.name);
    match f.attr {
        Some(Attr::ReadNone) => Effect::Pure,
        // A shader cannot write a constant buffer or resize a resource.
        Some(Attr::ReadOnly)
            if name.starts_with("cbufferLoadLegacy") || name.starts_with("getDimensions") =>
        {
            Effect::Pure
        }
        Some(Attr::ReadOnly) if name.starts_with("wave") => Effect::Lanes,
        Some(Attr::ReadOnly) => Effect::Reads,
        // Declared with the attribute of calls that have effects, and have
        // none: they read the other lanes' values.
        _ if name.starts_with("wave") || name.starts_with("quad") => Effect::Lanes,
        _ => Effect::Writes,
    }
}

impl Module {
    /// Runs the clean-up passes named in `passes` ([`ALL`]: every one).
    pub(crate) fn optimise(&mut self, passes: u32) {
        if passes & (FOLD | REUSE | FACTS | LOADS) != 0 {
            // Values merged in one round can make two joins the same in
            // the next (a loop's registers that only go round), and a
            // branch that goes one way leaves blocks with one way in.
            // Run exact float sign-of-zero / finiteness elision between rounds
            // so removing an outer tie-breaker's `bitcast` unlocks eliding the
            // inner `legacy_mul` or `max_min` on the next round.
            for _ in 0..4 {
                let mut changed = self.merge(passes);
                if passes & FOLD != 0 {
                    changed |= self.elide_unobserved_zero_sign();
                    changed |= self.prune();
                }
                if !changed {
                    break;
                }
            }
        }
        if passes & DEAD != 0 {
            self.drop_dead();
        }
    }

    /// Per function: what a call of it does.
    pub(super) fn effects(&self) -> Vec<Effect> {
        self.funcs.iter().map(effect).collect()
    }

    /// The blocks reached from the entry.
    pub(super) fn reached(&self) -> Vec<bool> {
        let mut seen = vec![false; self.blocks.len()];
        if self.blocks.is_empty() {
            return seen;
        }
        let mut stack = vec![0usize];
        seen[0] = true;
        while let Some(b) = stack.pop() {
            for &s in successors(self.blocks[b].term).iter() {
                if !seen[s] {
                    seen[s] = true;
                    stack.push(s);
                }
            }
        }
        seen
    }

    fn dominator_tree(&self) -> Tree {
        let n = self.blocks.len();
        let succs = |b: usize| successors(self.blocks[b].term);
        // Reverse postorder, as `write` lays the blocks out.
        let mut seen = vec![false; n];
        let mut post = Vec::new();
        let mut stack = vec![(0usize, 0usize)];
        seen[0] = true;
        while let Some(top) = stack.last_mut() {
            let (b, k) = *top;
            top.1 += 1;
            match succs(b).get(k) {
                Some(&s) if !seen[s] => {
                    seen[s] = true;
                    stack.push((s, 0));
                }
                Some(_) => {}
                None => {
                    post.push(b);
                    stack.pop();
                }
            }
        }
        post.reverse();
        let order = post;
        let mut number = vec![usize::MAX; n];
        for (k, &b) in order.iter().enumerate() {
            number[b] = k;
        }
        let mut preds = vec![Vec::new(); n];
        for &b in &order {
            for &s in succs(b).iter() {
                preds[s].push(b);
            }
        }
        // Immediate dominators (Cooper, Harvey and Kennedy's iteration).
        let mut idom = vec![usize::MAX; n];
        idom[0] = 0;
        let mut changed = true;
        while changed {
            changed = false;
            for &b in &order[1..] {
                let mut new = usize::MAX;
                for &p in &preds[b] {
                    if idom[p] == usize::MAX {
                        continue;
                    }
                    if new == usize::MAX {
                        new = p;
                        continue;
                    }
                    let (mut x, mut y) = (p, new);
                    while x != y {
                        while number[x] > number[y] {
                            x = idom[x];
                        }
                        while number[y] > number[x] {
                            y = idom[y];
                        }
                    }
                    new = x;
                }
                if idom[b] != new {
                    idom[b] = new;
                    changed = true;
                }
            }
        }
        let mut children = vec![Vec::new(); n];
        for &b in &order[1..] {
            children[idom[b]].push(b);
        }
        Tree {
            order,
            children,
            preds,
        }
    }

    /// Notes that `v` (an i1) is `holds` in the block being walked, and what
    /// follows from it for the values `v` was made of.
    fn learn(&self, v: V, holds: bool, facts: &mut Facts, learned: &mut Learned, depth: u32) {
        let V::Inst(i) = v else { return };
        if facts.bits.contains_key(&v) {
            return;
        }
        facts.bits.insert(v, holds);
        learned.bits.push(v);
        if depth == 0 {
            return;
        }
        match self.insts[i as usize].op {
            // Both terms of an `and` that holds hold; neither of an `or`
            // that does not.
            Op::Bin(Bin::And, a, b) if holds => {
                self.learn(a, true, facts, learned, depth - 1);
                self.learn(b, true, facts, learned, depth - 1);
            }
            Op::Bin(Bin::Or, a, b) if !holds => {
                self.learn(a, false, facts, learned, depth - 1);
                self.learn(b, false, facts, learned, depth - 1);
            }
            Op::Bin(Bin::Xor, a, b) => {
                let (x, k) = match (self.int_k(a), self.int_k(b)) {
                    (None, Some((1, k))) => (a, k),
                    (Some((1, k)), None) => (b, k),
                    _ => return,
                };
                self.learn(x, holds != (k != 0), facts, learned, depth - 1);
            }
            // The opposite comparison of the same operands.
            Op::Cmp(p, a, b) => {
                let key = (opposite(p), a, b);
                if let std::collections::hash_map::Entry::Vacant(e) = facts.compares.entry(key) {
                    e.insert(!holds);
                    learned.compares.push(key);
                }
            }
            _ => {}
        }
    }

    fn int_bits(&self, ty: Ty) -> Option<u32> {
        match self.types[ty.0 as usize] {
            TypeDef::Int(bits) => Some(bits),
            _ => None,
        }
    }

    /// An integer constant's width and value (the bits of that width).
    fn int_k(&self, v: V) -> Option<(u32, u64)> {
        let V::Const(i) = v else { return None };
        let (ty, Const::Int(x)) = &self.consts[i as usize] else {
            return None;
        };
        let bits = self.int_bits(*ty)?;
        Some((bits, *x as u64 & mask(bits)))
    }

    fn is_f32(&self, ty: Ty) -> bool {
        matches!(self.types[ty.0 as usize], TypeDef::Float)
    }

    /// If `(f, args)` is a `dx.op.{unary,binary,tertiary}.f32` call, returns its
    /// arity tag and DXIL opcode.
    fn dx_f32_op(&self, f: u32, args: &[V]) -> Option<(&'static str, u32)> {
        let kind = match self.funcs[f as usize].name.as_str() {
            "dx.op.unary.f32" => "unary",
            "dx.op.binary.f32" => "binary",
            "dx.op.tertiary.f32" => "tertiary",
            _ => return None,
        };
        let (_, code) = self.int_k(*args.first()?)?;
        Some((kind, code as u32))
    }

    fn float_k_inner(&self, v: V, depth: u32) -> Option<f32> {
        match v {
            V::Const(i) => match self.consts[i as usize].1 {
                Const::Float(bits) => Some(f32::from_bits(bits)),
                _ => None,
            },
            V::Inst(id) => {
                let depth = depth.checked_sub(1)?;
                let inst = &self.insts[id as usize];
                match &inst.op {
                    Op::Cast(Cast::Bitcast, x) if self.is_f32(inst.ty) => match *x {
                        V::Const(_) => {
                            let (bits, k) = self.int_k(*x)?;
                            (bits == 32).then_some(f32::from_bits(k as u32))
                        }
                        V::Inst(j) => match self.insts[j as usize].op {
                            Op::Cast(Cast::Bitcast, y) if self.is_f32(self.type_of(y)) => {
                                self.float_k_inner(y, depth)
                            }
                            _ => None,
                        },
                        _ => None,
                    },
                    // `fsrc` emits `v.neg` as `fsub -0.0, x`.
                    Op::Bin(Bin::Sub, z, x)
                        if self.is_f32(inst.ty)
                            && matches!(
                                *z,
                                V::Const(ci) if matches!(self.consts[ci as usize].1, Const::Float(0x8000_0000))
                            ) =>
                    {
                        let k = self.float_k_inner(*x, depth)?;
                        (!k.is_nan()).then_some(f32::from_bits(k.to_bits() ^ 0x8000_0000))
                    }
                    // `fsrc` emits `v.abs` as `dx.op.unary.f32(6, x)`.
                    Op::Call(f, args) if self.dx_f32_op(*f, args) == Some(("unary", 6)) => {
                        let k = self.float_k_inner(*args.get(1)?, depth)?;
                        (!k.is_nan()).then_some(f32::from_bits(k.to_bits() & 0x7FFF_FFFF))
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// The value of a 32-bit float constant (directly, via `i32 -> f32` bitcast,
    /// or folded through `fsrc`'s `-0.0 - k` / `FAbs(k)` modifiers).
    pub(crate) fn float_k(&self, v: V) -> Option<f32> {
        self.float_k_inner(v, 4)
    }

    /// For a float that is provably finite (never `NaN` or `±Inf`) whatever the
    /// shader reads: a power of two its magnitude does not pass (`< 126.0`).
    pub(crate) fn bounded(&self, v: V, depth: u32) -> Option<f32> {
        if let Some(k) = self.float_k(v) {
            return k.is_finite().then(|| k.abs().max(1e-30).log2().ceil());
        }
        let V::Inst(i) = v else { return None };
        let depth = depth.checked_sub(1)?;
        let of = |x: V| self.bounded(x, depth);
        // A power that overflows, or might with rounding, is no bound.
        let held = |b: f32| (b < 126.0).then_some(b);
        let inst = &self.insts[i as usize];
        if !self.is_f32(inst.ty) {
            return None;
        }
        match &inst.op {
            Op::Cast(Cast::UiToFp | Cast::SiToFp, _) => Some(32.0),
            Op::Cast(Cast::Bitcast, x) => match *x {
                V::Inst(j) => match self.insts[j as usize].op {
                    Op::Cast(Cast::Bitcast, y) if self.type_of(y) == inst.ty => of(y),
                    Op::Select(_, a_i, b_i) if self.int_bits(self.insts[j as usize].ty) == Some(32) => {
                        let arm = |arm_v: V| -> Option<f32> {
                            if let Some((_, k)) = self.int_k(arm_v) {
                                let fk = f32::from_bits(k as u32);
                                return fk.is_finite().then(|| fk.abs().max(1e-30).log2().ceil());
                            }
                            if let V::Inst(ai) = arm_v
                                && let Op::Cast(Cast::Bitcast, y) = self.insts[ai as usize].op
                                && self.type_of(y) == inst.ty
                            {
                                return of(y);
                            }
                            None
                        };
                        Some(arm(a_i)?.max(arm(b_i)?))
                    }
                    _ => None,
                },
                _ => None,
            },
            Op::Bin(Bin::Mul, a, b) => held(of(*a)? + of(*b)?),
            Op::Bin(Bin::Add | Bin::Sub, a, b) => held(of(*a)?.max(of(*b)?) + 1.0),
            Op::Select(_, a, b) => Some(of(*a)?.max(of(*b)?)),
            Op::Call(f, args) => {
                let (kind, code) = self.dx_f32_op(*f, args)?;
                match (kind, code, &args[1..]) {
                    // Saturate clamps any float (including Inf/NaN) to [0, 1].
                    ("unary", 7, [_]) => Some(0.0),
                    ("unary", 6, [x]) => of(*x),
                    // Cos, Sin, Frc of a finite input lie in [-1, 1].
                    ("unary", 12 | 13 | 22, [x]) => of(*x).map(|_| 0.0),
                    // Exp (2^x) of |x| <= 64 cannot overflow f32.
                    ("unary", 21, [x]) => (of(*x)? <= 6.0).then_some(64.0),
                    // Sqrt of a non-negative finite input is finite.
                    ("unary", 24, [x]) if self.float_non_negative_allow_nz(*x, depth) => {
                        held(of(*x)? * 0.5 + 1.0)
                    }
                    ("unary", 26..=29, [x]) => held(of(*x)? + 1.0),
                    // FMax/FMin: either both operands are finite, or one clamps
                    // the other side of an inner FMin/FMax with a finite constant.
                    ("binary", 35 | 36, [a, b]) => match (of(*a), of(*b)) {
                        (Some(x), Some(y)) => Some(x.max(y)),
                        _ => {
                            let other = if code == 35 { 36 } else { 35 };
                            let (inner, k) = if self.float_k(*b).is_some() {
                                (*a, *b)
                            } else {
                                (*b, *a)
                            };
                            let kb = self.float_k(k).and_then(|_| of(k))?;
                            let V::Inst(j) = inner else { return None };
                            let Op::Call(jf, jargs) = &self.insts[j as usize].op else {
                                return None;
                            };
                            if self.dx_f32_op(*jf, jargs) != Some(("binary", other)) {
                                return None;
                            }
                            let low = jargs[1..].iter().find(|x| self.float_k(**x).is_some())?;
                            Some(kb.max(of(*low)?))
                        }
                    },
                    ("tertiary", 46, [a, b, c]) => {
                        held((of(*a)? + of(*b)?).max(of(*c)?) + 1.0)
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// Whether `v` is provably never `NaN` (may be finite or `±Inf`).
    pub(crate) fn not_nan(&self, v: V, depth: u32) -> bool {
        if let Some(k) = self.float_k(v) {
            return !k.is_nan();
        }
        if self.bounded(v, depth).is_some() {
            return true;
        }
        let V::Inst(i) = v else { return false };
        let Some(depth) = depth.checked_sub(1) else {
            return false;
        };
        let inst = &self.insts[i as usize];
        if !self.is_f32(inst.ty) {
            return false;
        }
        match &inst.op {
            Op::Cast(Cast::Bitcast, x) => match *x {
                V::Inst(j) => match self.insts[j as usize].op {
                    Op::Cast(Cast::Bitcast, y) if self.type_of(y) == inst.ty => {
                        self.not_nan(y, depth)
                    }
                    _ => false,
                },
                _ => false,
            },
            Op::Bin(Bin::Sub, z, x)
                if matches!(
                    *z,
                    V::Const(ci) if matches!(self.consts[ci as usize].1, Const::Float(0x8000_0000))
                ) =>
            {
                self.not_nan(*x, depth)
            }
            Op::Select(_, a, b) => self.not_nan(*a, depth) && self.not_nan(*b, depth),
            Op::Call(f, args) => match self.dx_f32_op(*f, args) {
                Some(("unary", 7)) => true,
                Some(("unary", 6)) => args.get(1).is_some_and(|x| self.not_nan(*x, depth)),
                // DXIL FMax/FMin implements IEEE-754 maxNum/minNum (skips NaN).
                Some(("binary", 35 | 36)) => {
                    args.get(1).is_some_and(|x| self.not_nan(*x, depth))
                        || args.get(2).is_some_and(|x| self.not_nan(*x, depth))
                }
                _ => false,
            },
            _ => false,
        }
    }

    /// Whether `v` is provably never `±0.0` (`(bits & 0x7FFF_FFFF) != 0`).
    pub(crate) fn float_nonzero(&self, v: V, depth: u32) -> bool {
        if let Some(k) = self.float_k(v) {
            return k.to_bits() & 0x7FFF_FFFF != 0;
        }
        let V::Inst(i) = v else { return false };
        let Some(depth) = depth.checked_sub(1) else {
            return false;
        };
        let inst = &self.insts[i as usize];
        if !self.is_f32(inst.ty) {
            return false;
        }
        match &inst.op {
            Op::Cast(Cast::Bitcast, x) => match *x {
                V::Inst(j) => match self.insts[j as usize].op {
                    Op::Cast(Cast::Bitcast, y) if self.type_of(y) == inst.ty => {
                        self.float_nonzero(y, depth)
                    }
                    _ => false,
                },
                _ => false,
            },
            Op::Bin(Bin::Sub, z, x)
                if matches!(
                    *z,
                    V::Const(ci) if matches!(self.consts[ci as usize].1, Const::Float(0x8000_0000))
                ) =>
            {
                self.float_nonzero(*x, depth)
            }
            Op::Bin(Bin::Add, a, b) => {
                (self.float_k(*a).is_some_and(|k| k > 0.0 && k.is_finite())
                    && self.float_non_negative_allow_nz(*b, depth))
                    || (self.float_k(*b).is_some_and(|k| k > 0.0 && k.is_finite())
                        && self.float_non_negative_allow_nz(*a, depth))
            }
            Op::Select(_, a, b) => {
                self.float_nonzero(*a, depth) && self.float_nonzero(*b, depth)
            }
            Op::Call(f, args) => match (self.dx_f32_op(*f, args), args.get(1..)) {
                (Some(("unary", 6)), Some([x])) => self.float_nonzero(*x, depth),
                (Some(("binary", 35)), Some([a, b])) => {
                    self.float_k(*a).is_some_and(|k| k > 0.0)
                        || self.float_k(*b).is_some_and(|k| k > 0.0)
                }
                (Some(("binary", 36)), Some([a, b])) => {
                    self.float_k(*a).is_some_and(|k| k < 0.0)
                        || self.float_k(*b).is_some_and(|k| k < 0.0)
                }
                _ => false,
            },
            _ => false,
        }
    }

    /// Whether `v` is provably never `-0.0` (`0x8000_0000`). May be `+0.0`,
    /// any non-zero finite value, `±Inf`, or `NaN`.
    pub(crate) fn float_not_neg_zero(&self, v: V, depth: u32) -> bool {
        if let Some(k) = self.float_k(v) {
            return k.to_bits() != 0x8000_0000;
        }
        if self.float_nonzero(v, depth) {
            return true;
        }
        let V::Inst(i) = v else { return false };
        let Some(depth) = depth.checked_sub(1) else {
            return false;
        };
        let inst = &self.insts[i as usize];
        if !self.is_f32(inst.ty) {
            return false;
        }
        match &inst.op {
            Op::Cast(Cast::UiToFp | Cast::SiToFp, _) => true,
            Op::Cast(Cast::Bitcast, x) => match *x {
                V::Inst(j) => match self.insts[j as usize].op {
                    Op::Cast(Cast::Bitcast, y) if self.type_of(y) == inst.ty => {
                        self.float_not_neg_zero(y, depth)
                    }
                    _ => false,
                },
                _ => false,
            },
            // IEEE-754 §6.3 (round-to-nearest-even): `a + b == -0.0` iff BOTH
            // operands are `-0.0`.
            Op::Bin(Bin::Add, a, b) => {
                self.float_not_neg_zero(*a, depth) || self.float_not_neg_zero(*b, depth)
            }
            Op::Bin(Bin::Mul, a, b) => {
                self.same_bits(*a, *b)
                    || (self.float_pos_or_pos_zero(*a, depth)
                        && self.float_pos_or_pos_zero(*b, depth))
            }
            Op::Select(_, a, b) => {
                self.float_not_neg_zero(*a, depth) && self.float_not_neg_zero(*b, depth)
            }
            Op::Call(f, args) => match (self.dx_f32_op(*f, args), args.get(1..)) {
                (Some(("unary", 6 | 21)), Some([_])) => true,
                (Some(("binary", 35 | 36)), Some([a, b])) => {
                    self.float_not_neg_zero(*a, depth) && self.float_not_neg_zero(*b, depth)
                }
                _ => false,
            },
            _ => false,
        }
    }

    /// Whether `v` is provably non-negative with sign bit 0 (`>= +0.0`, never
    /// negative and never `-0.0`).
    pub(crate) fn float_pos_or_pos_zero(&self, v: V, depth: u32) -> bool {
        if let Some(k) = self.float_k(v) {
            return !k.is_nan() && (k.to_bits() & 0x8000_0000 == 0);
        }
        let V::Inst(i) = v else { return false };
        let Some(depth) = depth.checked_sub(1) else {
            return false;
        };
        let inst = &self.insts[i as usize];
        if !self.is_f32(inst.ty) {
            return false;
        }
        match &inst.op {
            Op::Cast(Cast::UiToFp, _) => true,
            Op::Cast(Cast::Bitcast, x) => match *x {
                V::Inst(j) => match self.insts[j as usize].op {
                    Op::Cast(Cast::Bitcast, y) if self.type_of(y) == inst.ty => {
                        self.float_pos_or_pos_zero(y, depth)
                    }
                    _ => false,
                },
                _ => false,
            },
            Op::Bin(Bin::Add, a, b) => {
                self.float_pos_or_pos_zero(*a, depth) && self.float_pos_or_pos_zero(*b, depth)
            }
            Op::Bin(Bin::Mul, a, b) => {
                self.same_bits(*a, *b)
                    || (self.float_pos_or_pos_zero(*a, depth)
                        && self.float_pos_or_pos_zero(*b, depth))
            }
            Op::Select(_, a, b) => {
                self.float_pos_or_pos_zero(*a, depth) && self.float_pos_or_pos_zero(*b, depth)
            }
            Op::Call(f, args) => match (self.dx_f32_op(*f, args), args.get(1..)) {
                (Some(("unary", 6 | 21)), Some([_])) => true,
                (Some(("binary", 35)), Some([a, b])) => {
                    self.float_k(*a).is_some_and(|k| k > 0.0)
                        || self.float_k(*b).is_some_and(|k| k > 0.0)
                        || (self.float_pos_or_pos_zero(*a, depth)
                            && self.float_pos_or_pos_zero(*b, depth))
                }
                (Some(("binary", 36)), Some([a, b])) => {
                    self.float_pos_or_pos_zero(*a, depth) && self.float_pos_or_pos_zero(*b, depth)
                }
                _ => false,
            },
            _ => false,
        }
    }

    /// Whether `v` is provably never `< 0.0` (`-0.0`, `+0.0`, `> 0`, `+Inf`,
    /// and `NaN` are all allowed). Used to prove `v_fract_f32` cannot hit the
    /// tiny-negative `1.0` rounding corner case.
    pub(crate) fn float_non_negative_allow_nz(&self, v: V, depth: u32) -> bool {
        if let Some(k) = self.float_k(v) {
            return !(k < 0.0);
        }
        if self.float_pos_or_pos_zero(v, depth) {
            return true;
        }
        let V::Inst(i) = v else { return false };
        let Some(depth) = depth.checked_sub(1) else {
            return false;
        };
        let inst = &self.insts[i as usize];
        if !self.is_f32(inst.ty) {
            return false;
        }
        match &inst.op {
            Op::Cast(Cast::Bitcast, x) => match *x {
                V::Inst(j) => match self.insts[j as usize].op {
                    Op::Cast(Cast::Bitcast, y) if self.type_of(y) == inst.ty => {
                        self.float_non_negative_allow_nz(y, depth)
                    }
                    _ => false,
                },
                _ => false,
            },
            Op::Bin(Bin::Add | Bin::Mul, a, b) => {
                self.float_non_negative_allow_nz(*a, depth)
                    && self.float_non_negative_allow_nz(*b, depth)
            }
            Op::Select(_, a, b) => {
                self.float_non_negative_allow_nz(*a, depth)
                    && self.float_non_negative_allow_nz(*b, depth)
            }
            Op::Call(f, args) => match (self.dx_f32_op(*f, args), args.get(1..)) {
                (Some(("unary", 6 | 7 | 21 | 22 | 24)), Some([_])) => true,
                // FMax skips NaN, so require the non-negative operand to also
                // be non-NaN unless both operands are non-negative.
                (Some(("binary", 35)), Some([a, b])) => {
                    (self.float_non_negative_allow_nz(*a, depth) && self.not_nan(*a, depth))
                        || (self.float_non_negative_allow_nz(*b, depth) && self.not_nan(*b, depth))
                        || (self.float_non_negative_allow_nz(*a, depth)
                            && self.float_non_negative_allow_nz(*b, depth))
                }
                (Some(("binary", 36)), Some([a, b])) => {
                    self.float_non_negative_allow_nz(*a, depth)
                        && self.float_non_negative_allow_nz(*b, depth)
                }
                _ => false,
            },
            _ => false,
        }
    }

    /// Whether `v` is provably never `+0.0` (`0x0000_0000`).
    fn float_not_pos_zero(&self, v: V, depth: u32) -> bool {
        if let Some(k) = self.float_k(v) {
            return k.to_bits() != 0;
        }
        self.float_nonzero(v, depth)
    }

    /// The constant `x` of the integer type `ty`, kept sign-extended as
    /// [`Module::i32`] and [`Module::bool`] keep theirs.
    fn k(&mut self, ty: Ty, bits: u32, x: u64) -> V {
        self.int_const(ty, signed(bits, x & mask(bits)))
    }

    /// The value instruction `i` is known to have without it, if any: a
    /// constant or an earlier value. May instead give `i` simpler operands
    /// (`changed` says so).
    fn simplify(&mut self, i: u32, changed: &mut bool) -> Option<V> {
        let ty = self.insts[i as usize].ty;
        match self.insts[i as usize].op {
            Op::Bin(op, a, b) => {
                // Floats: `x + 0.0`, `x * 1.0` and the like are not `x` for
                // every `x` (a negative zero, a signalling NaN).
                let bits = self.int_bits(ty)?;
                let (ka, kb) = (self.int_k(a).map(|k| k.1), self.int_k(b).map(|k| k.1));
                let all = mask(bits);
                if let (Some(x), Some(y)) = (ka, kb) {
                    // A shift by the width or more and a division by zero
                    // have no defined result: left as they are.
                    let amount = u32::try_from(y).ok().filter(|&y| y < bits);
                    let r = match op {
                        Bin::Add => x.wrapping_add(y),
                        Bin::Sub => x.wrapping_sub(y),
                        Bin::Mul => x.wrapping_mul(y),
                        Bin::And => x & y,
                        Bin::Or => x | y,
                        Bin::Xor => x ^ y,
                        Bin::Shl => x << amount?,
                        Bin::LShr => x >> amount?,
                        Bin::AShr => (signed(bits, x) >> amount?) as u64,
                        Bin::UDiv if y != 0 => x / y,
                        Bin::URem if y != 0 => x % y,
                        _ => return None,
                    };
                    return Some(self.k(ty, bits, r));
                }
                match (op, ka, kb) {
                    (
                        Bin::Add | Bin::Sub | Bin::Or | Bin::Xor | Bin::Shl | Bin::LShr | Bin::AShr,
                        _,
                        Some(0),
                    ) => Some(a),
                    (Bin::Add | Bin::Or | Bin::Xor, Some(0), _) => Some(b),
                    (Bin::Mul | Bin::UDiv, _, Some(1)) => Some(a),
                    (Bin::Mul, Some(1), _) => Some(b),
                    (Bin::Mul | Bin::And, _, Some(0)) => Some(b),
                    (Bin::Mul | Bin::And, Some(0), _) => Some(a),
                    (Bin::And, _, Some(m)) if m == all => Some(a),
                    (Bin::And, Some(m), _) if m == all => Some(b),
                    (Bin::Or, _, Some(m)) if m == all => Some(b),
                    (Bin::Or, Some(m), _) if m == all => Some(a),
                    (Bin::And | Bin::Or, _, _) if a == b => Some(a),
                    // Associative integer add-of-add with constants:
                    // `(x + c1) + c2 == x + (c1 + c2)` (exposes common bases
                    // to `coalesce_raw_loads` and drops chained offset adds).
                    (Bin::Add, Some(c2), None) | (Bin::Add, None, Some(c2)) => {
                        let inner = if ka.is_some() { b } else { a };
                        if let V::Inst(ii) = inner
                            && let Op::Bin(Bin::Add, x, y) = self.insts[ii as usize].op
                        {
                            let inner_const = if let Some((_, c1)) = self.int_k(y) {
                                Some((x, c1))
                            } else {
                                self.int_k(x).map(|(_, c1)| (y, c1))
                            };
                            if let Some((base, c1)) = inner_const {
                                let sum = c1.wrapping_add(c2) & all;
                                if sum == 0 {
                                    return Some(base);
                                }
                                let sum_v = self.k(ty, bits, sum);
                                self.insts[i as usize].op = Op::Bin(Bin::Add, base, sum_v);
                                *changed = true;
                            }
                        }
                        None
                    }
                    _ => {
                        // Drop a dead masked half of `or(and(x, c1), y)` when an
                        // outer shift or mask discards every bit in `c1` (common
                        // after 16-bit VALU writes that preserve the high 16 bits
                        // of a dead register such as an entry barycentric VGPR).
                        let dead_or_side = |or_v: V, kills: &dyn Fn(u64) -> bool| -> Option<V> {
                            let V::Inst(oi) = or_v else { return None };
                            let Op::Bin(Bin::Or, p, q) = self.insts[oi as usize].op else {
                                return None;
                            };
                            let masked_dead = |side: V| -> bool {
                                let V::Inst(ai) = side else { return false };
                                let Op::Bin(Bin::And, x, y) = self.insts[ai as usize].op else {
                                    return false;
                                };
                                self.int_k(x).is_some_and(|(_, m)| kills(m))
                                    || self.int_k(y).is_some_and(|(_, m)| kills(m))
                            };
                            if masked_dead(p) {
                                Some(q)
                            } else if masked_dead(q) {
                                Some(p)
                            } else {
                                None
                            }
                        };
                        match (op, ka, kb) {
                            (Bin::Shl, _, Some(k)) if k < u64::from(bits) => {
                                let k = k as u32;
                                if let Some(keep) = dead_or_side(a, &|c1| (c1 << k) & all == 0) {
                                    self.insts[i as usize].op = Op::Bin(Bin::Shl, keep, b);
                                    *changed = true;
                                }
                            }
                            (Bin::LShr, _, Some(k)) if k < u64::from(bits) => {
                                let k = k as u32;
                                if let Some(keep) = dead_or_side(a, &|c1| (c1 & all) >> k == 0) {
                                    self.insts[i as usize].op = Op::Bin(Bin::LShr, keep, b);
                                    *changed = true;
                                }
                            }
                            (Bin::And, _, Some(m)) => {
                                if let Some(keep) = dead_or_side(a, &|c1| (c1 & m) == 0) {
                                    self.insts[i as usize].op = Op::Bin(Bin::And, keep, b);
                                    *changed = true;
                                }
                            }
                            (Bin::And, Some(m), _) => {
                                if let Some(keep) = dead_or_side(b, &|c1| (c1 & m) == 0) {
                                    self.insts[i as usize].op = Op::Bin(Bin::And, a, keep);
                                    *changed = true;
                                }
                            }
                            _ => {}
                        }
                        None
                    }
                }
            }
            Op::Cmp(p, a, b) if p <= pred::TRUE => {
                // Two float constants compare one way (`v_mul_legacy` asks
                // whether its constant factor is zero).
                let (V::Const(x), V::Const(y)) = (a, b) else {
                    return None;
                };
                let (Const::Float(x), Const::Float(y)) =
                    (&self.consts[x as usize].1, &self.consts[y as usize].1)
                else {
                    return None;
                };
                let (x, y) = (f32::from_bits(*x), f32::from_bits(*y));
                let unordered = x.is_nan() || y.is_nan();
                // Bits 0-2 of a predicate: equal, greater, less; bit 3:
                // or unordered.
                let holds = if unordered {
                    p & 8 != 0
                } else {
                    (p & 1 != 0 && x == y) || (p & 2 != 0 && x > y) || (p & 4 != 0 && x < y)
                };
                Some(self.bool(holds))
            }
            Op::Cmp(p, a, b) => {
                if !(pred::EQ..=pred::SLE).contains(&p) {
                    return None;
                }
                let holds = |x: u64, y: u64, bits: u32| {
                    let (sx, sy) = (signed(bits, x), signed(bits, y));
                    match p {
                        pred::EQ => x == y,
                        pred::NE => x != y,
                        pred::IUGT => x > y,
                        pred::IUGE => x >= y,
                        pred::IULT => x < y,
                        pred::IULE => x <= y,
                        pred::SGT => sx > sy,
                        pred::SGE => sx >= sy,
                        pred::SLT => sx < sy,
                        _ => sx <= sy,
                    }
                };
                if let (Some((bits, x)), Some((_, y))) = (self.int_k(a), self.int_k(b)) {
                    return Some(self.bool(holds(x, y, bits)));
                }
                if a == b {
                    return Some(self.bool(holds(0, 0, 32)));
                }
                // A bit widened and tested again is the bit.
                if p == pred::NE
                    && self.int_k(b).is_some_and(|k| k.1 == 0)
                    && let V::Inst(z) = a
                    && let Op::Cast(Cast::ZExt, x) = self.insts[z as usize].op
                    && self.type_of(x) == ty
                {
                    return Some(x);
                }
                None
            }
            Op::Select(c, mut t, mut f) => {
                if let Some((_, c)) = self.int_k(c) {
                    return Some(if c != 0 { t } else { f });
                }
                // Under the same condition an inner select is one of its
                // operands (two guarded writes of one register in a row).
                let mut inner = false;
                while let V::Inst(x) = t
                    && let Op::Select(c2, t2, _) = self.insts[x as usize].op
                    && c2 == c
                {
                    (t, inner) = (t2, true);
                }
                while let V::Inst(x) = f
                    && let Op::Select(c2, _, f2) = self.insts[x as usize].op
                    && c2 == c
                {
                    (f, inner) = (f2, true);
                }
                if t == f {
                    return Some(t);
                }
                if self.int_bits(ty) == Some(1)
                    && self.int_k(t).is_some_and(|k| k.1 == 1)
                    && self.int_k(f).is_some_and(|k| k.1 == 0)
                {
                    return Some(c);
                }
                // `select (fcmp oeq x, 0.0), +0.0, (fmul x, x)` is bit-exact `fmul x, x`
                // across all 2^32 f32 patterns (`(±0)^2 = +0`, `(±Inf)^2 = +Inf`, `NaN^2 = NaN`).
                if matches!(t, V::Const(ci) if matches!(self.consts[ci as usize].1, Const::Float(0)))
                    && let V::Inst(mi) = f
                    && let Op::Bin(Bin::Mul, x, y) = self.insts[mi as usize].op
                    && x == y
                    && let V::Inst(ci) = c
                    && let Op::Cmp(pred::OEQ, ca, cb) = self.insts[ci as usize].op
                    && ((ca == x
                        && self.float_value(cb).is_some_and(|v| v == 0.0))
                        || (cb == x
                            && self.float_value(ca).is_some_and(|v| v == 0.0)))
                {
                    return Some(f);
                }
                if inner {
                    self.insts[i as usize].op = Op::Select(c, t, f);
                    *changed = true;
                }
                None
            }
            Op::Cast(op, x) => {
                let from = self.type_of(x);
                let through = match x {
                    V::Inst(y) => match self.insts[y as usize].op {
                        Op::Cast(op2, z) => Some((op2, z)),
                        _ => None,
                    },
                    _ => None,
                };
                match op {
                    Cast::Bitcast => {
                        if from == ty {
                            return Some(x);
                        }
                        // There and back: the bits are the same.
                        if let Some((Cast::Bitcast, z)) = through
                            && self.type_of(z) == ty
                        {
                            return Some(z);
                        }
                        let V::Const(c) = x else { return None };
                        let to_float = matches!(self.types[ty.0 as usize], TypeDef::Float);
                        match self.consts[c as usize].1 {
                            Const::Int(bits) if to_float && self.int_bits(from) == Some(32) => {
                                Some(self.f32(bits as u32))
                            }
                            Const::Float(bits) if self.int_bits(ty) == Some(32) => {
                                Some(self.int_const(ty, i64::from(bits as i32)))
                            }
                            _ => None,
                        }
                    }
                    Cast::ZExt | Cast::SExt | Cast::Trunc => {
                        let to = self.int_bits(ty)?;
                        if let Some((bits, k)) = self.int_k(x) {
                            let k = if op == Cast::SExt {
                                signed(bits, k) as u64
                            } else {
                                k
                            };
                            return Some(self.k(ty, to, k));
                        }
                        // Widened and narrowed again.
                        if op == Cast::Trunc
                            && let Some((Cast::ZExt | Cast::SExt, z)) = through
                            && self.type_of(z) == ty
                        {
                            return Some(z);
                        }
                        None
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// Whether a `dx.op.rawBufferLoad.*` call reads one of Victory's immutable
    /// draw-constant metadata tables (`SLOT_TABLE_STRIDES = 250`,
    /// `SLOT_VIEWS = 252`, `SLOT_MEMORY = 254`, `SLOT_STRIDES = 255`), whose
    /// contents never change during a draw or dispatch.
    fn is_pure_metadata_table_load(&self, f: u32, args: &[V]) -> bool {
        use super::super::shader::{
            ROOT_CBV_SPACE, SLOT_MEMORY, SLOT_STRIDES, SLOT_TABLE_STRIDES, SLOT_VIEWS,
        };
        if !self.funcs[f as usize]
            .name
            .starts_with("dx.op.rawBufferLoad.")
        {
            return false;
        }
        let Some(&mut_handle) = args.get(1) else {
            return false;
        };
        let mut h = mut_handle;
        // Unwrap `dx.op.annotateHandle` -> `dx.op.createHandleFromBinding`.
        if let V::Inst(id) = h
            && let Op::Call(cf, cargs) = &self.insts[id as usize].op
            && self.funcs[*cf as usize].name == "dx.op.annotateHandle"
            && let Some(&inner) = cargs.get(1)
        {
            h = inner;
        }
        let V::Inst(id) = h else { return false };
        let Op::Call(cf, cargs) = &self.insts[id as usize].op else {
            return false;
        };
        if self.funcs[*cf as usize].name != "dx.op.createHandleFromBinding" {
            return false;
        }
        let Some(&V::Const(ci)) = cargs.get(1) else {
            return false;
        };
        let Const::Aggregate(elems) = &self.consts[ci as usize].1 else {
            return false;
        };
        if elems.len() != 4 {
            return false;
        }
        let int_elem = |k: usize| match self.consts[elems[k] as usize].1 {
            Const::Int(v) => Some(v as u32),
            _ => None,
        };
        let (Some(lo), Some(hi), Some(space), Some(class)) =
            (int_elem(0), int_elem(1), int_elem(2), int_elem(3))
        else {
            return false;
        };
        space == ROOT_CBV_SPACE
            && class == 0
            && lo == hi
            && matches!(
                lo,
                x if x == u32::from(SLOT_TABLE_STRIDES)
                    || x == u32::from(SLOT_VIEWS)
                    || x == u32::from(SLOT_MEMORY)
                    || x == u32::from(SLOT_STRIDES)
            )
    }

    /// What kind of instruction `i` is for merging.
    fn class(&self, effects: &[Effect], i: u32) -> Effect {
        match &self.insts[i as usize].op {
            Op::Bin(..)
            | Op::Cast(..)
            | Op::Cmp(..)
            | Op::Select(..)
            | Op::Extract(..)
            | Op::Gep(..) => Effect::Pure,
            Op::Load(p) => {
                // A constant table reads the same whatever is stored.
                let base = match *p {
                    V::Inst(g) => match &self.insts[g as usize].op {
                        Op::Gep(_, base, _) => *base,
                        _ => *p,
                    },
                    _ => *p,
                };
                match base {
                    V::Global(g) if self.globals[g as usize].init.is_some() => Effect::Pure,
                    _ => Effect::Reads,
                }
            }
            Op::Store(..) | Op::AtomicRmw(..) | Op::CmpXchg(..) => Effect::Writes,
            Op::Phi(_) => Effect::Lanes,
            Op::Call(f, args) => {
                let e = effects[*f as usize];
                if e == Effect::Reads && self.is_pure_metadata_table_load(*f, args) {
                    Effect::Pure
                } else {
                    e
                }
            }
        }
    }

    fn key(&self, i: u32) -> Option<Key> {
        let inst = &self.insts[i as usize];
        Some(match &inst.op {
            Op::Bin(op, a, b) => {
                // Integer sums, products and bit operations whichever way
                // round; float ones as written (which NaN an operation of
                // two NaNs gives is the hardware's choice by operand).
                let commutes = matches!(op, Bin::Add | Bin::Mul | Bin::And | Bin::Or | Bin::Xor)
                    && self.int_bits(inst.ty).is_some();
                let (a, b) = if commutes && b < a {
                    (*b, *a)
                } else {
                    (*a, *b)
                };
                Key::Bin(*op as u8, a, b)
            }
            Op::Cast(op, v) => Key::Cast(*op as u8, inst.ty, *v),
            Op::Cmp(p, a, b) => Key::Cmp(*p, *a, *b),
            Op::Select(c, t, f) => Key::Select(*c, *t, *f),
            Op::Extract(a, n) => Key::Extract(*a, *n),
            Op::Call(f, args) => Key::Call(*f, args.clone()),
            Op::Gep(ty, p, idx) => Key::Gep(*ty, *p, idx.clone()),
            Op::Load(p) => Key::Load(*p),
            _ => return None,
        })
    }

    /// [`FOLD`], [`REUSE`], [`FACTS`] and [`LOADS`]: one walk over the
    /// dominator tree. Returns whether anything changed.
    fn merge(&mut self, passes: u32) -> bool {
        if self.blocks.is_empty() {
            return false;
        }
        let effects = self.effects();
        let tree = self.dominator_tree();
        let mut repl: Vec<Option<V>> = vec![None; self.insts.len()];
        let mut changed = false;
        // Joins of one value: until none is left (one gone can leave
        // another with one value).
        if passes & FOLD != 0 {
            let mut again = true;
            while again {
                again = false;
                for &b in &tree.order {
                    for &p in &self.blocks[b].phis {
                        if repl[p as usize].is_some() {
                            continue;
                        }
                        let Op::Phi(incoming) = &self.insts[p as usize].op else {
                            continue;
                        };
                        let mut one = None;
                        let mut same = true;
                        for &(v, _) in incoming {
                            let v = resolve(&repl, v);
                            if v == V::Inst(p) {
                                continue;
                            }
                            match one {
                                None => one = Some(v),
                                Some(o) if o == v => {}
                                Some(_) => same = false,
                            }
                        }
                        if let (true, Some(v)) = (same, one) {
                            repl[p as usize] = Some(v);
                            again = true;
                            changed = true;
                        }
                    }
                }
            }
        }
        // The values every path to the block being walked has computed
        // (those of the blocks above it in the tree, and its own so far).
        let mut known: HashMap<Key, V> = HashMap::new();
        // The reads of the block being walked since its last store.
        let mut read: HashMap<Key, V> = HashMap::new();
        let mut facts = Facts::default();
        let (yes, no) = (self.bool(true), self.bool(false));
        enum Step {
            Enter(usize),
            Leave(Vec<Key>, Learned),
        }
        let mut steps = vec![Step::Enter(0)];
        while let Some(step) = steps.pop() {
            let b = match step {
                Step::Enter(b) => b,
                Step::Leave(keys, learned) => {
                    for k in keys {
                        known.remove(&k);
                    }
                    for v in learned.bits {
                        facts.bits.remove(&v);
                    }
                    for k in learned.compares {
                        facts.compares.remove(&k);
                    }
                    continue;
                }
            };
            let mut added = Vec::new();
            // Reached from one block alone, by one side of its branch: the
            // condition is that side's, here and in every block below.
            let mut learned = Learned::default();
            if passes & FACTS != 0
                && let [p] = tree.preds[b][..]
                && let Some(Term::CondBr(c, t, f)) = self.blocks[p].term
                && t != f
            {
                self.learn(resolve(&repl, c), b == t, &mut facts, &mut learned, 8);
            }
            let known_bit = |facts: &Facts, v: V| match facts.bits.get(&v) {
                Some(true) => yes,
                Some(false) => no,
                None => v,
            };
            read.clear();
            // Two joins of the same values from the same blocks are one.
            if passes & REUSE != 0 && self.blocks[b].phis.len() > 1 {
                let mut joins: HashMap<Vec<(usize, V)>, V> = HashMap::new();
                for &p in &self.blocks[b].phis {
                    if repl[p as usize].is_some() {
                        continue;
                    }
                    let ty = self.insts[p as usize].ty;
                    let Op::Phi(incoming) = &self.insts[p as usize].op else {
                        continue;
                    };
                    let mut key: Vec<(usize, V)> = incoming
                        .iter()
                        .map(|&(v, from)| (from, resolve(&repl, v)))
                        .collect();
                    key.sort();
                    // The type too: an undefined value of one type is not
                    // that of another.
                    key.push((ty.0 as usize, V::Global(u32::MAX)));
                    match joins.get(&key) {
                        Some(&first) => {
                            repl[p as usize] = Some(first);
                            changed = true;
                        }
                        None => {
                            joins.insert(key, V::Inst(p));
                        }
                    }
                }
            }
            let body = std::mem::take(&mut self.blocks[b].body);
            let mut kept = Vec::with_capacity(body.len());
            for i in body {
                map_operands(&mut self.insts[i as usize].op, |v| {
                    let v = resolve(&repl, v);
                    let bit = known_bit(&facts, v);
                    changed |= bit != v;
                    bit
                });
                if let Op::Cmp(p, x, y) = self.insts[i as usize].op
                    && let Some(&holds) = facts.compares.get(&(p, x, y))
                {
                    repl[i as usize] = Some(if holds { yes } else { no });
                    changed = true;
                    continue;
                }
                if passes & FOLD != 0
                    && let Some(v) = self.simplify(i, &mut changed)
                {
                    repl[i as usize] = Some(v);
                    changed = true;
                    continue;
                }
                let class = self.class(&effects, i);
                if passes & REUSE != 0
                    && class == Effect::Pure
                    && let Some(key) = self.key(i)
                {
                    if let Some(&first) = known.get(&key) {
                        repl[i as usize] = Some(first);
                        changed = true;
                        continue;
                    }
                    known.insert(key.clone(), V::Inst(i));
                    added.push(key);
                }
                // Reads of memory, and wave operations (the same lanes run
                // the whole block): the block's own earlier one.
                if passes & LOADS != 0
                    && matches!(class, Effect::Reads | Effect::Lanes)
                    && let Some(key) = self.key(i)
                {
                    if let Some(&first) = read.get(&key) {
                        repl[i as usize] = Some(first);
                        changed = true;
                        continue;
                    }
                    read.insert(key, V::Inst(i));
                }
                if class == Effect::Writes {
                    read.clear();
                }
                kept.push(i);
            }
            self.blocks[b].body = kept;
            if let Some(Term::CondBr(c, t, f)) = self.blocks[b].term {
                let c = known_bit(&facts, resolve(&repl, c));
                self.blocks[b].term = Some(Term::CondBr(c, t, f));
            }
            steps.push(Step::Leave(added, learned));
            for &c in tree.children[b].iter().rev() {
                steps.push(Step::Enter(c));
            }
        }
        if !changed {
            return false;
        }
        // Everything that read a replaced value reads its replacement
        // (joins read values from blocks walked after them).
        for blk in &mut self.blocks {
            blk.phis.retain(|&p| repl[p as usize].is_none());
            if let Some(Term::CondBr(c, t, f)) = blk.term {
                blk.term = Some(Term::CondBr(resolve(&repl, c), t, f));
            }
        }
        let (blocks, insts) = (&self.blocks, &mut self.insts);
        for blk in blocks {
            for &i in blk.phis.iter().chain(&blk.body) {
                map_operands(&mut insts[i as usize].op, |v| resolve(&repl, v));
            }
        }
        true
    }

    /// If `v` is `fcmp oeq x, ±0.0`, returns `x`.
    fn zero_fcmp_arg(&self, v: V) -> Option<V> {
        let V::Inst(id) = v else { return None };
        let Op::Cmp(pred::OEQ, a, b) = self.insts[id as usize].op else {
            return None;
        };
        if self.float_k(b).is_some_and(|k| k == 0.0) {
            Some(a)
        } else if self.float_k(a).is_some_and(|k| k == 0.0) {
            Some(b)
        } else {
            None
        }
    }

    /// Whether `cond` is a boolean that holds whenever `x == ±0.0` (either
    /// `fcmp oeq x, 0.0` directly or an `or` containing it).
    fn cond_checks_zero(&self, cond: V, x: V) -> bool {
        if self.zero_fcmp_arg(cond).is_some_and(|a| self.same_bits(a, x)) {
            return true;
        }
        if let V::Inst(id) = cond
            && let Op::Bin(Bin::Or, p, q) = self.insts[id as usize].op
        {
            return self.zero_fcmp_arg(p).is_some_and(|a| self.same_bits(a, x))
                || self.zero_fcmp_arg(q).is_some_and(|a| self.same_bits(a, x));
        }
        false
    }

    /// Matches a `max_min` `-0 < +0` tie-breaker `select zeros, signed, FMax/FMin(a, b)`
    /// (both unfolded 2-dynamic and `merge()`-folded constant-zero shapes).
    /// Returns `Some((a, b))` when `select` and `FMax/FMin(a, b)` can only ever
    /// differ in the sign bit of `±0.0`.
    fn match_max_min_tiebreaker(&self, zeros: V, signed_v: V, r_v: V) -> Option<(V, V)> {
        let V::Inst(ri) = r_v else { return None };
        let Op::Call(rf, rargs) = &self.insts[ri as usize].op else {
            return None;
        };
        if !matches!(self.dx_f32_op(*rf, rargs), Some(("binary", 35 | 36))) {
            return None;
        }
        let (&a, &b) = (rargs.get(1)?, rargs.get(2)?);
        let V::Inst(zi) = zeros else { return None };
        let Op::Cmp(pred::EQ, cl, cr) = self.insts[zi as usize].op else {
            return None;
        };
        let masked = if self.int_k(cr).is_some_and(|(_, k)| k == 0) {
            cl
        } else if self.int_k(cl).is_some_and(|(_, k)| k == 0) {
            cr
        } else {
            return None;
        };
        let V::Inst(mi) = masked else { return None };
        let Op::Bin(Bin::And, ml, mr) = self.insts[mi as usize].op else {
            return None;
        };
        let either = if self.int_k(mr).is_some_and(|(_, k)| k == 0x7FFF_FFFF) {
            ml
        } else if self.int_k(ml).is_some_and(|(_, k)| k == 0x7FFF_FFFF) {
            mr
        } else {
            return None;
        };
        let checks_both_zero = if let V::Inst(oi) = either
            && let Op::Bin(Bin::Or, x, y) = self.insts[oi as usize].op
        {
            (self.same_bits(x, a) && self.same_bits(y, b))
                || (self.same_bits(x, b) && self.same_bits(y, a))
                || (self.same_bits(x, a) && self.int_k(y).is_some_and(|(_, k)| k == 0x8000_0000) && self.float_k(b).is_some_and(|k| k == 0.0))
                || (self.same_bits(y, a) && self.int_k(x).is_some_and(|(_, k)| k == 0x8000_0000) && self.float_k(b).is_some_and(|k| k == 0.0))
                || (self.same_bits(x, b) && self.int_k(y).is_some_and(|(_, k)| k == 0x8000_0000) && self.float_k(a).is_some_and(|k| k == 0.0))
                || (self.same_bits(y, b) && self.int_k(x).is_some_and(|(_, k)| k == 0x8000_0000) && self.float_k(a).is_some_and(|k| k == 0.0))
        } else {
            (self.same_bits(either, a) && self.float_k(b).is_some_and(|k| k == 0.0))
                || (self.same_bits(either, b) && self.float_k(a).is_some_and(|k| k == 0.0))
        };
        if !checks_both_zero {
            return None;
        }
        // Verify `signed_v` is always `±0.0` when `zeros` holds.
        let zero_bit_src = |v: V| {
            self.same_bits(v, a)
                || self.same_bits(v, b)
                || self.int_k(v).is_some_and(|(_, k)| k == 0 || k == 0x8000_0000)
        };
        let signed_is_zero = self.float_k(signed_v).is_some_and(|k| k == 0.0)
            || self.same_bits(signed_v, a)
            || self.same_bits(signed_v, b)
            || (matches!(
                signed_v,
                V::Inst(si)
                    if matches!(
                        self.insts[si as usize].op,
                        Op::Cast(Cast::Bitcast, V::Inst(bi))
                            if matches!(
                                self.insts[bi as usize].op,
                                Op::Bin(Bin::And | Bin::Or, x, y) if zero_bit_src(x) && zero_bit_src(y)
                            )
                    )
            ));
        signed_is_zero.then_some((a, b))
    }

    /// Whether replacing `-0.0` with `+0.0` (or vice versa) in `root` is 100%
    /// unobservable by every transitive live consumer in the shader.
    fn sign_of_zero_unobserved(
        &self,
        root: u32,
        uses: &[Vec<u32>],
        used_in_term: &[bool],
    ) -> bool {
        // `(inst, is_i32_carrier)`: when `is_i32_carrier` is true, `inst` is a
        // 32-bit integer value carrying the unmodified bit pattern of the float
        // across `write_f` bitcasts, lane-guard `Select`s, or `Phi`s; any
        // non-transport integer op (bit math, compare, store, call) immediately
        // rejects elision.
        let mut work = vec![(root, false)];
        let mut seen = std::collections::HashSet::new();
        seen.insert((root, false));
        while let Some((cur, is_i32)) = work.pop() {
            if used_in_term[cur as usize] {
                return false;
            }
            let cur_v = V::Inst(cur);
            for &u in &uses[cur as usize] {
                let u_inst = &self.insts[u as usize];
                let next: Option<bool> = if is_i32 {
                    match &u_inst.op {
                        Op::Cast(Cast::Bitcast, _) if self.is_f32(u_inst.ty) => Some(false),
                        Op::Select(_, t, f)
                            if self.int_bits(u_inst.ty) == Some(32)
                                && (*t == cur_v || *f == cur_v) =>
                        {
                            Some(true)
                        }
                        Op::Phi(_) if self.int_bits(u_inst.ty) == Some(32) => Some(true),
                        _ => return false,
                    }
                } else {
                    match &u_inst.op {
                        // Float comparisons treat -0.0 and +0.0 identically.
                        Op::Cmp(p, _, _) if *p <= pred::TRUE => None,
                        // Integer conversions map both -0.0 and +0.0 to 0.
                        Op::Cast(Cast::FpToUi | Cast::FpToSi, _) => None,
                        // Bitcast to i32: only sound if every transitive i32
                        // consumer merely transports the bits back to f32.
                        Op::Cast(Cast::Bitcast, _) if self.int_bits(u_inst.ty) == Some(32) => {
                            Some(true)
                        }
                        // Float add: `±0.0 + other` absorbs sign-of-zero whenever
                        // `other != -0.0`; otherwise propagates sign-of-zero only.
                        Op::Bin(Bin::Add, x, y) if self.is_f32(u_inst.ty) => {
                            if *x == cur_v && *y != cur_v && self.float_not_neg_zero(*y, 6) {
                                None
                            } else if *y == cur_v && *x != cur_v && self.float_not_neg_zero(*x, 6) {
                                None
                            } else {
                                Some(false)
                            }
                        }
                        // Float sub: `(±0.0) - y` absorbs when `y != +0.0`;
                        // `x - (±0.0)` absorbs when `x != -0.0`.
                        Op::Bin(Bin::Sub, x, y) if self.is_f32(u_inst.ty) => {
                            if *x == cur_v && *y != cur_v && self.float_not_pos_zero(*y, 6) {
                                None
                            } else if *y == cur_v && *x != cur_v && self.float_not_neg_zero(*x, 6) {
                                None
                            } else {
                                Some(false)
                            }
                        }
                        // Float mul: `x * x` always produces +0.0; `x * bounded`
                        // cannot create `0 * Inf` NaN and at worst flips sign-of-zero;
                        // an un-elided `legacy_mul` select overrides `±0.0 * other`
                        // with `+0.0`.
                        Op::Bin(Bin::Mul, x, y) if self.is_f32(u_inst.ty) => {
                            if self.same_bits(*x, *y) {
                                None
                            } else {
                                let other = if *x == cur_v { *y } else { *x };
                                if self.bounded(other, 8).is_some() {
                                    Some(false)
                                } else if let [sel] = uses[u as usize][..]
                                    && !used_in_term[u as usize]
                                    && let Op::Select(cond, t, f) = self.insts[sel as usize].op
                                    && f == V::Inst(u)
                                    && matches!(
                                        t,
                                        V::Const(ci) if matches!(self.consts[ci as usize].1, Const::Float(0))
                                    )
                                    && self.cond_checks_zero(cond, cur_v)
                                {
                                    None
                                } else {
                                    return false;
                                }
                            }
                        }
                        Op::Select(_, t, f)
                            if self.is_f32(u_inst.ty) && (*t == cur_v || *f == cur_v) =>
                        {
                            Some(false)
                        }
                        Op::Phi(_) if self.is_f32(u_inst.ty) => Some(false),
                        Op::Call(f, args) => match (self.dx_f32_op(*f, args), args.get(1..)) {
                            // FAbs, Cos, Exp, Frc produce bit-identical results for ±0.0
                            // (Frc(±0.0) = (±0.0) - floor(±0.0) = +0.0 per IEEE-754 §6.3).
                            (Some(("unary", 6 | 12 | 21 | 22)), Some([_])) => None,
                            // Saturate at worst preserves ±0.0.
                            (Some(("unary", 7)), Some([_])) => Some(false),
                            // FMax with positive const / FMin with negative const
                            // absorbs ±0.0; otherwise propagates at worst ±0.0.
                            (Some(("binary", 35)), Some([x, y])) => {
                                let other = if *x == cur_v { *y } else { *x };
                                (!self.float_k(other).is_some_and(|k| k > 0.0)).then_some(false)
                            }
                            (Some(("binary", 36)), Some([x, y])) => {
                                let other = if *x == cur_v { *y } else { *x };
                                (!self.float_k(other).is_some_and(|k| k < 0.0)).then_some(false)
                            }
                            (Some(("tertiary", 46)), Some([a, b, c])) => {
                                let in_a = *a == cur_v;
                                let in_b = *b == cur_v;
                                let in_c = *c == cur_v;
                                if in_a && in_b {
                                    // `v * v` is always `+0.0` for `v = ±0.0`.
                                    in_c.then_some(false)
                                } else if in_a || in_b {
                                    let other = if in_a { *b } else { *a };
                                    if self.bounded(other, 8).is_none() {
                                        return false;
                                    }
                                    if !in_c && self.float_not_neg_zero(*c, 6) {
                                        None
                                    } else {
                                        Some(false)
                                    }
                                } else {
                                    Some(false)
                                }
                            }
                            _ => return false,
                        },
                        _ => return false,
                    }
                };
                if let Some(next_i32) = next
                    && seen.insert((u, next_i32))
                {
                    if seen.len() > 48 {
                        return false;
                    }
                    work.push((u, next_i32));
                }
            }
        }
        true
    }

    /// Exact post-merge elision of `legacy_mul`, `max_min` `-0 < +0` tie-breakers,
    /// and `fract` clamps whose guards are provably redundant after constant
    /// folding/CSE or whose `+0.0`/`-0.0` sign bit is unobservable by every
    /// transitive live consumer.
    fn elide_unobserved_zero_sign(&mut self) -> bool {
        if self.blocks.is_empty() {
            return false;
        }
        let effects = self.effects();
        let reached = self.reached();
        let n = self.insts.len();
        // Compute live instructions so dead `write_f` bitcasts and dead
        // tie-breakers from earlier rounds do not poison `uses`.
        let mut live = vec![false; n];
        let mut lwork: Vec<u32> = Vec::new();
        let mark_live = |live: &mut [bool], lwork: &mut Vec<u32>, v: V| {
            if let V::Inst(i) = v
                && !live[i as usize]
            {
                live[i as usize] = true;
                lwork.push(i);
            }
        };
        for (b, blk) in self.blocks.iter().enumerate() {
            if !reached[b] {
                continue;
            }
            for &i in &blk.body {
                if self.has_effect(&effects, i) {
                    mark_live(&mut live, &mut lwork, V::Inst(i));
                }
            }
            if let Some(Term::CondBr(c, ..)) = blk.term {
                mark_live(&mut live, &mut lwork, c);
            }
        }
        while let Some(i) = lwork.pop() {
            for_each_operand(&self.insts[i as usize].op, |v| {
                mark_live(&mut live, &mut lwork, v);
            });
        }
        let mut uses: Vec<Vec<u32>> = vec![Vec::new(); n];
        let mut used_in_term = vec![false; n];
        for (b, blk) in self.blocks.iter().enumerate() {
            if !reached[b] {
                continue;
            }
            for &i in blk.phis.iter().chain(&blk.body) {
                if !live[i as usize] {
                    continue;
                }
                for_each_operand(&self.insts[i as usize].op, |v| {
                    if let V::Inst(def) = v {
                        uses[def as usize].push(i);
                    }
                });
            }
            if let Some(Term::CondBr(V::Inst(def), ..)) = blk.term {
                used_in_term[def as usize] = true;
            }
        }

        let mut repl: Vec<Option<V>> = vec![None; n];
        let mut changed = false;
        let mut any_repl = false;

        for b in 0..self.blocks.len() {
            if !reached[b] {
                continue;
            }
            let body = std::mem::take(&mut self.blocks[b].body);
            let mut kept = Vec::with_capacity(body.len());
            for i in body {
                let Op::Select(cond, t, f) = self.insts[i as usize].op else {
                    kept.push(i);
                    continue;
                };
                if !self.is_f32(self.insts[i as usize].ty) {
                    kept.push(i);
                    continue;
                }
                // Pattern A: `legacy_mul` select -> `fmul` (or 1-operand check).
                if matches!(
                    t,
                    V::Const(ci) if matches!(self.consts[ci as usize].1, Const::Float(0))
                ) && let V::Inst(mi) = f
                    && let Op::Bin(Bin::Mul, a, b) = self.insts[mi as usize].op
                {
                    let fa = self.bounded(a, 12).is_some();
                    let fb = self.bounded(b, 12).is_some();
                    let nz_a = self.float_nonzero(a, 8);
                    let nz_b = self.float_nonzero(b, 8);
                    let pos_finite = |v: V| {
                        self.float_k(v).is_some_and(|k| k > 0.0 && k.is_finite())
                    };
                    // Single-operand check: `select (fcmp oeq checked, 0.0), +0.0, (fmul a, b)`.
                    if let Some(checked) = self.zero_fcmp_arg(cond) {
                        let pair = if self.same_bits(checked, a) {
                            Some((a, b, nz_a, fb))
                        } else if self.same_bits(checked, b) {
                            Some((b, a, nz_b, fa))
                        } else {
                            None
                        };
                        if let Some((chk, other, nz_chk, finite_other)) = pair {
                            let bit_exact = self.same_bits(a, b)
                                || nz_chk
                                || (finite_other
                                    && self.float_pos_or_pos_zero(chk, 8)
                                    && self.float_pos_or_pos_zero(other, 8))
                                || (pos_finite(other) && self.float_not_neg_zero(chk, 8));
                            if bit_exact
                                || (finite_other
                                    && self.sign_of_zero_unobserved(i, &uses, &used_in_term))
                            {
                                repl[i as usize] = Some(f);
                                any_repl = true;
                                changed = true;
                                continue;
                            }
                        }
                    }
                    // Two-operand check: `select (or az, bz), +0.0, (fmul a, b)`.
                    if let V::Inst(oi) = cond
                        && let Op::Bin(Bin::Or, p, q) = self.insts[oi as usize].op
                        && let (Some(pa), Some(qa)) =
                            (self.zero_fcmp_arg(p), self.zero_fcmp_arg(q))
                    {
                        let ab_cmps = if self.same_bits(pa, a) && self.same_bits(qa, b) {
                            Some((p, q))
                        } else if self.same_bits(pa, b) && self.same_bits(qa, a) {
                            Some((q, p))
                        } else {
                            None
                        };
                        if let Some((cmp_a, cmp_b)) = ab_cmps {
                            let bit_exact = self.same_bits(a, b)
                                || (nz_a && nz_b)
                                || (fa
                                    && fb
                                    && self.float_pos_or_pos_zero(a, 8)
                                    && self.float_pos_or_pos_zero(b, 8))
                                || (pos_finite(a) && self.float_not_neg_zero(b, 8))
                                || (pos_finite(b) && self.float_not_neg_zero(a, 8));
                            if bit_exact {
                                repl[i as usize] = Some(f);
                                any_repl = true;
                                changed = true;
                                continue;
                            }
                            if self.sign_of_zero_unobserved(i, &uses, &used_in_term) {
                                if (fa && fb) || (fa && nz_a) || (fb && nz_b) {
                                    repl[i as usize] = Some(f);
                                    any_repl = true;
                                    changed = true;
                                    continue;
                                }
                                if fa || fb {
                                    let only = if fa { cmp_a } else { cmp_b };
                                    self.insts[i as usize].op = Op::Select(only, t, f);
                                    changed = true;
                                    kept.push(i);
                                    continue;
                                }
                            }
                        }
                    }
                }

                // Pattern B: `max_min` `-0 < +0` tie-breaker select -> `FMax`/`FMin`.
                if let Some((a, b)) = self.match_max_min_tiebreaker(cond, t, f) {
                    if self.float_nonzero(a, 8)
                        || self.float_nonzero(b, 8)
                        || (self.float_not_neg_zero(a, 8) && self.float_not_neg_zero(b, 8))
                        || self.sign_of_zero_unobserved(i, &uses, &used_in_term)
                    {
                        repl[i as usize] = Some(f);
                        any_repl = true;
                        changed = true;
                        continue;
                    }
                }

                // Pattern C: `fract` `< 1.0` clamp when input became provably non-negative.
                if matches!(
                    t,
                    V::Const(ci) if matches!(self.consts[ci as usize].1, Const::Float(0x3F7F_FFFF))
                ) && let V::Inst(ri) = f
                    && let Op::Call(rf, rargs) = &self.insts[ri as usize].op
                    && self.dx_f32_op(*rf, rargs) == Some(("unary", 22))
                    && let Some(&arg) = rargs.get(1)
                    && let V::Inst(ci) = cond
                    && let Op::Cmp(pred::OGE, cl, cr) = self.insts[ci as usize].op
                    && cl == f
                    && self.float_k(cr).is_some_and(|k| k == 1.0)
                    && self.float_non_negative_allow_nz(arg, 8)
                {
                    repl[i as usize] = Some(f);
                    any_repl = true;
                    changed = true;
                    continue;
                }

                kept.push(i);
            }
            self.blocks[b].body = kept;
        }

        if any_repl {
            let (blocks, insts) = (&mut self.blocks, &mut self.insts);
            for (b, blk) in blocks.iter_mut().enumerate() {
                if !reached[b] {
                    continue;
                }
                if let Some(Term::CondBr(c, t, f)) = blk.term {
                    blk.term = Some(Term::CondBr(resolve(&repl, c), t, f));
                }
                for &i in blk.phis.iter().chain(&blk.body) {
                    map_operands(&mut insts[i as usize].op, |v| resolve(&repl, v));
                }
            }
        }
        changed
    }

    /// [`FOLD`]: a branch on a constant goes one way. The side not taken
    /// loses the branching block's values at its joins, and so do the joins
    /// of blocks that nothing reaches any more. Returns whether a branch
    /// changed.
    fn prune(&mut self) -> bool {
        let mut changed = false;
        for b in 0..self.blocks.len() {
            let Some(Term::CondBr(c, t, f)) = self.blocks[b].term else {
                continue;
            };
            // Both sides one block: two edges into it, left as they are.
            if t == f {
                continue;
            }
            let Some((_, k)) = self.int_k(c) else {
                continue;
            };
            let (taken, not) = if k != 0 { (t, f) } else { (f, t) };
            self.blocks[b].term = Some(Term::Br(taken));
            changed = true;
            for n in 0..self.blocks[not].phis.len() {
                let p = self.blocks[not].phis[n];
                if let Op::Phi(incoming) = &mut self.insts[p as usize].op {
                    incoming.retain(|&(_, from)| from != b);
                }
            }
        }
        if !changed {
            return false;
        }
        let reached = self.reached();
        let (blocks, insts) = (&self.blocks, &mut self.insts);
        for (_, blk) in blocks.iter().enumerate().filter(|(b, _)| reached[*b]) {
            for &p in &blk.phis {
                if let Op::Phi(incoming) = &mut insts[p as usize].op {
                    incoming.retain(|&(_, from)| reached[from]);
                }
            }
        }
        true
    }

    /// Whether instruction `i` must stay whatever reads it.
    fn has_effect(&self, effects: &[Effect], i: u32) -> bool {
        match &self.insts[i as usize].op {
            Op::Store(..) | Op::AtomicRmw(..) | Op::CmpXchg(..) => true,
            Op::Call(f, _) => effects[*f as usize] == Effect::Writes,
            _ => false,
        }
    }

    /// [`DEAD`]: keeps the instructions with an effect, the branch
    /// conditions, and what they read, transitively; the rest leave their
    /// blocks. Blocks no branch reaches are emptied (they are not written).
    fn drop_dead(&mut self) {
        let effects = self.effects();
        let reached = self.reached();
        let mut live = vec![false; self.insts.len()];
        let mut work: Vec<u32> = Vec::new();
        let mark = |live: &mut [bool], work: &mut Vec<u32>, v: V| {
            if let V::Inst(i) = v
                && !live[i as usize]
            {
                live[i as usize] = true;
                work.push(i);
            }
        };
        for (b, blk) in self.blocks.iter().enumerate() {
            if !reached[b] {
                continue;
            }
            for &i in &blk.body {
                if self.has_effect(&effects, i) {
                    mark(&mut live, &mut work, V::Inst(i));
                }
            }
            if let Some(Term::CondBr(c, ..)) = blk.term {
                mark(&mut live, &mut work, c);
            }
        }
        while let Some(i) = work.pop() {
            for_each_operand(&self.insts[i as usize].op, |v| {
                mark(&mut live, &mut work, v)
            });
        }
        for (b, blk) in self.blocks.iter_mut().enumerate() {
            if reached[b] {
                blk.body.retain(|&i| live[i as usize]);
                blk.phis.retain(|&i| live[i as usize]);
            } else {
                blk.body.clear();
                blk.phis.clear();
            }
        }
        self.drop_unused_functions();
        self.drop_unused_tables();
    }

    /// See [`Usage`]: over the blocks reached from the entry.
    pub(crate) fn usage(&self) -> Usage {
        let mut u = Usage::default();
        let reached = self.reached();
        let wide = |ty: Ty| self.types[ty.0 as usize] == TypeDef::Int(64);
        // The first field of a handle's annotation: its kind, with the UAV
        // bit; the second: the component type and count.
        let annotation = |h: V| -> Option<(i64, i64)> {
            let V::Inst(a) = h else { return None };
            let Op::Call(f, args) = &self.insts[a as usize].op else {
                return None;
            };
            if self.funcs[*f as usize].name != "dx.op.annotateHandle" {
                return None;
            }
            let V::Const(props) = *args.get(2)? else {
                return None;
            };
            let Const::Aggregate(fields) = &self.consts[props as usize].1 else {
                return None;
            };
            match (
                &self.consts[fields[0] as usize].1,
                &self.consts[fields[1] as usize].1,
            ) {
                (Const::Int(kind), Const::Int(components)) => Some((*kind, *components)),
                _ => None,
            }
        };
        for (_, blk) in self.blocks.iter().enumerate().filter(|(b, _)| reached[*b]) {
            for &i in blk.phis.iter().chain(&blk.body) {
                let inst = &self.insts[i as usize];
                u.int64 |= wide(inst.ty);
                for_each_operand(&inst.op, |v| u.int64 |= wide(self.type_of(v)));
                let Op::Call(f, args) = &inst.op else {
                    continue;
                };
                let name = self.funcs[*f as usize].name.as_str();
                let name = name.strip_prefix("dx.op.").unwrap_or(name);
                if name.starts_with("wave") || name.starts_with("quad") {
                    u.wave_ops = true;
                } else if name == "createHandleFromHeap" {
                    match args.get(2).and_then(|&s| self.int_k(s)) {
                        Some((_, 0)) => u.heap_views = true,
                        _ => u.heap_samplers = true,
                    }
                } else if name == "annotateHandle" {
                    u.rovs |= annotation(V::Inst(i)).is_some_and(|(kind, _)| kind & 0x2000 != 0);
                } else if name.starts_with("atomic") && name.ends_with(".i64") {
                    u.atomic64 = true;
                } else if name.starts_with("textureLoad") || name.starts_with("bufferLoad") {
                    u.typed_uav_loads |= args.get(1).and_then(|&h| annotation(h)).is_none_or(
                        |(kind, components)| kind & 0x1000 != 0 && components >> 8 & 0xff > 1,
                    );
                } else if name.starts_with("loadInput")
                    || name.starts_with("evalSnapped")
                    || name.starts_with("evalSampleIndex")
                    || name.starts_with("evalCentroid")
                    || name.starts_with("attributeAtVertex")
                {
                    if name.starts_with("attributeAtVertex") {
                        u.barycentrics_attr = true;
                    }
                    match args.get(1).and_then(|&id| self.int_k(id)) {
                        Some((_, id)) if id < 64 => u.live_inputs |= 1u64 << id,
                        _ => u.live_inputs = u64::MAX,
                    }
                }
            }
        }
        u
    }

    /// A constant table no remaining load reads leaves the module (the
    /// validator refuses an internal declaration nothing uses). Shared
    /// memory stays declared whether it is read or not.
    fn drop_unused_tables(&mut self) {
        let mut used: Vec<bool> = self.globals.iter().map(|g| g.init.is_none()).collect();
        if used.iter().all(|&u| u) {
            return;
        }
        for blk in &self.blocks {
            for &i in blk.phis.iter().chain(&blk.body) {
                for_each_operand(&self.insts[i as usize].op, |v| {
                    if let V::Global(g) = v {
                        used[g as usize] = true;
                    }
                });
            }
        }
        for md in &self.mds {
            if let MdDef::Value(_, V::Global(g)) = md {
                used[*g as usize] = true;
            }
        }
        if used.iter().all(|&u| u) {
            return;
        }
        let mut new = vec![u32::MAX; used.len()];
        for (next, (old, _)) in used.iter().enumerate().filter(|u| *u.1).enumerate() {
            new[old] = next as u32;
        }
        let mut old = 0;
        self.globals.retain(|_| {
            old += 1;
            used[old - 1]
        });
        let renumber = |v: V| match v {
            V::Global(g) if used[g as usize] => V::Global(new[g as usize]),
            v => v,
        };
        for inst in &mut self.insts {
            map_operands(&mut inst.op, renumber);
        }
        for md in &mut self.mds {
            if let MdDef::Value(_, v) = md {
                *v = renumber(*v);
            }
        }
    }

    /// Declarations no remaining call names leave the module (the validator
    /// refuses an external function nothing uses); the others keep their
    /// order, so `main` stays function 0.
    fn drop_unused_functions(&mut self) {
        let mut used: Vec<bool> = self.funcs.iter().map(|f| f.body).collect();
        for blk in &self.blocks {
            for &i in blk.phis.iter().chain(&blk.body) {
                if let Op::Call(f, _) = self.insts[i as usize].op {
                    used[f as usize] = true;
                }
            }
        }
        for md in &self.mds {
            if let MdDef::Value(_, V::Func(f)) = md {
                used[*f as usize] = true;
            }
        }
        if used.iter().all(|&u| u) {
            return;
        }
        let mut new = vec![u32::MAX; used.len()];
        for (next, (old, _)) in used.iter().enumerate().filter(|u| *u.1).enumerate() {
            new[old] = next as u32;
        }
        let mut old = 0;
        self.funcs.retain(|_| {
            old += 1;
            used[old - 1]
        });
        self.func_ids = self
            .funcs
            .iter()
            .enumerate()
            .map(|(i, f)| (f.name.clone(), i as u32))
            .collect();
        // Instructions that left their block keep a stale number: nothing
        // reads them again.
        for inst in &mut self.insts {
            if let Op::Call(f, _) = &mut inst.op
                && used[*f as usize]
            {
                *f = new[*f as usize];
            }
        }
        for md in &mut self.mds {
            if let MdDef::Value(_, V::Func(f)) = md {
                *f = new[*f as usize];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `main` with one block and a function of each kind of effect.
    struct Kit {
        m: Module,
        b: usize,
        pure: V,
        reads: V,
        writes: V,
    }

    fn kit() -> Kit {
        let mut m = Module::default();
        let i32t = m.int(32);
        let void = m.void();
        let fty = m.func_ty(i32t, vec![i32t, i32t]);
        m.ptr(fty);
        let pure = m.function("dx.op.binary.i32", fty, Some(Attr::ReadNone), false);
        let reads = m.function("dx.op.bufferLoad.i32", fty, Some(Attr::ReadOnly), false);
        let writes = m.function("dx.op.bufferStore.i32", fty, Some(Attr::NoUnwind), false);
        let main_ty = m.func_ty(void, vec![]);
        m.ptr(main_ty);
        m.function("main", main_ty, None, true);
        let b = m.new_block();
        Kit {
            m,
            b,
            pure,
            reads,
            writes,
        }
    }

    fn body(m: &Module, b: usize) -> usize {
        m.blocks[b].body.len()
    }

    #[test]
    fn unread_results_without_effect_are_dropped() {
        let Kit {
            mut m,
            b,
            pure,
            reads,
            writes,
        } = kit();
        let (one, two) = (m.i32(1), m.i32(2));
        let used = m.call(b, reads, vec![one, two]);
        let dead = m.call(b, reads, vec![two, one]);
        let chain = m.bin(b, Bin::Add, dead, one);
        let _unread = m.call(b, pure, vec![chain, two]);
        // A store's result is unread and stays, with what it reads.
        let kept = m.bin(b, Bin::Add, used, two);
        m.call(b, writes, vec![kept, one]);
        m.terminate(b, Term::Ret);
        m.optimise(DEAD);
        assert_eq!(body(&m, b), 3, "the load, the add and the store");
        m.write();
    }

    #[test]
    fn branch_conditions_and_phi_cycles() {
        let Kit {
            mut m,
            b,
            reads,
            writes,
            ..
        } = kit();
        let i32t = m.int(32);
        let (zero, one) = (m.i32(0), m.i32(1));
        let (head, exit) = (m.new_block(), m.new_block());
        m.terminate(b, Term::Br(head));
        // A counter that decides the branch, and a register that only
        // feeds itself round the loop.
        let count = m.phi(head, i32t);
        let idle = m.phi(head, i32t);
        let next = m.bin(head, Bin::Add, count, one);
        let idle_next = m.call(head, reads, vec![idle, one]);
        let more = m.cmp(head, pred::IULT, next, one);
        m.terminate(head, Term::CondBr(more, head, exit));
        m.set_phi(count, vec![(zero, b), (next, head)]);
        m.set_phi(idle, vec![(zero, b), (idle_next, head)]);
        m.call(exit, writes, vec![zero, one]);
        m.terminate(exit, Term::Ret);
        m.optimise(DEAD);
        assert_eq!(m.blocks[head].phis.len(), 1, "the counter");
        assert_eq!(body(&m, head), 2, "the step and the test");
        assert_eq!(body(&m, exit), 1);
        m.write();
    }

    #[test]
    fn wave_operations_have_no_effect_of_their_own() {
        let mut k = kit();
        let i32t = k.m.int(32);
        let fty = k.m.func_ty(i32t, vec![i32t, i32t]);
        let first = k.m.function(
            "dx.op.waveReadLaneFirst.i32",
            fty,
            Some(Attr::NoUnwind),
            false,
        );
        let barrier =
            k.m.function("dx.op.barrier", fty, Some(Attr::NoDuplicate), false);
        let one = k.m.i32(1);
        k.m.call(k.b, first, vec![one, one]);
        k.m.call(k.b, barrier, vec![one, one]);
        k.m.terminate(k.b, Term::Ret);
        k.m.optimise(DEAD);
        assert_eq!(body(&k.m, k.b), 1, "the barrier");
        // The declarations nothing calls any more are gone, and the call
        // that stayed names its function by the new number.
        let names: Vec<&str> = k.m.funcs.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(names, ["main", "dx.op.barrier"]);
        let Op::Call(f, _) = k.m.insts[k.m.blocks[k.b].body[0] as usize].op else {
            panic!("a call");
        };
        assert_eq!(k.m.funcs[f as usize].name, "dx.op.barrier");
        k.m.write();
    }

    /// The operands of each store left in block `b`.
    fn stores(m: &Module, b: usize) -> Vec<Vec<V>> {
        m.blocks[b]
            .body
            .iter()
            .filter_map(|&i| match &m.insts[i as usize].op {
                Op::Call(g, args) if m.funcs[*g as usize].name.contains("Store") => {
                    Some(args.clone())
                }
                _ => None,
            })
            .collect()
    }

    #[test]
    fn constants_and_identities_fold() {
        let Kit {
            mut m,
            b,
            reads,
            writes,
            ..
        } = kit();
        let (i32t, f32t) = (m.int(32), m.float());
        let (zero, one, all) = (m.i32(0), m.i32(1), m.i32(u32::MAX));
        let x = m.call(b, reads, vec![one, one]);
        // x + 0, -1 & x, x >> 0, x * 1, x | x: all x.
        let a = m.bin(b, Bin::Add, x, zero);
        let a = m.bin(b, Bin::And, all, a);
        let a = m.bin(b, Bin::LShr, a, zero);
        let a = m.bin(b, Bin::Mul, a, one);
        let a = m.bin(b, Bin::Or, a, a);
        // There and back.
        let f = m.cast(b, Cast::Bitcast, a, f32t);
        let a = m.cast(b, Cast::Bitcast, f, i32t);
        // Constants: (262168 >> 16) & 127 = 4, 7 & 31 = 7, -8 >>s 1 = -4.
        let k = m.i32(262168);
        let sixteen = m.i32(16);
        let k = m.bin(b, Bin::LShr, k, sixteen);
        let k127 = m.i32(127);
        let k = m.bin(b, Bin::And, k, k127);
        let minus8 = m.i32(-8i32 as u32);
        let s = m.bin(b, Bin::AShr, minus8, one);
        // A shift by the width is left alone.
        let k32 = m.i32(32);
        let wide = m.bin(b, Bin::Shl, one, k32);
        // A select between a value and itself, and on a constant.
        let c = m.cmp(b, pred::IULT, zero, one);
        let sel = m.select(b, c, a, wide);
        let same = m.select(b, c, sel, sel);
        m.call(b, writes, vec![same, k]);
        m.call(b, writes, vec![s, wide]);
        m.terminate(b, Term::Ret);
        m.optimise(FOLD | DEAD);
        assert_eq!(body(&m, b), 4, "the read, the wide shift and two stores");
        let four = m.i32(4);
        let minus4 = m.i32(-4i32 as u32);
        let stores = stores(&m, b);
        assert_eq!(stores, [vec![x, four], vec![minus4, wide]]);
        m.write();
    }

    #[test]
    fn float_arithmetic_is_not_folded() {
        let Kit {
            mut m, b, writes, ..
        } = kit();
        let f32t = m.float();
        let fty = m.func_ty(f32t, vec![f32t, f32t]);
        let input = m.function("dx.op.loadInput.f32", fty, Some(Attr::ReadNone), false);
        let (zero, one, two) = (m.f32(0), m.f32(1.0f32.to_bits()), m.f32(2.0f32.to_bits()));
        let x = m.call(b, input, vec![zero, zero]);
        let y = m.call(b, input, vec![one, one]);
        let a = m.bin(b, Bin::Add, x, zero);
        let p = m.bin(b, Bin::Mul, a, one);
        // The same sum the other way round is another instruction.
        let xy = m.bin(b, Bin::Add, x, y);
        let yx = m.bin(b, Bin::Add, y, x);
        // Constants compare here: 1 < 2, and nothing equals a NaN.
        let nan = m.f32(f32::NAN.to_bits());
        let lt = m.cmp(b, pred::OLT, one, two);
        let eq = m.cmp(b, pred::OEQ, nan, nan);
        let une = m.cmp(b, pred::UNE, nan, one);
        let sel = m.select(b, lt, xy, yx);
        let sel = m.select(b, eq, p, sel);
        let sel = m.select(b, une, sel, p);
        m.call(b, writes, vec![sel, p]);
        m.terminate(b, Term::Ret);
        m.optimise(ALL);
        // Two inputs, x + 0.0, * 1.0, x + y and the store: y + x and the
        // selects are gone.
        assert_eq!(body(&m, b), 6);
        assert_eq!(stores(&m, b), [[xy, p]]);
        m.write();
    }

    /// entry: `c = x < y`, branch to `then` or `other`, which join.
    struct Diamond {
        k: Kit,
        x: V,
        y: V,
        c: V,
        then: usize,
        other: usize,
        join: usize,
    }

    fn diamond() -> Diamond {
        let mut k = kit();
        let (one, two) = (k.m.i32(1), k.m.i32(2));
        let x = k.m.call(k.b, k.reads, vec![one, one]);
        let y = k.m.call(k.b, k.reads, vec![two, two]);
        let c = k.m.cmp(k.b, pred::IULT, x, y);
        let (then, other, join) = (k.m.new_block(), k.m.new_block(), k.m.new_block());
        k.m.terminate(k.b, Term::CondBr(c, then, other));
        k.m.terminate(then, Term::Br(join));
        k.m.terminate(other, Term::Br(join));
        k.m.terminate(join, Term::Ret);
        Diamond {
            k,
            x,
            y,
            c,
            then,
            other,
            join,
        }
    }

    #[test]
    fn a_value_is_reused_below_its_block_and_not_beside_it() {
        let Diamond {
            k:
                Kit {
                    mut m,
                    b,
                    pure,
                    writes,
                    ..
                },
            x,
            y,
            then,
            other,
            join,
            ..
        } = diamond();
        // The entry's value again in a branch; a branch's value again in
        // the other branch and in the join (neither is on every path).
        let top = m.call(b, pure, vec![x, y]);
        m.call(b, writes, vec![top, top]);
        let again = m.call(then, pure, vec![x, y]);
        let side = m.bin(then, Bin::Add, y, x);
        m.call(then, writes, vec![again, side]);
        let beside = m.bin(other, Bin::Add, x, y);
        m.call(other, writes, vec![beside, beside]);
        let after = m.bin(join, Bin::Add, x, y);
        m.call(join, writes, vec![after, after]);
        m.optimise(REUSE);
        assert_eq!(stores(&m, then), [[top, side]]);
        assert_eq!(body(&m, then), 2);
        assert_eq!(body(&m, other), 2);
        assert_eq!(body(&m, join), 2);
        m.write();
    }

    #[test]
    fn a_branch_taken_says_what_its_condition_is() {
        let Diamond {
            k:
                Kit {
                    mut m,
                    b,
                    reads,
                    writes,
                    ..
                },
            x,
            y,
            c,
            then,
            other,
            join,
        } = diamond();
        let three = m.i32(3);
        let d = m.call(b, reads, vec![three, three]);
        let d = m.cmp(b, pred::NE, d, three);
        let both = m.bin(b, Bin::And, c, d);
        // Behind the branch: the condition again (a guard), its opposite
        // comparison, and a guarded write.
        let guard = m.bin(then, Bin::And, c, d);
        let old = m.select(then, guard, x, y);
        let not = m.cmp(then, pred::IUGE, x, y);
        let new = m.select(then, not, y, old);
        m.call(then, writes, vec![new, new]);
        let kept = m.select(other, c, x, y);
        m.call(other, writes, vec![kept, kept]);
        // After the join nothing is known.
        let open = m.select(join, c, x, y);
        m.call(join, writes, vec![open, open]);
        m.optimise(FOLD | FACTS);
        // then: select(d, x, y) and the store; other: y is stored.
        assert_eq!(body(&m, then), 2);
        let V::Inst(sel) = stores(&m, then)[0][0] else {
            panic!()
        };
        assert!(matches!(m.insts[sel as usize].op, Op::Select(g, a, z) if (g, a, z) == (d, x, y)));
        assert_eq!(stores(&m, other), [[y, y]]);
        assert_eq!(body(&m, join), 2);
        let _ = both;
        m.write();

        // Both terms of an `and` that held hold behind it.
        let Diamond {
            k:
                Kit {
                    mut m,
                    b,
                    reads,
                    writes,
                    ..
                },
            x,
            y,
            c,
            ..
        } = diamond();
        let three = m.i32(3);
        let d = m.call(b, reads, vec![three, three]);
        let d = m.cmp(b, pred::NE, d, three);
        let both = m.bin(b, Bin::And, c, d);
        let (inner, out) = (m.new_block(), m.new_block());
        m.terminate(b, Term::CondBr(both, inner, out));
        let a = m.select(inner, c, x, y);
        let z = m.select(inner, d, a, y);
        m.call(inner, writes, vec![z, z]);
        m.terminate(inner, Term::Br(out));
        // Where the `and` did not hold either term may.
        let w = m.select(out, d, x, y);
        m.call(out, writes, vec![w, w]);
        m.terminate(out, Term::Ret);
        m.optimise(FOLD | FACTS);
        assert_eq!(stores(&m, inner), [[x, x]]);
        assert_eq!(body(&m, out), 2);
        m.write();
    }

    #[test]
    fn a_branch_on_a_constant_goes_one_way() {
        let Diamond {
            k: Kit {
                mut m, b, writes, ..
            },
            x,
            y,
            c,
            then,
            other,
            join,
        } = diamond();
        let i32t = m.int(32);
        // Behind the branch the same test again: its other side is never
        // run, and the join behind it has one value.
        let (deep, never, meet) = (m.new_block(), m.new_block(), m.new_block());
        m.terminate(then, Term::CondBr(c, deep, never));
        m.terminate(deep, Term::Br(meet));
        m.call(never, writes, vec![y, y]);
        m.terminate(never, Term::Br(meet));
        let p = m.phi(meet, i32t);
        m.set_phi(p, vec![(x, deep), (y, never)]);
        m.call(meet, writes, vec![p, p]);
        m.terminate(meet, Term::Br(join));
        let q = m.phi(join, i32t);
        m.set_phi(q, vec![(p, meet), (y, other)]);
        m.call(join, writes, vec![q, q]);
        m.optimise(ALL);
        assert!(matches!(m.blocks[then].term, Some(Term::Br(t)) if t == deep));
        assert_eq!(stores(&m, meet), [[x, x]]);
        assert!(m.blocks[meet].phis.is_empty());
        assert_eq!(body(&m, never), 0, "emptied: nothing reaches it");
        let Op::Phi(incoming) = &m.insts[m.blocks[join].phis[0] as usize].op else {
            panic!()
        };
        assert_eq!(incoming, &vec![(x, meet), (y, other)]);
        let _ = b;
        m.write();
    }

    #[test]
    fn a_read_is_reused_within_its_block_until_a_store() {
        let Diamond {
            k:
                Kit {
                    mut m,
                    b,
                    reads,
                    writes,
                    ..
                },
            x,
            y,
            then,
            ..
        } = diamond();
        let i32t = m.int(32);
        // The entry reads (1, 1) for x: again in the entry it is x; in the
        // next block it is read again.
        let one = m.i32(1);
        let again = m.call(b, reads, vec![one, one]);
        m.call(b, writes, vec![again, y]);
        let after = m.call(b, reads, vec![one, one]);
        let twice = m.call(b, reads, vec![one, one]);
        let sum = m.bin(b, Bin::Add, after, twice);
        m.call(b, writes, vec![sum, sum]);
        let below = m.call(then, reads, vec![one, one]);
        m.call(then, writes, vec![below, below]);
        // A constant buffer cannot change: one load for the function.
        let cfty = m.func_ty(i32t, vec![i32t, i32t]);
        let cb = m.function(
            "dx.op.cbufferLoadLegacy.i32",
            cfty,
            Some(Attr::ReadOnly),
            false,
        );
        let top = m.call(b, cb, vec![one, one]);
        m.call(b, writes, vec![top, top]);
        let under = m.call(then, cb, vec![one, one]);
        m.call(then, writes, vec![under, under]);
        m.optimise(REUSE | LOADS);
        let stores = stores(&m, b);
        assert_eq!(stores[0], [x, y], "the second read is the first");
        let V::Inst(sum) = stores[1][0] else { panic!() };
        assert!(
            matches!(m.insts[sum as usize].op, Op::Bin(Bin::Add, a, z) if a == after && z == after),
            "read again after the store, once"
        );
        assert_eq!(stores[2], [top, top]);
        assert_eq!(
            super::tests::stores(&m, then),
            [vec![below, below], vec![top, top]]
        );
        m.write();
    }

    #[test]
    fn a_join_of_one_value_and_two_equal_joins() {
        let Diamond {
            k: Kit { mut m, writes, .. },
            x,
            y,
            then,
            other,
            join,
            ..
        } = diamond();
        let i32t = m.int(32);
        let one = m.phi(join, i32t);
        m.set_phi(one, vec![(x, then), (x, other)]);
        let p = m.phi(join, i32t);
        m.set_phi(p, vec![(x, then), (y, other)]);
        let q = m.phi(join, i32t);
        m.set_phi(q, vec![(y, other), (x, then)]);
        m.call(join, writes, vec![one, p]);
        let sum = m.bin(join, Bin::Add, p, q);
        m.call(join, writes, vec![sum, q]);
        m.optimise(FOLD | REUSE);
        assert_eq!(
            m.blocks[join].phis,
            [match p {
                V::Inst(i) => i,
                _ => unreachable!(),
            }]
        );
        let stores = stores(&m, join);
        assert_eq!(stores[0], [x, p]);
        assert_eq!(stores[1][1], p);
        m.write();
    }

    // ---- the passes against an interpreter, on random programs ----

    /// Runs `m` on `inputs`: the stores it makes, in order (and how it
    /// ended). Reads a value before its definition ran: a panic.
    fn run(m: &Module, inputs: &[u32]) -> Vec<(String, Vec<u64>)> {
        let mut vals: Vec<Option<u64>> = vec![None; m.insts.len()];
        let mut memory = [0u64; 16];
        let mut trace = Vec::new();
        let eval = |vals: &[Option<u64>], v: V| -> u64 {
            match v {
                V::Inst(i) => vals[i as usize].expect("a value defined before its use"),
                V::Const(c) => {
                    match (&m.consts[c as usize].1, m.int_bits(m.consts[c as usize].0)) {
                        (Const::Int(x), Some(bits)) => *x as u64 & mask(bits),
                        (Const::Float(bits), _) => u64::from(*bits),
                        _ => 0,
                    }
                }
                _ => panic!("a global or function as a value"),
            }
        };
        let (mut b, mut from) = (0usize, usize::MAX);
        for _ in 0..4000 {
            let blk = &m.blocks[b];
            // Joins take their values together, before any is assigned.
            let joined: Vec<(u32, u64)> = blk
                .phis
                .iter()
                .map(|&p| {
                    let Op::Phi(incoming) = &m.insts[p as usize].op else {
                        panic!("a join")
                    };
                    let v = incoming
                        .iter()
                        .find(|x| x.1 == from)
                        .expect("a value for the edge taken");
                    (p, eval(&vals, v.0))
                })
                .collect();
            for (p, x) in joined {
                vals[p as usize] = Some(x);
            }
            for &i in &blk.body {
                let inst = &m.insts[i as usize];
                let bits = m.int_bits(inst.ty);
                let float = |x: u64| f32::from_bits(x as u32);
                let r = match &inst.op {
                    Op::Bin(op, x, y) => {
                        let (x, y) = (eval(&vals, *x), eval(&vals, *y));
                        match bits {
                            Some(bits) => {
                                let r = match op {
                                    Bin::Add => x.wrapping_add(y),
                                    Bin::Sub => x.wrapping_sub(y),
                                    Bin::Mul => x.wrapping_mul(y),
                                    Bin::And => x & y,
                                    Bin::Or => x | y,
                                    Bin::Xor => x ^ y,
                                    Bin::Shl => x << y,
                                    Bin::LShr => x >> y,
                                    Bin::AShr => (signed(bits, x) >> y) as u64,
                                    _ => panic!("no divisions here"),
                                };
                                r & mask(bits)
                            }
                            None => {
                                let (x, y) = (float(x), float(y));
                                let r = match op {
                                    Bin::Add => x + y,
                                    Bin::Sub => x - y,
                                    _ => x * y,
                                };
                                u64::from(r.to_bits())
                            }
                        }
                    }
                    Op::Cmp(p, x, y) => {
                        let wide = m.int_bits(m.type_of(*x));
                        let (x, y) = (eval(&vals, *x), eval(&vals, *y));
                        let holds = match wide {
                            Some(bits) => {
                                let (sx, sy) = (signed(bits, x), signed(bits, y));
                                match *p {
                                    pred::EQ => x == y,
                                    pred::NE => x != y,
                                    pred::IUGT => x > y,
                                    pred::IUGE => x >= y,
                                    pred::IULT => x < y,
                                    pred::IULE => x <= y,
                                    pred::SGT => sx > sy,
                                    pred::SGE => sx >= sy,
                                    pred::SLT => sx < sy,
                                    _ => sx <= sy,
                                }
                            }
                            None => {
                                let (x, y) = (float(x), float(y));
                                if x.is_nan() || y.is_nan() {
                                    p & 8 != 0
                                } else {
                                    (p & 1 != 0 && x == y)
                                        || (p & 2 != 0 && x > y)
                                        || (p & 4 != 0 && x < y)
                                }
                            }
                        };
                        u64::from(holds)
                    }
                    Op::Select(c, t, f) => {
                        if eval(&vals, *c) != 0 {
                            eval(&vals, *t)
                        } else {
                            eval(&vals, *f)
                        }
                    }
                    Op::Cast(op, x) => {
                        let from_bits = m.int_bits(m.type_of(*x));
                        let x = eval(&vals, *x);
                        match (op, from_bits, bits) {
                            (Cast::Bitcast, _, _) => x,
                            (Cast::ZExt, _, _) => x,
                            (Cast::SExt, Some(f), Some(t)) => signed(f, x) as u64 & mask(t),
                            (Cast::Trunc, _, Some(t)) => x & mask(t),
                            _ => panic!("a cast the generator does not make"),
                        }
                    }
                    Op::Call(f, args) => {
                        let a: Vec<u64> = args.iter().map(|&v| eval(&vals, v)).collect();
                        let name = m.funcs[*f as usize].name.as_str();
                        match name {
                            "dx.op.loadInput.i32" => {
                                u64::from(inputs[(a[0] ^ a[1]) as usize % inputs.len()])
                            }
                            "dx.op.binary.i32" => a[0].min(a[1]),
                            "dx.op.cbufferLoadLegacy.i32" => u64::from(
                                inputs[a[0] as usize % inputs.len()].rotate_left(a[1] as u32 & 31),
                            ),
                            "dx.op.bufferLoad.i32" => memory[(a[0] ^ a[1]) as usize % 16],
                            "dx.op.waveReadLaneFirst.i32" => a[0] ^ a[1],
                            _ => {
                                memory[a[0] as usize % 16] = a[1];
                                trace.push((name.to_owned(), a));
                                0
                            }
                        }
                    }
                    _ => panic!("an instruction the generator does not make"),
                };
                vals[i as usize] = Some(r);
            }
            from = b;
            b = match blk.term.expect("a terminator") {
                Term::Ret => {
                    trace.push(("ret".into(), Vec::new()));
                    return trace;
                }
                Term::Br(t) => t,
                Term::CondBr(c, t, f) => {
                    if eval(&vals, c) != 0 {
                        t
                    } else {
                        f
                    }
                }
            };
        }
        trace.push(("still running".into(), Vec::new()));
        trace
    }

    /// A random structured program over a few registers, as the lowering
    /// makes them: every register joined at every join, conditions tested
    /// again behind their branch, values computed again.
    struct Gen {
        k: Kit,
        rng: u64,
        regs: Vec<V>,
        conds: Vec<V>,
        cur: usize,
        input: V,
        table: V,
        wave: V,
        /// The operands of the block's last read (to read the place again,
        /// and to store to it).
        last: Option<(V, V)>,
    }

    impl Gen {
        fn next(&mut self, n: u64) -> u64 {
            self.rng ^= self.rng << 13;
            self.rng ^= self.rng >> 7;
            self.rng ^= self.rng << 17;
            (self.rng >> 11) % n
        }

        fn reg(&mut self) -> V {
            let n = self.next(self.regs.len() as u64) as usize;
            self.regs[n]
        }

        /// A register or a small constant.
        fn value(&mut self) -> V {
            match self.next(4) {
                0 => {
                    let k = [0u32, 1, 2, 31, u32::MAX, 0x8000_0000, 7][self.next(7) as usize];
                    self.k.m.i32(k)
                }
                _ => self.reg(),
            }
        }

        fn cond(&mut self) -> V {
            let c = self.cur;
            let made = match self.next(6) {
                0 if !self.conds.is_empty() => {
                    let n = self.next(self.conds.len() as u64) as usize;
                    return self.conds[n];
                }
                1 if self.conds.len() > 1 => {
                    let (a, b) = (
                        self.next(self.conds.len() as u64),
                        self.next(self.conds.len() as u64),
                    );
                    let op = [Bin::And, Bin::Or, Bin::Xor][self.next(3) as usize];
                    self.k
                        .m
                        .bin(c, op, self.conds[a as usize], self.conds[b as usize])
                }
                2 if !self.conds.is_empty() => {
                    let n = self.next(self.conds.len() as u64) as usize;
                    let t = self.k.m.bool(true);
                    self.k.m.bin(c, Bin::Xor, self.conds[n], t)
                }
                _ => {
                    let p = pred::EQ + self.next(10) as u8;
                    let (x, y) = (self.reg(), self.value());
                    self.k.m.cmp(c, p, x, y)
                }
            };
            self.conds.push(made);
            made
        }

        fn statement(&mut self) {
            let c = self.cur;
            let d = self.next(self.regs.len() as u64) as usize;
            let i32t = self.k.m.int(32);
            let f32t = self.k.m.float();
            let r = match self.next(14) {
                0..=3 => {
                    let op = [Bin::Add, Bin::Sub, Bin::Mul, Bin::And, Bin::Or, Bin::Xor]
                        [self.next(6) as usize];
                    let (x, y) = (self.value(), self.value());
                    self.k.m.bin(c, op, x, y)
                }
                4 => {
                    let op = [Bin::Shl, Bin::LShr, Bin::AShr][self.next(3) as usize];
                    let x = self.value();
                    let by = self.next(32) as u32;
                    let by = self.k.m.i32(by);
                    self.k.m.bin(c, op, x, by)
                }
                5 | 6 => {
                    // A guarded write: the old value where the guard fails.
                    let g = self.cond();
                    let x = self.value();
                    self.k.m.select(self.cur, g, x, self.regs[d])
                }
                7 => {
                    let (x, y) = (self.value(), self.value());
                    let f = [self.k.pure, self.input, self.table, self.wave][self.next(4) as usize];
                    self.k.m.call(c, f, vec![x, y])
                }
                8 | 9 => {
                    let again = self.next(2) == 0;
                    let (x, y) = match self.last {
                        Some(last) if again => last,
                        _ => (self.value(), self.value()),
                    };
                    self.last = Some((x, y));
                    self.k.m.call(c, self.k.reads, vec![x, y])
                }
                10 => {
                    // A store, half the time to the place read last.
                    let (mut x, y) = (self.value(), self.value());
                    let there = self.next(2) == 0;
                    if let Some((a, b)) = self.last
                        && there
                    {
                        x = self.k.m.bin(c, Bin::Xor, a, b);
                    }
                    self.k.m.call(c, self.k.writes, vec![x, y]);
                    return;
                }
                11 => {
                    // Float arithmetic, read back as bits.
                    let (x, y) = (self.value(), self.value());
                    let (x, y) = (
                        self.k.m.cast(c, Cast::Bitcast, x, f32t),
                        self.k.m.cast(c, Cast::Bitcast, y, f32t),
                    );
                    let op = [Bin::Add, Bin::Sub, Bin::Mul][self.next(3) as usize];
                    let f = self.k.m.bin(c, op, x, y);
                    if self.next(2) == 0 {
                        let p = self.next(16) as u8;
                        let t = self.k.m.cmp(c, p, f, y);
                        self.conds.push(t);
                    }
                    self.k.m.cast(c, Cast::Bitcast, f, i32t)
                }
                12 => {
                    let g = self.cond();
                    self.k.m.cast(self.cur, Cast::ZExt, g, i32t)
                }
                _ => {
                    let x = self.value();
                    let f = self.k.m.cast(c, Cast::Bitcast, x, f32t);
                    self.k.m.cast(c, Cast::Bitcast, f, i32t)
                }
            };
            self.regs[d] = r;
        }

        fn code(&mut self, depth: u32) {
            for _ in 0..1 + self.next(6) {
                match self.next(8) {
                    0 | 1 if depth > 0 => self.branch(depth - 1),
                    2 if depth > 0 => self.repeat(depth - 1),
                    _ => self.statement(),
                }
            }
        }

        /// if / else, every register joined.
        fn branch(&mut self, depth: u32) {
            let c = self.cond();
            let i32t = self.k.m.int(32);
            let (then, other, join) = (
                self.k.m.new_block(),
                self.k.m.new_block(),
                self.k.m.new_block(),
            );
            self.k.m.terminate(self.cur, Term::CondBr(c, then, other));
            let (before, known) = (self.regs.clone(), self.conds.len());
            self.cur = then;
            self.last = None;
            self.code(depth);
            let (then_regs, then_end) = (std::mem::replace(&mut self.regs, before), self.cur);
            self.k.m.terminate(then_end, Term::Br(join));
            self.conds.truncate(known);
            self.cur = other;
            self.last = None;
            if self.next(3) != 0 {
                self.code(depth);
            }
            let other_end = self.cur;
            self.k.m.terminate(other_end, Term::Br(join));
            self.conds.truncate(known);
            for (r, &t) in then_regs.iter().enumerate() {
                let p = self.k.m.phi(join, i32t);
                self.k
                    .m
                    .set_phi(p, vec![(t, then_end), (self.regs[r], other_end)]);
                self.regs[r] = p;
            }
            self.cur = join;
            self.last = None;
        }

        /// A loop that runs at most a few times, every register joined at
        /// its head.
        fn repeat(&mut self, depth: u32) {
            let i32t = self.k.m.int(32);
            let (head, exit) = (self.k.m.new_block(), self.k.m.new_block());
            let before = self.cur;
            self.k.m.terminate(before, Term::Br(head));
            let entering = self.regs.clone();
            let joins: Vec<V> = (0..self.regs.len())
                .map(|_| self.k.m.phi(head, i32t))
                .collect();
            let count = self.k.m.phi(head, i32t);
            self.regs = joins.clone();
            let known = self.conds.len();
            self.cur = head;
            self.last = None;
            self.code(depth);
            let one = self.k.m.i32(1);
            let next = self.k.m.bin(self.cur, Bin::Add, count, one);
            let limit = 1 + self.next(3) as u32;
            let limit = self.k.m.i32(limit);
            let mut more = self.k.m.cmp(self.cur, pred::IULT, next, limit);
            if self.next(2) == 0 {
                let c = self.cond();
                more = self.k.m.bin(self.cur, Bin::And, more, c);
            }
            let latch = self.cur;
            self.k.m.terminate(latch, Term::CondBr(more, head, exit));
            for (r, &p) in joins.iter().enumerate() {
                self.k
                    .m
                    .set_phi(p, vec![(entering[r], before), (self.regs[r], latch)]);
            }
            let zero = self.k.m.i32(0);
            self.k.m.set_phi(count, vec![(zero, before), (next, latch)]);
            self.conds.truncate(known);
            self.cur = exit;
            self.last = None;
        }
    }

    fn random_program(seed: u64) -> Module {
        let mut k = kit();
        let i32t = k.m.int(32);
        let fty = k.m.func_ty(i32t, vec![i32t, i32t]);
        let input =
            k.m.function("dx.op.loadInput.i32", fty, Some(Attr::ReadNone), false);
        let table = k.m.function(
            "dx.op.cbufferLoadLegacy.i32",
            fty,
            Some(Attr::ReadOnly),
            false,
        );
        let wave = k.m.function(
            "dx.op.waveReadLaneFirst.i32",
            fty,
            Some(Attr::NoUnwind),
            false,
        );
        let b = k.b;
        let mut g = Gen {
            k,
            rng: seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1,
            regs: Vec::new(),
            conds: Vec::new(),
            cur: b,
            input,
            table,
            wave,
            last: None,
        };
        for r in 0..5 {
            let (x, y) = (g.k.m.i32(r), g.k.m.i32(r * 3));
            let v = g.k.m.call(b, input, vec![x, y]);
            g.regs.push(v);
        }
        g.code(3);
        g.code(2);
        // Every register is read at the end.
        for r in 0..g.regs.len() {
            let at = g.k.m.i32(r as u32);
            g.k.m.call(g.cur, g.k.writes, vec![at, g.regs[r]]);
        }
        g.k.m.terminate(g.cur, Term::Ret);
        g.k.m
    }

    #[test]
    fn every_pass_leaves_what_random_programs_store() {
        let mut removed = 0;
        for seed in 1..=400u64 {
            let plain = random_program(seed);
            let inputs: Vec<Vec<u32>> = (0..6u32)
                .map(|n| {
                    (0..7u32)
                        .map(|i| match n {
                            0 => 0,
                            1 => i,
                            2 => u32::MAX - i,
                            _ => {
                                (seed as u32 ^ n)
                                    .wrapping_mul(0x9E37_79B9)
                                    .rotate_left(i * 5)
                                    ^ (i << 28)
                            }
                        })
                        .collect()
                })
                .collect();
            let expected: Vec<_> = inputs.iter().map(|i| run(&plain, i)).collect();
            let written: usize = plain
                .blocks
                .iter()
                .map(|b| b.body.len() + b.phis.len())
                .sum();
            for passes in [
                DEAD,
                FOLD,
                REUSE,
                FACTS,
                LOADS,
                FOLD | FACTS,
                FOLD | REUSE | DEAD,
                ALL,
            ] {
                let mut m = random_program(seed);
                m.optimise(passes);
                for (i, want) in inputs.iter().zip(&expected) {
                    assert_eq!(&run(&m, i), want, "seed {seed}, passes {passes}");
                }
                if passes == ALL {
                    let left: usize = m.blocks.iter().map(|b| b.body.len() + b.phis.len()).sum();
                    removed += written - left;
                }
                m.write();
            }
        }
        // The programs are of the wasteful kind: the passes find work.
        assert!(removed > 4000, "{removed} instructions removed");
    }

    #[test]
    fn folds_chained_add_constants_and_exact_square_legacy_mul() {
        let mut m = Module::default();
        let i32t = m.int(32);
        let f32t = m.float();
        let void = m.void();
        let tid_ty = m.func_ty(i32t, vec![i32t, i32t]);
        let store_ty = m.func_ty(void, vec![i32t, f32t]);
        let main_ty = m.func_ty(void, vec![]);
        m.ptr(tid_ty);
        m.ptr(store_ty);
        m.ptr(main_ty);
        let tid_f = m.function("dx.op.threadId.i32", tid_ty, Some(Attr::ReadNone), false);
        let sink_f = m.function("dx.op.bufferStore.f32", store_ty, Some(Attr::NoUnwind), false);
        m.function("main", main_ty, None, true);

        let b = m.new_block();
        let (op_tid, z0, k16, k4, fzero) = (m.i32(93), m.i32(0), m.i32(16), m.i32(4), m.f32(0));
        let tid = m.call(b, tid_f, vec![op_tid, z0]);
        let a16 = m.bin(b, Bin::Add, tid, k16);
        let a20 = m.bin(b, Bin::Add, a16, k4);
        let xf = m.cast(b, Cast::Bitcast, a20, f32t);
        let sq = m.bin(b, Bin::Mul, xf, xf);
        let is_z = m.cmp(b, pred::OEQ, xf, fzero);
        let sq_sel = m.select(b, is_z, fzero, sq);
        m.call(b, sink_f, vec![a20, sq_sel]);
        m.terminate(b, Term::Ret);

        m.optimise(ALL);

        let adds: Vec<u64> = m.blocks[0]
            .body
            .iter()
            .filter_map(|&id| match m.insts[id as usize].op {
                Op::Bin(Bin::Add, _, c) => Some(m.int_k(c)?.1),
                _ => None,
            })
            .collect();
        assert_eq!(adds, vec![20], "chained (tid + 16) + 4 must fold to tid + 20");
        assert!(
            !m.blocks[0]
                .body
                .iter()
                .any(|&id| matches!(m.insts[id as usize].op, Op::Select(..) | Op::Cmp(..))),
            "bit-exact x * x legacy_mul select and fcmp must simplify to plain fmul"
        );
    }

    #[test]
    fn pure_metadata_table_loads_cse_across_stores_and_blocks() {
        use crate::emit::shader::{ROOT_CBV_SPACE, SLOT_MEMORY, SLOT_STRIDES};
        let mut m = Module::default();
        let i1t = m.int(1);
        let i8t = m.int(8);
        let i32t = m.int(32);
        let f32t = m.float();
        let void = m.void();
        let i8p = m.ptr(i8t);
        let handle = m.named_struct("dx.types.Handle", vec![i8p]);
        let bind_ty = m.named_struct("dx.types.ResBind", vec![i32t, i32t, i32t, i8t]);
        let props_ty = m.named_struct("dx.types.ResourceProperties", vec![i32t, i32t]);
        let res_ret = m.named_struct("dx.types.ResRet.f32", vec![f32t, f32t, f32t, f32t, i32t]);

        let bind_f_ty = m.func_ty(handle, vec![i32t, bind_ty, i32t, i1t]);
        let ann_f_ty = m.func_ty(handle, vec![i32t, handle, props_ty]);
        let rld_f_ty = m.func_ty(res_ret, vec![i32t, handle, i32t, i32t, i8t, i32t]);
        let bar_f_ty = m.func_ty(void, vec![i32t, i32t]);
        let sink_f_ty = m.func_ty(void, vec![f32t, f32t, f32t]);
        let main_ty = m.func_ty(void, vec![]);
        for ty in [bind_f_ty, ann_f_ty, rld_f_ty, bar_f_ty, sink_f_ty, main_ty] {
            m.ptr(ty);
        }
        let bind_f = m.function("dx.op.createHandleFromBinding", bind_f_ty, Some(Attr::ReadNone), false);
        let ann_f = m.function("dx.op.annotateHandle", ann_f_ty, Some(Attr::ReadNone), false);
        let rld_f = m.function("dx.op.rawBufferLoad.f32", rld_f_ty, Some(Attr::ReadOnly), false);
        let bar_f = m.function("dx.op.barrier", bar_f_ty, Some(Attr::NoDuplicate), false);
        let sink_f = m.function("dx.op.bufferStore.f32", sink_f_ty, Some(Attr::NoUnwind), false);
        m.function("main", main_ty, None, true);

        let b0 = m.new_block();
        let b1 = m.new_block();
        let make_srv = |m: &mut Module, blk: usize, slot: u32| {
            let r = m.i32(slot);
            let sp = m.i32(ROOT_CBV_SPACE);
            let cls = m.i8(0);
            let bind = m.aggregate(bind_ty, &[r, r, sp, cls]);
            let op_b = m.i32(217);
            let f_false = m.bool(false);
            let h = m.call(blk, bind_f, vec![op_b, bind, r, f_false]);
            let p0 = m.i32(11);
            let p1 = m.i32(0);
            let props = m.aggregate(props_ty, &[p0, p1]);
            let op_a = m.i32(216);
            m.call(blk, ann_f, vec![op_a, h, props])
        };
        let h_strides = make_srv(&mut m, b0, u32::from(SLOT_STRIDES));
        let h_memory = make_srv(&mut m, b0, u32::from(SLOT_MEMORY));
        let h_regular = make_srv(&mut m, b0, 0);

        let (op_rld, op_bar, k0, k4, iu, m1, bar9) = (
            m.i32(139),
            m.i32(80),
            m.i32(0),
            m.i32(4),
            m.undef(i32t),
            m.i8(1),
            m.i32(9),
        );
        // First loads in entry block `b0`.
        let r_s1 = m.call(b0, rld_f, vec![op_rld, h_strides, k0, iu, m1, k4]);
        let r_m1 = m.call(b0, rld_f, vec![op_rld, h_memory, k4, iu, m1, k4]);
        let r_u1 = m.call(b0, rld_f, vec![op_rld, h_regular, k0, iu, m1, k4]);
        let v_s1 = m.extract(b0, r_s1, 0);
        let v_m1 = m.extract(b0, r_m1, 0);
        let v_u1 = m.extract(b0, r_u1, 0);
        // Barrier + store between first and second loads, then branch to dominated `b1`.
        m.call(b0, bar_f, vec![op_bar, bar9]);
        m.call(b0, sink_f, vec![v_s1, v_m1, v_u1]);
        m.terminate(b0, Term::Br(b1));

        // Duplicate loads in dominated child block `b1`.
        let r_s2 = m.call(b1, rld_f, vec![op_rld, h_strides, k0, iu, m1, k4]);
        let r_m2 = m.call(b1, rld_f, vec![op_rld, h_memory, k4, iu, m1, k4]);
        let r_u2 = m.call(b1, rld_f, vec![op_rld, h_regular, k0, iu, m1, k4]);
        let v_s2 = m.extract(b1, r_s2, 0);
        let v_m2 = m.extract(b1, r_m2, 0);
        let v_u2 = m.extract(b1, r_u2, 0);
        m.call(b1, sink_f, vec![v_s2, v_m2, v_u2]);
        m.terminate(b1, Term::Ret);

        m.optimise(ALL);

        let raw_loads: Vec<V> = m
            .blocks
            .iter()
            .flat_map(|blk| blk.body.iter())
            .filter_map(|&id| match &m.insts[id as usize].op {
                Op::Call(cf, args) if m.funcs[*cf as usize].name == "dx.op.rawBufferLoad.f32" => {
                    Some(args[1])
                }
                _ => None,
            })
            .collect();
        // `SLOT_STRIDES` (1) + `SLOT_MEMORY` (1) CSE'd across barrier/store/blocks;
        // regular SRV `t0` kept both loads (2) -> 4 total instead of 6!
        assert_eq!(
            raw_loads.len(),
            4,
            "pure metadata table loads must CSE across barriers/stores/blocks while regular SRV keeps both"
        );
    }

    #[test]
    fn elide_unobserved_zero_sign_elides_fadd_chains_and_preserves_bitcast_and_rcp() {
        let mut m = Module::default();
        let i32t = m.int(32);
        let f32t = m.float();
        let void = m.void();
        let ld_ty = m.func_ty(f32t, vec![i32t]);
        let bin_f32_ty = m.func_ty(f32t, vec![i32t, f32t, f32t]);
        let st_i32_ty = m.func_ty(void, vec![i32t, i32t]);
        let st_f32_ty = m.func_ty(void, vec![i32t, f32t]);
        let main_ty = m.func_ty(void, vec![]);
        for ty in [ld_ty, bin_f32_ty, st_i32_ty, st_f32_ty, main_ty] {
            m.ptr(ty);
        }
        let ld = m.function("dx.op.loadInput.f32", ld_ty, Some(Attr::ReadNone), false);
        let bin_f32 = m.function("dx.op.binary.f32", bin_f32_ty, Some(Attr::ReadNone), false);
        let st_i32 = m.function("dx.op.bufferStore.i32", st_i32_ty, Some(Attr::NoUnwind), false);
        let st_f32 = m.function("dx.op.bufferStore.f32", st_f32_ty, Some(Attr::NoUnwind), false);
        m.function("main", main_ty, None, true);

        let b = m.new_block();
        let (k0, k1, k2, k3, k35) = (m.i32(0), m.i32(1), m.i32(2), m.i32(3), m.i32(35));
        let f0 = m.f32(0);
        let f1 = m.f32(1.0f32.to_bits());
        let fm2 = m.f32((-2.0f32).to_bits());
        let kmask = m.i32(0x7FFF_FFFF);

        // Input 0: `legacy_mul(-2.0, x0)` -> bitcast i32 -> bitcast f32 -> `+ 1.0` -> store
        // Sign of `-0.0` is absorbed by `+ 1.0`, so the select MUST be elided.
        let x0 = m.call(b, ld, vec![k0]);
        let mul0 = m.bin(b, Bin::Mul, fm2, x0);
        let z0 = m.cmp(b, pred::OEQ, x0, f0);
        let lm0 = m.select(b, z0, f0, mul0);
        let lm0_i = m.cast(b, Cast::Bitcast, lm0, i32t);
        let lm0_f = m.cast(b, Cast::Bitcast, lm0_i, f32t);
        let sum0 = m.bin(b, Bin::Add, lm0_f, f1);
        m.call(b, st_f32, vec![k0, sum0]);

        // Input 1: `max_min(true, x1, +0.0)` -> `+ 1.0` -> store
        // Sign of `-0.0` is absorbed by `+ 1.0`, so the 8-op tie-breaker MUST be elided.
        let x1 = m.call(b, ld, vec![k1]);
        let fmax1 = m.call(b, bin_f32, vec![k35, x1, f0]);
        let ix1 = m.cast(b, Cast::Bitcast, x1, i32t);
        let mag1 = m.bin(b, Bin::And, ix1, kmask);
        let zeros1 = m.cmp(b, pred::EQ, mag1, k0);
        let mm1 = m.select(b, zeros1, f0, fmax1);
        let sum1 = m.bin(b, Bin::Add, mm1, f1);
        m.call(b, st_f32, vec![k1, sum1]);

        // Input 2: `legacy_mul(-2.0, x2)` -> bitcast i32 -> store (OBSERVES sign bit!)
        // Must KEEP its select!
        let x2 = m.call(b, ld, vec![k2]);
        let mul2 = m.bin(b, Bin::Mul, fm2, x2);
        let z2 = m.cmp(b, pred::OEQ, x2, f0);
        let lm2 = m.select(b, z2, f0, mul2);
        let lm2_i = m.cast(b, Cast::Bitcast, lm2, i32t);
        m.call(b, st_i32, vec![k2, lm2_i]);

        // Input 3: `legacy_mul(-2.0, x3)` -> `1.0 / r` (OBSERVES sign of zero: ±Inf!)
        // Must KEEP its select!
        let x3 = m.call(b, ld, vec![k3]);
        let mul3 = m.bin(b, Bin::Mul, fm2, x3);
        let z3 = m.cmp(b, pred::OEQ, x3, f0);
        let lm3 = m.select(b, z3, f0, mul3);
        let rcp3 = m.bin(b, Bin::SDiv, f1, lm3);
        m.call(b, st_f32, vec![k3, rcp3]);

        m.terminate(b, Term::Ret);
        m.optimise(ALL);

        let selects: Vec<u32> = m.blocks[0]
            .body
            .iter()
            .copied()
            .filter(|&id| matches!(m.insts[id as usize].op, Op::Select(..)))
            .collect();
        assert_eq!(
            selects.len(),
            2,
            "expected exactly 2 selects kept (bitcast-store and 1/x) and 2 elided (fadd +1.0 chains), got {}",
            selects.len()
        );
    }
}
