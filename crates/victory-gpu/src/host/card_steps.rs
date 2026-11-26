//! Steps of a submission that read title memory flights in the air may
//! still write, done on the card in the flights' order instead of on the
//! CPU after a wait for them (`VICTORY_CARD_STEPS`, see [`stage`]).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use super::*;
use crate::shadow::Shadow;

/// Resolves `VICTORY_CARD_STEPS_COMPUTE`: on by default (`0` forces legacy
/// per-fill `CopyBufferRegion` loops instead of the batched compute copy).
fn resolve_compute_copy(env: Option<&str>) -> bool {
    env.is_none_or(|v| v.trim() != "0")
}

fn compute_copy_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let env = std::env::var("VICTORY_CARD_STEPS_COMPUTE").ok();
        resolve_compute_copy(env.as_deref())
    })
}

/// Resolves `VICTORY_CARD_STEPS` (default `2`: constant buffers and buffer
/// copies; `1`: constant buffers only; `0`: disabled).
pub(super) fn resolve_stage(env: Option<&str>) -> u32 {
    match env.map(str::trim) {
        Some("0") => 0,
        Some("1") => 1,
        Some("2") => 2,
        _ => 2,
    }
}

/// The stage (`VICTORY_CARD_STEPS`; default `2`): `1` constant buffers of
/// descriptor tables (`Step::Fill` with a slot of its own), `2` (default) the
/// title's buffer copies (`Step::Copy`) and upload-buffer fills too, with the
/// steps of the same submission that depend on them. Only with the relay's
/// second stage (`relay_exact`). `0` disables.
pub(super) fn stage() -> u32 {
    static STAGE: OnceLock<u32> = OnceLock::new();
    *STAGE.get_or_init(|| {
        let env = std::env::var("VICTORY_CARD_STEPS").ok();
        resolve_stage(env.as_deref())
    })
}

/// Resolves `VICTORY_PLACED_ORDER`: on by default (`0` disables). When on,
/// an inline `Step::Placed` copy only blocks `card_steps` for steps that stand
/// AFTER it in the submission, and `cut_for` splits the submission cleanly
/// behind the placed copy's list instead of stalling 40-60 ms.
fn resolve_placed_order(env: Option<&str>) -> bool {
    env.is_none_or(|v| v.trim() != "0")
}

fn placed_order() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let env = std::env::var("VICTORY_PLACED_ORDER").ok();
        resolve_placed_order(env.as_deref())
    })
}

/// Why [`plan`] refuses for [`place`].
const BEHIND_PLACED: &str = "a copy at its place in a list writes what it reads, or reads or writes what it writes";

/// The bytes of our buffer between reading and writing: what one submission
/// may do this way (each place read once, however many fills take it).
const BETWEEN: u64 = 4 << 20;

/// The most steps of a submission done this way.
const MOST: usize = 12288;

/// How a fill's bytes get where its list reads them, if the card brings
/// them.
#[derive(Clone)]
pub(super) enum FillTo {
    /// A descriptor slot that shows the fill's memory, and nothing else does
    /// (title heap, slot): it is made to show our buffer instead.
    Slot(usize, u32),
    /// A buffer the card can write, filled from its beginning.
    Buffer(ID3D12Resource),
    /// A slice `(buffer, byte_offset)` of a card-writable buffer (e.g. a
    /// `shared_buffer` scratch chunk backing an unaligned root CBV).
    BufferOffset(ID3D12Resource, u64),
}

/// Title memory something in the air writes: the host copy it goes through
/// (`None`: none, the bytes reach title memory when the flight lands), from,
/// to, and whether it is the destination of a copy done here.
pub(super) struct Writer {
    shadow: Option<Arc<Shadow>>,
    from: usize,
    to: usize,
    copy: bool,
}

/// What `items` of the line write.
pub(super) fn writers(items: &[Late]) -> Vec<Writer> {
    let mut out = Vec::new();
    for item in items {
        match item {
            Late::Run(flight) => {
                for (shadow, ranges) in &flight.back {
                    out.extend(ranges.iter().map(|&(from, to)| Writer {
                        shadow: Some(shadow.clone()),
                        from,
                        to,
                        copy: false,
                    }));
                }
                for &i in &flight.order {
                    for &(_, at, [row, rows, pitch]) in &flight.lists[i].host().deferred().backs {
                        out.push(Writer {
                            shadow: None,
                            from: at,
                            to: at + rows.saturating_sub(1) * pitch + row,
                            copy: false,
                        });
                    }
                }
                out.extend(flight.own.iter().flat_map(Taken::copied));
            }
            Late::Write(at, _, bytes) => out.push(Writer {
                shadow: None,
                from: *at,
                to: *at + *bytes,
                copy: false,
            }),
            Late::Signal(..) => {}
        }
    }
    out
}

/// The destinations of copies done here that have not landed.
pub(super) fn copies(line: &Line) -> Vec<Writer> {
    if stage() < 2 {
        return Vec::new();
    }
    let flying = line.line.iter().filter_map(|l| match l {
        Late::Run(flight) => flight.own.as_ref(),
        _ => None,
    });
    let landing = line.landing.iter().filter(|w| w.copy).map(|w| Writer {
        shadow: w.shadow.clone(),
        ..*w
    });
    flying.flat_map(Taken::copied).chain(landing).collect()
}

/// Whether `runs` refresh another host copy of a destination in `copies`,
/// or read it in a step that flights are not waited for otherwise
/// (`Step::Host`): that reads title memory, where the copy is not yet.
pub(super) fn copied_read(copies: &[Writer], runs: &[&HostList]) -> bool {
    if copies.is_empty() {
        return false;
    }
    runs.iter().any(|list| {
        list.deferred().copies.iter().any(|step| match step {
            Step::Refresh(shadow, at, len) => copies
                .iter()
                .any(|w| w.from < *at + *len && *at < w.to && w.shadow.as_ref().is_none_or(|s| !Arc::ptr_eq(s, shadow))),
            Step::Host(_, (from, to)) if from < to => copies.iter().any(|w| w.from < *to && *from < w.to),
            _ => false,
        })
    })
}

/// What a step of a submission does to title memory, as the plan sees it.
/// `asked`: flights in the air stop it from being done on the CPU now.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Does {
    Copy { dst: usize, src: usize, len: usize, asked: bool },
    /// `to`: its memory can be given the bytes on the card.
    Fill { src: usize, len: usize, asked: bool, to: bool },
    /// Another read on the CPU (a texture, vertices expanded).
    Read { from: usize, to: usize, asked: bool },
    /// A refresh of a host copy (named by its address).
    Refresh { shadow: usize, from: usize, to: usize },
    /// A copy that its list does on the card ([`place`]).
    Placed { dst: usize, src: usize, len: usize },
    /// Anything, anywhere.
    Run,
}

fn meet(a: (usize, usize), b: (usize, usize)) -> bool {
    a.0 < b.1 && b.0 < a.1
}

/// The bytes `d` reads and writes as a copy or a fill (empty ranges for
/// what it does not).
fn reads_writes(d: &Does) -> Option<((usize, usize), (usize, usize))> {
    match *d {
        Does::Copy { dst, src, len, .. } => Some(((src, src + len), (dst, dst + len))),
        Does::Fill { src, len, .. } => Some(((src, src + len), (0, 0))),
        _ => None,
    }
}

fn touched(ranges: &[(usize, usize)], r: (usize, usize)) -> bool {
    touches(ranges, r.0, r.1)
}

/// The steps of `does` (by place) to be done on the card, in order, each
/// with its level; or why not. All that are asked for, and every copy and
/// fill that is in the way of one that is done there: it writes what that
/// one reads or writes, or reads what it writes, whichever comes first. So
/// what is left on the CPU writes none of their bytes, and reads of them
/// only what a taken copy has not been told to write yet: bytes a step
/// ahead of it reads, in title memory as it is.
///
/// Steps of a level are done together, every read before every write; a
/// step that writes what an earlier one reads or writes, or reads what it
/// writes, is of a higher level. Our list does them level by level.
fn plan(
    does: &[Does],
    stage: u32,
    by_order: bool,
    behind_placed: &mut Option<(usize, usize)>,
) -> Result<Vec<(usize, u32)>, &'static str> {
    let takes = |d: &Does| match *d {
        Does::Copy { .. } if stage < 2 => Err("a buffer copy (VICTORY_CARD_STEPS=2 does it on the card)"),
        Does::Fill { to: false, .. } => Err("a fill of memory the card cannot be given the bytes of"),
        _ => Ok(()),
    };
    let mut taken = vec![false; does.len()];
    let (mut read, mut written) = (Vec::new(), Vec::new());
    let mut more = Vec::new();
    for (i, d) in does.iter().enumerate() {
        match *d {
            Does::Run => return Err("a step of ours runs"),
            Does::Read { asked: true, .. } => return Err("a texture reads it"),
            Does::Copy { asked: true, .. } | Does::Fill { asked: true, .. } => more.push(i),
            _ => {}
        }
    }
    let has_copies = does.iter().any(|d| matches!(d, Does::Copy { .. }));
    while !more.is_empty() {
        let (mut reads, mut writes) = (Vec::new(), Vec::new());
        for i in std::mem::take(&mut more) {
            takes(&does[i])?;
            taken[i] = true;
            if let Some((r, w)) = reads_writes(&does[i]) {
                if w.0 < w.1 && meet(r, w) {
                    return Err("a copy onto itself");
                }
                reads.push(r);
                writes.push(w);
            }
        }
        join_ranges(&mut read, reads);
        if writes.iter().any(|w| w.0 < w.1) {
            join_ranges(&mut written, writes);
        }
        if taken.iter().filter(|t| **t).count() > MOST {
            return Err("too many steps depend on it");
        }
        // When no taken step writes and no `Does::Copy` exists in `does`,
        // fills have empty write ranges `(0, 0)` and cannot conflict with `read`.
        if written.is_empty() && !has_copies {
            break;
        }
        for (i, d) in does.iter().enumerate() {
            // A fill that cannot be taken is looked at below.
            if taken[i] || matches!(d, Does::Fill { to: false, .. }) {
                continue;
            }
            if let Some((r, w)) = reads_writes(d)
                && (touched(&written, r) || (w.0 < w.1 && (touched(&written, w) || touched(&read, w))))
            {
                more.push(i);
            }
        }
    }
    // A copy at its place in a list comes after what is done in front of all
    // of them: as it should after the steps ahead of it, but not after those
    // behind it that read or write its bytes.
    for (p, d) in does.iter().enumerate() {
        if let Does::Placed { dst, src, len } = *d
            && (touched(&read, (dst, dst + len))
                || touched(&written, (dst, dst + len))
                || touched(&written, (src, src + len)))
        {
            let (to, from) = ((dst, dst + len), (src, src + len));
            let in_the_way = |i: usize| {
                reads_writes(&does[i])
                    .is_some_and(|(r, w)| meet(r, to) || (w.0 < w.1 && (meet(w, to) || meet(w, from))))
            };
            let found = (p + 1..does.len()).find(|&i| taken[i] && in_the_way(i));
            *behind_placed = found.map(|i| (p, i));
            if !by_order || found.is_some() {
                return Err(BEHIND_PLACED);
            }
        }
    }
    let order: Vec<usize> = (0..does.len()).filter(|&i| taken[i]).collect();
    // What stays on the CPU and reads what an earlier taken copy writes
    // would read it too early.
    let mut behind: Vec<(usize, usize)> = Vec::new();
    for (i, d) in does.iter().enumerate() {
        if taken[i] {
            if let Some((_, w)) = reads_writes(d)
                && w.0 < w.1
            {
                join_ranges(&mut behind, [w]);
            }
            continue;
        }
        match *d {
            Does::Read { from, to, .. } if touched(&behind, (from, to)) => {
                return Err("a texture or expanded vertices read what it writes");
            }
            Does::Fill { src, len, .. } if touched(&behind, (src, src + len)) => {
                return Err("a fill of memory the card cannot be given the bytes of reads what it writes");
            }
            _ => {}
        }
    }
    let mut out: Vec<(usize, u32)> = Vec::with_capacity(order.len());
    // The copies among them, which alone put others behind them or have to
    // stand behind others.
    let mut copies: Vec<usize> = Vec::new();
    for &i in &order {
        let Some((r, w)) = reads_writes(&does[i]) else {
            continue;
        };
        let writes = w.0 < w.1;
        let mut level = 0;
        // Behind what writes what this reads or writes, and, if this
        // writes, behind what reads it. (No `Box<dyn Iterator>` heap alloc.)
        if writes {
            for &(j, at) in &out {
                if at < level {
                    continue;
                }
                if let Some((jr, jw)) = reads_writes(&does[j])
                    && (meet(jw, r) || meet(jw, w) || meet(jr, w))
                {
                    level = at + 1;
                }
            }
            copies.push(out.len());
        } else {
            for &k in &copies {
                let (j, at) = out[k];
                if at < level {
                    continue;
                }
                if let Some((_, jw)) = reads_writes(&does[j])
                    && meet(jw, r)
                {
                    level = at + 1;
                }
            }
        }
        out.push((i, level));
    }
    Ok(out)
}

/// Whether a refresh of `does` brings a host copy other than `shadow` up to
/// date in `[from, to)`.
fn elsewhere(does: &[Does], shadow: usize, from: usize, to: usize) -> bool {
    does.iter().any(|d| match *d {
        Does::Refresh { shadow: s, from: a, to: b } => s != shadow && meet((a, b), (from, to)),
        _ => false,
    })
}

/// The host copy that holds `[from, to)` on the card once what `writers`
/// write is there: the one they all write it through, or without any the
/// one shaders would be given.
fn holder(writers: &[Writer], from: usize, to: usize) -> Result<Arc<Shadow>, &'static str> {
    let mut found: Option<&Arc<Shadow>> = None;
    for w in writers.iter().filter(|w| meet((w.from, w.to), (from, to))) {
        match (&w.shadow, found) {
            (None, _) => return Err("a read-back or a value behind a flight is written there"),
            (Some(s), Some(f)) if !Arc::ptr_eq(s, f) => return Err("flights write it through two host copies"),
            (Some(s), _) => found = Some(s),
        }
    }
    let shadow = match found {
        Some(s) => s.clone(),
        None => crate::shadow::find(from as u64, (to - from) as u64).ok_or("no host copy of it")?,
    };
    let (start, len) = shadow.span();
    if shadow.in_place() || (from as u64) < start || to as u64 > start + len {
        return Err("no host copy holds all of it");
    }
    Ok(shadow)
}

/// What the steps of `runs` do, and which step each is (its list by address,
/// its place in it). `written`: what flights may write.
/// When `collect_fill_to` is true, also captures each `Step::Fill`'s `FillTo`
/// in-place to avoid re-scanning `runs` and re-locking `list.deferred()` per fill.
fn does_of(
    written: &[(usize, usize)],
    runs: &[&HostList],
    collect_fill_to: bool,
) -> (Vec<Does>, Vec<(usize, usize)>, Vec<Option<FillTo>>) {
    let mut does = Vec::new();
    // Which step each is. One list's lock at a time: a submission may
    // name a list twice.
    let mut steps = Vec::new();
    let mut fill_tos = Vec::new();
    for list in runs {
        let list_ptr = std::ptr::from_ref::<HostList>(list) as usize;
        for (i, step) in list.deferred().copies.iter().enumerate() {
            let mut fill_to = None;
            does.push(match step {
                &Step::Copy(dst, src, len) => Does::Copy {
                    dst,
                    src,
                    len,
                    asked: touches(written, src, src + len) || touches(written, dst, dst + len),
                },
                Step::Fill(_, src, len, to) => {
                    if collect_fill_to {
                        fill_to = to.clone();
                    }
                    Does::Fill {
                        src: *src,
                        len: *len,
                        asked: touches(written, *src, *src + *len),
                        to: to.is_some(),
                    }
                }
                Step::Texture(staging) => {
                    let (at, len) = staging.memory();
                    Does::Read {
                        from: at,
                        to: at + len,
                        asked: touches(written, at, at + len),
                    }
                }
                &Step::Host(_, (from, to)) if from < to => Does::Read { from, to, asked: false },
                Step::Refresh(shadow, at, len) => Does::Refresh {
                    shadow: Arc::as_ptr(shadow) as usize,
                    from: *at,
                    to: *at + *len,
                },
                Step::Run(_) => Does::Run,
                Step::Placed(p) => Does::Placed { dst: p.dst, src: p.src, len: p.len },
                Step::GpuWrite(..) | Step::Host(..) => continue,
            });
            steps.push((list_ptr, i));
            if collect_fill_to {
                fill_tos.push(fill_to);
            }
        }
    }
    (does, steps, fill_tos)
}

/// With flights in the air: before which of `runs` the submission is cut so
/// that what reads or writes the bytes of a copy at its place in an earlier
/// list ([`BEHIND_PLACED`]) is done on the card in front of a submission of
/// its own, behind that copy.
pub(super) fn cut_for(runs: &[&HostList]) -> Option<usize> {
    if stage() == 0 || !relay_exact() || !placed_order() || runs.len() < 2 {
        return None;
    }
    let first = runs
        .iter()
        .position(|l| l.deferred().copies.iter().any(|s| matches!(s, Step::Placed(_))))?;
    if first + 1 == runs.len() {
        return None;
    }
    let written = line().written.clone();
    blocked_by(&written, runs, false)?;
    let (does, steps, _) = does_of(&written, runs, false);
    let mut behind = None;
    if plan(&does, stage(), true, &mut behind) != Err(BEHIND_PLACED) {
        return None;
    }
    let (placed, reader) = behind?;
    let place = |i: usize| runs.iter().position(|l| std::ptr::from_ref::<HostList>(l) as usize == steps[i].0);
    let cut = place(placed)? + 1;
    (place(reader)? >= cut).then_some(cut)
}

/// Whether every page of `[from, to)` can be read (through the kernel:
/// asking Windows walks the page tables, milliseconds in the title's heap).
/// `known`: pages that could be, of the same submission.
fn mapped(from: usize, to: usize, known: &mut HashSet<usize>) -> bool {
    const PAGE: usize = 4096;
    (from / PAGE..to.div_ceil(PAGE)).all(|page| {
        known.contains(&page)
            || lazy_readback::is_gpu_resident((page * PAGE) / lazy_readback::BLOCK)
            || {
                let there = read_into((page * PAGE).max(from) as *const u8, &mut [0]);
                if there {
                    known.insert(page);
                }
                there
            }
    })
}

/// A step done on the card.
struct Op {
    from: (Arc<Shadow>, usize),
    len: usize,
    /// Its place in the buffer between.
    at: u64,
    level: u32,
    to: To,
}

enum To {
    /// A host copy, and the title address.
    Copy(Arc<Shadow>, usize),
    /// A descriptor slot (title heap, slot).
    Slot(usize, u32),
    /// A buffer's beginning.
    Buffer(ID3D12Resource),
    /// A byte offset inside a card-writable buffer (e.g. unaligned root CBV).
    BufferOffset(ID3D12Resource, u64),
}

/// 16-byte descriptor for one 4-byte-aligned chunk executed by [`CARD_STEPS_COPY`].
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CopyJob {
    src_off: u32,
    dst_off: u32,
    dwords: u32,
    _pad: u32,
}

/// Chunk size (4 KiB = 1,024 dwords = 256 `uint4`s = 4 iterations/wave) so
/// large copies fan out across all GPU SMs instead of stalling one workgroup.
const COMPUTE_CHUNK_BYTES: u64 = 4096;

/// Initial capacity of [`JobBuf`] in [`CopyJob`] descriptors (`16,384` = 256 KiB).
const INITIAL_JOB_CAP: usize = 16_384;

/// Batched GPU compute copy shader replacing thousands of tiny scalar
/// `CopyBufferRegion` commands with a single `Dispatch` per buffer pair.
///
/// Uses root descriptors only (`b0`, `t0`, `t1`, `u0`) so no descriptor heap
/// switch is required. Uses 128-bit `Load4`/`Store4` whenever both offsets are
/// 16-byte aligned, falling back to 32-bit `Load`/`Store` for 4-byte alignment.
const CARD_STEPS_COPY: &str = r#"
#define RS "RootConstants(num32BitConstants=2, b0), SRV(t0), SRV(t1), UAV(u0)"
cbuffer Meta : register(b0) {
    uint first_job;
    uint job_count;
};
struct CopyJob {
    uint src_off;
    uint dst_off;
    uint dwords;
    uint _pad;
};
StructuredBuffer<CopyJob> jobs : register(t0);
ByteAddressBuffer src_buf : register(t1);
RWByteAddressBuffer dst_buf : register(u0);
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    uint idx = first_job + gid.x;
    if (idx >= job_count) return;
    CopyJob j = jobs[idx];
    uint n = j.dwords;
    if (((j.src_off | j.dst_off) & 15u) == 0u) {
        uint vecs = n >> 2u;
        [loop]
        for (uint v = tid; v < vecs; v += 64u) {
            uint off = v << 4u;
            dst_buf.Store4(j.dst_off + off, src_buf.Load4(j.src_off + off));
        }
        uint tail = vecs << 2u;
        if (tail + tid < n) {
            uint off = (tail + tid) << 2u;
            dst_buf.Store(j.dst_off + off, src_buf.Load(j.src_off + off));
        }
    } else {
        [loop]
        for (uint i = tid; i < n; i += 64u) {
            uint off = i << 2u;
            dst_buf.Store(j.dst_off + off, src_buf.Load(j.src_off + off));
        }
    }
}
"#;

#[derive(Clone)]
struct CopyShader {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

// SAFETY: D3D12 root signatures and pipeline states are COM free-threaded.
unsafe impl Send for CopyShader {}
// SAFETY: as above.
unsafe impl Sync for CopyShader {}

/// Compiles and caches [`CARD_STEPS_COPY`] keyed by `device` pointer so
/// standalone Debug-Layer test devices never share PSOs with `host()`.
fn copy_shader(device: &ID3D12Device) -> Option<CopyShader> {
    if !compute_copy_enabled() {
        return None;
    }
    static CACHE: Mutex<Option<(usize, Option<CopyShader>)>> = Mutex::new(None);
    let key = device.as_raw() as usize;
    let mut slot = CACHE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((cached_key, shader)) = slot.as_ref()
        && *cached_key == key
    {
        return shader.clone();
    }
    let compiled = compile_copy_shader(device).ok();
    *slot = Some((key, compiled.clone()));
    compiled
}

fn compile_copy_shader(device: &ID3D12Device) -> windows::core::Result<CopyShader> {
    use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
    use windows::Win32::Graphics::Direct3D::ID3DBlob;
    let mut code: Option<ID3DBlob> = None;
    // SAFETY: static HLSL source and entry strings live through the call.
    unsafe {
        D3DCompile(
            CARD_STEPS_COPY.as_ptr().cast(),
            CARD_STEPS_COPY.len(),
            windows::core::s!("card steps copy"),
            None,
            None,
            windows::core::s!("main"),
            windows::core::s!("cs_5_1"),
            0,
            0,
            &mut code,
            None,
        )?;
    }
    let code = code.ok_or_else(windows::core::Error::empty)?;
    // SAFETY: valid compiled bytecode with embedded root signature.
    unsafe {
        let bytes = std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
        let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
        let pso: ID3D12PipelineState = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: bytes.as_ptr().cast(),
                BytecodeLength: bytes.len(),
            },
            ..Default::default()
        })?;
        Ok(CopyShader { root, pso })
    }
}

/// Persistently mapped upload buffer holding [`CopyJob`] descriptors for one
/// `card_steps` command list (reused across submissions, grown lazily).
struct JobBuf {
    buffer: ID3D12Resource,
    gpu: u64,
    ptr: *mut CopyJob,
    cap: usize,
    used: usize,
}

// SAFETY: `ID3D12Resource` is free-threaded; `JobBuf` is owned by a single
// `Own` (or test) and accessed by one thread at a time while recording.
unsafe impl Send for JobBuf {}

impl JobBuf {
    fn new(device: &ID3D12Device, cap: usize) -> Option<Self> {
        let cap = cap.max(256);
        let bytes = (cap * size_of::<CopyJob>()) as u64;
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_UPLOAD,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: bytes,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        let mut data = std::ptr::null_mut();
        // SAFETY: plain UPLOAD buffer creation and persistent CPU write mapping.
        unsafe {
            device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                    None,
                    &mut buffer,
                )
                .ok()?;
            let buffer = buffer?;
            buffer.Map(0, None, Some(&mut data)).ok()?;
            let gpu = buffer.GetGPUVirtualAddress();
            Some(Self {
                buffer,
                gpu,
                ptr: data.cast(),
                cap,
                used: 0,
            })
        }
    }

    fn reset(&mut self) {
        self.used = 0;
    }

    /// Appends `jobs` to the upload buffer, growing if needed (keeping any
    /// replaced buffer alive via `keep_old`). Returns `(buffer_gpu_va, first_job, count)`.
    fn push(
        &mut self,
        device: &ID3D12Device,
        jobs: &[CopyJob],
        mut keep_old: impl FnMut(ID3D12Resource),
    ) -> Option<(u64, u32, u32)> {
        if jobs.is_empty() {
            return None;
        }
        if self.used + jobs.len() > self.cap {
            let new_cap = (self.cap * 2).max(jobs.len().next_power_of_two());
            let next = Self::new(device, new_cap)?;
            let old = std::mem::replace(self, next);
            keep_old(old.buffer);
        }
        let first = self.used;
        // SAFETY: `first + jobs.len() <= self.cap` inside persistently mapped upload buffer.
        unsafe {
            std::ptr::copy_nonoverlapping(jobs.as_ptr(), self.ptr.add(first), jobs.len());
        }
        self.used += jobs.len();
        Some((self.gpu, first as u32, jobs.len() as u32))
    }
}

/// Tracks written byte intervals `[start, end)` within one compute wave so
/// overlapping writes split into ordered waves separated by a targeted UAV barrier.
#[derive(Default)]
struct OverlapTracker {
    max_end: u64,
    /// Disjoint non-overlapping intervals `start -> end` in the active wave.
    intervals: BTreeMap<u64, u64>,
}

impl OverlapTracker {
    fn overlaps(&self, start: u64, end: u64) -> bool {
        if start >= end || start >= self.max_end {
            return false;
        }
        if let Some((_, &prev_end)) = self.intervals.range(..end).next_back()
            && prev_end > start
        {
            return true;
        }
        self.intervals.range(start..end).next().is_some()
    }

    fn insert(&mut self, start: u64, end: u64) {
        if start >= end {
            return;
        }
        self.max_end = self.max_end.max(end);
        self.intervals.insert(start, end);
    }

    fn clear(&mut self) {
        self.max_end = 0;
        self.intervals.clear();
    }
}

/// Whether `(src_off, dst_off, len)` is eligible for [`CARD_STEPS_COPY`]:
/// 4-byte aligned offsets and length, non-empty, within 32-bit `ByteAddressBuffer` offsets.
fn is_compute_aligned(src_off: u64, dst_off: u64, len: u64) -> bool {
    len > 0
        && ((src_off | dst_off | len) & 3) == 0
        && src_off.checked_add(len).is_some_and(|e| e <= u32::MAX as u64)
        && dst_off.checked_add(len).is_some_and(|e| e <= u32::MAX as u64)
}

/// Pushes chunked [`CopyJob`]s for an aligned copy so large fills fan out across SMs.
fn push_chunked_jobs(out: &mut Vec<CopyJob>, mut src_off: u64, mut dst_off: u64, mut len: u64) {
    while len > 0 {
        let piece = len.min(COMPUTE_CHUNK_BYTES);
        out.push(CopyJob {
            src_off: src_off as u32,
            dst_off: dst_off as u32,
            dwords: (piece / 4) as u32,
            _pad: 0,
        });
        src_off += piece;
        dst_off += piece;
        len -= piece;
    }
}

/// Whether `device` supports D3D12 Enhanced Barriers (`OPTIONS12`).
fn enhanced_barriers_on(device: &ID3D12Device) -> bool {
    let mut o = D3D12_FEATURE_DATA_D3D12_OPTIONS12::default();
    // SAFETY: plain feature query on live device.
    let ok = unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS12,
            (&raw mut o).cast(),
            size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS12>() as u32,
        )
    };
    ok.is_ok() && o.EnhancedBarriersSupported.as_bool()
}

/// Targeted legacy UAV barrier for a single buffer (never global `UAV(NULL)`).
fn uav_barrier(resource: &ID3D12Resource) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                // SAFETY: borrowed pointer; never released by the barrier.
                pResource: unsafe { std::mem::transmute_copy(resource) },
            }),
        },
    }
}

/// Targeted Enhanced Buffer Barrier for a single buffer.
fn buffer_barrier(
    resource: &ID3D12Resource,
    sync_before: D3D12_BARRIER_SYNC,
    sync_after: D3D12_BARRIER_SYNC,
    access_before: D3D12_BARRIER_ACCESS,
    access_after: D3D12_BARRIER_ACCESS,
) -> D3D12_BUFFER_BARRIER {
    D3D12_BUFFER_BARRIER {
        SyncBefore: sync_before,
        SyncAfter: sync_after,
        AccessBefore: access_before,
        AccessAfter: access_after,
        // SAFETY: borrowed pointer; lives across `Barrier`.
        pResource: unsafe { std::mem::transmute_copy(resource) },
        Offset: 0,
        Size: u64::MAX,
    }
}

const SYNC_ALL_BUF: D3D12_BARRIER_SYNC = D3D12_BARRIER_SYNC(
    D3D12_BARRIER_SYNC_ALL_SHADING.0
        | D3D12_BARRIER_SYNC_INDEX_INPUT.0
        | D3D12_BARRIER_SYNC_EXECUTE_INDIRECT.0
        | D3D12_BARRIER_SYNC_COPY.0,
);
const SYNC_WRITE_BUF: D3D12_BARRIER_SYNC =
    D3D12_BARRIER_SYNC(D3D12_BARRIER_SYNC_ALL_SHADING.0 | D3D12_BARRIER_SYNC_COPY.0);
const SYNC_READ_BUF: D3D12_BARRIER_SYNC =
    D3D12_BARRIER_SYNC(D3D12_BARRIER_SYNC_COMPUTE_SHADING.0 | D3D12_BARRIER_SYNC_COPY.0);
const ACCESS_ALL_BUF: D3D12_BARRIER_ACCESS = D3D12_BARRIER_ACCESS(
    D3D12_BARRIER_ACCESS_UNORDERED_ACCESS.0
        | D3D12_BARRIER_ACCESS_SHADER_RESOURCE.0
        | D3D12_BARRIER_ACCESS_CONSTANT_BUFFER.0
        | D3D12_BARRIER_ACCESS_VERTEX_BUFFER.0
        | D3D12_BARRIER_ACCESS_INDEX_BUFFER.0
        | D3D12_BARRIER_ACCESS_INDIRECT_ARGUMENT.0
        | D3D12_BARRIER_ACCESS_COPY_SOURCE.0
        | D3D12_BARRIER_ACCESS_COPY_DEST.0,
);
const ACCESS_ALL_BUF_NO_UAV: D3D12_BARRIER_ACCESS = D3D12_BARRIER_ACCESS(
    D3D12_BARRIER_ACCESS_SHADER_RESOURCE.0
        | D3D12_BARRIER_ACCESS_CONSTANT_BUFFER.0
        | D3D12_BARRIER_ACCESS_VERTEX_BUFFER.0
        | D3D12_BARRIER_ACCESS_INDEX_BUFFER.0
        | D3D12_BARRIER_ACCESS_INDIRECT_ARGUMENT.0
        | D3D12_BARRIER_ACCESS_COPY_SOURCE.0
        | D3D12_BARRIER_ACCESS_COPY_DEST.0,
);
const ACCESS_WRITE_BUF: D3D12_BARRIER_ACCESS =
    D3D12_BARRIER_ACCESS(D3D12_BARRIER_ACCESS_UNORDERED_ACCESS.0 | D3D12_BARRIER_ACCESS_COPY_DEST.0);
const ACCESS_WRITE_BUF_NO_UAV: D3D12_BARRIER_ACCESS = D3D12_BARRIER_ACCESS_COPY_DEST;
const ACCESS_READ_SRC: D3D12_BARRIER_ACCESS =
    D3D12_BARRIER_ACCESS(D3D12_BARRIER_ACCESS_SHADER_RESOURCE.0 | D3D12_BARRIER_ACCESS_COPY_SOURCE.0);
const ACCESS_READ_BETWEEN: D3D12_BARRIER_ACCESS = D3D12_BARRIER_ACCESS(
    D3D12_BARRIER_ACCESS_SHADER_RESOURCE.0
        | D3D12_BARRIER_ACCESS_CONSTANT_BUFFER.0
        | D3D12_BARRIER_ACCESS_COPY_SOURCE.0,
);

#[derive(Clone, Copy, PartialEq, Eq)]
enum WriteMode {
    None,
    Compute,
    Copy,
}

impl WriteMode {
    fn sync_access(self) -> (D3D12_BARRIER_SYNC, D3D12_BARRIER_ACCESS) {
        match self {
            Self::Compute => (D3D12_BARRIER_SYNC_COMPUTE_SHADING, D3D12_BARRIER_ACCESS_UNORDERED_ACCESS),
            Self::Copy => (D3D12_BARRIER_SYNC_COPY, D3D12_BARRIER_ACCESS_COPY_DEST),
            Self::None => (D3D12_BARRIER_SYNC_NONE, D3D12_BARRIER_ACCESS_NO_ACCESS),
        }
    }

    fn legacy_state(self) -> D3D12_RESOURCE_STATES {
        match self {
            Self::Compute => D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            Self::Copy => D3D12_RESOURCE_STATE_COPY_DEST,
            Self::None => D3D12_RESOURCE_STATE_COMMON,
        }
    }
}

/// Emits a non-empty slice of Enhanced Buffer Barriers on `list7`.
///
/// # Safety
/// `list7` is open and every resource referenced in `barriers` is alive.
unsafe fn emit_enhanced(list7: &ID3D12GraphicsCommandList7, barriers: &[D3D12_BUFFER_BARRIER]) {
    if barriers.is_empty() {
        return;
    }
    let group = [D3D12_BARRIER_GROUP {
        Type: D3D12_BARRIER_TYPE_BUFFER,
        NumBarriers: barriers.len() as u32,
        Anonymous: D3D12_BARRIER_GROUP_0 {
            pBufferBarriers: barriers.as_ptr(),
        },
    }];
    // SAFETY: caller's invariants.
    unsafe { list7.Barrier(&group) };
}

/// Dispatches `count` jobs starting at `first` from `jobs_gpu` copying `src -> dst`.
///
/// # Safety
/// `list` is open with `shader` bound; `jobs_gpu`, `src`, and `dst` are valid and in
/// SRV/UAV-compatible states.
unsafe fn dispatch_copy_jobs(
    list: &ID3D12GraphicsCommandList,
    shader: &CopyShader,
    bound: &mut bool,
    jobs_gpu: u64,
    src: &ID3D12Resource,
    dst: &ID3D12Resource,
    first: u32,
    count: u32,
) {
    if count == 0 {
        return;
    }
    // SAFETY: caller's invariants.
    unsafe {
        if !*bound {
            list.SetComputeRootSignature(&shader.root);
            list.SetPipelineState(&shader.pso);
            *bound = true;
        }
        list.SetComputeRootShaderResourceView(1, jobs_gpu);
        list.SetComputeRootShaderResourceView(2, src.GetGPUVirtualAddress());
        list.SetComputeRootUnorderedAccessView(3, dst.GetGPUVirtualAddress());
        let max_groups = D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION;
        let mut done = 0u32;
        while done < count {
            let n = (count - done).min(max_groups);
            let meta = [first + done, first + count];
            list.SetComputeRoot32BitConstants(0, 2, meta.as_ptr().cast(), 0);
            list.Dispatch(n, 1, 1);
            done += n;
        }
    }
}

/// Records a level of what [`stage`] describes into `list` (open): `reads`
/// are copied into `between` (source, offset in it, offset in `between`,
/// bytes), then `writes` out of it (destination, offset in it, offset in
/// `between`, bytes).
///
/// Uses O(N) resource dedup, batched GPU compute copy ([`CARD_STEPS_COPY`]) for
/// 4-byte-aligned fills/copies (splitting overlapping writes into ordered waves),
/// `CopyBufferRegion` fallback for unaligned/non-UAV edge cases, and targeted
/// per-buffer Enhanced Barriers (or targeted legacy barriers — zero `UAV(NULL)`).
fn record_level(
    first: bool,
    device: Option<&ID3D12Device>,
    list: &ID3D12GraphicsCommandList,
    keep_list: Option<&HostList>,
    between: &ID3D12Resource,
    mut jobs_buf: Option<&mut JobBuf>,
    reads: &[(&ID3D12Resource, u64, u64, u64)],
    writes: &[(&ID3D12Resource, u64, u64, u64)],
) {
    if reads.is_empty() && writes.is_empty() {
        return;
    }
    // O(N) deduplication of unique source and destination buffers.
    let mut source_set: HashSet<usize> = HashSet::with_capacity(reads.len().min(64));
    let mut sources: Vec<&ID3D12Resource> = Vec::new();
    for &(buffer, ..) in reads {
        if source_set.insert(buffer.as_raw() as usize) {
            sources.push(buffer);
        }
    }
    let mut dest_set: HashSet<usize> = HashSet::with_capacity(writes.len().min(64));
    let mut dests: Vec<&ID3D12Resource> = Vec::new();
    let mut dest_uav: HashMap<usize, bool> = HashMap::new();
    // Group writes per unique destination buffer while preserving exact per-buffer order.
    let mut writes_by_dest: HashMap<usize, Vec<(u64, u64, u64)>> = HashMap::new();
    for &(buffer, dst_off, at, len) in writes {
        let raw = buffer.as_raw() as usize;
        if dest_set.insert(raw) {
            dests.push(buffer);
            // SAFETY: plain COM query on a live buffer.
            let has_uav = unsafe { buffer.GetDesc() }
                .Flags
                .contains(D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
            dest_uav.insert(raw, has_uav);
        }
        writes_by_dest.entry(raw).or_default().push((dst_off, at, len));
    }

    let shader = device.and_then(copy_shader);
    let list7 = device
        .filter(|d| enhanced_barriers_on(d))
        .and_then(|_| list.cast::<ID3D12GraphicsCommandList7>().ok());
    let use_compute = shader.is_some() && jobs_buf.is_some();

    // SAFETY: `between` is a live buffer created with `ALLOW_UNORDERED_ACCESS`.
    let between_uav = unsafe { between.GetDesc() }
        .Flags
        .contains(D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

    // Precompute each destination's first Phase-2 write mode so mid-barriers
    // transition directly to the right state/access without an extra hop.
    let first_write_mode_of = |raw: usize| -> WriteMode {
        let Some(ops) = writes_by_dest.get(&raw) else {
            return WriteMode::None;
        };
        let Some(&(dst_off, at, len)) = ops.first() else {
            return WriteMode::None;
        };
        if use_compute && *dest_uav.get(&raw).unwrap_or(&false) && is_compute_aligned(at, dst_off, len) {
            WriteMode::Compute
        } else {
            WriteMode::Copy
        }
    };

    let mut bound_compute = false;
    let mut keep_old = |old: ID3D12Resource| {
        if let Some(hl) = keep_list {
            hl.keep(&old);
        }
    };

    // --- Enhanced Barriers path (zero legacy ResourceBarrier mixing, zero UAV(NULL)) ---
    if let Some(list7) = list7 {
        // 1. Pre-Phase-1 targeted source barriers.
        if first && !sources.is_empty() {
            let pre: Vec<D3D12_BUFFER_BARRIER> = sources
                .iter()
                .map(|src| {
                    // SAFETY: plain COM query on live buffer.
                    let has_uav = unsafe { src.GetDesc() }
                        .Flags
                        .contains(D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
                    let ab = if has_uav { ACCESS_WRITE_BUF } else { ACCESS_WRITE_BUF_NO_UAV };
                    buffer_barrier(src, SYNC_WRITE_BUF, SYNC_READ_BUF, ab, ACCESS_READ_SRC)
                })
                .collect();
            // SAFETY: open command list and live buffers.
            unsafe { emit_enhanced(&list7, &pre) };
        }

        // 2. Phase 1 (`reads`: `src -> between`), batching non-overlapping aligned reads.
        let mut between_mode = WriteMode::None;
        let mut between_was_read = !first;
        let mut between_tracker = OverlapTracker::default();
        let mut wave_order: Vec<usize> = Vec::new();
        let mut wave_jobs: HashMap<usize, (&ID3D12Resource, Vec<CopyJob>)> = HashMap::new();

        let ensure_between = |target: WriteMode,
                              cur: &mut WriteMode,
                              was_read: &mut bool,
                              force_uav: bool| {
            let (sa, aa) = target.sync_access();
            if *was_read {
                let b = [buffer_barrier(
                    between,
                    SYNC_WRITE_BUF,
                    sa,
                    ACCESS_READ_BETWEEN,
                    aa,
                )];
                // SAFETY: open command list and live `between`.
                unsafe { emit_enhanced(&list7, &b) };
                *was_read = false;
                *cur = target;
            } else if *cur == WriteMode::None {
                let b = [buffer_barrier(between, SYNC_ALL_BUF, sa, ACCESS_ALL_BUF, aa)];
                // SAFETY: open command list and live `between`.
                unsafe { emit_enhanced(&list7, &b) };
                *cur = target;
            } else if *cur != target || (force_uav && target == WriteMode::Compute) {
                let (sb, ab) = cur.sync_access();
                let b = [buffer_barrier(between, sb, sa, ab, aa)];
                // SAFETY: open command list and live `between`.
                unsafe { emit_enhanced(&list7, &b) };
                *cur = target;
            }
        };

        let flush_reads = |wave_order: &mut Vec<usize>,
                           wave_jobs: &mut HashMap<usize, (&ID3D12Resource, Vec<CopyJob>)>,
                           between_mode: &mut WriteMode,
                           between_was_read: &mut bool,
                           bound_compute: &mut bool,
                           jobs_buf: &mut Option<&mut JobBuf>,
                           keep_old: &mut dyn FnMut(ID3D12Resource),
                           force_uav: bool| {
            if wave_order.is_empty() {
                return;
            }
            let (Some(sh), Some(dev), Some(jb)) = (shader.as_ref(), device, jobs_buf.as_deref_mut()) else {
                wave_order.clear();
                wave_jobs.clear();
                return;
            };
            ensure_between(WriteMode::Compute, between_mode, between_was_read, force_uav);
            for raw in wave_order.drain(..) {
                if let Some((src, jobs)) = wave_jobs.remove(&raw)
                    && let Some((jobs_gpu, first_job, count)) = jb.push(dev, &jobs, |old| keep_old(old))
                {
                    // SAFETY: open list, valid shader/jobs/src/between in SRV/UAV states.
                    unsafe {
                        dispatch_copy_jobs(list, sh, bound_compute, jobs_gpu, src, between, first_job, count);
                    }
                }
            }
        };

        let mut between_need_uav_sep = false;
        for &(src, src_off, at, len) in reads {
            if use_compute && between_uav && is_compute_aligned(src_off, at, len) {
                if between_tracker.overlaps(at, at + len) {
                    flush_reads(
                        &mut wave_order,
                        &mut wave_jobs,
                        &mut between_mode,
                        &mut between_was_read,
                        &mut bound_compute,
                        &mut jobs_buf,
                        &mut keep_old,
                        between_need_uav_sep,
                    );
                    between_need_uav_sep = true;
                    between_tracker.clear();
                }
                let raw = src.as_raw() as usize;
                let entry = wave_jobs.entry(raw).or_insert_with(|| {
                    wave_order.push(raw);
                    (src, Vec::new())
                });
                push_chunked_jobs(&mut entry.1, src_off, at, len);
                between_tracker.insert(at, at + len);
            } else {
                if !wave_order.is_empty() {
                    flush_reads(
                        &mut wave_order,
                        &mut wave_jobs,
                        &mut between_mode,
                        &mut between_was_read,
                        &mut bound_compute,
                        &mut jobs_buf,
                        &mut keep_old,
                        between_need_uav_sep,
                    );
                    between_need_uav_sep = false;
                    between_tracker.clear();
                }
                ensure_between(WriteMode::Copy, &mut between_mode, &mut between_was_read, false);
                // SAFETY: open command list and live buffers.
                unsafe { list.CopyBufferRegion(between, at, src, src_off, len) };
            }
        }
        flush_reads(
            &mut wave_order,
            &mut wave_jobs,
            &mut between_mode,
            &mut between_was_read,
            &mut bound_compute,
            &mut jobs_buf,
            &mut keep_old,
            between_need_uav_sep,
        );

        // 3. Mid-barriers: publish `between` for reads/constants and prepare `dests`.
        let mut mid = Vec::with_capacity(1 + dests.len());
        if between_mode != WriteMode::None {
            let (sb, ab) = between_mode.sync_access();
            mid.push(buffer_barrier(between, sb, SYNC_WRITE_BUF, ab, ACCESS_READ_BETWEEN));
        }
        for dst in &dests {
            let raw = dst.as_raw() as usize;
            let first_mode = first_write_mode_of(raw);
            let (sa, aa) = first_mode.sync_access();
            if source_set.contains(&raw) {
                mid.push(buffer_barrier(dst, SYNC_READ_BUF, sa, ACCESS_READ_SRC, aa));
            } else {
                let ab = if *dest_uav.get(&raw).unwrap_or(&false) {
                    ACCESS_ALL_BUF
                } else {
                    ACCESS_ALL_BUF_NO_UAV
                };
                mid.push(buffer_barrier(dst, SYNC_ALL_BUF, sa, ab, aa));
            }
        }
        // SAFETY: open command list and live buffers.
        unsafe { emit_enhanced(&list7, &mid) };

        // 4. Phase 2 (`writes`: `between -> dst`), batched per destination with exact overlap waves.
        let mut post = Vec::with_capacity(dests.len());
        for &dst in &dests {
            let raw = dst.as_raw() as usize;
            let has_uav = *dest_uav.get(&raw).unwrap_or(&false);
            let Some(ops) = writes_by_dest.get(&raw) else {
                continue;
            };
            let mut cur_mode = first_write_mode_of(raw);
            let mut tracker = OverlapTracker::default();
            let mut wave: Vec<CopyJob> = Vec::new();
            let mut need_uav_sep = false;

            let flush_dst = |wave: &mut Vec<CopyJob>,
                             cur_mode: &mut WriteMode,
                             bound_compute: &mut bool,
                             jobs_buf: &mut Option<&mut JobBuf>,
                             keep_old: &mut dyn FnMut(ID3D12Resource),
                             need_uav_sep: bool| {
                if wave.is_empty() {
                    return;
                }
                let (Some(sh), Some(dev), Some(jb)) = (shader.as_ref(), device, jobs_buf.as_deref_mut()) else {
                    wave.clear();
                    return;
                };
                if *cur_mode == WriteMode::Copy {
                    let b = [buffer_barrier(
                        dst,
                        D3D12_BARRIER_SYNC_COPY,
                        D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                        D3D12_BARRIER_ACCESS_COPY_DEST,
                        D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                    )];
                    // SAFETY: open list and live `dst`.
                    unsafe { emit_enhanced(&list7, &b) };
                    *cur_mode = WriteMode::Compute;
                } else if need_uav_sep {
                    let b = [buffer_barrier(
                        dst,
                        D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                        D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                        D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                        D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                    )];
                    // SAFETY: open list and live `dst`.
                    unsafe { emit_enhanced(&list7, &b) };
                }
                if let Some((jobs_gpu, first_job, count)) = jb.push(dev, wave, |old| keep_old(old)) {
                    // SAFETY: open list, valid shader/jobs/between/dst.
                    unsafe {
                        dispatch_copy_jobs(list, sh, bound_compute, jobs_gpu, between, dst, first_job, count);
                    }
                }
                wave.clear();
            };

            for &(dst_off, at, len) in ops {
                if use_compute && has_uav && is_compute_aligned(at, dst_off, len) {
                    if tracker.overlaps(dst_off, dst_off + len) {
                        flush_dst(&mut wave, &mut cur_mode, &mut bound_compute, &mut jobs_buf, &mut keep_old, need_uav_sep);
                        need_uav_sep = true;
                        tracker.clear();
                    }
                    push_chunked_jobs(&mut wave, at, dst_off, len);
                    tracker.insert(dst_off, dst_off + len);
                } else {
                    if !wave.is_empty() {
                        flush_dst(&mut wave, &mut cur_mode, &mut bound_compute, &mut jobs_buf, &mut keep_old, need_uav_sep);
                        need_uav_sep = false;
                        tracker.clear();
                    }
                    if cur_mode == WriteMode::Compute {
                        let b = [buffer_barrier(
                            dst,
                            D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                            D3D12_BARRIER_SYNC_COPY,
                            D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                            D3D12_BARRIER_ACCESS_COPY_DEST,
                        )];
                        // SAFETY: open list and live `dst`.
                        unsafe { emit_enhanced(&list7, &b) };
                        cur_mode = WriteMode::Copy;
                    }
                    // SAFETY: open list and live buffers.
                    unsafe { list.CopyBufferRegion(dst, dst_off, between, at, len) };
                }
            }
            flush_dst(&mut wave, &mut cur_mode, &mut bound_compute, &mut jobs_buf, &mut keep_old, need_uav_sep);
            let (sb, ab) = cur_mode.sync_access();
            let aa = if has_uav { ACCESS_ALL_BUF } else { ACCESS_ALL_BUF_NO_UAV };
            post.push(buffer_barrier(dst, sb, SYNC_ALL_BUF, ab, aa));
        }

        // 5. Post-Phase-2 targeted destination barriers.
        // SAFETY: open command list and live buffers.
        unsafe { emit_enhanced(&list7, &post) };
        return;
    }

    // --- Legacy ResourceBarrier fallback (targeted per-buffer UAV barriers, zero UAV(NULL)) ---
    let common = D3D12_RESOURCE_STATE_COMMON;
    let generic_read = D3D12_RESOURCE_STATE_GENERIC_READ;
    let mut pre = Vec::new();
    if first {
        for src in &sources {
            pre.push(uav_barrier(src));
        }
        for dst in &dests {
            if !source_set.contains(&(dst.as_raw() as usize)) {
                pre.push(uav_barrier(dst));
            }
        }
    }
    for src in &sources {
        pre.push(transition(src, common, generic_read));
    }
    if !pre.is_empty() {
        // SAFETY: open command list and live buffers.
        unsafe { list.ResourceBarrier(&pre) };
    }

    let mut between_state = if first { common } else { generic_read };
    let mut between_tracker = OverlapTracker::default();
    let mut wave_order: Vec<usize> = Vec::new();
    let mut wave_jobs: HashMap<usize, (&ID3D12Resource, Vec<CopyJob>)> = HashMap::new();

    let legacy_between = |target: WriteMode, cur_state: &mut D3D12_RESOURCE_STATES, force_uav: bool| {
        let want = target.legacy_state();
        if *cur_state != want {
            let b = [transition(between, *cur_state, want)];
            // SAFETY: open list and live `between`.
            unsafe { list.ResourceBarrier(&b) };
            *cur_state = want;
        } else if force_uav && target == WriteMode::Compute {
            let b = [uav_barrier(between)];
            // SAFETY: open list and live `between`.
            unsafe { list.ResourceBarrier(&b) };
        }
    };

    let flush_legacy_reads = |wave_order: &mut Vec<usize>,
                              wave_jobs: &mut HashMap<usize, (&ID3D12Resource, Vec<CopyJob>)>,
                              between_state: &mut D3D12_RESOURCE_STATES,
                              bound_compute: &mut bool,
                              jobs_buf: &mut Option<&mut JobBuf>,
                              keep_old: &mut dyn FnMut(ID3D12Resource),
                              force_uav: bool| {
        if wave_order.is_empty() {
            return;
        }
        let (Some(sh), Some(dev), Some(jb)) = (shader.as_ref(), device, jobs_buf.as_deref_mut()) else {
            wave_order.clear();
            wave_jobs.clear();
            return;
        };
        legacy_between(WriteMode::Compute, between_state, force_uav);
        for raw in wave_order.drain(..) {
            if let Some((src, jobs)) = wave_jobs.remove(&raw)
                && let Some((jobs_gpu, first_job, count)) = jb.push(dev, &jobs, |old| keep_old(old))
            {
                // SAFETY: open list, valid shader/jobs/src/between.
                unsafe {
                    dispatch_copy_jobs(list, sh, bound_compute, jobs_gpu, src, between, first_job, count);
                }
            }
        }
    };

    let mut need_sep = false;
    for &(src, src_off, at, len) in reads {
        if use_compute && between_uav && is_compute_aligned(src_off, at, len) {
            if between_tracker.overlaps(at, at + len) {
                flush_legacy_reads(
                    &mut wave_order,
                    &mut wave_jobs,
                    &mut between_state,
                    &mut bound_compute,
                    &mut jobs_buf,
                    &mut keep_old,
                    need_sep,
                );
                need_sep = true;
                between_tracker.clear();
            }
            let raw = src.as_raw() as usize;
            let entry = wave_jobs.entry(raw).or_insert_with(|| {
                wave_order.push(raw);
                (src, Vec::new())
            });
            push_chunked_jobs(&mut entry.1, src_off, at, len);
            between_tracker.insert(at, at + len);
        } else {
            if !wave_order.is_empty() {
                flush_legacy_reads(
                    &mut wave_order,
                    &mut wave_jobs,
                    &mut between_state,
                    &mut bound_compute,
                    &mut jobs_buf,
                    &mut keep_old,
                    need_sep,
                );
                need_sep = false;
                between_tracker.clear();
            }
            legacy_between(WriteMode::Copy, &mut between_state, false);
            // SAFETY: open command list and live buffers.
            unsafe { list.CopyBufferRegion(between, at, src, src_off, len) };
        }
    }
    flush_legacy_reads(
        &mut wave_order,
        &mut wave_jobs,
        &mut between_state,
        &mut bound_compute,
        &mut jobs_buf,
        &mut keep_old,
        need_sep,
    );

    let mut mid = Vec::new();
    if between_state != generic_read {
        mid.push(transition(between, between_state, generic_read));
    }
    for src in &sources {
        let raw = src.as_raw() as usize;
        if dest_set.contains(&raw) {
            mid.push(transition(src, generic_read, first_write_mode_of(raw).legacy_state()));
        } else {
            mid.push(transition(src, generic_read, common));
        }
    }
    for dst in &dests {
        let raw = dst.as_raw() as usize;
        if !source_set.contains(&raw) {
            mid.push(transition(dst, common, first_write_mode_of(raw).legacy_state()));
        }
    }
    if !mid.is_empty() {
        // SAFETY: open list and live buffers.
        unsafe { list.ResourceBarrier(&mid) };
    }

    let mut done = Vec::with_capacity(dests.len() * 2);
    for &dst in &dests {
        let raw = dst.as_raw() as usize;
        let has_uav = *dest_uav.get(&raw).unwrap_or(&false);
        let Some(ops) = writes_by_dest.get(&raw) else {
            continue;
        };
        let mut cur_state = first_write_mode_of(raw).legacy_state();
        let mut tracker = OverlapTracker::default();
        let mut wave: Vec<CopyJob> = Vec::new();
        let mut need_uav_sep = false;

        let flush_legacy_dst = |wave: &mut Vec<CopyJob>,
                                cur_state: &mut D3D12_RESOURCE_STATES,
                                bound_compute: &mut bool,
                                jobs_buf: &mut Option<&mut JobBuf>,
                                keep_old: &mut dyn FnMut(ID3D12Resource),
                                need_uav_sep: bool| {
            if wave.is_empty() {
                return;
            }
            let (Some(sh), Some(dev), Some(jb)) = (shader.as_ref(), device, jobs_buf.as_deref_mut()) else {
                wave.clear();
                return;
            };
            if *cur_state != D3D12_RESOURCE_STATE_UNORDERED_ACCESS {
                let b = [transition(dst, *cur_state, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)];
                // SAFETY: open list and live `dst`.
                unsafe { list.ResourceBarrier(&b) };
                *cur_state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            } else if need_uav_sep {
                let b = [uav_barrier(dst)];
                // SAFETY: open list and live `dst`.
                unsafe { list.ResourceBarrier(&b) };
            }
            if let Some((jobs_gpu, first_job, count)) = jb.push(dev, wave, |old| keep_old(old)) {
                // SAFETY: open list, valid shader/jobs/between/dst.
                unsafe {
                    dispatch_copy_jobs(list, sh, bound_compute, jobs_gpu, between, dst, first_job, count);
                }
            }
            wave.clear();
        };

        for &(dst_off, at, len) in ops {
            if use_compute && has_uav && is_compute_aligned(at, dst_off, len) {
                if tracker.overlaps(dst_off, dst_off + len) {
                    flush_legacy_dst(&mut wave, &mut cur_state, &mut bound_compute, &mut jobs_buf, &mut keep_old, need_uav_sep);
                    need_uav_sep = true;
                    tracker.clear();
                }
                push_chunked_jobs(&mut wave, at, dst_off, len);
                tracker.insert(dst_off, dst_off + len);
            } else {
                if !wave.is_empty() {
                    flush_legacy_dst(&mut wave, &mut cur_state, &mut bound_compute, &mut jobs_buf, &mut keep_old, need_uav_sep);
                    need_uav_sep = false;
                    tracker.clear();
                }
                if cur_state != D3D12_RESOURCE_STATE_COPY_DEST {
                    let b = [transition(dst, cur_state, D3D12_RESOURCE_STATE_COPY_DEST)];
                    // SAFETY: open list and live `dst`.
                    unsafe { list.ResourceBarrier(&b) };
                    cur_state = D3D12_RESOURCE_STATE_COPY_DEST;
                }
                // SAFETY: open list and live buffers.
                unsafe { list.CopyBufferRegion(dst, dst_off, between, at, len) };
            }
        }
        flush_legacy_dst(&mut wave, &mut cur_state, &mut bound_compute, &mut jobs_buf, &mut keep_old, need_uav_sep);
        if cur_state == D3D12_RESOURCE_STATE_UNORDERED_ACCESS {
            done.push(uav_barrier(dst));
        }
        done.push(transition(dst, cur_state, common));
    }
    if !done.is_empty() {
        // SAFETY: open command list and live buffers.
        unsafe { list.ResourceBarrier(&done) };
    }
}

/// Wrapper exposing `record_level()` to sibling live D3D12 Debug Layer tests.
#[cfg(test)]
pub(super) fn record_for_test(
    list: &ID3D12GraphicsCommandList,
    between: &ID3D12Resource,
    reads: &[(&ID3D12Resource, u64, u64, u64)],
    writes: &[(&ID3D12Resource, u64, u64, u64)],
) {
    let dev = host().map(|h| h.device.clone());
    record_level(true, dev.as_ref(), list, None, between, None, reads, writes);
}

/// The list in front of a submission's, its buffer ([`BETWEEN`] bytes), and its
/// reusable GPU compute-copy descriptor upload buffer.
struct Own {
    list: HostList,
    between: ID3D12Resource,
    jobs: Option<JobBuf>,
}

// SAFETY: D3D12 objects are free-threaded, and the list is touched by one
// thread at a time (the submitting one until it is sent, then the one that
// lands it, then nobody until it is taken out of `POOL`).
unsafe impl Send for Own {}

/// Wrapper to share a free-threaded `ID3D12Resource` across worker threads
/// while updating descriptor slots in parallel.
struct SharedResource(ID3D12Resource);

// SAFETY: `ID3D12Resource` is COM free-threaded (`CreateConstantBufferView` + `AddRef`).
unsafe impl Send for SharedResource {}
unsafe impl Sync for SharedResource {}

/// Those that have landed: closed, their steps dropped.
static POOL: Mutex<Vec<Own>> = Mutex::new(Vec::new());

/// Since the last frame-rate line: fills and copies done on the card, their
/// bytes, and submissions that waited instead, refused by a check.
static DONE: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// Per-reason counts of submissions refused by [`Host::card_steps`], for
/// rate-limited diagnostics and the frame-rate summary.
static REFUSED_WHY: Mutex<Option<HashMap<&'static str, u64>>> = Mutex::new(None);

fn note_refused(why: &'static str) {
    let mut map = REFUSED_WHY.lock().unwrap_or_else(|e| e.into_inner());
    let entry = map.get_or_insert_with(HashMap::new).entry(why).or_insert(0);
    *entry += 1;
    let count = *entry;
    drop(map);
    if count <= 8 || count.is_power_of_two() {
        tracing::info!(
            reason = why,
            count,
            stage = stage(),
            "card_steps refused submission; falling back to flight wait"
        );
    }
}

/// What [`Host::card_steps`] took off a submission: the list that does it,
/// for the submission to run first, and the steps the CPU leaves out (their
/// list by address, their place in it).
pub(super) struct Taken {
    own: Own,
    steps: HashSet<(usize, usize)>,
    copied: Vec<(Arc<Shadow>, usize, usize)>,
}

impl Taken {
    pub(super) fn list(&self) -> &HostList {
        &self.own.list
    }

    /// Whether step `i` of `list` is done by this.
    pub(super) fn takes(&self, list: &HostList, i: usize) -> bool {
        self.steps.contains(&(std::ptr::from_ref(list) as usize, i))
    }

    fn copied(&self) -> impl Iterator<Item = Writer> {
        self.copied.iter().map(|(shadow, from, to)| Writer {
            shadow: Some(shadow.clone()),
            from: *from,
            to: *to,
            copy: true,
        })
    }

    /// The submission has run and landed: the list and its buffer are free.
    pub(super) fn landed(self) {
        // The host copies its steps held go; its commands stay until the
        // next use resets it.
        drop(std::mem::take(&mut *self.own.list.deferred()));
        POOL.lock().unwrap_or_else(|e| e.into_inner()).push(self.own);
    }
}

impl Host {
    /// With flights in the air: if what keeps `runs` from being prepared
    /// now is reads of title memory that can all be done on the card
    /// ([`stage`]), the list that does them. `None` if nothing is in the
    /// way or the old wait is due. Under the submission lock.
    pub(super) fn card_steps(&self, runs: &[&HostList]) -> Option<Taken> {
        if stage() == 0 || !relay_exact() {
            return None;
        }
        let (written, copies) = {
            let line = line();
            if line.written.is_empty() {
                return None;
            }
            (line.written.clone(), copies(&line))
        };
        blocked_by(&written, runs, false)?;
        let taken = if copied_read(&copies, runs) {
            Err("a copy done on the card is read through another host copy")
        } else {
            self.take_steps(&written, runs)
        };
        match taken {
            Ok(taken) => taken,
            Err(why) => {
                DONE[3].fetch_add(1, Ordering::Relaxed);
                note_refused(why);
                None
            }
        }
    }

    fn take_steps(&self, written: &[(usize, usize)], runs: &[&HostList]) -> Result<Option<Taken>, &'static str> {
        let (does, steps, mut fill_tos) = does_of(written, runs, true);
        let mut behind_placed = None;
        let chosen = plan(&does, stage(), placed_order(), &mut behind_placed)?;
        if chosen.is_empty() {
            return Ok(None);
        }
        let mut writers = {
            let line = line();
            let mut all = writers(&line.line);
            all.extend(line.landing.iter().map(|w| Writer {
                shadow: w.shadow.clone(),
                ..*w
            }));
            all
        };
        let mut ops: Vec<Op> = Vec::with_capacity(chosen.len());
        // The places read (host copy, address, bytes, level): where in the
        // buffer between their bytes are.
        let mut places: HashMap<(usize, usize, usize, u32), u64> = HashMap::new();
        let mut held: HashMap<(usize, usize), Arc<Shadow>> = HashMap::new();
        let mut room: HashSet<usize> = HashSet::new();
        let mut end = 0u64;
        for &(i, level) in &chosen {
            let Some(((src, _), (dst, _))) = reads_writes(&does[i]) else {
                continue;
            };
            let (Does::Copy { len, .. } | Does::Fill { len, .. }) = does[i] else {
                continue;
            };
            if len == 0 {
                return Err("nothing to read");
            }
            let from = match held.get(&(src, len)) {
                Some(from) => from.clone(),
                None => {
                    let from = holder(&writers, src, src + len)?;
                    held.insert((src, len), from.clone());
                    from
                }
            };
            let place = (Arc::as_ptr(&from) as usize, src, len, level);
            let at = match places.get(&place) {
                Some(&at) => at,
                None => {
                    if end + len as u64 > BETWEEN {
                        return Err("too much at once");
                    }
                    if !mapped(src, src + len, &mut room) {
                        return Err("not all of what it reads is mapped");
                    }
                    let at = end;
                    end = (end + len as u64).next_multiple_of(256);
                    places.insert(place, at);
                    at
                }
            };
            let to = match does[i] {
                Does::Copy { .. } => {
                    let to = holder(&writers, dst, dst + len)?;
                    if elsewhere(&does, Arc::as_ptr(&to) as usize, dst, dst + len) {
                        return Err("another host copy of what it writes is read");
                    }
                    if !mapped(dst, dst + len, &mut room) {
                        return Err("not all of what it writes is mapped");
                    }
                    // What the steps behind it read of it is in that host
                    // copy, or nowhere.
                    writers.push(Writer {
                        shadow: Some(to.clone()),
                        from: dst,
                        to: dst + len,
                        copy: true,
                    });
                    held.clear();
                    To::Copy(to, dst)
                }
                _ => match fill_tos[i].take() {
                    Some(FillTo::Slot(heap, index)) => To::Slot(heap, index),
                    Some(FillTo::Buffer(buffer)) => To::Buffer(buffer),
                    Some(FillTo::BufferOffset(buffer, offset)) => To::BufferOffset(buffer, offset),
                    None => continue,
                },
            };
            ops.push(Op {
                from: (from, src),
                len,
                at,
                level,
                to,
            });
        }
        let Some(mut own) = self.own_list() else {
            return Err("no host list or buffer for it");
        };
        // First every refresh, then the writes: the bytes compared with are
        // the title's. (One step for each host copy, over all that is read of
        // it: a flush brings a copy up to date from the first range asked for
        // to the last, whatever the steps.)
        let mut spans: HashMap<usize, (Arc<Shadow>, usize, usize)> = HashMap::new();
        for op in &ops {
            let to = match &op.to {
                To::Copy(to, dst) => Some((to, *dst)),
                _ => None,
            };
            for (shadow, at) in std::iter::once((&op.from.0, op.from.1)).chain(to) {
                let span = spans
                    .entry(Arc::as_ptr(shadow) as usize)
                    .or_insert_with(|| (shadow.clone(), at, at + op.len));
                (span.1, span.2) = (span.1.min(at), span.2.max(at + op.len));
            }
        }
        for (shadow, from, to) in spans.into_values() {
            shadow.refresh(&own.list, from as u64, (to - from) as u64);
        }
        let mut copied = Vec::new();
        for op in &ops {
            if let To::Copy(to, dst) = &op.to {
                to.gpu_writes(&own.list, *dst as u64, op.len as u64);
                copied.push((to.clone(), *dst, *dst + op.len));
            }
        }
        #[cfg(feature = "diagnostics")]
        own.list.note_op(|| {
            format!(
                "{} copies of title memory done on the card, {end} bytes (VICTORY_CARD_STEPS)",
                ops.len()
            )
        });
        let levels = ops.iter().map(|op| op.level).max().unwrap_or(0);
        for level in 0..=levels {
            let mut reads = Vec::new();
            let mut writes = Vec::new();
            for op in ops.iter().filter(|op| op.level == level) {
                let len = op.len as u64;
                let read = (&op.from.0.buffer, op.from.0.offset(op.from.1 as u64), op.at, len);
                // (Each place once.)
                if places.remove(&(Arc::as_ptr(&op.from.0) as usize, op.from.1, op.len, level)).is_some() {
                    reads.push(read);
                }
                match &op.to {
                    To::Copy(to, dst) => writes.push((&to.buffer, to.offset(*dst as u64), op.at, len)),
                    To::Buffer(buffer) => writes.push((buffer, 0, op.at, len)),
                    To::BufferOffset(buffer, offset) => writes.push((buffer, *offset, op.at, len)),
                    To::Slot(..) => {}
                }
            }
            record_level(
                level == 0,
                Some(&self.device),
                &own.list.list,
                Some(&own.list),
                &own.between,
                own.jobs.as_mut(),
                &reads,
                &writes,
            );
        }
        own.list.close();
        // SAFETY: a live buffer.
        let gpu = unsafe { own.between.GetGPUVirtualAddress() };
        let slots: Vec<(usize, u32, u64, u32)> = ops
            .iter()
            .filter_map(|op| match op.to {
                To::Slot(heap, index) => Some((heap, index, gpu + op.at, op.len as u32)),
                _ => None,
            })
            .collect();
        if slots.len() < 256 {
            for &(heap, index, addr, len) in &slots {
                crate::descriptor::constant_slot(heap, index, &own.between, addr, len);
            }
        } else {
            let shared = Arc::new(SharedResource(own.between.clone()));
            let slots = Arc::new(slots);
            let parts = 8usize;
            let jobs: Vec<Box<dyn FnOnce() + Send>> = (0..parts)
                .map(|part| {
                    let shared = shared.clone();
                    let slots = slots.clone();
                    Box::new(move || {
                        for &(heap, index, addr, len) in slots.iter() {
                            if index as usize / 4 % parts == part {
                                crate::descriptor::constant_slot(heap, index, &shared.0, addr, len);
                            }
                        }
                    }) as Box<dyn FnOnce() + Send>
                })
                .collect();
            crate::shadow::wide_run(jobs);
        }
        let copies = ops.iter().filter(|op| matches!(op.to, To::Copy(..))).count();
        DONE[0].fetch_add((ops.len() - copies) as u64, Ordering::Relaxed);
        DONE[1].fetch_add(copies as u64, Ordering::Relaxed);
        DONE[2].fetch_add(ops.iter().map(|op| op.len as u64).sum(), Ordering::Relaxed);
        Ok(Some(Taken {
            own,
            steps: chosen.iter().map(|&(i, _)| steps[i]).collect(),
            copied,
        }))
    }

    /// A list of ours open for recording, and its buffer.
    fn own_list(&self) -> Option<Own> {
        let pooled = POOL.lock().unwrap_or_else(|e| e.into_inner()).pop();
        if let Some(mut own) = pooled {
            own.list.reset();
            if let Some(jobs) = own.jobs.as_mut() {
                jobs.reset();
            }
            return Some(own);
        }
        let between = self.uav_buffer(BETWEEN)?;
        name(&between, || "copies of title memory on the card".into());
        let list = self.new_list().ok()?;
        let jobs = compute_copy_enabled()
            .then(|| JobBuf::new(&self.device, INITIAL_JOB_CAP))
            .flatten();
        Some(Own { list, between, jobs })
    }
}

/// For the frame-rate line.
pub(super) fn log(frames: f64) {
    if stage() == 0 {
        return;
    }
    let d: Vec<f64> = DONE.iter().map(|d| d.swap(0, Ordering::Relaxed) as f64 / frames).collect();
    if d[0] + d[1] + d[3] > 0.0 {
        let reasons: String = if d[3] > 0.0 {
            let map = REFUSED_WHY.lock().unwrap_or_else(|e| e.into_inner());
            let mut items: Vec<(&&'static str, &u64)> = map
                .as_ref()
                .map(|m| m.iter().collect())
                .unwrap_or_default();
            items.sort_by(|a, b| b.1.cmp(a.1));
            let top: Vec<String> = items.into_iter().take(4).map(|(k, v)| format!("{k} ({v})")).collect();
            if top.is_empty() {
                String::new()
            } else {
                format!(" [reasons: {}]", top.join(", "))
            }
        } else {
            String::new()
        };
        tracing::info!(
            "steps done on the card instead of waiting for flights, per frame (VICTORY_CARD_STEPS={}): {:.2} fills, {:.2} buffer \
             copies, {:.1} KB; {:.2} submissions refused and waited{reasons}",
            stage(),
            d[0],
            d[1],
            d[2] / 1024.0,
            d[3],
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn copy(dst: usize, src: usize, len: usize, asked: bool) -> Does {
        Does::Copy { dst, src, len, asked }
    }

    fn fill(src: usize, len: usize, asked: bool) -> Does {
        Does::Fill { src, len, asked, to: true }
    }

    fn stuck(src: usize, len: usize) -> Does {
        Does::Fill { src, len, asked: false, to: false }
    }

    fn plan_simple(does: &[Does], stage: u32) -> Result<Vec<(usize, u32)>, &'static str> {
        let mut behind = None;
        plan(does, stage, true, &mut behind)
    }

    fn planned(does: &[Does]) -> Vec<(usize, u32)> {
        plan_simple(does, 2).unwrap()
    }

    #[test]
    fn card_steps_defaults_to_stage_2_and_honors_overrides() {
        assert_eq!(resolve_stage(None), 2);
        assert_eq!(resolve_stage(Some("")), 2);
        assert_eq!(resolve_stage(Some("0")), 0);
        assert_eq!(resolve_stage(Some("1")), 1);
        assert_eq!(resolve_stage(Some("2")), 2);
        assert!(resolve_placed_order(None));
        assert!(resolve_placed_order(Some("1")));
        assert!(!resolve_placed_order(Some("0")));
    }

    #[test]
    fn steps_apart_from_one_another_are_taken() {
        let does = [
            copy(1000, 5000, 100, false),
            fill(2000, 64, true),
            Does::Refresh { shadow: 1, from: 0, to: 9000 },
            copy(3000, 6000, 100, true),
            fill(1000, 64, false),
            Does::Read { from: 3100, to: 4000, asked: false },
            stuck(3100, 5),
        ];
        assert_eq!(planned(&does), [(1, 0), (3, 0)]);
        assert_eq!(plan_simple(&[fill(0, 64, true), fill(32, 64, true)], 1), Ok(vec![(0, 0), (1, 0)]));
        assert_eq!(plan_simple(&[fill(0, 64, false)], 1), Ok(vec![]));
        let does = [
            copy(1000, 5000, 100, true),
            copy(900, 9000, 100, false),
            copy(1100, 9000, 5, false),
            copy(4900, 9000, 100, false),
        ];
        assert_eq!(planned(&does), [(0, 0)]);
    }

    #[test]
    fn placed_copy_allows_earlier_fills_and_flags_later_reads_for_split() {
        let placed = Does::Placed { dst: 1000, src: 5000, len: 100 };
        // A fill ahead of Placed reading Placed's dst is safe under by_order=true,
        // because own.list runs ahead of Placed in GPU order!
        let mut behind = None;
        assert_eq!(
            plan(&[fill(1000, 64, true), placed], 1, true, &mut behind),
            Ok(vec![(0, 0)])
        );
        assert_eq!(behind, None);
        // Under by_order=false, even an earlier fill is refused.
        assert_eq!(
            plan(&[fill(1000, 64, true), placed], 1, false, &mut behind),
            Err(BEHIND_PLACED)
        );
        // A fill AFTER Placed reading Placed's dst must split behind Placed!
        let mut behind = None;
        assert_eq!(
            plan(&[placed, fill(1000, 64, true)], 1, true, &mut behind),
            Err(BEHIND_PLACED)
        );
        assert_eq!(behind, Some((0, 1)));
    }

    #[test]
    fn steps_that_depend_on_one_another_are_taken_in_order() {
        let taken = copy(1000, 5000, 100, true);
        assert_eq!(planned(&[taken, fill(1099, 64, false)]), [(0, 0), (1, 1)]);
        assert_eq!(planned(&[fill(1099, 64, false), taken]), [(0, 0), (1, 1)]);
        assert_eq!(planned(&[taken, copy(7000, 1050, 10, false)]), [(0, 0), (1, 1)]);
        assert_eq!(planned(&[copy(5099, 9000, 1, false), taken]), [(0, 0), (1, 1)]);
        assert_eq!(planned(&[taken, copy(5099, 9000, 1, false)]), [(0, 0), (1, 1)]);
        assert_eq!(planned(&[taken, copy(999, 9000, 2, false)]), [(0, 0), (1, 1)]);
        assert_eq!(planned(&[taken, taken]), [(0, 0), (1, 1)]);
        let does = [
            copy(8000, 7000, 50, false),
            taken,
            copy(2000, 1010, 20, false),
            fill(90_000, 10, false),
            fill(2000, 20, false),
            fill(2010, 20, true),
            copy(1000, 2005, 5, false),
            fill(1000, 100, false),
        ];
        assert_eq!(planned(&does), [(1, 0), (2, 1), (4, 2), (5, 2), (6, 2), (7, 3)]);
        assert_eq!(planned(&[taken, fill(5000, 100, false)]), [(0, 0)]);
        assert_eq!(planned(&[copy(7000, 5000, 10, false), taken]), [(1, 0)]);
    }

    #[test]
    fn levels_do_what_the_order_does() {
        let mut seed = 12345u32;
        let mut random = move |n: u32| {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
            (seed >> 16) % n
        };
        let (mut done, mut refused) = (0, 0);
        for _ in 0..4000 {
            let does: Vec<Does> = (0..2 + random(9))
                .map(|_| {
                    let (at, other, len) = (random(60) as usize, random(60) as usize, 1 + random(8) as usize);
                    let asked = random(4) == 0;
                    match random(8) {
                        0..4 => copy(at, other, len, asked),
                        4..6 => fill(at, len, asked),
                        6 => stuck(at, len),
                        _ => Does::Read { from: at, to: at + len, asked: false },
                    }
                })
                .collect();
            let Ok(chosen) = plan_simple(&does, 2) else {
                refused += 1;
                continue;
            };
            done += 1;
            let start: Vec<u32> = (0..80).collect();
            let mut memory = start.clone();
            let mut wanted = Vec::new();
            for d in &does {
                let (from, to) = match *d {
                    Does::Read { from, to, .. } => (from, to),
                    _ => reads_writes(d).unwrap().0,
                };
                wanted.push(memory[from..to].to_vec());
                if let Does::Copy { dst, src, len, .. } = *d {
                    memory.copy_within(src..src + len, dst);
                }
            }
            let mut cpu = start.clone();
            let taken = |i: usize| chosen.iter().any(|c| c.0 == i);
            for (i, d) in does.iter().enumerate().filter(|&(i, _)| !taken(i)) {
                let (from, to) = match *d {
                    Does::Read { from, to, .. } => (from, to),
                    _ => reads_writes(d).unwrap().0,
                };
                assert_eq!(cpu[from..to], wanted[i][..], "step {i} of {does:?}");
                if let Does::Copy { dst, src, len, .. } = *d {
                    cpu.copy_within(src..src + len, dst);
                }
            }
            let mut card = cpu.clone();
            for level in 0..=chosen.iter().map(|c| c.1).max().unwrap_or(0) {
                let of: Vec<usize> = chosen.iter().filter(|c| c.1 == level).map(|c| c.0).collect();
                let before = card.clone();
                for i in of {
                    let ((from, to), (dst, _)) = reads_writes(&does[i]).unwrap();
                    assert_eq!(before[from..to], wanted[i][..], "step {i} of {does:?}: {chosen:?}");
                    if matches!(does[i], Does::Copy { .. }) {
                        card[dst..dst + to - from].copy_from_slice(&before[from..to]);
                    }
                }
            }
            assert_eq!(card, memory, "{does:?}: {chosen:?}");
        }
        assert!(done > 500 && refused > 100, "{done} {refused}");
    }

    #[test]
    fn steps_in_the_way_are_refused() {
        let taken = copy(1000, 5000, 100, true);
        assert!(plan_simple(&[taken, Does::Run], 2).is_err());
        assert!(plan_simple(&[taken, Does::Read { from: 1, to: 2, asked: true }], 2).is_err());
        assert!(plan_simple(&[Does::Fill { src: 1, len: 1, asked: true, to: false }], 2).is_err());
        for read in [Does::Read { from: 900, to: 1001, asked: false }, stuck(1050, 3)] {
            assert!(plan_simple(&[taken, read], 2).is_err(), "{read:?} behind");
            assert_eq!(planned(&[read, taken]), [(1, 0)], "{read:?} ahead");
        }
        assert!(plan_simple(&[taken, copy(3000, 1000, 10, false), stuck(3005, 1)], 2).is_err());
        assert!(plan_simple(&[copy(1000, 1050, 100, true)], 2).is_err());
        assert!(plan_simple(&[taken], 1).is_err());
        assert!(plan_simple(&[fill(1000, 4, true), copy(1000, 5000, 100, false)], 1).is_err());
    }

    #[test]
    fn a_refresh_of_another_host_copy_is_seen() {
        let does = [Does::Refresh { shadow: 7, from: 100, to: 200 }];
        assert!(!elsewhere(&does, 7, 100, 200));
        assert!(elsewhere(&does, 8, 199, 300));
        assert!(!elsewhere(&does, 8, 200, 300));
    }

    #[test]
    fn what_no_host_copy_holds_is_refused() {
        let writers = [Writer { shadow: None, from: 100, to: 108, copy: false }];
        assert!(holder(&writers, 104, 120).is_err());
    }

    #[test]
    fn the_card_copies_what_is_recorded() {
        let Some(host) = host() else {
            return;
        };
        let (a, a_data) = host.shared_buffer(1 << 16).unwrap();
        let (b, b_data) = host.shared_buffer(1 << 16).unwrap();
        // SAFETY: mappings of 64 KiB.
        let (a_bytes, b_bytes) =
            unsafe { (std::slice::from_raw_parts_mut(a_data, 1 << 16), std::slice::from_raw_parts_mut(b_data, 1 << 16)) };
        for (i, (x, y)) in a_bytes.iter_mut().zip(b_bytes.iter_mut()).enumerate() {
            (*x, *y) = (i as u8 ^ (i >> 8) as u8, 0xee);
        }
        let was = a_bytes.to_vec();
        let mut own = host.own_list().unwrap();
        record_level(
            true,
            Some(&host.device),
            &own.list.list,
            Some(&own.list),
            &own.between,
            own.jobs.as_mut(),
            &[(&a, 33, 0, 100), (&a, 2000, 256, 100), (&a, 5001, 512, 7), (&a, 9, 1024, 4)],
            &[(&a, 2000, 0, 100), (&a, 33, 256, 100), (&b, 77, 512, 7)],
        );
        record_level(
            false,
            Some(&host.device),
            &own.list.list,
            Some(&own.list),
            &own.between,
            own.jobs.as_mut(),
            &[(&b, 77, 2048, 7)],
            &[(&b, 1077, 2048, 7)],
        );
        own.list.close();
        let next = host.new_list().unwrap();
        let (out, out_data) = host.shared_buffer(1 << 16).unwrap();
        // SAFETY: an open list; live buffers.
        unsafe {
            next.list.CopyBufferRegion(&out, 0, &own.between, 1024, 4);
            next.list.CopyBufferRegion(&b, 4096, &a, 33, 100);
        }
        next.close();
        host.execute(&[&own.list, &next]);
        assert_eq!(a_bytes[2000..2100], was[33..133]);
        assert_eq!(a_bytes[33..133], was[2000..2100]);
        assert_eq!(b_bytes[77..84], was[5001..5008]);
        assert_eq!(b_bytes[1077..1084], was[5001..5008]);
        assert_eq!(b_bytes[4096..4196], was[2000..2100]);
        // SAFETY: a mapping of 64 KiB.
        assert_eq!(unsafe { std::slice::from_raw_parts(out_data, 4) }, &was[9..13]);
        assert!(b_bytes[..77].iter().all(|&x| x == 0xee));
        a_bytes[2000..2100].copy_from_slice(&was[2000..2100]);
        a_bytes[33..133].copy_from_slice(&was[33..133]);
        assert_eq!(a_bytes, &was[..]);
        POOL.lock().unwrap().push(own);
        let own = host.own_list().unwrap();
        own.list.close();
    }

    #[test]
    fn compute_copy_toggle_and_overlap_tracker_work() {
        assert!(resolve_compute_copy(None));
        assert!(resolve_compute_copy(Some("1")));
        assert!(!resolve_compute_copy(Some("0")));
        assert!(!resolve_compute_copy(Some(" 0 ")));

        assert!(is_compute_aligned(0, 256, 4));
        assert!(is_compute_aligned(12, 4, 100));
        assert!(!is_compute_aligned(1, 0, 4));
        assert!(!is_compute_aligned(0, 2, 4));
        assert!(!is_compute_aligned(0, 0, 7));
        assert!(!is_compute_aligned(0, 0, 0));

        let mut jobs = Vec::new();
        push_chunked_jobs(&mut jobs, 16, 256, 9000);
        assert_eq!(jobs.len(), 3);
        assert_eq!(jobs[0], CopyJob { src_off: 16, dst_off: 256, dwords: 1024, _pad: 0 });
        assert_eq!(jobs[1], CopyJob { src_off: 4112, dst_off: 4352, dwords: 1024, _pad: 0 });
        assert_eq!(jobs[2], CopyJob { src_off: 8208, dst_off: 8448, dwords: 202, _pad: 0 });

        let mut t = OverlapTracker::default();
        assert!(!t.overlaps(0, 16));
        t.insert(0, 16);
        // Strictly adjacent [16, 32) does NOT overlap [0, 16).
        assert!(!t.overlaps(16, 32));
        t.insert(16, 32);
        // Non-monotonic disjoint hole test.
        t.insert(100, 200);
        assert!(!t.overlaps(32, 100));
        assert!(t.overlaps(15, 20));
        assert!(t.overlaps(150, 250));
        assert!(t.overlaps(50, 120));
        assert!(t.overlaps(90, 210));
        t.clear();
        assert!(!t.overlaps(0, 1000));
    }

    #[test]
    fn plan_handles_thousands_of_fills_quickly() {
        let mut does = Vec::with_capacity(2048);
        for i in 0..2048 {
            does.push(fill(0x1000 + i * 256, 256, true));
        }
        let got = planned(&does);
        assert_eq!(got.len(), 2048);
        assert!(got.iter().all(|&(_, level)| level == 0));
    }

    /// Live RTX 3090 D3D12 Debug Layer validation of `record_level`'s batched GPU
    /// compute copy shader (`CARD_STEPS_COPY`), targeted Enhanced Barriers (zero
    /// `UAV(NULL)`), O(N) dedup, overlapping/adjacent destinations, unaligned
    /// `CopyBufferRegion` fallback, non-UAV destination fallback, same-buffer
    /// swap, and multi-level chaining across 1,100+ mixed-size fills.
    #[test]
    #[ignore = "enables D3D12 Debug Layer on live RTX 3090 (run standalone with -- --ignored)"]
    fn live_rtx3090_card_steps_batched_copy_debug_layer() {
        use windows::Win32::Graphics::Direct3D12::{
            D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_QUEUE_DESC, D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
            D3D12_FENCE_FLAG_NONE, D3D12_HEAP_FLAG_NONE, D3D12_HEAP_PROPERTIES, D3D12_HEAP_TYPE_CUSTOM,
            D3D12_HEAP_TYPE_DEFAULT, D3D12_MEMORY_POOL_L0, D3D12_MESSAGE, D3D12_MESSAGE_SEVERITY_CORRUPTION,
            D3D12_MESSAGE_SEVERITY_ERROR, D3D12_RESOURCE_DESC, D3D12_RESOURCE_DIMENSION_BUFFER,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_COMMON, D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12CreateDevice,
            D3D12GetDebugInterface, ID3D12CommandAllocator, ID3D12CommandList, ID3D12CommandQueue,
            ID3D12Debug, ID3D12Device, ID3D12Fence, ID3D12GraphicsCommandList, ID3D12InfoQueue,
            ID3D12Resource,
        };

        // SAFETY: standalone D3D12 Debug-Layer device + queue + buffers for live validation.
        unsafe {
            let mut debug: Option<ID3D12Debug> = None;
            D3D12GetDebugInterface(&mut debug).expect("D3D12 Debug Layer must be installed");
            debug.unwrap().EnableDebugLayer();

            let mut device: Option<ID3D12Device> = None;
            D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut device).expect("D3D12CreateDevice");
            let device = device.unwrap();
            let info_queue: ID3D12InfoQueue = device.cast().expect("ID3D12InfoQueue");
            info_queue.ClearStoredMessages();

            let queue: ID3D12CommandQueue = device
                .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                    ..Default::default()
                })
                .expect("CreateCommandQueue");
            let alloc: ID3D12CommandAllocator = device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                .expect("CreateCommandAllocator");
            let list: ID3D12GraphicsCommandList = device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)
                .expect("CreateCommandList");
            let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("CreateFence");
            let event = CreateEventW(None, false, false, None).expect("CreateEventW");

            let make_shared = |size: u64, allow_uav: bool| -> (ID3D12Resource, *mut u8) {
                let heap = D3D12_HEAP_PROPERTIES {
                    Type: D3D12_HEAP_TYPE_CUSTOM,
                    CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
                    MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
                    ..Default::default()
                };
                let desc = D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: size,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    Flags: if allow_uav {
                        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
                    } else {
                        D3D12_RESOURCE_FLAG_NONE
                    },
                    ..Default::default()
                };
                let mut res: Option<ID3D12Resource> = None;
                device
                    .CreateCommittedResource(
                        &heap,
                        D3D12_HEAP_FLAG_NONE,
                        &desc,
                        D3D12_RESOURCE_STATE_COMMON,
                        None,
                        &mut res,
                    )
                    .expect("CreateCommittedResource(shared)");
                let res = res.unwrap();
                let mut ptr = std::ptr::null_mut();
                res.Map(0, None, Some(&mut ptr)).expect("Map");
                (res, ptr.cast())
            };

            let make_uav = |size: u64| -> ID3D12Resource {
                let heap = D3D12_HEAP_PROPERTIES {
                    Type: D3D12_HEAP_TYPE_DEFAULT,
                    ..Default::default()
                };
                let desc = D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: size,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                    ..Default::default()
                };
                let mut res: Option<ID3D12Resource> = None;
                device
                    .CreateCommittedResource(
                        &heap,
                        D3D12_HEAP_FLAG_NONE,
                        &desc,
                        D3D12_RESOURCE_STATE_COMMON,
                        None,
                        &mut res,
                    )
                    .expect("CreateCommittedResource(between)");
                res.unwrap()
            };

            // 3 distinct sources (1 MiB each), `between` (4 MiB), 3 destinations.
            const SRC_SZ: usize = 1 << 20;
            const DST_BIG_SZ: usize = 2 << 20;
            const DST_OVL_SZ: usize = 512 << 10;
            const DST_NOUAV_SZ: usize = 64 << 10;

            let (src0, p_src0) = make_shared(SRC_SZ as u64, true);
            let (src1, p_src1) = make_shared(SRC_SZ as u64, true);
            let (src2, p_src2) = make_shared(SRC_SZ as u64, true);
            let between = make_uav(BETWEEN);
            let (dst_big, p_big) = make_shared(DST_BIG_SZ as u64, true);
            let (dst_ovl, p_ovl) = make_shared(DST_OVL_SZ as u64, true);
            let (dst_nouav, p_nouav) = make_shared(DST_NOUAV_SZ as u64, false);

            let s0 = std::slice::from_raw_parts_mut(p_src0, SRC_SZ);
            let s1 = std::slice::from_raw_parts_mut(p_src1, SRC_SZ);
            let s2 = std::slice::from_raw_parts_mut(p_src2, SRC_SZ);
            for i in 0..SRC_SZ {
                s0[i] = ((i.wrapping_mul(31) ^ (i >> 8) ^ 0x11) & 0xff) as u8;
                s1[i] = ((i.wrapping_mul(17) ^ (i >> 7) ^ 0x55) & 0xff) as u8;
                s2[i] = ((i.wrapping_mul(13) ^ (i >> 9) ^ 0xA3) & 0xff) as u8;
            }
            let d_big = std::slice::from_raw_parts_mut(p_big, DST_BIG_SZ);
            let d_ovl = std::slice::from_raw_parts_mut(p_ovl, DST_OVL_SZ);
            let d_nouav = std::slice::from_raw_parts_mut(p_nouav, DST_NOUAV_SZ);
            d_big.fill(0xCC);
            d_ovl.fill(0xDD);
            d_nouav.fill(0xEE);

            let mut ref_s0 = s0.to_vec();
            let ref_s1 = s1.to_vec();
            let ref_s2 = s2.to_vec();
            let mut ref_big = d_big.to_vec();
            let mut ref_ovl = d_ovl.to_vec();
            let mut ref_nouav = d_nouav.to_vec();
            let mut ref_between = vec![0u8; BETWEEN as usize];

            let mut reads_l0: Vec<(&ID3D12Resource, u64, u64, u64)> = Vec::new();
            let mut writes_l0: Vec<(&ID3D12Resource, u64, u64, u64)> = Vec::new();
            let mut cursor: u64 = 0;

            fn push_op<'a>(
                reads: &mut Vec<(&'a ID3D12Resource, u64, u64, u64)>,
                writes: &mut Vec<(&'a ID3D12Resource, u64, u64, u64)>,
                ref_between: &mut [u8],
                cursor: &mut u64,
                src_res: &'a ID3D12Resource,
                src_ref: &[u8],
                src_off: u64,
                dst_res: &'a ID3D12Resource,
                dst_ref: &mut [u8],
                dst_off: u64,
                len: u64,
            ) {
                let at = *cursor;
                *cursor = (at + len).next_multiple_of(256);
                assert!(*cursor <= BETWEEN);
                reads.push((src_res, src_off, at, len));
                writes.push((dst_res, dst_off, at, len));
                ref_between[at as usize..(at + len) as usize]
                    .copy_from_slice(&src_ref[src_off as usize..(src_off + len) as usize]);
                dst_ref[dst_off as usize..(dst_off + len) as usize]
                    .copy_from_slice(&ref_between[at as usize..(at + len) as usize]);
            }

            // A. 1,100 mixed-size 4-byte-aligned fills (4B..8192B, including 4B-aligned
            //    non-16B-aligned offsets, >4096B chunked fills, and strictly adjacent pairs).
            let sizes: [u64; 10] = [4, 12, 16, 28, 64, 256, 384, 512, 1024, 8192];
            let mut dst_pos: u64 = 0;
            for idx in 0..1100usize {
                let len = if idx < 1050 { sizes[idx % 9] } else { sizes[idx % 10] };
                let (s_res, s_ref): (&ID3D12Resource, &[u8]) = match idx % 3 {
                    0 => (&src0, &ref_s0),
                    1 => (&src1, &ref_s1),
                    _ => (&src2, &ref_s2),
                };
                // Mix 16-byte aligned and 4-byte (non-16-byte) aligned offsets.
                let src_off = ((idx as u64 * 197) & !3u64) % (SRC_SZ as u64 - len);
                let dst_off = dst_pos;
                dst_pos += len; // Strictly adjacent with zero gap!
                assert!(dst_pos <= DST_BIG_SZ as u64);
                push_op(
                    &mut reads_l0,
                    &mut writes_l0,
                    &mut ref_between,
                    &mut cursor,
                    s_res,
                    s_ref,
                    src_off,
                    &dst_big,
                    &mut ref_big,
                    dst_off,
                    len,
                );
            }

            // B. Overlapping aligned writes + interleaved unaligned & adjacent unaligned writes.
            let ovl_cases: &[(usize, u64, u64, u64)] = &[
                (0, 0x100, 0, 1024),        // aligned base write [0..1024)
                (1, 0x200, 256, 512),       // overlaps middle [256..768) -> wave 2 must win
                (2, 0x304, 512, 768),       // overlaps tail [512..1280) -> wave 3 must win
                (0, 0x400, 64, 128),        // overlaps start [64..192) -> wave 4 must win
                (1, 33, 13, 7),             // unaligned src/dst/len inside [0..1024)
                (2, 5001, 20, 11),          // strictly adjacent unaligned [20..31)
                (0, 0x800, 2048, 256),      // aligned write after unaligned
                (1, 0x901, 2050, 15),       // unaligned overwrite inside [2048..2304)
                (2, 0xA00, 2064, 64),       // aligned overwrite after unaligned
                (0, 0xB00, 4096, 6000),     // >4096B chunked base
                (1, 0xC00, 5000, 2000),     // overlapping chunked write
            ];
            for &(which, src_off, dst_off, len) in ovl_cases {
                let (s_res, s_ref): (&ID3D12Resource, &[u8]) = match which {
                    0 => (&src0, &ref_s0),
                    1 => (&src1, &ref_s1),
                    _ => (&src2, &ref_s2),
                };
                push_op(
                    &mut reads_l0,
                    &mut writes_l0,
                    &mut ref_between,
                    &mut cursor,
                    s_res,
                    s_ref,
                    src_off,
                    &dst_ovl,
                    &mut ref_ovl,
                    dst_off,
                    len,
                );
            }

            // C. Non-UAV destination buffer (must automatically fall back to CopyBufferRegion).
            push_op(
                &mut reads_l0,
                &mut writes_l0,
                &mut ref_between,
                &mut cursor,
                &src2,
                &ref_s2,
                0x4000,
                &dst_nouav,
                &mut ref_nouav,
                0,
                4096,
            );
            push_op(
                &mut reads_l0,
                &mut writes_l0,
                &mut ref_between,
                &mut cursor,
                &src1,
                &ref_s1,
                0x5003,
                &dst_nouav,
                &mut ref_nouav,
                4096,
                19,
            );

            // D. Same-buffer swap on `src0` (reads `[0xF0000..+512)` and `[0xF1000..+512)`, swaps them).
            let at_sw0 = cursor;
            cursor = (cursor + 512).next_multiple_of(256);
            let at_sw1 = cursor;
            cursor = (cursor + 512).next_multiple_of(256);
            reads_l0.push((&src0, 0xF0000, at_sw0, 512));
            reads_l0.push((&src0, 0xF1000, at_sw1, 512));
            writes_l0.push((&src0, 0xF1000, at_sw0, 512));
            writes_l0.push((&src0, 0xF0000, at_sw1, 512));
            let sw0_copy = ref_s0[0xF0000..0xF0200].to_vec();
            let sw1_copy = ref_s0[0xF1000..0xF1200].to_vec();
            ref_s0[0xF1000..0xF1200].copy_from_slice(&sw0_copy);
            ref_s0[0xF0000..0xF0200].copy_from_slice(&sw1_copy);

            // Level 1 (`first = false`): read back from `dst_ovl[0..1024)` (written in Level 0)
            // into `dst_ovl[16384..17408)` to validate multi-level barrier ordering!
            let at_l1 = cursor;
            let reads_l1 = [(&dst_ovl, 0u64, at_l1, 1024u64)];
            let writes_l1 = [(&dst_ovl, 16384u64, at_l1, 1024u64)];
            let l1_snap = ref_ovl[0..1024].to_vec();
            ref_ovl[16384..17408].copy_from_slice(&l1_snap);

            let mut jb = JobBuf::new(&device, INITIAL_JOB_CAP).expect("JobBuf::new");
            let t0 = std::time::Instant::now();
            record_level(
                true,
                Some(&device),
                &list,
                None,
                &between,
                Some(&mut jb),
                &reads_l0,
                &writes_l0,
            );
            record_level(
                false,
                Some(&device),
                &list,
                None,
                &between,
                Some(&mut jb),
                &reads_l1,
                &writes_l1,
            );
            let record_us = t0.elapsed().as_micros();
            let close_res = list.Close();
            if let Err(ref e) = close_res {
                let n = info_queue.GetNumStoredMessagesAllowedByRetrievalFilter();
                for i in 0..n {
                    let mut len = 0usize;
                    let _ = info_queue.GetMessage(i, None, &mut len);
                    if len == 0 { continue; }
                    let mut buf = vec![0u8; len];
                    let ptr = buf.as_mut_ptr().cast::<D3D12_MESSAGE>();
                    if info_queue.GetMessage(i, Some(ptr), &mut len).is_ok() {
                        let msg = &*ptr;
                        let text = std::ffi::CStr::from_ptr(msg.pDescription.cast()).to_string_lossy();
                        eprintln!("D3D12 DEBUG [{:?}/{:?}]: {text}", msg.Severity, msg.ID);
                    }
                }
                panic!("list.Close failed: {e}");
            }

            let exec: [Option<ID3D12CommandList>; 1] = [Some(list.cast().unwrap())];
            queue.ExecuteCommandLists(&exec);
            queue.Signal(&fence, 1).expect("Signal");
            if fence.GetCompletedValue() < 1 {
                fence.SetEventOnCompletion(1, event).expect("SetEventOnCompletion");
                WaitForSingleObject(event, INFINITE);
            }
            let _ = windows::Win32::Foundation::CloseHandle(event);

            assert_eq!(d_big, &ref_big[..], "1,100 mixed-size adjacent fills must match bit-for-bit");
            assert_eq!(d_ovl, &ref_ovl[..], "overlapping + unaligned + level-1 fills must match bit-for-bit");
            assert_eq!(d_nouav, &ref_nouav[..], "non-UAV destination fallback must match bit-for-bit");
            assert_eq!(s0, &ref_s0[..], "same-buffer swap on src0 must match bit-for-bit");

            let mut errors = Vec::new();
            let msg_count = info_queue.GetNumStoredMessagesAllowedByRetrievalFilter();
            for i in 0..msg_count {
                let mut len = 0usize;
                let _ = info_queue.GetMessage(i, None, &mut len);
                if len == 0 {
                    continue;
                }
                let mut buf = vec![0u8; len];
                let ptr = buf.as_mut_ptr().cast::<D3D12_MESSAGE>();
                if info_queue.GetMessage(i, Some(ptr), &mut len).is_ok() {
                    let msg = &*ptr;
                    if msg.Severity == D3D12_MESSAGE_SEVERITY_ERROR
                        || msg.Severity == D3D12_MESSAGE_SEVERITY_CORRUPTION
                    {
                        let text = std::ffi::CStr::from_ptr(msg.pDescription.cast())
                            .to_string_lossy()
                            .into_owned();
                        errors.push(format!("[id={:?}] {text}", msg.ID));
                    }
                }
            }
            eprintln!(
                "live_rtx3090_card_steps_batched_copy_debug_layer: PASS — {} level-0 fills + level-1 chain, \
                 {} compute jobs, recorded in {} us, enhanced_barriers={}, debug_errors={}",
                reads_l0.len(),
                jb.used,
                record_us,
                enhanced_barriers_on(&device),
                errors.len(),
            );
            assert!(
                errors.is_empty(),
                "D3D12 Debug Layer reported {} error(s): {errors:#?}",
                errors.len()
            );
        }
    }

    /// Standalone live RTX 3090 hardware probe for Phase 9C (`slot[3]` 2,199 MiB
    /// direct `D3D12_HEAP_TYPE_GPU_UPLOAD` ReBAR): verifies `OPTIONS16`, allocates
    /// a single 2,199 MiB `GPU_UPLOAD` buffer (probing both `ALLOW_UNORDERED_ACCESS`
    /// and fallback `NONE`), maps it, checks GPU & CPU VA bit-widths (`<= 47` bits),
    /// writes & verifies a deterministic pattern, and logs achieved GB/s.
    #[test]
    #[ignore = "allocates a live 2,199 MiB GPU_UPLOAD ReBAR buffer on RTX 3090 (run with -- --ignored --nocapture)"]
    fn probe_rtx3090_gpu_upload_2200mib() {
        use windows::Win32::Graphics::Direct3D12::{
            D3D12_FEATURE_D3D12_OPTIONS16, D3D12_FEATURE_DATA_D3D12_OPTIONS16, D3D12_HEAP_FLAG_NONE,
            D3D12_HEAP_PROPERTIES, D3D12_HEAP_TYPE_GPU_UPLOAD, D3D12_RESOURCE_DESC,
            D3D12_RESOURCE_DIMENSION_BUFFER, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON, D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            D3D12CreateDevice, ID3D12Device, ID3D12Resource,
        };

        const SIZE: u64 = 2199 << 20; // 2,199 MiB (`slot[3]` exact size)

        // SAFETY: standalone D3D12 device and committed `GPU_UPLOAD` probe buffer.
        unsafe {
            let mut device: Option<ID3D12Device> = None;
            D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut device).expect("D3D12CreateDevice");
            let device = device.unwrap();

            let mut o16 = D3D12_FEATURE_DATA_D3D12_OPTIONS16::default();
            device
                .CheckFeatureSupport(
                    D3D12_FEATURE_D3D12_OPTIONS16,
                    (&raw mut o16).cast(),
                    size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS16>() as u32,
                )
                .expect("CheckFeatureSupport(OPTIONS16)");
            let supported = o16.GPUUploadHeapSupported.as_bool();
            eprintln!("OPTIONS16::GPUUploadHeapSupported = {supported}");
            assert!(supported, "RTX 3090 ReBAR must report GPUUploadHeapSupported = true");

            let try_alloc = |allow_uav: bool| -> windows::core::Result<ID3D12Resource> {
                let heap = D3D12_HEAP_PROPERTIES {
                    Type: D3D12_HEAP_TYPE_GPU_UPLOAD,
                    ..Default::default()
                };
                let desc = D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: SIZE,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    Flags: if allow_uav {
                        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
                    } else {
                        D3D12_RESOURCE_FLAG_NONE
                    },
                    ..Default::default()
                };
                let mut res: Option<ID3D12Resource> = None;
                device.CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut res,
                )?;
                Ok(res.unwrap())
            };

            let (buf, has_uav) = match try_alloc(true) {
                Ok(b) => {
                    eprintln!("CreateCommittedResource(GPU_UPLOAD, 2199 MiB, ALLOW_UNORDERED_ACCESS): OK");
                    (b, true)
                }
                Err(e) => {
                    eprintln!("CreateCommittedResource(GPU_UPLOAD, 2199 MiB, ALLOW_UNORDERED_ACCESS): FAILED ({e}); retrying with FLAG_NONE");
                    let b = try_alloc(false).expect("CreateCommittedResource(GPU_UPLOAD, 2199 MiB, NONE)");
                    (b, false)
                }
            };

            let gpu_va = buf.GetGPUVirtualAddress();
            let mut ptr = std::ptr::null_mut();
            buf.Map(0, None, Some(&mut ptr)).expect("Map(GPU_UPLOAD, 2199 MiB)");
            let cpu_va = ptr as u64;
            let gpu_end = gpu_va + SIZE - 1;
            let cpu_end = cpu_va + SIZE - 1;
            let gpu_bits = 64 - gpu_end.leading_zeros();
            let cpu_bits = 64 - cpu_end.leading_zeros();
            eprintln!(
                "Mapped 2,199 MiB GPU_UPLOAD (uav={has_uav}): gpu_va={gpu_va:#x}..={gpu_end:#x} ({gpu_bits} bits, <=40b={}), \
                 cpu_va={cpu_va:#x}..={cpu_end:#x} ({cpu_bits} bits, <=40b={})",
                gpu_bits <= 40,
                cpu_bits <= 40,
            );
            assert!(gpu_bits <= 47, "GPU VA exceeded 47 bits: {gpu_va:#x} ({gpu_bits} bits)");

            // Sequential streaming write across the full 2,199 MiB (64-bit words) to measure ReBAR write GB/s.
            let words = std::slice::from_raw_parts_mut(ptr.cast::<u64>(), (SIZE / 8) as usize);
            let t_write = std::time::Instant::now();
            for (i, w) in words.iter_mut().enumerate() {
                *w = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xA55A_5AA5_1234_5678;
            }
            let write_s = t_write.elapsed().as_secs_f64();
            let write_gbs = (SIZE as f64 / 1e9) / write_s;

            // Verify pattern across start, middle, and end windows (16 MiB each = 48 MiB WC readback
            // so we also measure exact CPU WC read GB/s without stalling for seconds).
            const CHECK_WORDS: usize = (16 << 20) / 8;
            let mid = words.len() / 2;
            let tail = words.len() - CHECK_WORDS;
            let t_read = std::time::Instant::now();
            for base in [0usize, mid, tail] {
                for off in 0..CHECK_WORDS {
                    let i = base + off;
                    let expected = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xA55A_5AA5_1234_5678;
                    assert_eq!(words[i], expected, "mismatch at word index {i}");
                }
            }
            let read_s = t_read.elapsed().as_secs_f64();
            let read_bytes = (3 * CHECK_WORDS * 8) as f64;
            let read_gbs = (read_bytes / 1e9) / read_s;

            buf.Unmap(0, None);
            drop(buf);
            eprintln!(
                "probe_rtx3090_gpu_upload_2200mib: PASS — 2,199 MiB GPU_UPLOAD (uav={has_uav}), \
                 write={write_gbs:.2} GB/s ({:.1} ms), wc_read={read_gbs:.2} GB/s ({:.1} ms for 48 MiB), \
                 gpu_va_bits={gpu_bits}, cpu_va_bits={cpu_bits}",
                write_s * 1e3,
                read_s * 1e3,
            );
        }
    }
}
