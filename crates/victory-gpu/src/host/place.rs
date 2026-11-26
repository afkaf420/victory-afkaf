//! The title's buffer copies, done on the card at their place in the list
//! (`VICTORY_COPIES_IN_PLACE`, on by default; `0`: all at submission, on the
//! CPU, as before).
//!
//! A buffer copy was done in title memory when its list was submitted, before
//! any draw of the submission ran, and the host copies refreshed after it.
//! That is another order than the list's: draws ahead of the copy read what
//! it put; of two copies to the same bytes with a draw behind each, both
//! draws read the second; a source that a dispatch of the same submission
//! writes first was read as it was before it; and a destination the
//! submissions in the air still read changed under them.
//!
//! Now the list copies, in the destination's host copy: out of the source's
//! host copy where shaders write that one, else out of upload memory of the
//! list that takes the source's bytes when the list is submitted (a source
//! shaders never wrote is the title's CPU's, or a copy's put there earlier in
//! the submission). What is written there is a shader's write from then on:
//! the bytes are declared, and what the card made of them is put into title
//! memory before the title is told of the submission.
//!
//! Title memory takes the copy at submission as well, as it did: fills of
//! upload memory, textures and host copies that lie over the same bytes read
//! it there. The reference of the destination's host copy takes the same
//! bytes ([`Shadow::placed_copy`]), so what the landing finds is what the card
//! made of them afterwards (a counter copied to zero and counted up; a source
//! that a shader wrote in between), and nothing is sent up over it. Where
//! flights in the air may write the source or the destination this is left to
//! the landing: what they wrote is not in title memory yet, and what they
//! will put there must not stand over this copy's bytes. If a step of the
//! submission reads the destination on the CPU, it waits for them instead
//! ([`Placed::waits`]).
//!
//! Copies to memory that has no host copy yet are done as before: nothing on
//! the card has read or written it. So are those of more than 8 MiB out of
//! memory shaders do not write (streaming).

use super::*;
use crate::shadow::Shadow;

pub(super) fn resolve_on(env: Option<&str>) -> bool {
    env.is_none_or(|v| v.trim() != "0")
}

pub(super) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let env = std::env::var("VICTORY_COPIES_IN_PLACE").ok();
        resolve_on(env.as_deref())
    })
}

/// The most bytes copied out of memory shaders do not write.
const MOST: usize = 8 << 20;

/// Since the last frame-rate line: copies at their place, their bytes, and
/// those out of a shader-written host copy.
pub(super) static STATS: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// What the CPU does at submission for a copy at its place in a list.
pub(super) struct Placed {
    pub(super) dst: usize,
    pub(super) src: usize,
    pub(super) len: usize,
    /// The destination's host copy.
    pub(super) shadow: std::sync::Arc<Shadow>,
    /// Upload memory the list copies from, to be given the source's bytes.
    from: Option<usize>,
}

impl Placed {
    /// Whether flights that write `written` are waited for.
    pub(super) fn waits(&self, written: &[(usize, usize)]) -> Waits {
        waits_for(self.dst, self.src, self.len, self.from.is_some(), written)
    }

    /// At submission, in the order of the steps: the list's upload memory
    /// takes the source's bytes and, if `free` (no flight in the air may write
    /// source or destination), title memory at the destination does. Gives
    /// the bytes if it took them.
    pub(super) fn prepare(&self, free: bool) -> Option<Vec<u8>> {
        if !free && self.from.is_none() {
            return None;
        }
        let mut bytes = vec![0u8; self.len];
        // (What is not there reads as zeros, as any copy's source.)
        copy_mapped(bytes.as_mut_ptr(), self.src as *const u8, self.len, false);
        if let Some(to) = self.from {
            // SAFETY: the list's own upload memory of that many bytes.
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), to as *mut u8, self.len) };
        }
        if !free {
            return None;
        }
        copy_mapped(self.dst as *mut u8, bytes.as_ptr(), self.len, true);
        crate::watch::wrote(self.dst, self.len);
        // Title memory that is not all there was left alone.
        let mut there = vec![0u8; self.len];
        (read_into(self.dst as *const u8, &mut there) && there == bytes).then_some(bytes)
    }
}

fn waits_for(dst: usize, src: usize, len: usize, cpu_src: bool, written: &[(usize, usize)]) -> Waits {
    let source = touches(written, src, src.saturating_add(len));
    if source && cpu_src {
        return Waits::Yes;
    }
    if source || touches(written, dst, dst.saturating_add(len)) {
        Waits::ForReaders
    } else {
        Waits::No
    }
}

/// Whether a copy waits for flights before its submission is prepared: not;
/// yes, as it takes a source on the CPU that they may write; or only if
/// another step of the submission reads its destination on the CPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Waits {
    No,
    ForReaders,
    Yes,
}

/// Whether a step of `runs` reads one of `ranges` on the CPU.
pub(super) fn read_by(runs: &[&HostList], mut ranges: Vec<(usize, usize)>) -> bool {
    join_ranges(&mut ranges, []);
    runs.iter().any(|list| {
        list.deferred().copies.iter().any(|step| match step {
            &Step::Copy(_, src, len) | &Step::Fill(_, src, len, _) => touches(&ranges, src, src + len),
            Step::Texture(staging) => {
                let (at, len) = staging.memory();
                touches(&ranges, at, at + len)
            }
            Step::Host(_, (from, to)) => touches(&ranges, *from, *to),
            Step::Placed(p) => p.from.is_some() && touches(&ranges, p.src, p.src + p.len),
            _ => false,
        })
    })
}

/// Whether the host device supports D3D12 Enhanced Barriers (`OPTIONS12`).
pub(super) fn enhanced_barriers(host: &Host) -> bool {
    static ENHANCED: OnceLock<bool> = OnceLock::new();
    *ENHANCED.get_or_init(|| {
        let mut o = D3D12_FEATURE_DATA_D3D12_OPTIONS12::default();
        // SAFETY: a feature structure of its size.
        let known = unsafe {
            host.device.CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS12,
                (&raw mut o).cast(),
                size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS12>() as u32,
            )
        };
        known.is_ok() && o.EnhancedBarriersSupported.as_bool()
    })
}

/// A targeted legacy UAV barrier for a single buffer (never global `UAV(NULL)`).
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

/// A targeted D3D12 Enhanced Buffer Barrier for a single buffer (never global
/// `D3D12_BARRIER_TYPE_GLOBAL` / `SYNC_ALL`).
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
        // SAFETY: a borrowed pointer; the barrier never releases it.
        pResource: unsafe { std::mem::transmute_copy(resource) },
        Offset: 0,
        Size: u64::MAX,
    }
}

/// Shader + non-render-target pipeline stages that may read or write `Shadow::buffer`.
const BUFFER_SYNC_ALL: D3D12_BARRIER_SYNC = D3D12_BARRIER_SYNC(
    D3D12_BARRIER_SYNC_ALL_SHADING.0
        | D3D12_BARRIER_SYNC_INDEX_INPUT.0
        | D3D12_BARRIER_SYNC_EXECUTE_INDIRECT.0
        | D3D12_BARRIER_SYNC_COPY.0,
);

/// Stages that may write a `Shadow::buffer` (UAV shaders or copies).
const BUFFER_SYNC_WRITE: D3D12_BARRIER_SYNC =
    D3D12_BARRIER_SYNC(D3D12_BARRIER_SYNC_ALL_SHADING.0 | D3D12_BARRIER_SYNC_COPY.0);

/// Accesses that may read or write a `Shadow::buffer` outside a copy.
const BUFFER_ACCESS_ALL: D3D12_BARRIER_ACCESS = D3D12_BARRIER_ACCESS(
    D3D12_BARRIER_ACCESS_UNORDERED_ACCESS.0
        | D3D12_BARRIER_ACCESS_SHADER_RESOURCE.0
        | D3D12_BARRIER_ACCESS_CONSTANT_BUFFER.0
        | D3D12_BARRIER_ACCESS_VERTEX_BUFFER.0
        | D3D12_BARRIER_ACCESS_INDEX_BUFFER.0
        | D3D12_BARRIER_ACCESS_INDIRECT_ARGUMENT.0
        | D3D12_BARRIER_ACCESS_COPY_SOURCE.0
        | D3D12_BARRIER_ACCESS_COPY_DEST.0,
);

/// Accesses that may write a `Shadow::buffer`.
const BUFFER_ACCESS_WRITE: D3D12_BARRIER_ACCESS =
    D3D12_BARRIER_ACCESS(D3D12_BARRIER_ACCESS_UNORDERED_ACCESS.0 | D3D12_BARRIER_ACCESS_COPY_DEST.0);

/// Emits targeted per-buffer barriers before a placed copy on `list`.
///
/// # Safety
/// An open command list; `from` (if any) and `to` are live buffers.
unsafe fn barrier_before_copy(
    host: &Host,
    list: &ID3D12GraphicsCommandList,
    from: Option<&ID3D12Resource>,
    to: &ID3D12Resource,
) {
    let same = from.is_some_and(|f| f.as_raw() == to.as_raw());
    if enhanced_barriers(host)
        && let Ok(list7) = list.cast::<ID3D12GraphicsCommandList7>()
    {
        let buf_barriers = [
            buffer_barrier(
                to,
                BUFFER_SYNC_ALL,
                D3D12_BARRIER_SYNC_COPY,
                BUFFER_ACCESS_ALL,
                if same {
                    D3D12_BARRIER_ACCESS_COPY_SOURCE
                } else {
                    D3D12_BARRIER_ACCESS_COPY_DEST
                },
            ),
            buffer_barrier(
                from.unwrap_or(to),
                BUFFER_SYNC_WRITE,
                D3D12_BARRIER_SYNC_COPY,
                BUFFER_ACCESS_WRITE,
                D3D12_BARRIER_ACCESS_COPY_SOURCE,
            ),
        ];
        let count = if from.is_some() && !same { 2 } else { 1 };
        let group = [D3D12_BARRIER_GROUP {
            Type: D3D12_BARRIER_TYPE_BUFFER,
            NumBarriers: count,
            Anonymous: D3D12_BARRIER_GROUP_0 {
                pBufferBarriers: buf_barriers.as_ptr(),
            },
        }];
        // SAFETY: open command list; borrowed resource pointers stay alive across `Barrier`.
        unsafe { list7.Barrier(&group) };
        return;
    }
    let legacy = [uav_barrier(to), uav_barrier(from.unwrap_or(to))];
    let count = if from.is_some() && !same { 2 } else { 1 };
    // SAFETY: open command list; targeted per-resource UAV barrier(s).
    unsafe { list.ResourceBarrier(&legacy[..count]) };
}

/// Emits targeted per-buffer barriers between the two copies of a same-buffer
/// placed copy (`from -> between`, then `between -> to` where `from == to`).
///
/// # Safety
/// An open command list; `between` and `target` are live buffers.
unsafe fn barrier_between_same_buffer(
    host: &Host,
    list: &ID3D12GraphicsCommandList,
    between: &ID3D12Resource,
    target: &ID3D12Resource,
) {
    if enhanced_barriers(host)
        && let Ok(list7) = list.cast::<ID3D12GraphicsCommandList7>()
    {
        let barriers = [
            buffer_barrier(
                between,
                D3D12_BARRIER_SYNC_COPY,
                D3D12_BARRIER_SYNC_COPY,
                D3D12_BARRIER_ACCESS_COPY_DEST,
                D3D12_BARRIER_ACCESS_COPY_SOURCE,
            ),
            buffer_barrier(
                target,
                D3D12_BARRIER_SYNC_COPY,
                D3D12_BARRIER_SYNC_COPY,
                D3D12_BARRIER_ACCESS_COPY_SOURCE,
                D3D12_BARRIER_ACCESS_COPY_DEST,
            ),
        ];
        let group = [D3D12_BARRIER_GROUP {
            Type: D3D12_BARRIER_TYPE_BUFFER,
            NumBarriers: barriers.len() as u32,
            Anonymous: D3D12_BARRIER_GROUP_0 {
                pBufferBarriers: barriers.as_ptr(),
            },
        }];
        // SAFETY: open command list; borrowed pointers live across `Barrier`.
        unsafe { list7.Barrier(&group) };
        return;
    }
    let barriers = [
        transition(between, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE),
        transition(target, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COPY_DEST),
    ];
    // SAFETY: open command list; both buffers were just copied into/out of.
    unsafe { list.ResourceBarrier(&barriers) };
}

/// Emits targeted per-buffer barriers after a placed copy on `list` so later
/// shaders or copies see `to` (and may safely overwrite `from`) without a
/// global GPU pipeline flush.
///
/// # Safety
/// An open command list; `from` (if any) and `to` are live buffers.
unsafe fn barrier_after_copy(
    host: &Host,
    list: &ID3D12GraphicsCommandList,
    from: Option<&ID3D12Resource>,
    to: &ID3D12Resource,
) {
    let same = from.is_some_and(|f| f.as_raw() == to.as_raw());
    if enhanced_barriers(host)
        && let Ok(list7) = list.cast::<ID3D12GraphicsCommandList7>()
    {
        let barriers = [
            buffer_barrier(
                to,
                D3D12_BARRIER_SYNC_COPY,
                BUFFER_SYNC_ALL,
                D3D12_BARRIER_ACCESS_COPY_DEST,
                BUFFER_ACCESS_ALL,
            ),
            buffer_barrier(
                from.unwrap_or(to),
                D3D12_BARRIER_SYNC_COPY,
                BUFFER_SYNC_WRITE,
                D3D12_BARRIER_ACCESS_COPY_SOURCE,
                BUFFER_ACCESS_WRITE,
            ),
        ];
        let count = if from.is_some() && !same { 2 } else { 1 };
        let group = [D3D12_BARRIER_GROUP {
            Type: D3D12_BARRIER_TYPE_BUFFER,
            NumBarriers: count,
            Anonymous: D3D12_BARRIER_GROUP_0 {
                pBufferBarriers: barriers.as_ptr(),
            },
        }];
        // SAFETY: open command list; borrowed pointers live across `Barrier`.
        unsafe { list7.Barrier(&group) };
        return;
    }
    if let Some(f) = from
        && !same
    {
        let barriers = [
            transition(to, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON),
            transition(f, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON),
            uav_barrier(to),
        ];
        // SAFETY: open command list; targeted transitions + UAV barrier on `to`.
        unsafe { list.ResourceBarrier(&barriers) };
    } else {
        let barriers = [
            transition(to, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON),
            uav_barrier(to),
        ];
        // SAFETY: open command list; targeted transition + UAV barrier on `to`.
        unsafe { list.ResourceBarrier(&barriers) };
    }
}

/// The host copy of `[src, src + len)` if shaders write it: what it holds at
/// a place in a list is not what title memory holds at submission.
fn shaders_own(src: usize, len: usize) -> Option<std::sync::Arc<Shadow>> {
    crate::shadow::peek(src as u64, len as u64).filter(|s| s.written() && !s.in_place())
}

impl HostList {
    /// Copies `len` bytes of title memory at `src` to `dst`, at this place of
    /// the list (see the file). `false` if it is to be done at submission.
    pub(crate) fn place_copy(&self, dst: usize, src: usize, len: usize) -> bool {
        if !on() || len == 0 {
            return false;
        }
        let Some(host) = host() else {
            return false;
        };
        let own = shaders_own(src, len);
        let Some(to) = crate::shadow::peek(dst as u64, len as u64)
            .or_else(|| own.is_some().then(|| crate::shadow::find(dst as u64, len as u64)).flatten())
            .filter(|s| !s.in_place())
        else {
            return false;
        };
        let scratch = match &own {
            Some(_) => None,
            None if len <= MOST => match self.scratch_at(len as u64) {
                Some(scratch) => Some(scratch),
                None => return false,
            },
            None => return false,
        };
        let at = to.offset(dst as u64);
        // What the title's CPU wrote around and under it so far goes first.
        to.refresh(self, dst as u64, len as u64);
        if let Some(from) = &own {
            from.refresh(self, src as u64, len as u64);
        }
        let same_buffer = own
            .as_ref()
            .is_some_and(|from| from.buffer.as_raw() == to.buffer.as_raw());
        let between = if same_buffer {
            match self.between(host, len as u64) {
                Some(b) => Some(b),
                None => return false,
            }
        } else {
            None
        };
        let from_res = own.as_ref().map(|from| &from.buffer);
        // SAFETY: an open list; live buffers (the list holds the copies and its
        // scratch/between memory), ranges inside them; targeted per-buffer barriers
        // avoid stalling unrelated render targets, depth buffers, and shaders.
        unsafe { barrier_before_copy(host, &self.list, from_res, &to.buffer) };
        match (&own, &scratch, &between) {
            (Some(from), _, Some(between)) => {
                let offset = from.offset(src as u64);
                unsafe {
                    self.list.CopyBufferRegion(between, 0, &from.buffer, offset, len as u64);
                    barrier_between_same_buffer(host, &self.list, between, &to.buffer);
                    self.list.CopyBufferRegion(&to.buffer, at, between, 0, len as u64);
                }
                STATS[2].fetch_add(1, Ordering::Relaxed);
            }
            (Some(from), _, None) => {
                let offset = from.offset(src as u64);
                unsafe {
                    self.list
                        .CopyBufferRegion(&to.buffer, at, &from.buffer, offset, len as u64);
                }
                STATS[2].fetch_add(1, Ordering::Relaxed);
            }
            (None, Some((from, offset, _)), _) => unsafe {
                self.list.CopyBufferRegion(&to.buffer, at, from, *offset, len as u64);
            },
            (None, None, _) => return false,
        }
        unsafe { barrier_after_copy(host, &self.list, from_res, &to.buffer) };
        STATS[0].fetch_add(1, Ordering::Relaxed);
        STATS[1].fetch_add(len as u64, Ordering::Relaxed);
        to.gpu_writes(self, dst as u64, len as u64);
        let mut deferred = self.deferred();
        deferred.copies.push(Step::Placed(Placed {
            dst,
            src,
            len,
            shadow: to,
            from: scratch.map(|s| s.2 as usize),
        }));
        deferred.refreshed.clear();
        self.changed_something();
        true
    }

    /// A buffer on the card for this recording alone.
    fn between(&self, host: &Host, len: u64) -> Option<ID3D12Resource> {
        let buffer = host.uav_buffer(len.next_multiple_of(1 << 16))?;
        name(&buffer, || "bytes between copies".into());
        self.keep(&buffer);
        Some(buffer)
    }
}

/// Logs per-frame placed buffer-copy counts when `VICTORY_COPIES_IN_PLACE` is active.
pub(super) fn log(frames: f64) {
    if !on() {
        return;
    }
    let t: Vec<f64> = STATS
        .iter()
        .map(|r| r.swap(0, Ordering::Relaxed) as f64 / frames)
        .collect();
    if t[0] > 0.0 {
        tracing::info!(
            "title buffer copies at their place in the lists, per frame (VICTORY_COPIES_IN_PLACE): \
             {:.1} of {:.1} KB, {:.1} of them out of what shaders wrote",
            t[0],
            t[1] / 1024.0,
            t[2],
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copies_in_place_defaults_on_and_respects_disable() {
        assert!(resolve_on(None));
        assert!(resolve_on(Some("")));
        assert!(resolve_on(Some("1")));
        assert!(!resolve_on(Some("0")));
        assert!(!resolve_on(Some(" 0 ")));
    }

    #[test]
    fn waits_for_distinguishes_gpu_src_cpu_src_and_dst_overlap() {
        let written = [(1000usize, 2000usize)];
        // Completely disjoint: no wait.
        assert_eq!(waits_for(3000, 4000, 64, true, &written), Waits::No);
        assert_eq!(waits_for(3000, 4000, 64, false, &written), Waits::No);
        // CPU-staged source overlaps in-flight shader write: must wait.
        assert_eq!(waits_for(3000, 1500, 64, true, &written), Waits::Yes);
        // GPU-copy source overlaps in-flight shader write: only wait if a later CPU reader reads dst.
        assert_eq!(waits_for(3000, 1500, 64, false, &written), Waits::ForReaders);
        // Destination overlaps in-flight shader write: only wait for CPU readers of dst.
        assert_eq!(waits_for(1500, 4000, 64, true, &written), Waits::ForReaders);
        assert_eq!(waits_for(1500, 4000, 64, false, &written), Waits::ForReaders);
    }

    /// Live RTX 3090 D3D12 Debug Layer verification of targeted per-buffer
    /// barriers in `place.rs` across chained `A -> B`, same-buffer `B -> B`
    /// (via `between`), `B -> C`, and upload `scratch -> B` copies in a SINGLE
    /// command list — asserting exact GPU bytes and ZERO Debug Layer errors.
    #[test]
    #[ignore = "requires live D3D12 GPU with VICTORY_D3D12_DEBUG=1"]
    fn debug_layer_place_copy_targeted_barriers_all_paths() {
        let Some(host) = host() else {
            return;
        };
        let (buf_a, ptr_a) = host.shared_buffer(1 << 16).expect("shared A");
        let (buf_b, ptr_b) = host.shared_buffer(1 << 16).expect("shared B");
        let (buf_c, ptr_c) = host.shared_buffer(1 << 16).expect("shared C");
        let between = host.uav_buffer(1 << 16).expect("between uav");
        let list = host.new_list().expect("host list");
        let (scratch, scratch_off, scratch_ptr) = list.scratch_at(64).expect("scratch");

        // SAFETY: mapped 64 KiB shared buffers and 64-byte scratch upload range.
        unsafe {
            std::slice::from_raw_parts_mut(ptr_a, 128).fill(0x3A);
            std::slice::from_raw_parts_mut(ptr_b, 512).fill(0x00);
            std::slice::from_raw_parts_mut(ptr_c, 128).fill(0x00);
            std::slice::from_raw_parts_mut(scratch_ptr, 64).fill(0x9C);
        }

        // SAFETY: open command list, live buffers, valid 64-byte ranges.
        unsafe {
            // 1. GPU->GPU (`buf_a -> buf_b[0..64]`)
            barrier_before_copy(host, &list.list, Some(&buf_a), &buf_b);
            list.list.CopyBufferRegion(&buf_b, 0, &buf_a, 0, 64);
            barrier_after_copy(host, &list.list, Some(&buf_a), &buf_b);

            // 2. Same-buffer (`buf_b[0..64] -> between -> buf_b[256..320]`)
            barrier_before_copy(host, &list.list, Some(&buf_b), &buf_b);
            list.list.CopyBufferRegion(&between, 0, &buf_b, 0, 64);
            barrier_between_same_buffer(host, &list.list, &between, &buf_b);
            list.list.CopyBufferRegion(&buf_b, 256, &between, 0, 64);
            barrier_after_copy(host, &list.list, Some(&buf_b), &buf_b);

            // 3. Chained GPU->GPU (`buf_b[256..320] -> buf_c[0..64]`)
            barrier_before_copy(host, &list.list, Some(&buf_b), &buf_c);
            list.list.CopyBufferRegion(&buf_c, 0, &buf_b, 256, 64);
            barrier_after_copy(host, &list.list, Some(&buf_b), &buf_c);

            // 4. CPU-staged upload -> `buf_b[0..64]` (overwriting after step 3 read)
            barrier_before_copy(host, &list.list, None, &buf_b);
            list.list.CopyBufferRegion(&buf_b, 0, &scratch, scratch_off, 64);
            barrier_after_copy(host, &list.list, None, &buf_b);
        }
        list.close();
        host.execute(&[&list]);

        // SAFETY: GPU execution finished; read back mapped shared buffers.
        unsafe {
            assert!(
                std::slice::from_raw_parts(ptr_b, 64).iter().all(|&b| b == 0x9C),
                "step 4 upload must land in buf_b[0..64]"
            );
            assert!(
                std::slice::from_raw_parts(ptr_b.add(256), 64).iter().all(|&b| b == 0x3A),
                "step 2 same-buffer copy must preserve 0x3A in buf_b[256..320]"
            );
            assert!(
                std::slice::from_raw_parts(ptr_c, 64).iter().all(|&b| b == 0x3A),
                "step 3 chained copy must land 0x3A in buf_c[0..64]"
            );
        }
        if let Some(err) = host.log_messages() {
            panic!("D3D12 Debug Layer reported error on targeted place_copy barriers: {err}");
        }
    }
}
