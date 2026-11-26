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

/// Everything recorded so far is done, and seen, before what follows.
///
/// # Safety
/// An open list.
unsafe fn wait_for_all(host: &Host, list: &ID3D12GraphicsCommandList) {
    static ENHANCED: OnceLock<bool> = OnceLock::new();
    let enhanced = *ENHANCED.get_or_init(|| {
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
    });
    // SAFETY: the caller's.
    unsafe {
        if enhanced && let Ok(list) = list.cast::<ID3D12GraphicsCommandList7>() {
            let all = [D3D12_GLOBAL_BARRIER {
                SyncBefore: D3D12_BARRIER_SYNC_ALL,
                SyncAfter: D3D12_BARRIER_SYNC_ALL,
                AccessBefore: D3D12_BARRIER_ACCESS_COMMON,
                AccessAfter: D3D12_BARRIER_ACCESS_COMMON,
            }];
            list.Barrier(&[D3D12_BARRIER_GROUP {
                Type: D3D12_BARRIER_TYPE_GLOBAL,
                NumBarriers: 1,
                Anonymous: D3D12_BARRIER_GROUP_0 {
                    pGlobalBarriers: all.as_ptr(),
                },
            }]);
            return;
        }
        let alias = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_ALIASING,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Aliasing: std::mem::ManuallyDrop::new(D3D12_RESOURCE_ALIASING_BARRIER {
                    pResourceBefore: std::mem::ManuallyDrop::new(None),
                    pResourceAfter: std::mem::ManuallyDrop::new(None),
                }),
            },
        };
        let uav = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                    pResource: std::mem::ManuallyDrop::new(None),
                }),
            },
        };
        list.ResourceBarrier(&[alias, uav]);
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
        let Some(to) = crate::shadow::peek(dst as u64, len as u64).filter(|s| !s.in_place()) else {
            return false;
        };
        let own = shaders_own(src, len);
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
        // SAFETY (both): an open list; live buffers (the list holds the
        // copies and its scratch memory), ranges inside them.
        unsafe { wait_for_all(host, &self.list) };
        match (&own, &scratch) {
            (Some(from), _) => {
                from.refresh(self, src as u64, len as u64);
                let offset = from.offset(src as u64);
                if from.buffer.as_raw() == to.buffer.as_raw() {
                    let Some(between) = self.between(host, len as u64) else {
                        return false;
                    };
                    unsafe {
                        self.list.CopyBufferRegion(&between, 0, &from.buffer, offset, len as u64);
                        self.list.ResourceBarrier(&[transition(
                            &between,
                            D3D12_RESOURCE_STATE_COPY_DEST,
                            D3D12_RESOURCE_STATE_COPY_SOURCE,
                        )]);
                        wait_for_all(host, &self.list);
                        self.list.CopyBufferRegion(&to.buffer, at, &between, 0, len as u64);
                    }
                } else {
                    unsafe {
                        self.list
                            .CopyBufferRegion(&to.buffer, at, &from.buffer, offset, len as u64);
                    }
                }
                STATS[2].fetch_add(1, Ordering::Relaxed);
            }
            (None, Some((from, offset, _))) => unsafe {
                self.list.CopyBufferRegion(&to.buffer, at, from, *offset, len as u64);
            },
            (None, None) => return false,
        }
        unsafe { wait_for_all(host, &self.list) };
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
}
