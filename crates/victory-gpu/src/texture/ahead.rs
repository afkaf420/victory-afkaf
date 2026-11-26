//! First uploads of textures picked by index on a copy queue of their own
//! (`VICTORY_UPLOAD_AHEAD`, on by default; `=0` disables): read, detiled and
//! copied by threads of ours as soon as the texture pass has made them, and
//! the slots that wait for them shown when the copy has run. Recorded into the
//! pass's list, as every other upload is, the copy ran with the frame that
//! needs it, so a slot that shows its texture only after its upload
//! ([`crate::descriptor::host::show_after`]) was black or grey in that frame,
//! and in the next.
//!
//! Only a texture that no list has used is taken: made in the last second,
//! in the common state, and never asked for an upload before
//! ([`crate::resource::HostTexture::ahead`]). It stays in the common state
//! (a copy queue's copies leave it there), from which the card's shaders read
//! it as any texture.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU8, AtomicU64, AtomicUsize, Ordering::Relaxed};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::Interface;

use super::Staging;
use crate::object::This;
use crate::resource::HostTexture;

/// [`HostTexture::ahead`]: no upload was asked for yet; one is on its way
/// here; it has run; one was recorded into a list.
pub(crate) const NONE: u8 = 0;
const PENDING: u8 = 1;
const DONE: u8 = 2;
pub(crate) const LISTED: u8 = 3;

/// Bytes of title memory on their way at most: past it uploads go into the
/// lists, as before.
const MOST: u64 = 256 << 20;
/// Bytes and textures of one submission at most.
const BATCH: (u64, usize) = (32 << 20, 128);

/// Whether it is on (`VICTORY_UPLOAD_AHEAD`; on by default, `=0` disables).
pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_UPLOAD_AHEAD").is_none_or(|v| v != "0"))
}

/// An upload on its way.
pub(crate) struct Job {
    staging: Arc<Staging>,
    copy: Arc<HostTexture>,
    asked: std::time::Instant,
    /// What to do when it has run; `None`: it has.
    then: Mutex<Option<Vec<Box<dyn FnOnce() + Send>>>>,
}

impl Job {
    /// Calls `f` when the copy has run: now if it has.
    pub(crate) fn then(&self, f: impl FnOnce() + Send + 'static) {
        let mut then = self.then.lock().unwrap_or_else(|e| e.into_inner());
        match then.as_mut() {
            Some(then) => then.push(Box::new(f)),
            None => {
                drop(then);
                f();
            }
        }
    }
}

/// What a texture asked for here gets.
pub(crate) enum Taken {
    /// A job of its own, to be [`send`] once what follows it is added.
    New(Arc<Job>),
    /// One already on its way, or that has run.
    Known(Arc<Job>),
}

static QUEUE: (Mutex<VecDeque<Arc<Job>>>, Condvar) = (Mutex::new(VecDeque::new()), Condvar::new());
/// Those on their way, by host copy ([`HostTexture::serial`]).
static ON_THEIR_WAY: Mutex<Vec<(u64, Arc<Job>)>> = Mutex::new(Vec::new());
static IN_FLIGHT: AtomicUsize = AtomicUsize::new(0);
static LANDED: Condvar = Condvar::new();
static BYTES: AtomicU64 = AtomicU64::new(0);
/// Since the start: uploads, bytes, microseconds from the asking until the
/// copy had run, the longest; askings turned down; waits of lists.
static STATS: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

/// Takes the first upload of `resource`'s levels from `first` on, `there` of
/// whose bytes are mapped (`u64::MAX`: not known). `None`: the list's.
pub(crate) fn take(resource: This, first: u32, there: u64) -> Option<Taken> {
    if on() { claim(resource, first, there) } else { None }
}

fn claim(resource: This, first: u32, there: u64) -> Option<Taken> {
    workers()?;
    let turned_down = || {
        STATS[4].fetch_add(1, Relaxed);
        None
    };
    let copy = crate::resource::host_copy(resource)?;
    if copy.ahead.load(Relaxed) == PENDING {
        let way = ON_THEIR_WAY.lock().unwrap_or_else(|e| e.into_inner());
        return way.iter().find(|(serial, _)| *serial == copy.serial).map(|(_, job)| Taken::Known(job.clone()));
    }
    let most = if crate::residency::limited_host_ram() { 32 << 20 } else { MOST };
    if !crate::residency::allow_staging_growth(false)
        || BYTES.load(Relaxed) >= most
        || copy.flags() & 0x7 != 0
        || copy.state.load(Relaxed) != 0
        || copy.grown.load(Relaxed)
        || copy.tiled.is_some()
        || copy.holds.lock().unwrap_or_else(|e| e.into_inner()).is_some()
        || !crate::resource::fresh(&copy.texture)
        || super::host_wrote(resource as usize)
        || crate::resource::alias_of(resource).is_some()
    {
        return turned_down();
    }
    if copy.ahead.compare_exchange(NONE, PENDING, Relaxed, Relaxed).is_err() {
        return turned_down();
    }
    super::PICKING.set(Some(there));
    let staging = super::staging(resource, &copy, first);
    super::PICKING.set(None);
    let Some(staging) = staging.filter(|s| {
        !s.on_demand && s.first == s.level0 && !s.uploaded.load(Relaxed) && s.once.load(Relaxed)
    }) else {
        copy.ahead.store(LISTED, Relaxed);
        return turned_down();
    };
    if super::no_wait() && super::fill_unlocked() {
        staging.expected.store(
            if there == u64::MAX { super::mapped_bytes(staging.memory, staging.read) } else { there },
            Relaxed,
        );
    }
    let job = Arc::new(Job { staging, copy, asked: std::time::Instant::now(), then: Mutex::new(Some(Vec::new())) });
    BYTES.fetch_add(job.staging.read as u64, Relaxed);
    {
        let mut way = ON_THEIR_WAY.lock().unwrap_or_else(|e| e.into_inner());
        way.push((job.copy.serial, job.clone()));
        IN_FLIGHT.store(way.len(), std::sync::atomic::Ordering::Release);
    }
    Some(Taken::New(job))
}

/// Sends a job of [`take`] on its way.
pub(crate) fn send(job: &Arc<Job>) {
    QUEUE.0.lock().unwrap_or_else(|e| e.into_inner()).push_back(job.clone());
    QUEUE.1.notify_one();
}

/// Waits until every upload currently on its way here has finished copying
/// and run its completion callbacks (queuing slot flips into `FLIPS`).
/// Returns how many jobs were waited for.
pub(crate) fn drain() -> usize {
    if IN_FLIGHT.load(std::sync::atomic::Ordering::Acquire) == 0 {
        return 0;
    }
    let mut way = ON_THEIR_WAY.lock().unwrap_or_else(|e| e.into_inner());
    let waiting = way.len();
    let started = std::time::Instant::now();
    while !way.is_empty() && started.elapsed().as_millis() < 2000 {
        way = LANDED
            .wait_timeout(way, std::time::Duration::from_millis(50))
            .unwrap_or_else(|e| e.into_inner())
            .0;
    }
    waiting
}

/// A list is about to record an upload into `copy`: one on its way here runs
/// first, and no other is taken after it.
pub(crate) fn listed(copy: &HostTexture) {
    if copy.ahead.compare_exchange(NONE, LISTED, Relaxed, Relaxed) != Err(PENDING) {
        return;
    }
    STATS[5].fetch_add(1, Relaxed);
    let mut way = ON_THEIR_WAY.lock().unwrap_or_else(|e| e.into_inner());
    let started = std::time::Instant::now();
    while way.iter().any(|(serial, _)| *serial == copy.serial) && started.elapsed().as_millis() < 1000 {
        way = LANDED.wait_timeout(way, std::time::Duration::from_millis(50)).unwrap_or_else(|e| e.into_inner()).0;
    }
}

/// The copy queue and its threads, made on first use.
fn workers() -> Option<&'static ID3D12CommandQueue> {
    /// Free-threaded.
    struct Queue(ID3D12CommandQueue);
    // SAFETY: a command queue is free-threaded.
    unsafe impl Send for Queue {}
    // SAFETY: as above.
    unsafe impl Sync for Queue {}
    static WORKERS: OnceLock<Option<Queue>> = OnceLock::new();
    WORKERS
        .get_or_init(|| {
            let host = crate::host::host()?;
            // SAFETY: plain object creation.
            let queue: ID3D12CommandQueue = unsafe {
                host.device
                    .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                        Type: D3D12_COMMAND_LIST_TYPE_COPY,
                        ..Default::default()
                    })
                    .ok()?
            };
            for n in 0..4 {
                let worker = Worker::new(&host.device, queue.clone()).ok()?;
                std::thread::Builder::new().name(format!("uploads ahead {n}")).spawn(move || worker.work()).ok()?;
            }
            Some(Queue(queue))
        })
        .as_ref()
        .map(|q| &q.0)
}

struct Worker {
    queue: ID3D12CommandQueue,
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    fence: ID3D12Fence,
}

// SAFETY: a worker's objects are its thread's own; the queue is free-threaded.
unsafe impl Send for Worker {}

impl Worker {
    fn new(device: &ID3D12Device, queue: ID3D12CommandQueue) -> windows::core::Result<Worker> {
        // SAFETY: plain object creation.
        unsafe {
            let allocator: ID3D12CommandAllocator = device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY)?;
            let list: ID3D12GraphicsCommandList =
                device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, &allocator, None)?;
            list.Close()?;
            Ok(Worker { queue, allocator, list, fence: device.CreateFence(0, D3D12_FENCE_FLAG_NONE)? })
        }
    }

    fn work(self) {
        let mut value = 0;
        loop {
            let mut jobs = Vec::new();
            {
                let mut queue = QUEUE.0.lock().unwrap_or_else(|e| e.into_inner());
                while queue.is_empty() {
                    queue = QUEUE.1.wait(queue).unwrap_or_else(|e| e.into_inner());
                }
                let mut bytes = 0;
                while bytes < BATCH.0 && jobs.len() < BATCH.1
                    && let Some(job) = queue.pop_front()
                {
                    bytes += job.staging.read as u64;
                    jobs.push(job);
                }
            }
            value += 1;
            if let Err(e) = self.copy(&jobs, value) {
                tracing::warn!(error = %e, "uploads ahead did not run");
            }
            finish(jobs);
        }
    }

    fn copy(&self, jobs: &[Arc<Job>], value: u64) -> windows::core::Result<()> {
        for job in jobs {
            job.staging.refresh();
        }
        // SAFETY: this worker's allocator and list, whose last submission has
        // run; the jobs hold what the list names until after the wait.
        unsafe {
            self.allocator.Reset()?;
            self.list.Reset(&self.allocator, None)?;
            for job in jobs {
                job.record(&self.list);
            }
            self.list.Close()?;
            self.queue.ExecuteCommandLists(&[Some(self.list.cast()?)]);
            self.queue.Signal(&self.fence, value)?;
            // (Without an event: this waits.)
            self.fence.SetEventOnCompletion(value, windows::Win32::Foundation::HANDLE::default())?;
        }
        Ok(())
    }
}

impl Job {
    /// # Safety
    /// `list` is an open copy list, and the job lives until it has run.
    unsafe fn record(&self, list: &ID3D12GraphicsCommandList) {
        let (s, texture) = (&self.staging, &self.copy.texture);
        for (i, footprint) in s.footprints.iter().enumerate() {
            let to = D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: a borrowed pointer the location never releases.
                pResource: unsafe { std::mem::transmute_copy(texture) },
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: s.first - s.level0 + i as u32 },
            };
            let from = D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: as above.
                pResource: unsafe { std::mem::transmute_copy(&s.buffer) },
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT { Offset: s.base + footprint.Offset, ..*footprint },
                },
            };
            // SAFETY: the caller's.
            unsafe { list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
        }
    }
}

/// The jobs' copies have run.
fn finish(jobs: Vec<Arc<Job>>) {
    for job in &jobs {
        job.staging.uploaded.store(true, Relaxed);
        job.staging.dirty.store(false, Relaxed);
        job.staging.note_held();
        job.copy.ahead.store(DONE, Relaxed);
        BYTES.fetch_sub(job.staging.read as u64, Relaxed);
    }
    for job in &jobs {
        let us = job.asked.elapsed().as_micros() as u64;
        let n = STATS[0].fetch_add(1, Relaxed) + 1;
        let bytes = STATS[1].fetch_add(job.staging.read as u64, Relaxed) + job.staging.read as u64;
        let all = STATS[2].fetch_add(us, Relaxed) + us;
        let most = STATS[3].fetch_max(us, Relaxed).max(us);
        if n == 1 || n.is_multiple_of(5000) {
            tracing::info!(
                "textures uploaded ahead of the lists (VICTORY_UPLOAD_AHEAD=1), since the start: {n}, {} MB, {:.1} ms from the asking until they were there, the longest {:.0}; {} turned down, {} waited for by lists",
                bytes >> 20,
                all as f64 / n as f64 / 1e3,
                most as f64 / 1e3,
                STATS[4].load(Relaxed),
                STATS[5].load(Relaxed)
            );
        }
        let then = job.then.lock().unwrap_or_else(|e| e.into_inner()).take();
        for f in then.into_iter().flatten() {
            f();
        }
    }
    // Remove from `ON_THEIR_WAY` only after `job.then` callbacks have queued
    // their `FLIPS` entries so `drain()` observes every flip ready to apply.
    {
        let mut way = ON_THEIR_WAY.lock().unwrap_or_else(|e| e.into_inner());
        way.retain(|(serial, _)| !jobs.iter().any(|job| job.copy.serial == *serial));
        IN_FLIGHT.store(way.len(), std::sync::atomic::Ordering::Release);
        LANDED.notify_all();
    }
    crate::descriptor::host::flip_now();
}

/// Whether `resource`'s host copy was filled here: no list still brings what
/// it shows.
pub(crate) fn there(resource: This) -> bool {
    crate::resource::made_copy(resource).is_some_and(|copy| copy.ahead.load(Relaxed) == DONE)
}

/// The state of [`HostTexture::ahead`] of a new host copy.
pub(crate) fn fresh() -> AtomicU8 {
    AtomicU8::new(NONE)
}
