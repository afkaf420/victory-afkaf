//! Clamps of many indirect executions in one compute pass (`VICTORY_INDIRECT_BATCH`):
//! the arguments of every single-record indirect execution between two points where
//! arguments may change are bounded together, at the first of them.
//!
//! Because later executions are not known yet when the batch dispatch is recorded,
//! the compute pass reads its job list from a CPU-populated upload table (`Table`),
//! which the CPU continues filling as commands are recorded into the list. Executions
//! sharing the same `(source, offset, kind)` reuse the same clamped slot and multiply
//! their budget charge cleanly.

use windows::Win32::Graphics::Direct3D12::*;

use super::{Clamp, uav_barrier};
use crate::host::transition;

/// Maximum jobs in one batched clamp pass.
pub(crate) const JOBS: u32 = 256;
/// Bytes of the upload job table: 16-byte header (job count) + 16 bytes/job.
pub(crate) const TABLE: u64 = 16 + 16 * JOBS as u64;
/// Scratch bytes per clamped record slot.
pub(crate) const SLOT: u32 = 32;
/// Number of distinct source argument buffers bound to one batch pass.
pub(crate) const SOURCES: usize = 4;
/// Maximum reuse count per job (so `cost * times` fits cleanly in 32 bits).
pub(crate) const TIMES: u32 = 31;

pub(crate) const SOURCE: &str = r#"
#define RS "SRV(t0), SRV(t1), SRV(t2), SRV(t3), SRV(t4), UAV(u0), UAV(u1)"
ByteAddressBuffer src0 : register(t0);
ByteAddressBuffer src1 : register(t1);
ByteAddressBuffer src2 : register(t2);
ByteAddressBuffer src3 : register(t3);
// Header: count (u32), 12 bytes pad.
// Per job: x = source (0..1) | (kind << 4) | (words << 8) | (times << 16),
//          y = src byte offset, z = dst scratch byte offset, w = 0.
ByteAddressBuffer jobs : register(t4);
RWByteAddressBuffer dst_args : register(u0);
RWByteAddressBuffer budget : register(u1);
static const uint BUDGET = 1u << 27;
static const uint GROUPS = 1u << 20;
groupshared uint costs[64];
groupshared uint fits[64];
groupshared uint largest_asked;
groupshared uint all_asked;
uint load(uint source, uint at) {
    if (source == 0) return src0.Load(at);
    if (source == 1) return src1.Load(at);
    if (source == 2) return src2.Load(at);
    return src3.Load(at);
}
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID, uint gi : SV_GroupIndex) {
    uint count = jobs.Load(0);
    bool active = id.x < count;
    uint4 job = uint4(0, 0, 0, 0);
    uint record[5] = {0, 0, 0, 0, 0};
    uint cost = 0;
    uint asked = 0;
    uint kind = 0;
    uint words = 0;
    if (gi == 0) {
        largest_asked = 0;
        all_asked = 0;
    }
    if (active) {
        job = jobs.Load4(16 + id.x * 16);
        kind = (job.x >> 4) & 15;
        words = min((job.x >> 8) & 15, 5);
        for (uint w = 0; w < words; w++) {
            record[w] = load(job.x & 3, job.y + w * 4);
        }
        if (kind <= 1) {
            if (record[0] >= 1u << 23) {
                record[0] = 0;
                record[1] = 0;
            }
            record[1] = min(record[1], (1u << 26) / max(record[0], 1u));
            cost = record[0] * record[1];
        } else if (kind == 2) {
            uint3 g = uint3(record[0], record[1], record[2]);
            g = min(g, uint3(g.x <= WIDEST ? WIDEST : 65535u, 65535u, 65535u));
            if ((g.y != 0 && g.x > GROUPS / g.y) || (g.z != 0 && g.x * g.y > GROUPS / g.z)) {
                g = uint3(0, 0, 0);
            }
            cost = g.x * g.y * g.z * 64u;
            record[0] = g.x;
            record[1] = g.y;
            record[2] = g.z;
        } else {
            if (record[0] > 65535u) {
                record[0] = 0;
                record[1] = 0;
                record[2] = 0;
            }
            asked = record[0];
            record[1] = min(record[1], 1u);
            record[2] = min(record[2], 1u);
            cost = record[0] * record[1] * record[2] * 128u;
        }
        cost *= job.x >> 16;
    }
    costs[gi] = cost;
    GroupMemoryBarrierWithGroupSync();
    if (active && kind > 2) {
        uint was;
        InterlockedMax(largest_asked, asked, was);
        InterlockedAdd(all_asked, min(asked, 1u << 24), was);
    }
    uint first = id.x - gi;
    uint n = count > first ? min(64u, count - first) : 0u;
    if (gi == 0 && n > 0) {
        uint before;
        budget.InterlockedOr(0, 0, before);
        [allow_uav_condition]
        while (before < BUDGET) {
            uint next = before;
            for (uint j = 0; j < n; j++) {
                uint c = costs[j];
                if (c > 0 && next < BUDGET) {
                    next = (BUDGET - next < c) ? (REST ? BUDGET : next) : (next + c);
                }
            }
            if (next == before) break;
            uint prev;
            budget.InterlockedCompareExchange(0, before, next, prev);
            if (prev == before) break;
            before = prev;
        }
        uint at = before;
        for (uint k = 0; k < n; k++) {
            uint c = costs[k];
            uint fit = 1;
            if (c > 0) {
                fit = (at < BUDGET && BUDGET - at >= c) ? 1u : 0u;
                at = (at >= BUDGET || BUDGET - at < c) ? (REST ? BUDGET : at) : (at + c);
            }
            fits[k] = fit;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (gi == 0 && n > 0 && all_asked > 0) {
        uint was;
        budget.InterlockedMax(4, largest_asked, was);
        budget.InterlockedAdd(8, all_asked, was);
    }
    if (!active) {
        return;
    }
    if (fits[gi] == 0) {
        record[0] = 0;
        record[1] = 0;
        if (kind > 1) {
            record[2] = 0;
        }
    }
    for (uint s = 0; s < words; s++) {
        dst_args.Store(job.z + s * 4, record[s]);
    }
}
"#;

/// Words per record for PC indirect argument `kind`.
pub(crate) fn words(kind: u32) -> Option<u32> {
    super::min_stride(kind).map(|bytes| bytes / 4)
}

/// Packs a job's first word.
pub(crate) fn job(source: u32, kind: u32, times: u32) -> u32 {
    source | (kind << 4) | (words(kind).unwrap_or(0) << 8) | (times << 16)
}

/// CPU-side builder for a batch pass's job table in mapped upload memory.
#[derive(Default)]
pub(crate) struct Table {
    at: usize,
    base: u32,
    sources: [u64; SOURCES],
    times: Vec<u32>,
    known: std::collections::HashMap<(u32, u32, u32), u32>,
}

impl Table {
    /// Initializes an empty table at `at`.
    ///
    /// # Safety
    /// `at` points to [`TABLE`] writable, 4-byte-aligned bytes of mapped upload
    /// memory that live until the recorded list finishes executing on the GPU.
    pub(crate) unsafe fn new(at: *mut u8, base: u32, sources: [u64; SOURCES]) -> Table {
        // SAFETY: caller's contract.
        unsafe { at.cast::<u32>().write_volatile(0) };
        Table { at: at as usize, base, sources, ..Table::default() }
    }

    /// Places or reuses `(id, offset, kind)` in this batch, returning its byte
    /// offset in the batch scratch buffer (`None` if source isn't bound or full).
    pub(crate) fn place(&mut self, id: u64, offset: u64, kind: u32) -> Option<u64> {
        let source = self.sources.iter().position(|&s| s == id)? as u32;
        let offset = u32::try_from(offset).ok().filter(|o| o.is_multiple_of(4))?;
        offset.checked_add(words(kind)? * 4)?;
        let key = (source, offset, kind);
        let table = self.at as *mut u32;
        let place = |i: u32| u64::from(self.base + i * SLOT);
        if let Some(&i) = self.known.get(&key)
            && self.times[i as usize] < TIMES
        {
            self.times[i as usize] += 1;
            // SAFETY: inside `TABLE` bounds (`i < JOBS`).
            unsafe { table.add(4 + 4 * i as usize).write_volatile(job(source, kind, self.times[i as usize])) };
            return Some(place(i));
        }
        let i = self.times.len() as u32;
        if i == JOBS {
            return None;
        }
        self.times.push(1);
        self.known.insert(key, i);
        // SAFETY: inside `TABLE` bounds (`i < JOBS`).
        unsafe {
            let to = table.add(4 + 4 * i as usize);
            to.write_volatile(job(source, kind, 1));
            to.add(1).write_volatile(offset);
            to.add(2).write_volatile(self.base + i * SLOT);
            to.add(3).write_volatile(0);
            table.write_volatile(i + 1);
        }
        Some(place(i))
    }
}

/// Records the batched clamp pass over `jobs` into `dst` (entering in `before`
/// and leaving in `INDIRECT_ARGUMENT`).
pub(super) fn record(
    c: &Clamp,
    l: &ID3D12GraphicsCommandList,
    (dst, before): (&ID3D12Resource, D3D12_RESOURCE_STATES),
    budget: u64,
    sources: [u64; SOURCES],
    jobs: u64,
) {
    let after = transition(
        dst,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
    );
    // SAFETY: open list, live buffers, our root signature and compute PSO.
    unsafe {
        l.ResourceBarrier(&[
            uav_barrier(),
            transition(dst, before, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
        ]);
        l.SetComputeRootSignature(&c.root);
        l.SetPipelineState(&c.pso);
        for (i, source) in sources.into_iter().enumerate() {
            l.SetComputeRootShaderResourceView(i as u32, source);
        }
        l.SetComputeRootShaderResourceView(SOURCES as u32, jobs);
        l.SetComputeRootUnorderedAccessView(SOURCES as u32 + 1, dst.GetGPUVirtualAddress());
        l.SetComputeRootUnorderedAccessView(SOURCES as u32 + 2, budget);
        l.Dispatch(JOBS.div_ceil(64), 1, 1);
        l.ResourceBarrier(&[after]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_dedups_and_bounds_sources_and_capacity() {
        let mut raw = vec![0u32; (TABLE / 4) as usize];
        // SAFETY: aligned writable backing slice of `TABLE` bytes.
        let mut table = unsafe { Table::new(raw.as_mut_ptr().cast(), 64, [10, 20, 0, 0]) };
        // First insert for source 10 at offset 32, kind 10 (mesh).
        assert_eq!(table.place(10, 32, 10), Some(64));
        // Duplicate reuses slot 0 and bumps times=2.
        assert_eq!(table.place(10, 32, 10), Some(64));
        assert_eq!(raw[0], 1);
        assert_eq!(raw[4], job(0, 10, 2));
        // Second source 20 allocates slot 1.
        assert_eq!(table.place(20, 16, 2), Some(64 + u64::from(SLOT)));
        assert_eq!(raw[0], 2);
        // Unknown source returns None.
        assert_eq!(table.place(99, 0, 2), None);
        // Unaligned offset returns None.
        assert_eq!(table.place(10, 2, 2), None);
    }
}
