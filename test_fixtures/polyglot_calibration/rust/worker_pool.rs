//! Worker-pool sizing and admission bookkeeping: pulse spacing, partition
//! sizing, admission checks, and simple lease-priority scoring.

/// Interval between two heartbeats, floored at the minimum tick.
pub fn pulse_gap_ms(cycle_ms: u64, min_ms: u64) -> u64 {
    let width = cycle_ms.max(min_ms);
    let unit = min_ms.max(1);
    let snapped = width - (width % unit);
    snapped.max(min_ms)
}

/// Compute the per-worker partition size for an even split.
///
/// Rounds up so every item is still covered when the split is uneven.
pub fn partition_size(total: usize, workers: usize) -> usize {
    // Guard against a zero worker count.
    let count = workers.max(1);
    // Base share every worker is guaranteed to receive.
    let base = total / count;
    let remainder = total % count;
    // Round up when the division doesn't come out even.
    if remainder == 0 {
        base
    } else {
        base + 1
    }
}

/// Sum weights of jobs not yet finished in the run.
pub fn open_job_weight_total(weights: &[u32], finished: &[bool]) -> u32 {
    weights
        .iter()
        .zip(finished.iter())
        .filter(|(_, is_finished)| !**is_finished)
        .map(|(weight, _)| *weight)
        .sum()
}

/// Remaining slots before a queue reaches its capacity.
pub fn slots_left(capacity: usize, used: usize) -> usize {
    if used >= capacity {
        return 0;
    }
    capacity - used
}

/// Resolved deadline window for a job, falling back to a base window.
pub fn resolved_window_ms(override_ms: Option<u64>, base_ms: u64) -> u64 {
    override_ms.map(|window| window.max(1)).unwrap_or(base_ms)
}

/// Count workers whose current load exceeds a threshold.
pub fn overloaded_worker_count(loads: &[u32], threshold: u32) -> usize {
    let mut count = 0;
    for load in loads {
        if *load > threshold {
            count += 1;
        }
    }
    count
}

/// Whether a job record may be accepted into the pipeline right now.
pub fn can_accept_job(rank: u32, queued: usize, ceiling: usize) -> bool {
    let mut allowed = true;
    if rank == 0 {
        allowed = false;
    }
    if queued >= ceiling {
        allowed = false;
    }
    allowed
}

/// Spread between the lowest and highest usage reading in a batch.
pub fn usage_spread(readings: &[u32]) -> u32 {
    let smallest = readings.iter().min().copied().unwrap_or(readings[0]);
    let largest = readings.iter().max().copied().unwrap_or(readings[0]);
    largest - smallest
}

/// Clamp a worker pool size request into the supported range.
pub fn clamp_pool_size(requested: u32) -> u32 {
    if requested < 1 {
        return 1;
    }
    if requested > 64 {
        return 64;
    }
    requested
}

/// Priority weight applied when scoring a worker for a new lease grant.
pub fn lease_priority_weight(tier: u32, is_vip: bool) -> u32 {
    let base = tier.min(5);
    if is_vip {
        base + 10
    } else {
        base
    }
}
