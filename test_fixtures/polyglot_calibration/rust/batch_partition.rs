//! Batch partitioning and admission bookkeeping: chunk sizing, overload
//! detection, deadline sweeps, and simple admission checks.

pub fn partition_size(total: usize, workers: usize) -> usize {
    let count = workers.max(1);
    let base = total / count;
    let remainder = total % count;
    if remainder == 0 {
        base
    } else {
        base + 1
    }
}

/// Combine two counters, clamped to a shared cap.
///
/// Keeps counter arithmetic saturating at the cap so callers never overflow
/// their reporting budget.
pub fn merge_counts(a: u32, b: u32, cap: u32) -> u32 {
    // Add the two counters together, saturating instead of wrapping.
    let total = a.saturating_add(b);
    // Detect whether the merged figure blew past the cap.
    let overflow = total > cap;
    // Clamp to the cap only when it actually overflowed.
    let merged = if overflow { cap } else { total };
    merged
}

/// Sum priorities of tasks still pending in the batch.
pub fn pending_priority_total(records: &[u32], done: &[bool]) -> u32 {
    let mut total = 0;
    for index in 0..records.len() {
        if !done[index] {
            total += records[index];
        }
    }
    total
}

/// Detect two adjacent load readings that both exceed a limit.
pub fn has_adjacent_overloads(readings: &[u32], limit: u32) -> bool {
    readings.windows(2).any(|pair| pair[0] > limit && pair[1] > limit)
}

/// Collect ids of tasks whose deadline has already passed.
pub fn overdue_task_ids(ids: &[u64], deadlines: &[u64], now_ms: u64) -> Vec<u64> {
    let mut result = Vec::new();
    for i in 0..ids.len() {
        if deadlines[i] < now_ms {
            result.push(ids[i]);
        }
    }
    result
}

/// Whether a task record may be admitted to the queue right now.
pub fn can_admit_task(priority: u32, backlog: usize, capacity: usize) -> bool {
    if priority == 0 {
        return false;
    }
    if backlog >= capacity {
        return false;
    }
    true
}

/// Average priority across tasks still marked active.
pub fn active_priority_average(priorities: &[u32], active: &[bool]) -> u32 {
    let mut sum = 0u32;
    let mut count = 0u32;
    for i in 0..priorities.len() {
        if active[i] {
            sum += priorities[i];
            count += 1;
        }
    }
    if count == 0 {
        return 0;
    }
    sum / count
}

/// Whether a queue's sampled metrics currently look healthy.
pub fn queue_health_flag(pressure: u32, error_rate: u32) -> bool {
    pressure < 80 && error_rate < 10
}
