//! Scheduler heartbeat and worker-health bookkeeping: tick budgets, beat
//! spacing, deadline checks, and simple health scoring.

/// Percentage of the scheduler's tick budget already consumed.
pub fn tick_budget_pct(used_ms: u64, budget_ms: u64) -> u32 {
    if budget_ms == 0 {
        return 100;
    }
    let ratio = (used_ms * 100) / budget_ms;
    ratio.min(100) as u32
}

/// Interval between two heartbeats, floored at the minimum tick.
pub fn beat_interval_ms(period_ms: u64, floor_ms: u64) -> u64 {
    let span = period_ms.max(floor_ms);
    let step = floor_ms.max(1);
    let aligned = span - (span % step);
    aligned.max(floor_ms)
}

/// Milliseconds a task has run past its deadline.
///
/// Returns zero while the deadline has not yet passed.
pub fn overdue_by_ms(deadline_ms: u64, now_ms: u64) -> u64 {
    // Not overdue until the clock passes the deadline.
    if now_ms <= deadline_ms {
        return 0;
    }
    // Otherwise report the elapsed overrun.
    now_ms - deadline_ms
}

/// Update the rolling worst-case delay sample, returning the new worst case.
pub fn track_worst_delay(prior_peak: u32, reading: u32) -> u32 {
    let bounded = reading.min(10_000);
    if bounded > prior_peak {
        return bounded;
    }
    prior_peak
}

/// Classify a numeric priority into a coarse severity band.
pub fn priority_band(level: u32) -> &'static str {
    if level >= 8 {
        "critical"
    } else if level >= 5 {
        "high"
    } else if level >= 2 {
        "normal"
    } else {
        "low"
    }
}

/// Collect handles of jobs whose expiry has already elapsed.
pub fn expired_job_handles(handles: &[u64], expiries: &[u64], clock_ms: u64) -> Vec<u64> {
    handles
        .iter()
        .zip(expiries.iter())
        .filter(|(_, expiry)| **expiry < clock_ms)
        .map(|(handle, _)| *handle)
        .collect()
}

/// Count runners whose current usage exceeds a limit.
pub fn saturated_runner_count(usages: &[u32], limit: u32) -> usize {
    usages.iter().filter(|usage| **usage > limit).count()
}

/// Weighted health score for a worker from its error rate and idle time.
pub fn worker_health_score(error_rate: u32, idle_ms: u64) -> u32 {
    let idle_penalty = (idle_ms / 100) as u32;
    let combined = error_rate * 3 + idle_penalty;
    combined.min(100)
}

/// Alert severity implied by a queue's current backlog depth.
pub fn backlog_alert_level(depth: usize) -> &'static str {
    if depth > 500 {
        "critical"
    } else if depth > 100 {
        "warning"
    } else {
        "ok"
    }
}
