//! Deadline and retry-budget bookkeeping: overdue checks, timeout
//! resolution, capped wait totals, and runner fitness scoring.

pub fn overdue_by_ms(deadline_ms: u64, now_ms: u64) -> u64 {
    if now_ms <= deadline_ms {
        return 0;
    }
    now_ms - deadline_ms
}

/// Classify a numeric urgency score into a coarse tier name.
pub fn urgency_tier(score: u32) -> &'static str {
    match score {
        n if n >= 8 => "critical",
        5..=7 => "high",
        2..=4 => "normal",
        _ => "low",
    }
}

/// Total delay across pending entries, each bounded individually.
pub fn capped_wait_total(waits: &[u64], cap_ms: u64) -> u64 {
    let mut total = 0u64;
    let mut idx = 0;
    while idx < waits.len() {
        total += waits[idx].min(cap_ms);
        idx += 1;
    }
    total
}

/// Effective timeout for a task, falling back to a default when unset.
pub fn effective_timeout_ms(custom: Option<u64>, default_ms: u64) -> u64 {
    if let Some(value) = custom {
        value.max(1)
    } else {
        default_ms
    }
}

/// Clamp a runner pool size request into the supported range.
pub fn clamp_runner_size(count: u32) -> u32 {
    if count < 1 {
        return 1;
    }
    if count > 32 {
        return 32;
    }
    count
}

fn idle_penalty_points(idle_ms: u64) -> u32 {
    (idle_ms / 100) as u32
}

/// Weighted fitness score for a runner from its fault rate and idle span.
pub fn runner_fitness_score(fault_rate: u32, idle_span_ms: u64) -> u32 {
    let penalty = idle_penalty_points(idle_span_ms);
    let score = fault_rate * 3 + penalty;
    score.min(100)
}

/// Short label describing how much retry budget a task has left.
pub fn retry_budget_label(attempts: u32, max_attempts: u32) -> &'static str {
    if attempts == 0 {
        "fresh"
    } else if attempts >= max_attempts {
        "exhausted"
    } else {
        "retrying"
    }
}
