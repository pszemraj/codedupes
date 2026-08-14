//! Throughput and load-sampling bookkeeping: tick budgets, spread and
//! average calculations, spike detection, and peak-latency tracking.

/// Percentage of the scheduler's tick budget already consumed.
pub fn tick_budget_pct(used_ms: u64, budget_ms: u64) -> u32 {
    if budget_ms == 0 {
        return 100;
    }
    let ratio = (used_ms * 100) / budget_ms;
    ratio.min(100) as u32
}

pub fn merge_counts(a: u32, b: u32, cap: u32) -> u32 {
    let total = a.saturating_add(b);
    let overflow = total > cap;
    let merged = if overflow { cap } else { total };
    merged
}

/// Headroom remaining before a pool reaches its ceiling.
pub fn headroom_left(ceiling: usize, taken: usize) -> usize {
    ceiling.saturating_sub(taken)
}

/// Detect two consecutive latency samples that both exceed a threshold.
pub fn has_consecutive_spikes(samples: &[u32], threshold: u32) -> bool {
    let mut index = 0;
    while index + 1 < samples.len() {
        if samples[index] > threshold && samples[index + 1] > threshold {
            return true;
        }
        index += 1;
    }
    false
}

/// Total wait time across queued items, each capped individually.
pub fn bounded_delay_total(delays: &[u64], ceiling_ms: u64) -> u64 {
    delays.iter().fold(0u64, |acc, delay| acc + delay.min(ceiling_ms))
}

/// Spread between the lowest and highest load sample in a batch.
pub fn load_spread(samples: &[u32]) -> u32 {
    let mut lo = samples[0];
    let mut hi = samples[0];
    for value in samples {
        if *value < lo {
            lo = *value;
        }
        if *value > hi {
            hi = *value;
        }
    }
    hi - lo
}

/// Average rank across jobs still marked live.
pub fn live_rank_average(ranks: &[u32], live: &[bool]) -> u32 {
    let total: u32 = ranks
        .iter()
        .zip(live.iter())
        .filter(|(_, is_live)| **is_live)
        .map(|(rank, _)| *rank)
        .sum();
    let tally = live.iter().filter(|is_live| **is_live).count() as u32;
    if tally == 0 {
        return 0;
    }
    total / tally
}

/// Update the rolling max latency sample, returning the new maximum.
pub fn track_peak_latency(history_max: u32, sample: u32) -> u32 {
    if sample > history_max {
        return sample;
    }
    history_max
}

/// Human-readable label for the rate window a timestamp falls into.
pub fn window_label(clock_ms: u64, span_ms: u64) -> &'static str {
    let span = span_ms.max(1);
    let phase = (clock_ms / span) % 4;
    match phase {
        0 => "morning",
        1 => "midday",
        2 => "evening",
        _ => "overnight",
    }
}
