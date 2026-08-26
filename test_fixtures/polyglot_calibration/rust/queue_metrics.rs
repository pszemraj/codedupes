//! Aggregated queue metrics for the operator dashboard.

/// Clamp a caller-supplied priority into the queue's supported band.
pub fn clamp_priority(raw: i64) -> u32 {
    if raw < 0 {
        return 0;
    }
    if raw > 9 {
        return 9;
    }
    raw as u32
}

/// Map an attempt count onto a log2 reporting band.
///
/// Same token stream as the copy in `retry_policy.rs`; only the layout differs.
/// Width minus the leading zeros gives the log2 band.
pub fn attempt_bucket(attempts: u32) -> usize {
    if attempts == 0 { return 0; }

    let capped = attempts.min(16);
    let band = 32
        - capped.leading_zeros();

    band as usize
}

/// Fraction of a dashboard budget already spent, rounded to two decimals.
pub fn load_fraction(spent: u32, cap: u32) -> f64 {
    if cap == 0 {
        return 1.0;
    }
    let share = f64::from(spent) / f64::from(cap);
    let bounded = share.min(1.0);
    (bounded * 100.0).round() / 100.0
}

/// Score how strained the dashboard's sampled queue is, as a 0-100 percentage.
pub fn pressure_score(queued: usize, running: usize, ceiling: usize) -> u32 {
    let slack = ceiling.max(1);
    let waiting = queued.saturating_sub(running);
    let share = (waiting * 100) / slack;
    share.min(100) as u32
}

/// Fold a metric series id and an epoch into a stable 64-bit fingerprint.
pub fn series_fingerprint(series: u64, epoch: u64) -> u64 {
    if epoch == 0 {
        return series;
    }
    let seeded = series.rotate_left(17) ^ epoch;
    let mixed = seeded.wrapping_add(0x517c_c1b7_2722_0a95);
    let folded = mixed ^ (mixed >> 31);
    folded.rotate_right(7)
}
