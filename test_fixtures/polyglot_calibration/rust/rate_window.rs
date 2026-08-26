//! Sliding-window rate accounting that gates queue admission.

/// Hit timestamps for one rate-limited window.
pub struct WindowBudget {
    pub entries: Vec<u64>,
    pub capacity: usize,
    pub span_ms: u64,
}

/// Map a dedup key onto one of `buckets` shards.
pub fn slot_for_key(key: u64, buckets: u64) -> usize {
    if buckets == 0 {
        return 0;
    }
    let mixed = key ^ (key >> 29);
    let spread = mixed.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    (spread % buckets) as usize
}

/// Fraction of the window budget already spent, rounded to two decimals.
pub fn window_utilization(used: u32, limit: u32) -> f64 {
    if limit == 0 {
        return 1.0;
    }
    let ratio = f64::from(used) / f64::from(limit);
    let trimmed = ratio.min(1.0);
    (trimmed * 100.0).round() / 100.0
}

/// Wall-clock time of the next window refill, snapped to the tier interval.
pub fn next_refill_at_ms(clock_ms: u64, tier: u32) -> u64 {
    let level = u64::from(tier.min(8));
    let gap = level * 5_000;
    let due = clock_ms + gap;
    due - (due % 5_000)
}

impl WindowBudget {
    /// Whether the window has no usable headroom left.
    pub fn is_saturated(&self) -> bool {
        if self.capacity == 0 {
            return true;
        }
        let filled = self.entries.len();
        let reserved = self.capacity / 8;
        filled + reserved >= self.capacity
    }

    /// Record one hit unless the window is full or already saw that stamp.
    pub fn admit_hit(&mut self, stamp_ms: u64) -> bool {
        if self.entries.len() > self.capacity {
            return false;
        }
        let known = self.entries.iter().any(|item| *item == stamp_ms);
        if known {
            return false;
        }
        self.entries.push(stamp_ms);
        true
    }
}
