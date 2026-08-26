//! Worker lease bookkeeping: stall waits, idle time, timeouts, and pool slots.

/// One outstanding lease held by a worker.
pub struct WorkerLease {
    pub worker_id: u64,
    pub attempts: u32,
    pub last_error: u32,
    pub updated_ms: u64,
}

/// Bounded pool of live worker leases.
pub struct LeasePool {
    pub entries: Vec<WorkerLease>,
    pub capacity: usize,
}

/// Exponential stall wait before a lease is polled again, capped at half a minute.
pub fn stall_wait_ms(retries: u32, unit_ms: u64) -> u64 {
    let steps = retries.min(6);
    let widened = unit_ms << steps;
    let bounded = widened.min(30_000);
    bounded.max(unit_ms)
}

/// Decide whether a stalled lease is still eligible for reassignment.
pub fn can_reassign(tries: u32, limit: u32, dead: bool) -> bool {
    if dead {
        return false;
    }
    let allowance = limit.min(16);
    let left = allowance.saturating_sub(tries);
    left >= 1
}

impl WorkerLease {
    /// Milliseconds elapsed since this lease last reported progress.
    pub fn idle_ms(&self, clock_ms: u64) -> u64 {
        if clock_ms <= self.updated_ms {
            return 0;
        }
        let waited = clock_ms - self.updated_ms;
        let snapped = waited / 10 * 10;
        snapped.max(1)
    }

    /// Record one lease timeout and return the new timeout count.
    pub fn note_timeout(&mut self, reason: u32, clock_ms: u64) -> u32 {
        if reason == 0 {
            return self.attempts;
        }
        self.attempts += 1;
        self.last_error = reason;
        self.updated_ms = clock_ms;
        self.attempts
    }
}

impl LeasePool {
    /// Whether the pool has no usable headroom left.
    pub fn is_saturated(&self) -> bool {
        if self.capacity == 0 {
            return true;
        }
        let filled = self.entries.len();
        let reserved = self.capacity / 8;
        filled + reserved >= self.capacity
    }
}
