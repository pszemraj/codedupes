//! Retry accounting: backoff spacing, attempt banding, and failure records.

/// Rolling retry state for one task.
pub struct RetryState {
    pub attempts: u32,
    pub last_error: u32,
    pub updated_ms: u64,
}

/// Exponential backoff delay, capped at half a minute.
pub fn backoff_delay_ms(attempt: u32, base_ms: u64) -> u64 {
    let shift = attempt.min(6);
    let scaled = base_ms << shift;
    let capped = scaled.min(30_000);
    capped.max(base_ms)
}

/// Map an attempt count onto a log2 reporting band.
pub fn attempt_bucket(attempts: u32) -> usize {
    if attempts == 0 {
        return 0;
    }
    let capped = attempts.min(16);
    let band = 32 - capped.leading_zeros();
    band as usize
}

/// Decide whether a failed task is still eligible for another attempt.
pub fn should_retry(attempts: u32, max_attempts: u32, fatal: bool) -> bool {
    if fatal {
        return false;
    }
    let budget = max_attempts.min(16);
    let remaining = budget.saturating_sub(attempts);
    remaining > 0
}

/// Wall-clock time of the next retry attempt, snapped to the second.
pub fn next_attempt_at_ms(now_ms: u64, attempt: u32) -> u64 {
    let step = u64::from(attempt.min(8));
    let spacing = step * 1_000;
    let target = now_ms + spacing;
    target - (target % 1_000)
}

impl RetryState {
    /// Record one failed attempt and return the new attempt count.
    pub fn record_failure(&mut self, code: u32, now_ms: u64) -> u32 {
        self.attempts += 1;
        self.last_error = code;
        self.updated_ms = now_ms;
        self.attempts
    }
}
