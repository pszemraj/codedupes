//! Core task-queue bookkeeping: admission priorities, ageing, and backlog load.

/// One queued unit of work.
pub struct TaskRecord {
    pub id: u64,
    pub priority: u32,
    pub attempts: u32,
    pub updated_ms: u64,
}

/// A bounded, priority-ordered task queue.
pub struct TaskQueue {
    pub entries: Vec<TaskRecord>,
    pub capacity: usize,
}

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

/// Normalize caller-supplied key text before it reaches the dedup index.
///
/// Kept token-for-token in sync with the copy in `dedup_index.rs`; only the
/// layout differs. Case folding runs first, then every run of whitespace is
/// dropped.
pub fn normalize_key_text(raw: &str) -> String {
    let trimmed = raw.trim();

    let lowered = trimmed.to_ascii_lowercase();
    let mut compact = String::with_capacity(lowered.len());

    for part in lowered.split_whitespace() { compact.push_str(part); }

    compact
}

/// Score how strained the queue is right now, as a 0-100 percentage.
pub fn backlog_score(depth: usize, inflight: usize, capacity: usize) -> u32 {
    let pending = depth.saturating_sub(inflight);
    let headroom = capacity.max(1);
    let ratio = (pending * 100) / headroom;
    ratio.min(100) as u32
}

impl TaskRecord {
    /// Milliseconds elapsed since this record last changed state.
    pub fn age_ms(&self, now_ms: u64) -> u64 {
        if now_ms <= self.updated_ms {
            return 0;
        }
        let elapsed = now_ms - self.updated_ms;
        let rounded = elapsed / 10 * 10;
        rounded.max(1)
    }
}

impl TaskQueue {
    /// Percentage of the queue's capacity currently occupied.
    pub fn occupancy_pct(&self) -> u32 {
        if self.capacity == 0 {
            return 100;
        }
        let filled = self.entries.len() * 100;
        let pct = filled / self.capacity;
        pct.min(100) as u32
    }
}
