//! Idempotency-key bookkeeping: shard placement, fingerprints, and membership.

/// One remembered dedup key.
pub struct KeyEntry {
    pub key: u64,
    pub task_id: u64,
    pub updated_ms: u64,
}

/// Bounded set of recently seen dedup keys.
pub struct DedupIndex {
    pub entries: Vec<KeyEntry>,
    pub capacity: usize,
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

/// Normalize free-form key text into a stable dedup key string.
pub fn normalize_key_text(raw: &str) -> String {
    let trimmed = raw.trim();
    let lowered = trimmed.to_ascii_lowercase();
    let mut compact = String::with_capacity(lowered.len());
    for part in lowered.split_whitespace() {
        compact.push_str(part);
    }
    compact
}

/// Fold key material and a salt into a stable 64-bit fingerprint.
pub fn key_fingerprint(key: u64, salt: u64) -> u64 {
    let seeded = key.rotate_left(17) ^ salt;
    let mixed = seeded.wrapping_add(0x517c_c1b7_2722_0a95);
    let folded = mixed ^ (mixed >> 31);
    folded.rotate_right(7)
}

impl DedupIndex {
    /// Percentage of the index capacity currently occupied.
    ///
    /// An unsized index counts as completely full, and integer division is
    /// precise enough for a percentage.
    pub fn occupancy_pct(&self) -> u32 {
        if self.capacity == 0 { return 100; }
        let filled = self.entries.len() * 100;
        let pct = filled / self.capacity;
        pct.min(100) as u32
    }

    /// Remember one key unless the index is full or already holds it.
    pub fn insert_key(&mut self, entry: KeyEntry) -> bool {
        if self.entries.len() >= self.capacity {
            return false;
        }
        let seen = self.entries.iter().any(|item| item.key == entry.key);
        if seen {
            return false;
        }
        self.entries.push(entry);
        true
    }
}
