//! Selection of eligible jobs for an offline worker scheduler.

use std::cmp::Reverse;
use std::collections::{BTreeSet, BinaryHeap, HashSet};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum State {
    Ready,
    Running,
    Cooldown { ready_at: u64 },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Job {
    pub id: u64,
    pub priority: u8,
    pub created_at: u64,
    pub expires_at: u64,
    pub state: State,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SelectionError {
    DuplicateId(u64),
    InvalidWindow(u64),
}

/// Select every eligible job, then keep the highest-ranked entries.
pub fn select_for_worker(jobs: &[Job], now: u64, limit: usize) -> Result<Vec<u64>, SelectionError> {
    let mut identifiers = HashSet::new();
    let mut candidates = Vec::new();
    for job in jobs {
        if !identifiers.insert(job.id) {
            return Err(SelectionError::DuplicateId(job.id));
        }
        if job.expires_at <= job.created_at {
            return Err(SelectionError::InvalidWindow(job.id));
        }
        if job.created_at > now || job.expires_at <= now {
            continue;
        }
        let available = match job.state {
            State::Ready => true,
            State::Running => false,
            State::Cooldown { ready_at } => ready_at <= now,
        };
        if available {
            candidates.push(job);
        }
    }
    candidates.sort_by(|a, b| {
        b.priority
            .cmp(&a.priority)
            .then_with(|| a.created_at.cmp(&b.created_at))
            .then_with(|| a.id.cmp(&b.id))
    });
    candidates.truncate(limit);
    Ok(candidates.into_iter().map(|job| job.id).collect())
}

/// Retain only the best eligible ranks in a bounded priority heap.
pub fn reserve_candidates(
    jobs: &[Job],
    now: u64,
    limit: usize,
) -> Result<Vec<u64>, SelectionError> {
    type Rank = (u8, Reverse<u64>, Reverse<u64>);
    let mut known_ids = BTreeSet::new();
    let mut best: BinaryHeap<Reverse<Rank>> = BinaryHeap::new();
    for item in jobs {
        if !known_ids.insert(item.id) {
            return Err(SelectionError::DuplicateId(item.id));
        }
        if item.expires_at <= item.created_at {
            return Err(SelectionError::InvalidWindow(item.id));
        }
        let is_ready = match item.state {
            State::Running => false,
            State::Cooldown { ready_at } if ready_at > now => false,
            _ => true,
        };
        if !is_ready || now < item.created_at || now >= item.expires_at || limit == 0 {
            continue;
        }
        let rank = (item.priority, Reverse(item.created_at), Reverse(item.id));
        if best.len() < limit {
            best.push(Reverse(rank));
        } else if best.peek().is_some_and(|worst| rank > worst.0) {
            let _ = best.pop();
            best.push(Reverse(rank));
        }
    }
    let mut result = Vec::with_capacity(best.len());
    while let Some(Reverse((_, _, Reverse(id)))) = best.pop() {
        result.push(id);
    }
    result.reverse();
    Ok(result)
}

/// Run the small scheduler path exposed by the native fixture CLI.
pub fn demo_selection() -> Result<Vec<u64>, SelectionError> {
    let jobs = [
        Job {
            id: 1,
            priority: 3,
            created_at: 0,
            expires_at: 100,
            state: State::Ready,
        },
        Job {
            id: 2,
            priority: 7,
            created_at: 1,
            expires_at: 100,
            state: State::Ready,
        },
    ];
    reserve_candidates(&jobs, 10, 1)
}
