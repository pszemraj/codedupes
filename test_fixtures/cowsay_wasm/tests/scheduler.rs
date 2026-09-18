use cowsay_dupe_fixture::scheduler::{
    reserve_candidates, select_for_worker, Job, SelectionError, State,
};

fn job(id: u64, priority: u8, created_at: u64) -> Job {
    Job {
        id,
        priority,
        created_at,
        expires_at: 100,
        state: State::Ready,
    }
}

#[test]
fn ranking_and_ties_have_a_concrete_oracle() {
    let jobs = vec![job(4, 9, 2), job(3, 9, 1), job(2, 9, 1), job(1, 2, 0)];
    assert_eq!(select_for_worker(&jobs, 10, 3), Ok(vec![2, 3, 4]));
    assert_eq!(reserve_candidates(&jobs, 10, 3), Ok(vec![2, 3, 4]));
}

#[test]
fn state_and_expiry_boundaries() {
    let mut jobs = vec![job(1, 5, 0), job(2, 9, 0), job(3, 8, 0), job(4, 7, 0)];
    jobs[1].state = State::Running;
    jobs[2].state = State::Cooldown { ready_at: 10 };
    jobs[3].expires_at = 10;
    assert_eq!(select_for_worker(&jobs, 10, 10), Ok(vec![3, 1]));
    assert_eq!(reserve_candidates(&jobs, 10, 10), Ok(vec![3, 1]));
    jobs[2].state = State::Cooldown { ready_at: 11 };
    assert_eq!(select_for_worker(&jobs, 10, 10), Ok(vec![1]));
    assert_eq!(reserve_candidates(&jobs, 10, 10), Ok(vec![1]));
}

#[test]
fn empty_zero_limit_and_future_work() {
    assert_eq!(select_for_worker(&[], 0, 4), Ok(vec![]));
    assert_eq!(reserve_candidates(&[], 0, 4), Ok(vec![]));
    let jobs = vec![job(1, 5, 0), job(2, 9, 11)];
    assert_eq!(select_for_worker(&jobs, 10, 0), Ok(vec![]));
    assert_eq!(reserve_candidates(&jobs, 10, 0), Ok(vec![]));
    assert_eq!(select_for_worker(&jobs, 10, 10), Ok(vec![1]));
    assert_eq!(reserve_candidates(&jobs, 10, 10), Ok(vec![1]));
}

#[test]
fn validation_precedes_policy_filtering() {
    let repeated = vec![job(7, 1, 0), job(7, 2, 1)];
    for limit in [0, 1, 10] {
        assert_eq!(
            select_for_worker(&repeated, 200, limit),
            Err(SelectionError::DuplicateId(7))
        );
        assert_eq!(
            reserve_candidates(&repeated, 200, limit),
            Err(SelectionError::DuplicateId(7))
        );
    }
    let mut bad = job(8, 1, 0);
    bad.expires_at = 0;
    assert_eq!(
        select_for_worker(std::slice::from_ref(&bad), 200, 0),
        Err(SelectionError::InvalidWindow(8))
    );
    assert_eq!(
        reserve_candidates(&[bad], 200, 0),
        Err(SelectionError::InvalidWindow(8))
    );
}

#[test]
fn generated_sets_preserve_inputs_and_match_expected_prefixes() {
    for count in 0_u8..80 {
        let jobs: Vec<Job> = (0..count)
            .map(|id| Job {
                id: u64::from(id),
                priority: id.wrapping_mul(7) % 13,
                created_at: u64::from(id % 11),
                expires_at: 100,
                state: match id % 5 {
                    0 => State::Running,
                    1 => State::Cooldown { ready_at: 10 },
                    2 => State::Cooldown { ready_at: 11 },
                    _ => State::Ready,
                },
            })
            .collect();
        let snapshot = jobs.clone();
        let expected = select_for_worker(&jobs, 10, usize::MAX).unwrap();
        for limit in [0, 1, 2, 5, 15, 100] {
            let prefix: Vec<u64> = expected.iter().take(limit).copied().collect();
            assert_eq!(select_for_worker(&jobs, 10, limit), Ok(prefix.clone()));
            assert_eq!(reserve_candidates(&jobs, 10, limit), Ok(prefix.clone()));
            let reversed: Vec<Job> = jobs.iter().rev().cloned().collect();
            assert_eq!(reserve_candidates(&reversed, 10, limit), Ok(prefix));
            assert_eq!(jobs, snapshot);
        }
    }
}
