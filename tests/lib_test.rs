mod helpers;
use helpers::{make_tiny_train, make_tiny_probe};

use netflix_prize::{calc_gbias, calc_user_offsets};

#[test]
fn tiny_train_sorted_by_user_date() {
    let ds = make_tiny_train();
    for t in 1..ds.n_ratings {
        let prev = (ds.user_idxs[t - 1], ds.dates[t - 1]);
        let curr = (ds.user_idxs[t], ds.dates[t]);
        assert!(
            curr >= prev,
            "rating {t}: ({}, {}) comes after ({}, {}) but is smaller",
            curr.0, curr.1, prev.0, prev.1
        );
    }
}

#[test]
fn tiny_probe_sorted_by_item() {
    let ds = make_tiny_probe();
    for t in 1..ds.n_ratings {
        assert!(
            ds.item_idxs[t] >= ds.item_idxs[t - 1],
            "rating {t}: item {} comes after item {} but is smaller",
            ds.item_idxs[t], ds.item_idxs[t - 1]
        );
    }
}

#[test]
fn user_item_pairs_unique_across_train_and_probe() {
    use std::collections::HashSet;
    let train = make_tiny_train();
    let probe = make_tiny_probe();
    let mut seen = HashSet::new();
    for ds in [&train, &probe] {
        for t in 0..ds.n_ratings {
            let pair = (ds.user_idxs[t], ds.item_idxs[t]);
            assert!(seen.insert(pair), "duplicate (user, item) pair: {:?}", pair);
        }
    }
}

#[test]
#[should_panic(expected = "not user-sorted")]
fn user_offsets_panics_on_unsorted_dataset() {
    use helpers::make_tiny_probe;
    calc_user_offsets(&make_tiny_probe());
}

#[test]
fn user_offsets_span_equals_user_cnts() {
    let ds = make_tiny_train();
    let offsets = calc_user_offsets(&ds);
    for u in 0..ds.n_users {
        assert_eq!(
            offsets[u + 1] - offsets[u], ds.user_cnts[u] as usize,
            "user {u}"
        );
    }
}

#[test]
fn gbias_equals_residuals_mean() {
    let ds = make_tiny_train();
    let mut sum = 0.0_f64;
    for &r in ds.residuals.iter() {
        sum += r as f64;
    }
    let expected = (sum / ds.n_ratings as f64) as f32;
    assert_eq!(calc_gbias(&ds), expected);
}
