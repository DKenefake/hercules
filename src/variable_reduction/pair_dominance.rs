//! Strict two-variable derivative/co-derivative persistencies (Boros et al.,
//! RRR 10-2006, Section 3.2). Only root processing uses this bounded pass.

use crate::constraint::{Constraint, ConstraintType};
use crate::preprocess::PreparedPreprocess;
use crate::FixedVarMap;
use std::time::{Duration, Instant};

pub(crate) fn find_pair_dominance(
    prepared: &PreparedPreprocess,
    fixed: &FixedVarMap,
    budget: Duration,
) -> Vec<Constraint> {
    let start = Instant::now();
    if budget.is_zero() {
        return Vec::new();
    }
    let n = prepared.qubo().num_x();
    let values: Vec<_> = (0..n).map(|i| fixed.get(&i).copied()).collect();
    let free: Vec<_> = (0..n).filter(|&i| values[i].is_none()).collect();
    let mut linear = prepared.qubo().c.to_vec();
    let mut rows = vec![Vec::new(); n];
    let mut mass = vec![0.0; n];
    for &i in &free {
        if start.elapsed() >= budget {
            return Vec::new();
        }
        mass[i] = linear[i].abs();
        for &(j, weight) in &prepared.adjacency()[i] {
            mass[i] += weight.abs();
            if let Some(value) = values[j] {
                linear[i] += weight * value as f64;
            } else {
                rows[i].push((j, weight));
            }
        }
        rows[i].sort_unstable_by_key(|&(j, _)| j);
        rows[i].dedup_by(|a, b| {
            if a.0 == b.0 {
                b.1 += a.1;
                true
            } else {
                false
            }
        });
        rows[i].retain(|&(_, weight)| weight != 0.0);
    }
    let mut found = Vec::new();
    let mut pairs = 0;
    for (offset, &i) in free.iter().enumerate() {
        for &j in &free[offset + 1..] {
            // Bound both runtime and work on very large sparse instances.
            if pairs >= 250_000 || (pairs % 64 == 0 && start.elapsed() >= budget) {
                return found;
            }
            pairs += 1;
            let edge = rows[i]
                .binary_search_by_key(&j, |&(k, _)| k)
                .map_or(0.0, |position| rows[i][position].1);
            let sum = linear[i] + linear[j] + edge;
            let diff = linear[i] - linear[j];
            let (mut sum_lo, mut sum_hi, mut diff_lo, mut diff_hi) = (sum, sum, diff, diff);
            let tolerance = 1e-10 * (mass[i] + mass[j]).max(1.0);
            let (mut a, mut b) = (0, 0);
            while a < rows[i].len() || b < rows[j].len() {
                let left = rows[i].get(a).copied().unwrap_or((usize::MAX, 0.0));
                let right = rows[j].get(b).copied().unwrap_or((usize::MAX, 0.0));
                let k = left.0.min(right.0);
                let wi = if left.0 == k {
                    a += 1;
                    left.1
                } else {
                    0.0
                };
                let wj = if right.0 == k {
                    b += 1;
                    right.1
                } else {
                    0.0
                };
                if k == i || k == j {
                    continue;
                }
                // Combine first: opposite shared-neighbor terms can cancel.
                sum_lo += (wi + wj).min(0.0);
                sum_hi += (wi + wj).max(0.0);
                diff_lo += (wi - wj).min(0.0);
                diff_hi += (wi - wj).max(0.0);
                if sum_lo <= tolerance
                    && sum_hi >= -tolerance
                    && diff_lo <= tolerance
                    && diff_hi >= -tolerance
                {
                    break;
                }
            }
            // f(11,z)-f(00,z): a strict sign excludes one of these two patterns.
            if sum_lo > tolerance {
                found.push(Constraint::new(i, j, ConstraintType::NoMoreThanOne));
            } else if sum_hi < -tolerance {
                found.push(Constraint::new(i, j, ConstraintType::AtLeastOne));
            }
            // f(10,z)-f(01,z), with the other variables z unchanged.
            if diff_lo > tolerance {
                found.push(Constraint::new(i, j, ConstraintType::LessThan));
            } else if diff_hi < -tolerance {
                found.push(Constraint::new(i, j, ConstraintType::GreaterThan));
            }
        }
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preprocess::prepare_preprocess;
    use crate::qubo::Qubo;
    use ndarray::Array1;
    use smolprng::{JsfLarge, PRNG};

    #[test]
    fn joint_test_uses_cancellation_between_shared_neighbors() {
        let mut q = sprs::TriMat::new((3, 3));
        q.add_triplet(0, 2, 20.0);
        q.add_triplet(1, 2, -20.0);
        let qubo = Qubo::new_with_c(q.to_csr(), ndarray::array![0.5, 0.5, 0.0]);
        let prepared = prepare_preprocess(&qubo, false);
        let relations =
            find_pair_dominance(&prepared, &FixedVarMap::default(), Duration::from_secs(1));
        assert!(relations.contains(&Constraint::new(0, 1, ConstraintType::NoMoreThanOne)));
        assert!(find_pair_dominance(&prepared, &FixedVarMap::default(), Duration::ZERO).is_empty());
    }

    #[test]
    fn strict_pair_rules_hold_for_every_conditional_optimum() {
        let mut rng = PRNG {
            generator: JsfLarge::from(881991u64),
        };
        for sample in 0..128 {
            let mut qubo = Qubo::make_random_qubo(8, &mut rng, 0.5);
            qubo.q
                .data_mut()
                .iter_mut()
                .for_each(|v| *v = (*v * 8.0).round());
            qubo.c.mapv_inplace(|v| (v * 8.0).round());
            if sample % 2 == 0 {
                qubo.q = qubo.q.to_csc();
            }
            let prepared = prepare_preprocess(&qubo, false);
            for fixed in [
                FixedVarMap::default(),
                [(0, 1), (2, 0)].into_iter().collect(),
            ] {
                let relations = find_pair_dominance(&prepared, &fixed, Duration::from_secs(1));
                let mut best = f64::INFINITY;
                let mut optima = Vec::new();
                for mask in 0..1usize << 8 {
                    let full: FixedVarMap = (0..8).map(|i| (i, (mask >> i) & 1)).collect();
                    if fixed.iter().any(|(i, v)| full[i] != *v) {
                        continue;
                    }
                    let value = qubo.eval_usize(&Array1::from_iter((0..8).map(|i| full[&i])));
                    if value < best {
                        best = value;
                        optima.clear();
                    }
                    if value == best {
                        optima.push(full);
                    }
                }
                assert!(optima
                    .iter()
                    .all(|full| relations.iter().all(|relation| relation.check(full))));
            }
        }
    }

    #[test]
    fn tied_pair_comparisons_do_not_generate_strong_relations() {
        let qubo = Qubo::new(sprs::CsMat::zero((3, 3)));
        assert!(find_pair_dominance(
            &prepare_preprocess(&qubo, false),
            &FixedVarMap::default(),
            Duration::from_secs(1)
        )
        .is_empty());
    }
}
