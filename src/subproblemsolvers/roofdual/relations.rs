//! Nonnegative posiform penalties for proven root optimality relations.
//!
//! Each forbidden pair contributes M [x_i=a][x_j=b]. Combine these terms with
//! the original binary polynomial once at the root. Leaving opposing terms as
//! separate graph arcs misses cancellations and can give a weaker relaxation.

use crate::constraint::Constraint;

#[derive(Clone, Default)]
pub(super) struct RelationPenalties {
    model: Option<PenalizedModel>,
}

#[derive(Clone)]
struct PenalizedModel {
    linear: Vec<f64>,
    edges: Vec<(usize, usize, f64)>,
    constant: f64,
}

impl RelationPenalties {
    pub fn new(linear: &[f64], edges: &[(usize, usize, f64)], relations: &[Constraint]) -> Self {
        if relations.is_empty() {
            return Self::default();
        }
        let mut forbidden = Vec::new();
        for relation in relations {
            for (a, b) in relation.forbidden_assignments() {
                let (i, j) = (relation.x_i, relation.x_j);
                forbidden.push(if i <= j { (i, a, j, b) } else { (j, b, i, a) });
            }
        }
        forbidden.sort_unstable();
        forbidden.dedup();
        // R bounds the objective spread on every binary node domain. M>R
        // ensures a relation-violating point cannot beat a feasible completion.
        let range: f64 = linear.iter().map(|v| v.abs()).sum::<f64>()
            + edges.iter().map(|&(_, _, v)| v.abs()).sum::<f64>();
        let penalty = 2.0 * (range + (range * 1e-12).max(1.0));
        if !penalty.is_finite() {
            return Self::default();
        }
        let mut linear = linear.to_vec();
        let mut pairs: rustc_hash::FxHashMap<_, _> =
            edges.iter().map(|&(i, j, w)| ((i, j), w)).collect();
        let mut constant = 0.0;
        for (i, a, j, b) in forbidden {
            if i == j {
                if a == 0 {
                    constant += penalty;
                    linear[i] -= penalty;
                } else {
                    linear[i] += penalty;
                }
                continue;
            }
            match (a, b) {
                (0, 0) => {
                    constant += penalty;
                    linear[i] -= penalty;
                    linear[j] -= penalty;
                    *pairs.entry((i, j)).or_default() += penalty;
                }
                (1, 1) => *pairs.entry((i, j)).or_default() += penalty,
                (1, 0) => {
                    linear[i] += penalty;
                    *pairs.entry((i, j)).or_default() -= penalty;
                }
                (0, 1) => {
                    linear[j] += penalty;
                    *pairs.entry((i, j)).or_default() -= penalty;
                }
                _ => unreachable!(),
            }
        }
        if !constant.is_finite()
            || linear.iter().any(|v| !v.is_finite())
            || pairs.values().any(|v| !v.is_finite())
        {
            return Self::default();
        }
        let mut edges: Vec<_> = pairs
            .into_iter()
            .filter(|(_, w)| *w != 0.0)
            .map(|((i, j), w)| (i, j, w))
            .collect();
        edges.sort_unstable_by_key(|&(i, j, _)| (i, j));
        Self {
            model: Some(PenalizedModel {
                linear,
                edges,
                constant,
            }),
        }
    }

    pub fn coefficients<'a>(
        &'a self,
        linear: &'a [f64],
        edges: &'a [(usize, usize, f64)],
    ) -> (&'a [f64], &'a [(usize, usize, f64)], f64) {
        self.model.as_ref().map_or((linear, edges, 0.0), |model| {
            (&model.linear, &model.edges, model.constant)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraint::{ConstraintType, ImplicationGraph};
    use crate::qubo::Qubo;
    use crate::subproblemsolvers::roofdual::PreparedRoofDual;
    use crate::FixedVarMap;
    use ndarray::Array1;
    use smolprng::{JsfLarge, PRNG};

    const KINDS: [ConstraintType; 6] = [
        ConstraintType::AtLeastOne,
        ConstraintType::ExactlyOne,
        ConstraintType::NoMoreThanOne,
        ConstraintType::GreaterThan,
        ConstraintType::LessThan,
        ConstraintType::Equal,
    ];

    #[test]
    fn canonical_penalty_projection_matches_every_truth_table_and_fixing() {
        let q = sprs::TriMat::<f64>::new((2, 2)).to_csr();
        let qubo = Qubo::new_with_c(q, ndarray::array![-2.0, 3.0]);
        let mut prepared = PreparedRoofDual::new(&qubo);
        for j in 0..2 {
            for kind in KINDS {
                let relation = Constraint::new(0, j, kind);
                prepared.set_strong_relations(&[relation, relation]);
                for pattern in 0..9 {
                    let fixed: FixedVarMap = [(0, pattern % 3), (1, pattern / 3)]
                        .into_iter()
                        .filter(|(_, v)| *v != 2)
                        .collect();
                    let reduced = prepared.project(&fixed);
                    for mask in 0..4 {
                        let full: FixedVarMap =
                            [(0, mask & 1), (1, mask >> 1)].into_iter().collect();
                        if !fixed.iter().all(|(i, v)| full[i] == *v) {
                            continue;
                        }
                        let mut assignment = vec![1; reduced.num_variables + 1];
                        for (&i, &r) in &reduced.original_to_reduced {
                            assignment[r + 1] = full[&i];
                        }
                        let value = reduced.constant
                            + reduced
                                .biterms
                                .iter()
                                .map(|term| {
                                    let equal = assignment[term.i] == assignment[term.j];
                                    let active = match term.kind {
                                crate::subproblemsolvers::roofdual::BiTermKind::Equal => !equal,
                                crate::subproblemsolvers::roofdual::BiTermKind::Different => equal,
                            };
                                    term.weight * usize::from(active) as f64
                                })
                                .sum::<f64>();
                        let expected = if relation.check(&full) {
                            0.0
                        } else {
                            12.0 // M=2*(|-2|+|3|+1), duplicate relations count once
                        };
                        assert_eq!(
                            value - qubo.eval_usize(&Array1::from_iter((0..2).map(|i| full[&i]))),
                            expected
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn relation_bounds_and_fixings_preserve_conditional_constrained_optima() {
        let mut rng = PRNG {
            generator: JsfLarge::from(194020u64),
        };
        for sample in 0..128 {
            let mut qubo = Qubo::make_random_qubo(7, &mut rng, 0.5);
            let scale = [1.0, 0.125, 0.013, 3.7][sample % 4];
            qubo.q
                .data_mut()
                .iter_mut()
                .for_each(|v| *v = (*v * 8.0).round() * scale);
            qubo.c.mapv_inplace(|v| (v * 8.0).round() * scale);
            let witness: FixedVarMap = (0..7).map(|i| (i, (sample >> i) & 1)).collect();
            let relations: Vec<_> = (0..6)
                .map(|i| Constraint::new(i, (i + 2) % 7, KINDS[(sample + i) % 6]))
                .filter(|r| r.check(&witness))
                .collect();
            let bare = PreparedRoofDual::new(&qubo);
            let mut augmented = bare.clone();
            augmented.set_strong_relations(&relations);
            for fixed in [
                FixedVarMap::default(),
                [(0, 0)].into_iter().collect(),
                [(0, 1), (2, 0)].into_iter().collect(),
            ] {
                let feasible: Vec<_> = (0..1usize << 7)
                    .filter_map(|mask| {
                        let full: FixedVarMap = (0..7).map(|i| (i, (mask >> i) & 1)).collect();
                        if fixed.iter().any(|(i, v)| full[i] != *v)
                            || relations.iter().any(|r| !r.check(&full))
                        {
                            return None;
                        }
                        let x = Array1::from_iter((0..7).map(|i| full[&i]));
                        Some((full, qubo.eval_usize(&x)))
                    })
                    .collect();
                if feasible.is_empty() {
                    continue;
                }
                let best = feasible
                    .iter()
                    .map(|(_, value)| *value)
                    .fold(f64::INFINITY, f64::min);
                let optima: Vec<_> = feasible
                    .iter()
                    // Scaling decimal coefficients changes the last bits of
                    // otherwise tied objective sums, not the reference optimum.
                    .filter(|(_, value)| (*value - best).abs() < 1e-12)
                    .collect();
                let original = bare.solve(&fixed).lower_bound.unwrap();
                assert!(augmented.solve(&fixed).lower_bound.unwrap() >= original - 1e-9);
                for weak in [false, true] {
                    let result = if weak {
                        augmented.solve_iterative_with_weak_persistencies(&fixed, 7)
                    } else {
                        augmented.solve_iterative(&fixed, 7)
                    };
                    assert!(result.lower_bound.unwrap() <= best + 1e-9);
                    assert!(optima.iter().all(|(full, _)| result
                        .fixed_variables
                        .iter()
                        .all(|(i, v)| full[i] == *v)));
                    assert!(
                        optima.iter().any(|(full, _)| result
                            .fixed_variables
                            .iter()
                            .chain(&result.weak_fixed_variables)
                            .all(|(i, v)| full[i] == *v)),
                        "sample={sample} scale={scale} weak={weak} fixed={fixed:?} best={best:?} strong={:?} weak_fixings={:?}",
                        result.fixed_variables, result.weak_fixed_variables
                    );
                }
            }
            augmented.set_strong_relations(&[]);
            assert_eq!(
                augmented.solve(&FixedVarMap::default()).lower_bound,
                bare.solve(&FixedVarMap::default()).lower_bound
            );
        }
    }

    #[test]
    fn relations_strengthen_a_bound_even_before_any_variable_can_propagate() {
        // Unconstrained f=-x0-x1 has roof bound -2. The additional valid domain
        // relation x0+x1<=1 strengthens it to -1 without an incoming fixation.
        let qubo = Qubo::new_with_c(sprs::CsMat::zero((2, 2)), ndarray::array![-1.0, -1.0]);
        let relation = Constraint::new(0, 1, ConstraintType::NoMoreThanOne);
        let mut fixed = FixedVarMap::default();
        assert!(ImplicationGraph::new(2, &[relation]).propagate(&mut fixed));
        assert!(fixed.is_empty());
        let mut prepared = PreparedRoofDual::new(&qubo);
        assert_eq!(prepared.solve(&fixed).lower_bound, Some(-2.0));
        prepared.set_strong_relations(&[relation]);
        assert_eq!(prepared.solve(&fixed).lower_bound, Some(-1.0));
    }

    #[test]
    fn joint_dominance_relations_tighten_real_unconstrained_qubos_safely() {
        let mut rng = PRNG {
            generator: JsfLarge::from(602411u64),
        };
        let mut improved = 0;
        for _ in 0..256 {
            let mut qubo = Qubo::make_random_qubo(8, &mut rng, 0.5);
            qubo.q
                .data_mut()
                .iter_mut()
                .for_each(|v| *v = (*v * 8.0).round());
            qubo.c.mapv_inplace(|v| (v * 8.0).round());
            let fixed = FixedVarMap::default();
            let (mut relations, _) = crate::variable_reduction::probe(&qubo, &fixed, false);
            relations
                .constraints
                .extend(crate::variable_reduction::find_pair_dominance(
                    &crate::preprocess::prepare_preprocess(&qubo, false),
                    &fixed,
                    std::time::Duration::from_secs(1),
                ));
            let mut prepared = PreparedRoofDual::new(&qubo);
            let before = prepared.solve(&fixed).lower_bound.unwrap();
            prepared.set_strong_relations(&relations.constraints);
            let after = prepared.solve(&fixed);
            let best = crate::subproblemsolvers::enumerate_qubo::enumerate_solve(&qubo).0;
            assert!(after.lower_bound.unwrap() <= best + 1e-9);
            assert!(after.lower_bound.unwrap() >= before - 1e-9);
            improved += usize::from(after.lower_bound.unwrap() > before + 1e-9);
        }
        assert!(
            improved > 0,
            "joint pair dominance should sometimes strengthen the roof bound"
        );
    }
}
