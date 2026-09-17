use crate::constraint::{Constraint, ConstraintType};
use crate::preprocess::{prepare_preprocess, PreparedPreprocess, ProbePreprocess};
use crate::qubo::Qubo;
use crate::FixedVarMap;
use ndarray::Array1;

mod pair_dominance;
pub(crate) use pair_dominance::find_pair_dominance;

#[derive(Debug, Clone)]
pub struct ProbedEquationSet {
    pub constraints: Vec<Constraint>,
}

impl ProbedEquationSet {
    pub const fn new(constraints: Vec<Constraint>) -> Self {
        Self { constraints }
    }
}

/// Probe strict gradient persistencies under both values of each variable.
/// Returned deductions hold for every optimum compatible with the input fixings;
/// arbitrary choices between tied component optima are not used as implications.
pub fn probe(
    qubo: &Qubo,
    fixed_vars: &FixedVarMap,
    in_standard_form: bool,
) -> (ProbedEquationSet, FixedVarMap) {
    let candidates = (0..qubo.num_x())
        .filter(|i| !fixed_vars.contains_key(i))
        .collect::<Vec<_>>();
    let prepared = prepare_preprocess(qubo, in_standard_form);

    probe_candidates(&prepared, &candidates, fixed_vars, qubo.num_x())
}

/// A cheaper probing variant that only explores the top scoring candidate variables.
/// Candidates are ranked by the absolute incident quadratic weight remaining in the QUBO.
pub fn probe_limited(
    qubo: &Qubo,
    fixed_vars: &FixedVarMap,
    in_standard_form: bool,
    max_candidates: usize,
) -> (ProbedEquationSet, FixedVarMap) {
    if max_candidates == 0 {
        return (ProbedEquationSet::new(Vec::new()), FixedVarMap::default());
    }

    let prepared = prepare_preprocess(qubo, in_standard_form);
    probe_limited_with_prepared(&prepared, fixed_vars, max_candidates)
}

pub(crate) fn probe_limited_with_prepared(
    prepared: &PreparedPreprocess,
    fixed_vars: &FixedVarMap,
    max_candidates: usize,
) -> (ProbedEquationSet, FixedVarMap) {
    let qubo = prepared.qubo();
    let candidates = select_probe_candidates(qubo, fixed_vars, max_candidates);
    probe_candidates(prepared, &candidates, fixed_vars, qubo.num_x())
}

fn probe_candidates(
    prepared: &PreparedPreprocess,
    candidates: &[usize],
    fixed_vars: &FixedVarMap,
    num_x: usize,
) -> (ProbedEquationSet, FixedVarMap) {
    let mut constraints = Vec::new();
    let mut new_fixed_vars = FixedVarMap::default();
    if candidates.is_empty() {
        return (ProbedEquationSet { constraints }, new_fixed_vars);
    }
    let mut presolver = ProbePreprocess::new(prepared, fixed_vars);

    for &i in candidates {
        let fixed_vars_0 = presolver.probe(i, 0);
        let fixed_vars_1 = presolver.probe(i, 1);

        let candidate_vars = fixed_vars_0.keys().copied().chain(
            fixed_vars_1
                .keys()
                .copied()
                .filter(|j| !fixed_vars_0.contains_key(j)),
        );

        for j in candidate_vars {
            if j >= num_x || i == j || fixed_vars.contains_key(&j) {
                continue;
            }

            let val_0 = fixed_vars_0.get(&j).copied();
            let val_1 = fixed_vars_1.get(&j).copied();

            // see if we have any equations
            if let (Some(v0), Some(v1)) = (val_0, val_1) {
                if v1 == 1 && v0 == 0 {
                    constraints.push(Constraint::new(i.min(j), j.max(i), ConstraintType::Equal));
                } else if v0 == 1 && v1 == 0 {
                    constraints.push(Constraint::new(
                        i.min(j),
                        j.max(i),
                        ConstraintType::ExactlyOne,
                    ));
                } else if v0 == 0 && v1 == 0 {
                    new_fixed_vars.insert(j, 0);
                } else {
                    new_fixed_vars.insert(j, 1);
                }
            // see if we have any inequalities
            } else {
                if let Some(v0) = val_0 {
                    if v0 == 1 {
                        constraints.push(Constraint::new(
                            i.min(j),
                            j.max(i),
                            ConstraintType::AtLeastOne,
                        ));
                    } else if v0 == 0 {
                        // x_i = 0 => x_j = 0, so x_j <= x_i.
                        constraints.push(Constraint::new(i, j, ConstraintType::GreaterThan));
                    }
                } else if let Some(v1) = val_1 {
                    if v1 == 1 {
                        // x_i = 1 => x_j = 1, so x_i <= x_j.
                        constraints.push(Constraint::new(i, j, ConstraintType::LessThan));
                    } else if v1 == 0 {
                        constraints.push(Constraint::new(
                            i.min(j),
                            j.max(i),
                            ConstraintType::NoMoreThanOne,
                        ));
                    }
                }
            }
        }
    }

    constraints.sort_unstable();
    constraints.dedup();
    (ProbedEquationSet { constraints }, new_fixed_vars)
}

pub(crate) fn select_probe_candidates(
    qubo: &Qubo,
    fixed_vars: &FixedVarMap,
    max_candidates: usize,
) -> Vec<usize> {
    let mut edge_mass = Array1::<f64>::zeros(qubo.num_x());

    for (&value, (i, j)) in &qubo.q {
        if fixed_vars.contains_key(&i) || fixed_vars.contains_key(&j) {
            continue;
        }

        let weight = value.abs();
        edge_mass[i] += weight;
        edge_mass[j] += weight;
    }

    let mut candidates = (0..qubo.num_x())
        .filter(|i| !fixed_vars.contains_key(i))
        .collect::<Vec<_>>();

    candidates.sort_by(|&i, &j| {
        edge_mass[j]
            .total_cmp(&edge_mass[i])
            .then_with(|| i.cmp(&j))
    });
    candidates.truncate(max_candidates.min(candidates.len()));
    candidates
}

#[cfg(test)]
mod tests {
    use super::ProbedEquationSet;
    use crate::constraint::{Constraint, ConstraintType};
    use crate::FixedVarMap as HashMap;

    fn assert_all_optima_satisfy_probes(qubo: &crate::qubo::Qubo, fixed: &HashMap) {
        let (relations, deductions) = super::probe(qubo, fixed, false);
        let mut best = f64::INFINITY;
        let mut optima = Vec::new();
        for mask in 0..(1usize << qubo.num_x()) {
            if !fixed.iter().all(|(&i, &v)| (mask >> i) & 1 == v) {
                continue;
            }
            let x = ndarray::Array1::from_iter((0..qubo.num_x()).map(|i| (mask >> i) & 1));
            let value = qubo.eval_usize(&x);
            if value < best {
                best = value;
                optima.clear();
            }
            if value == best {
                optima.push(mask);
            }
        }
        for mask in optima {
            let full: HashMap = (0..qubo.num_x()).map(|i| (i, (mask >> i) & 1)).collect();
            assert!(deductions.iter().all(|(i, v)| full.get(i) == Some(v)));
            assert!(relations
                .constraints
                .iter()
                .all(|relation| relation.check(&full)));
        }
    }

    #[test]
    fn strict_probe_deductions_hold_for_all_conditional_optima() {
        use smolprng::{JsfLarge, PRNG};
        let mut prng = PRNG {
            generator: JsfLarge::from(487102u64),
        };
        for _ in 0..100 {
            let mut qubo = crate::qubo::Qubo::make_random_qubo(8, &mut prng, 0.4);
            // Exact binary fractions deliberately create tied optima as well.
            qubo.q
                .data_mut()
                .iter_mut()
                .for_each(|v| *v = (*v * 16.0).round());
            qubo.c.mapv_inplace(|v| (v * 8.0).round());
            for fixed in [HashMap::default(), [(0, 1), (3, 0)].into_iter().collect()] {
                assert_all_optima_satisfy_probes(&qubo, &fixed);
            }
        }
    }

    #[test]
    fn probes_do_not_turn_tied_component_completions_into_facts() {
        let mut terms = sprs::TriMat::new((4, 4));
        terms.add_triplet(2, 3, -2.0);
        terms.add_triplet(3, 2, -2.0);
        let qubo =
            crate::qubo::Qubo::new_with_c(terms.to_csr(), ndarray::array![0.0, 1.0, 1.0, 1.0]);
        let (_, fixed) = super::probe(&qubo, &HashMap::default(), false);
        assert_eq!(fixed, [(1, 0)].into_iter().collect());
        assert_all_optima_satisfy_probes(&qubo, &HashMap::default());
    }

    fn contains_constraint(
        set: &ProbedEquationSet,
        x_i: usize,
        x_j: usize,
        constraint_type: ConstraintType,
    ) -> bool {
        let display = Constraint::new(x_i, x_j, constraint_type).to_string();
        set.constraints
            .iter()
            .any(|constraint| constraint.to_string() == display)
    }

    #[test]
    fn greater_than_constraint_matches_zero_to_zero_implication() {
        let set = ProbedEquationSet::new(vec![Constraint::new(2, 5, ConstraintType::GreaterThan)]);
        let mut persistent = HashMap::default();
        persistent.insert(2, 0);

        assert!(contains_constraint(&set, 2, 5, ConstraintType::GreaterThan));
        assert_eq!(set.constraints[0].make_inference(&persistent), Some((5, 0)));
    }

    #[test]
    fn less_than_constraint_matches_one_to_one_implication() {
        let set = ProbedEquationSet::new(vec![Constraint::new(2, 5, ConstraintType::LessThan)]);
        let mut persistent = HashMap::default();
        persistent.insert(2, 1);

        assert!(contains_constraint(&set, 2, 5, ConstraintType::LessThan));
        assert_eq!(set.constraints[0].make_inference(&persistent), Some((5, 1)));
    }
}
