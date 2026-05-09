use crate::constraint::{Constraint, ConstraintType};
use crate::preprocess::{prepare_preprocess, preprocess_with_prepared, PreparedPreprocess};
use crate::qubo::Qubo;
use ndarray::Array1;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct ProbedEquationSet {
    pub constraints: Vec<Constraint>
}

impl ProbedEquationSet{
    pub fn new(constraints: Vec<Constraint>) -> Self{
        Self{constraints}
    }

}

/// Find Equations and inequalities that can be used to strengthen the QUBO by probing using the presolver
/// This is a probing method that tries to fix each variable to 0 and 1 and see what other variables can be fixed as a result
pub fn probe(
    qubo: &Qubo,
    fixed_vars: &HashMap<usize, usize>,
    in_standard_form: bool,
) -> (ProbedEquationSet, HashMap<usize, usize>) {
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
    fixed_vars: &HashMap<usize, usize>,
    in_standard_form: bool,
    max_candidates: usize,
) -> (ProbedEquationSet, HashMap<usize, usize>) {
    if max_candidates == 0 {
        return (ProbedEquationSet::new(Vec::new()), HashMap::new());
    }

    let candidates = select_probe_candidates(qubo, fixed_vars, max_candidates);
    let prepared = prepare_preprocess(qubo, in_standard_form);

    probe_candidates(&prepared, &candidates, fixed_vars, qubo.num_x())
}

fn probe_candidates(
    prepared: &PreparedPreprocess,
    candidates: &[usize],
    fixed_vars: &HashMap<usize, usize>,
    num_x: usize,
) -> (ProbedEquationSet, HashMap<usize, usize>) {

    let mut constraints = Vec::new();
    let mut new_fixed_vars = HashMap::new();
    let probe_base = fixed_vars.clone();

    for &i in candidates {
        let mut fixed_vars_0 = probe_base.clone();
        let mut fixed_vars_1 = probe_base.clone();

        fixed_vars_0.insert(i, 0);
        fixed_vars_1.insert(i, 1);

        let fixed_vars_0 = preprocess_with_prepared(&prepared, &fixed_vars_0);
        let fixed_vars_1 = preprocess_with_prepared(&prepared, &fixed_vars_1);

        let mut candidate_vars = Vec::with_capacity(fixed_vars_0.len() + fixed_vars_1.len());
        candidate_vars.extend(fixed_vars_0.keys().copied());
        candidate_vars.extend(
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
                    constraints.push(Constraint::new(i.min(j), j.max(i), ConstraintType::ExactlyOne));
                } else if v0 == 0  && v1 == 0 {
                    new_fixed_vars.insert(j, 0);
                } else {
                    new_fixed_vars.insert(j, 1);
                }
            // see if we have any inequalities
            } else {
                if let Some(v0) = val_0 {
                    if v0 == 1 {
                        constraints.push(Constraint::new(i.min(j), j.max(i), ConstraintType::AtLeastOne));
                    } else if v0 == 0 {
                        constraints.push(Constraint::new(i, j, ConstraintType::LessThan));
                    }
                } else if let Some(v1) = val_1 {
                    if v1 == 1 {
                        constraints.push(Constraint::new(i, j, ConstraintType::GreaterThan));
                    } else if v1 == 0 {
                        constraints.push(Constraint::new(i.min(j), j.max(i), ConstraintType::NoMoreThanOne));
                    }
                }
            }
        }
    }

    (ProbedEquationSet{constraints}, new_fixed_vars)
}

fn select_probe_candidates(
    qubo: &Qubo,
    fixed_vars: &HashMap<usize, usize>,
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

    candidates.sort_by(|&i, &j| edge_mass[j].total_cmp(&edge_mass[i]).then_with(|| i.cmp(&j)));
    candidates.truncate(max_candidates.min(candidates.len()));
    candidates
}
