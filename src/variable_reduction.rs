use crate::constraint::{Constraint, ConstraintType};
use crate::preprocess::preprocess_qubo;
use crate::qubo::Qubo;
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

    let mut constraints = Vec::new();
    let mut new_fixed_vars = HashMap::new();
    let n = qubo.num_x();

    for i in 0..n {
        if fixed_vars.contains_key(&i) {
            continue;
        }

        let mut fixed_vars_0 = fixed_vars.clone();
        let mut fixed_vars_1 = fixed_vars.clone();

        fixed_vars_0.insert(i, 0);
        fixed_vars_1.insert(i, 1);

        let fixed_vars_0 = preprocess_qubo(qubo, &fixed_vars_0, in_standard_form);
        let fixed_vars_1 = preprocess_qubo(qubo, &fixed_vars_1, in_standard_form);

        for j in 0..n {
            if i == j || fixed_vars.contains_key(&j) {
                continue;
            }

            // see if we have any equations
            if fixed_vars_0.contains_key(&j) && fixed_vars_1.contains_key(&j) {
                if fixed_vars_1[&j] == 1 && fixed_vars_0[&j] == 0 {
                    constraints.push(Constraint::new(i.min(j), j.max(i), ConstraintType::Equal));
                } else if fixed_vars_0[&j] == 1 && fixed_vars_1[&j] == 0 {
                    constraints.push(Constraint::new(i.min(j), j.max(i), ConstraintType::ExactlyOne));
                } else if fixed_vars_0[&j] == 0  && fixed_vars_1[&j] == 0 {
                    new_fixed_vars.insert(j, 0);
                } else { new_fixed_vars.insert(j, 1); }
            // see if we have any inequalities
            }else{
                if fixed_vars_0.contains_key(&j) {
                    if fixed_vars_0[&j] == 1 {
                        constraints.push(Constraint::new(i.min(j), j.max(i), ConstraintType::AtLeastOne));
                    } else if fixed_vars_0[&j] == 0 {
                        constraints.push(Constraint::new(i, j, ConstraintType::LessThan));
                    }
                }
                else if fixed_vars_1.contains_key(&j) {
                    if fixed_vars_1[&j] == 1 {
                        constraints.push(Constraint::new(i, j, ConstraintType::GreaterThan));
                    } else if fixed_vars_1[&j] == 0 {
                        constraints.push(Constraint::new(i.min(j), j.max(i), ConstraintType::NoMoreThanOne));
                    }
                }
            }
        }
    }

    (ProbedEquationSet{constraints}, new_fixed_vars)
}
