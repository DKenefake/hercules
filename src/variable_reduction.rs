use crate::constraint::{Constraint, ConstraintType};
use crate::qubo::Qubo;
use std::collections::HashMap;
use crate::preprocess::preprocess_qubo;

/// Find Equations and inequalities that can be used to strengthen the QUBO by probing using the presolver
fn find_equations(qubo: &Qubo, fixed_vars: &HashMap<usize, usize>, in_standard_form: bool) -> Vec<Constraint>{

    let mut constraints = Vec::new();
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

            if fixed_vars_0.contains_key(&j) && fixed_vars_1.contains_key(&j) {
                if fixed_vars_1[&j] == 1 && fixed_vars_0[&j] == 0 {
                    constraints.push(Constraint::new(i, j, ConstraintType::Equal));
                }
                else if fixed_vars_0[&j] == 1 && fixed_vars_1[&j] == 0 {
                    constraints.push(Constraint::new(i, j, ConstraintType::ExactlyOne));
                }
            }
        }
    }

    constraints
}
