use crate::constraint::{Constraint, ConstraintType};
use crate::persistence::grad_bounds;
use crate::qubo::Qubo;
use ndarray::Array1;
use std::collections::HashMap;

/// Helper function to get d_ih, where d_ih = p.q[i,h] + p.q[h,i]
pub fn get_dih(p: &Qubo, i: usize, h: usize) -> f64 {
    let q_ih = *(p.q.get(i, h).unwrap_or(&0.0));
    let q_hi = *(p.q.get(i, h).unwrap_or(&0.0));
    q_ih + q_hi
}

/// Implement Rule 1.1 from the paper glover2018
///
/// Rule was originally implemented as the following
/// Assume d_ih > 0, if c_i +d_ih + D-_ >= 0 then x_i >= x_h
///
/// The rule we are implimenting is the following (this is due to us doing min while they are doing max)
/// Assume d_ih < 0, if c_i +d_ih + D+_ > 0 then x_i <= x_h
pub fn generate_rule_11(p: &Qubo, fixed: &HashMap<usize, usize>, i: usize) -> Vec<Constraint> {
    let d_i = (0..p.num_x())
        .map(|h| get_dih(p, i, h))
        .collect::<Array1<f64>>();
    let (D_plus, _) = grad_bounds(p, i, fixed);

    let mut generated_rules = vec![];

    for h in 0..p.num_x() {
        if d_i[h] < 0.0 && d_i[h] + D_plus > 0.0 {
            let rule = Constraint::new(i, h, ConstraintType::LessThan);
            generated_rules.push(rule);
        }
    }

    generated_rules
}

/// Implement Rule 2.1 from the paper glover2018
pub fn generate_rule_21(p: &Qubo, fixed: &HashMap<usize, usize>, i: usize) -> Vec<Constraint> {
    let d_i = (0..p.num_x())
        .map(|h| get_dih(p, i, h))
        .collect::<Array1<f64>>();
    let (_, D_minus) = grad_bounds(p, i, fixed);

    let mut generated_rules = vec![];

    for h in 0..p.num_x() {
        if h == i {
            continue;
        }

        if d_i[h] > 0.0 && d_i[h] + D_minus > 0.0 {
            let rule = Constraint::new(i, h, ConstraintType::AtLeastOne);
            generated_rules.push(rule);
        }
    }

    generated_rules
}
