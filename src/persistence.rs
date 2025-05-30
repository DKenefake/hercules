use crate::qubo::Qubo;
use std::cmp::min;
use std::collections::HashMap;
use ndarray::Array1;

/// This function takes a QUBO and a set of persistent variables and returns a new set of persistent variables by repeatedly re
/// computing the persistent variables until.
pub fn compute_iterative_persistence(
    qubo: &Qubo,
    persistent: &HashMap<usize, usize>,
    iter_lim: usize,
) -> HashMap<usize, usize> {
    // make a copy of the passed persistence variable
    let mut new_persistent = persistent.clone();

    // the number of required iterations is always below the number of variables
    let iters = min(iter_lim, qubo.num_x());

    // loop over the number of iters
    for _ in 0..iters {
        let incoming_persistent = compute_persistent(qubo, &new_persistent, true);
        if new_persistent == incoming_persistent {
            break;
        }
        new_persistent = incoming_persistent;
    }

    new_persistent
}

/// This function takes a QUBO and a set of persistent variables and returns a new set of persistent variables by computing the
/// persistent variables once.
pub fn compute_persistent(
    qubo: &Qubo,
    persistent: &HashMap<usize, usize>,
    keep_vars: bool,
) -> HashMap<usize, usize> {
    // create a new hashmap to store the new persistent variables
    let mut new_persistent = persistent.clone();

    // iterate over all the variables in the QUBO
    for i in 0..qubo.num_x() {
        if persistent.contains_key(&i) {
            continue;
        }

        // find the bounds of the gradient in each direction
        let (lower, upper) = grad_bounds(qubo, i, persistent, keep_vars);

        // if the lower bound it positive, then we can set the variable to 0
        if lower > 0.0 {
            new_persistent.insert(i, 0);
        }

        // if the upper bound is below 0, then we can set the variable to 1
        if upper < 0.0 {
            new_persistent.insert(i, 1);
        }
    }

    new_persistent
}

pub fn all_grad_bounds(
    qubo: &Qubo,
    persistent: &HashMap<usize, usize>,
    keep_vars: bool,
) -> (Array1<f64>, Array1<f64>) {
    let mut lb_bounds = Array1::<f64>::zeros(qubo.num_x());
    let mut ub_bounds = Array1::<f64>::zeros(qubo.num_x());

    for i in 0..qubo.num_x() {
        let x = grad_bounds(qubo, i, persistent, keep_vars);
        lb_bounds[i] = x.0;
        ub_bounds[i] = x.1;
    }

    (lb_bounds, ub_bounds)
}



/// Finds bounds of the i-th index of the gradients of the QUBO function
///
/// # Panics
/// This function should not panic as the unwraps are bounded on the size of the QUBO matrix
pub fn grad_bounds(
    qubo: &Qubo,
    i: usize,
    persistent: &HashMap<usize, usize>,
    keep_vars: bool,
) -> (f64, f64) {
    // set up tracking variables for each bound
    let mut lower = 0.0;
    let mut upper = 0.0;

    // get the i-th row of the Q matrix, this is safe as we are bounded by the size of the Q matrix
    let x = qubo.q.outer_view(i).unwrap();

    // get the i-th column of the Q matrix, this is safe for the same reason
    let binding = qubo.q.transpose_view();
    let y = binding.outer_view(i).unwrap();

    // add the row and column together
    let q_term = x + y;

    // loop over the variables in this row
    for (index, x_j) in q_term.iter() {
        // dereference and multiply by 0.5 to make an auxiliary variable that is clearer
        let mut value = *x_j;
        value *= 0.5;

        // if it is a fixed variable, we have effectively removed this variable from the QUBO
        if persistent.contains_key(&index) {
            if keep_vars {
                lower += value * (persistent[&index] as f64);
                upper += value * (persistent[&index] as f64);
            }
        } else {
            // if it is not in the persistent set, then we can choose the best value

            // if the value is negative, we would set it to 1 to minimize the gradient else 0
            if value <= 0.0 {
                lower += value;
            }

            // if the value is positive, we would set it to 1 to maximize the gradient else 0
            if value >= 0.0 {
                upper += value;
            }
        }
    }

    // add the contribution from the constant term
    lower += qubo.c[i];
    upper += qubo.c[i];

    (lower, upper)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qubo::Qubo;
    use crate::tests::make_solver_qubo;
    use ndarray::Array1;
    use sprs::CsMat;
    use std::collections::HashMap;

    #[test]
    fn test_persistence() {
        //build the problem
        let eye = CsMat::eye(3);
        let c = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let p = Qubo::new_with_c(eye, c);
        let persist = compute_iterative_persistence(&p, &HashMap::new(), 3);

        assert!(persist.contains_key(&0));
        assert!(persist.contains_key(&1));
        assert!(persist.contains_key(&2));

        assert!(persist[&0].eq(&0));
        assert!(persist[&1].eq(&0));
        assert!(persist[&2].eq(&0));
    }
    #[test]
    fn test_grad_bounds_1() {
        let eye = CsMat::eye(3);
        let c = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let p = Qubo::new_with_c(eye, c);
        assert_eq!(grad_bounds(&p, 0, &HashMap::new(), true), (1.0, 2.0));
        assert_eq!(grad_bounds(&p, 1, &HashMap::new(), true), (2.0, 3.0));
        assert_eq!(grad_bounds(&p, 2, &HashMap::new(), true), (3.0, 4.0));
    }

    #[test]
    fn test_grad_bounds_2() {
        let eye = CsMat::eye(3);
        let c = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let p = Qubo::new_with_c(eye, c);

        let mut fixed_vars = HashMap::new();
        fixed_vars.insert(0, 1);
        fixed_vars.insert(2, 1);

        assert_eq!(grad_bounds(&p, 0, &fixed_vars, true), (2.0, 2.0));
        assert_eq!(grad_bounds(&p, 1, &fixed_vars, true), (2.0, 3.0));
        assert_eq!(grad_bounds(&p, 2, &fixed_vars, true), (4.0, 4.0));
    }

    #[test]
    fn test_grad_bounds_3() {
        let zero = CsMat::zero((3, 3));
        let c = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let p = Qubo::new_with_c(zero, c);

        assert_eq!(grad_bounds(&p, 0, &HashMap::new(), true), (1.0, 1.0));
        assert_eq!(grad_bounds(&p, 1, &HashMap::new(), true), (2.0, 2.0));
        assert_eq!(grad_bounds(&p, 2, &HashMap::new(), true), (3.0, 3.0));
    }

    #[test]
    fn test_alternating_persistence() {
        let p = make_solver_qubo();
        let p_symm = p.make_symmetric();
        let persist = compute_iterative_persistence(&p_symm, &HashMap::new(), p.num_x());

        assert_eq!(persist.len(), 41);
    }
}
