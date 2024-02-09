use crate::qubo::Qubo;
use std::cmp::min;
use std::collections::HashMap;

/// This function takes a QUBO and a set of persistent variables and returns a new set of persistent variables by repeatedly re
/// computing the persistent variables until.
pub fn compute_iterative_persistence(
    qubo: &Qubo,
    persistent: &HashMap<usize, f64>,
    iter_lim: usize,
) -> HashMap<usize, f64> {
    // make a copy of the passed persistence variable
    let mut new_persistent = persistent.clone();

    // the number of required iterations is always below the number of variables
    let iters = min(iter_lim, qubo.num_x());

    // loop over the number of iters
    for _ in 0..iters {
        let incoming_persistent = compute_persistent(qubo, &new_persistent);
        if new_persistent == incoming_persistent {
            break;
        }
        new_persistent = incoming_persistent;
    }

    new_persistent
}

/// This function takes a QUBO and a set of persistent variables and returns a new set of persistent variables by computing the
/// persistent variables once.
pub fn compute_persistent(qubo: &Qubo, persistent: &HashMap<usize, f64>) -> HashMap<usize, f64> {
    // create a new hashmap to store the new persistent variables
    let mut new_persistent = persistent.clone();

    // iterate over all the variables in the QUBO
    for i in 0..qubo.num_x() {
        if persistent.contains_key(&i) {
            continue;
        }

        // find the bounds of the gradient in each direction
        let (lower, upper) = grad_bounds(qubo, i, persistent);

        // if the lower bound it positive, then we can set the variable to 0
        if lower > 0.0 {
            new_persistent.insert(i, 0.0);
        }

        // if the upper bound is below 0, then we can set the variable to 1
        if upper < 0.0 {
            new_persistent.insert(i, 1.0);
        }
    }

    new_persistent
}

/// Finds bounds of the i-th index of the gradients of the QUBO function
pub fn grad_bounds(qubo: &Qubo, i: usize, persistent: &HashMap<usize, f64>) -> (f64, f64) {
    // set up tracking variables for each bound
    let mut lower = 0.0;
    let mut upper = 0.0;

    // get the i-th row of the Q matrix
    let x = qubo.q.outer_view(i).unwrap();

    // get the i-th column of the Q matrix
    let binding = qubo.q.transpose_view();
    let y = binding.outer_view(i).unwrap();

    // add the row and column together
    let q_term = x + y;

    // loop over the variables in this row
    for (index, x_j) in q_term.iter() {
        // dereference and multiply by 0.5 to make an auxiliary variable that is clearer
        let mut value = *x_j;
        value = value * 0.5;

        // if it is a fixed variable, we have effectively removed this variable from the QUBO
        if persistent.contains_key(&index) {
            lower += 1.0 * value * persistent[&index];
            upper += 1.0 * value * persistent[&index];
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
    lower = lower + qubo.c[i];
    upper = upper + qubo.c[i];

    (lower, upper)
}
