//! This module contains utility functions for local search algorithms.
//!
//! This typically includes functions that are used by multiple local search algorithms.
//!
//! These include:
//! - 1-opt local search
//! - 1-step Opt criteria local search

use crate::qubo::Qubo;
use ndarray::Array1;

/// Auxiliary function to calculate the optimal criteria for each variable
///
/// This is essentially a helper function that calculates the gains of fliping bits for each variable and then flips in
/// the direction that gives the best gain.
pub fn get_opt_criteria(qubo: &Qubo, x: &Array1<f64>) -> Array1<f64> {
    // calculate the optimal criteria for each variable, given the point x
    // if the gradient is negative, then the optimal criteria is 1.0 for x_1
    // if the gradient is positive, then the optimal criteria is 0.0 for x_1

    let mut fixed = Array1::zeros(qubo.num_x());
    let grad = qubo.eval_grad(x);

    for i in 0..qubo.num_x() {
        if grad[i] <= 0.0 {
            fixed[i] = 1.0;
        } else {
            fixed[i] = 0.0;
        }
    }

    fixed
}

/// Auxiliary function to calculate Delta, as defined in Boros2007
pub fn compute_d(x_0: &Array1<f64>, grad: &Array1<f64>) -> Array1<f64> {
    // compute the variable importance function
    let mut d = Array1::<f64>::zeros(x_0.len());
    for i in 0..x_0.len() {
        // find the max of the two terms
        d[i] = f64::max(-x_0[i] * grad[i], (1.0 - x_0[i]) * grad[i]);
    }
    d
}

/// Auxiliary function to calculate I, as defined in Boros2007
pub fn compute_I(d: &Array1<f64>) -> Vec<usize> {
    // compute the variable selection function
    d.iter()
        .filter(|x| **x > 0.0)
        .enumerate()
        .map(|(i, _)| i)
        .collect()
}

/// Efficient calculation of the delta of the objective function for a single bit flip for each variable
/// more or less this is a helper function that allows for selecting the best bit to flip option without
/// having to calculate the objective function for each bit flip, independently.
///
/// Run time is O(|Q|) + O(|x|)
pub fn one_flip_objective(qubo: &Qubo, x_0: &Array1<f64>) -> (f64, Array1<f64>) {
    // set up the array to hold the objective function values
    let mut objs = Array1::<f64>::zeros(qubo.num_x());

    // calculate the objective function for each variable and each term in the delta formula
    let x_q = 0.5 * (&qubo.q * x_0);
    let q_x = 0.5 * (&qubo.q.transpose_view() * x_0);
    let q_jj = 0.5 * qubo.q.diag().to_dense();
    let delta = 1.0 - 2.0 * x_0;

    // generate the objective shifts for each variable
    for i in 0..qubo.num_x() {
        objs[i] = q_jj[i] + delta[i] * (x_q[i] + q_x[i] + qubo.c[i]);
    }

    // calculate the objective function for the original solution
    let obj_0 = x_0.dot(&x_q) + qubo.c.dot(x_0);

    (obj_0, objs)
}

/// Performs a single 1-opt local search, which is to say that it will flip a single bit and return the best solution out of all
/// of the possible bit flips.
/// This takes O(n|Q|) + O(n) time, where |Q| is the number of non-zero elements in the QUBO matrix.
pub fn one_step_local_search(qubo: &Qubo, x_0: &Array1<f64>, subset: &Vec<usize>) -> Array1<f64> {
    let current_obj = qubo.eval(x_0);

    let y = 1.0f64 - x_0;
    let mut objs = Array1::<f64>::zeros(qubo.num_x());

    // calculate the objective function for each variable in our selected subset and each term in the delta formula
    for i in subset.iter() {
        let mut x = x_0.clone();
        x[*i] = y[*i];
        objs[*i] = qubo.eval(&x);
    }

    // find the index of the best neighbor
    let best_neighbor = objs
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap()
        .0;

    // get the objective of this best neighbor
    let best_obj = objs[best_neighbor];

    // generate the vector relating to this best neighbor
    let mut x_1 = x_0.clone();
    x_1[best_neighbor] = 1.0 - x_1[best_neighbor];

    // return the best neighbor if it is better than the current solution
    match best_obj < current_obj {
        true => x_1,
        false => x_0.clone(),
    }
}
