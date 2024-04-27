//! This module contains utility functions for local search algorithms.
//!
//! This typically includes functions that are used by multiple local search algorithms.
//!
//! These include:
//! - 1-opt local search
//! - 1-step gain criteria local search

use crate::qubo::Qubo;
use ndarray::Array1;

/// Performs a single step of local search, which is to say that it will flip a single bit and return the best solution out of all
/// of the possible bit flips.
/// This takes O(n|Q|) + O(n) time, where |Q| is the number of non-zero elements in the QUBO matrix.
///
/// # Panics
///
/// Will panic is there are not any selected variables.
///
/// Example:
/// ``` rust
/// use hercules::qubo::Qubo;
/// use smolprng::{PRNG, JsfLarge};
/// use hercules::{initial_points, utils};
/// use hercules::local_search_utils;
///
/// // generate a random QUBO
/// let mut prng = PRNG {
///    generator: JsfLarge::default(),
/// };
/// let p = Qubo::make_random_qubo(10, &mut prng, 0.5);
///
/// // generate a random point inside with x in {0, 1}^10 with
/// let x_0 = initial_points::generate_random_binary_point(p.num_x(), &mut prng, 0.5);
///
/// // perform a single step of local search
/// let x_1 = local_search_utils::one_step_local_search_improved(&p, &x_0, &vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
/// ```
pub fn one_step_local_search_improved(
    qubo: &Qubo,
    x_0: &Array1<usize>,
    selected_vars: &Vec<usize>,
) -> Array1<usize> {
    // Do a neighborhood search of up to one bit flip and returns the best solution
    // found, this can include the original solution, out of the selected variables.

    let (_, objs) = one_flip_objective(qubo, x_0);

    let best_neighbor = objs
        .iter()
        .enumerate()
        .filter(|(i, _)| selected_vars.contains(i))
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap()
        .0;

    let best_obj = objs[best_neighbor];

    if best_obj < 0.0f64 {
        let mut x_1 = x_0.clone();
        x_1[best_neighbor] = 1 - x_1[best_neighbor];
        x_1
    } else {
        x_0.clone()
    }
}

/// Auxiliary function to calculate the gains from flipping each variable
///
/// This is essentially a helper function that calculates the gains of flipping bits for each variable and then flips in
/// the direction that gives the best gain.
pub fn get_gain_criteria(qubo: &Qubo, x: &Array1<usize>) -> Array1<usize> {
    // calculate the gain criteria for each variable, given the point x
    // if the gradient is negative, then the optimal criteria is 1.0 for x_1
    // if the gradient is positive, then the optimal criteria is 0.0 for x_1

    let mut fixed = Array1::zeros(qubo.num_x());
    let grad = qubo.eval_grad_usize(x);

    for i in 0..qubo.num_x() {
        if grad[i] <= 0.0 {
            fixed[i] = 1;
        } else {
            fixed[i] = 0;
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
pub fn one_flip_objective(qubo: &Qubo, x_0: &Array1<usize>) -> (f64, Array1<f64>) {
    // set up the array to hold the objective function values
    let mut objs = Array1::<f64>::zeros(qubo.num_x());
    let x_0f = x_0.mapv(|x| x as f64);

    // calculate the objective function for each variable and each term in the delta formula
    let x_q = 0.5 * (&qubo.q * &x_0f);
    let q_x = 0.5 * (&qubo.q.transpose_view() * &x_0f);
    let q_jj = 0.5 * qubo.q.diag().to_dense();
    let delta = 1.0 - 2.0 * &x_0f;

    // generate the objective shifts for each variable
    for i in 0..qubo.num_x() {
        objs[i] = q_jj[i] + delta[i] * (x_q[i] + q_x[i] + qubo.c[i]);
    }

    // calculate the objective function for the original solution
    let obj_0 = x_0f.dot(&x_q) + qubo.c.dot(&x_0f);

    (obj_0, objs)
}

/// Performs a single gain local search, which is to say that it will flip a single bit and return the best solution out of all
/// of the possible bit flips.
/// This takes O(n|Q|) + O(n) time, where |Q| is the number of non-zero elements in the QUBO matrix.
///
/// # Panics
///
/// Will panic is the subset of variables is zero.
pub fn one_step_local_search(
    qubo: &Qubo,
    x_0: &Array1<usize>,
    subset: &Vec<usize>,
) -> Array1<usize> {
    let current_obj = qubo.eval_usize(x_0);

    let y = 1 - x_0;
    let mut objs = Array1::<f64>::zeros(qubo.num_x());

    // calculate the objective function for each variable in our selected subset and each term in the delta formula
    for i in subset {
        let mut x = x_0.clone();
        x[*i] = y[*i];
        objs[*i] = qubo.eval_usize(&x);
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
    x_1[best_neighbor] = 1 - x_1[best_neighbor];

    // return the best neighbor if it is better than the current solution
    match best_obj < current_obj {
        true => x_1,
        false => x_0.clone(),
    }
}

/// This is a helper function for the basic particle swarm algorithm. It takes two points, x_0 and x_1, and sets up to num_contract
/// variables to be the same between the two points, and then returns the new point.
///
/// Example
/// ``` rust
/// use hercules::qubo::Qubo;
/// use smolprng::{PRNG, JsfLarge};
/// use hercules::{initial_points, utils};
/// use hercules::local_search_utils;
///
/// // generate a random QUBO
/// let mut prng = PRNG {
///    generator: JsfLarge::default(),
/// };
/// let p = Qubo::make_random_qubo(10, &mut prng, 0.5);
///
/// // generate random points inside with x in {0, 1}^10
/// let mut x_0 = initial_points::generate_random_binary_point(p.num_x(), &mut prng, 0.5);
/// let mut x_1 = initial_points::generate_random_binary_point(p.num_x(), &mut prng, 0.5);
///
/// let mut x_s = vec![x_0, x_1];
///
/// // find the point with the best objective
/// let x_best = utils::get_best_point(&p, &x_s);
///
/// // contract the point x_0 up to 4 bits
/// x_s[0] = local_search_utils::contract_point(&x_best, &x_s[0], 4);
/// x_s[1] = local_search_utils::contract_point(&x_best, &x_s[1], 4);
/// ```
pub fn contract_point(
    x_0: &Array1<usize>,
    x_1: &Array1<usize>,
    num_contract: usize,
) -> Array1<usize> {
    // contract the point x_0 to the subset of variables
    let mut x_1 = x_1.clone();
    let mut flipped = 0;

    for i in 0..x_0.len() {
        if x_0[i] != x_1[i] {
            flipped += 1;

            if x_0[i] != x_1[i] {
                if x_0[i] == 1 {
                    x_1[i] = 0;
                } else {
                    x_1[i] = 1;
                }
            }

            if flipped > num_contract {
                break;
            }
        }
    }

    x_1
}
