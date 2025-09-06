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
/// This takes O(|Q|) + O(n) time, where |Q| is the number of non-zero elements in the QUBO matrix.
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

/// Performs a two-step local search, which is to say that it will flip either one or two bits and
/// return the best solution out of all the possible bit flips.
/// This takes O(|Q|) + O(n) time, where |Q| is the number of non-zero elements in the QUBO matrix.
pub fn two_step_local_search_improved(qubo: &Qubo, x_0: &Array1<usize>) -> Array1<usize> {
    // Do a neighborhood search of up to two bit flips and returns the best solution
    let (_, obj_1d) = one_flip_objective(qubo, x_0);

    // Here we increase the neighborhood to include two bit flips
    let best_1d_neighbor = obj_1d
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap()
        .0;

    let best_obj_1d = obj_1d[best_1d_neighbor];

    let mut best_obj_2d = f64::INFINITY;
    let mut best_2d_neighbor = (0, 1);

    for (q_ij, (i, j)) in &qubo.q {
        if i > j {
            let current_obj_2d = obj_1d[i]
                + obj_1d[j]
                + q_ij * (1.0 - 2.0 * (x_0[i] as f64)) * (1.0 - 2.0 * (x_0[j] as f64));

            if current_obj_2d < best_obj_2d {
                best_obj_2d = current_obj_2d;
                best_2d_neighbor = (i, j);
            }
        }
    }

    // see if the best 2d neighbor is better than the best 1d neighbor
    if best_obj_2d < best_obj_1d && best_obj_2d < 0.0f64 {
        let mut x_1 = x_0.clone();
        x_1[best_2d_neighbor.0] = 1 - x_1[best_2d_neighbor.0];
        x_1[best_2d_neighbor.1] = 1 - x_1[best_2d_neighbor.1];
        x_1
    } else if best_obj_1d < 0.0f64 {
        let mut x_1 = x_0.clone();
        x_1[best_1d_neighbor] = 1 - x_1[best_1d_neighbor];
        x_1
    } else {
        x_0.clone()
    }
}

/// Auxiliary function to calculate the gains from flipping each variable
///
/// This is essentially a helper function that calculates the gains of flipping bits for each variable and then flips in
/// the direction that gives the best gain.
///
/// This runs in O(|Q|) + O(n) time, where |Q| is the number of non-zero elements in the QUBO matrix.
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

/// Efficient calculation of the delta of the objective function for a single bit flip for each variable
/// more or less this is a helper function that allows for selecting the best bit to flip option without
/// having to calculate the objective function for each bit flip.
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

#[cfg(test)]
mod tests {

    use crate::local_search_utils::{
        one_step_local_search_improved, two_step_local_search_improved,
    };
    use crate::qubo::Qubo;
    use smolprng::{JsfLarge, PRNG};

    #[test]
    fn test_one_step_local_search_improved() {
        // generate a random QUBO
        let mut prng = PRNG {
            generator: JsfLarge::default(),
        };

        let p = Qubo::make_random_qubo(50, &mut prng, 0.2);
        let selected_vars: Vec<usize> = (0..p.num_x()).collect();

        for _ in 0..100 {
            // generate a random point inside with x in {0, 1}^10 with
            let x_0 =
                crate::initial_points::generate_random_binary_point(p.num_x(), &mut prng, 0.5);

            // compute the next step
            let x_1 = one_step_local_search_improved(&p, &x_0, &selected_vars);

            // compute the objective function values
            let obj_0 = p.eval_usize(&x_0);
            let obj_1 = p.eval_usize(&x_1);

            // ensure that the objective has not increased
            assert!(obj_1 <= obj_0);
        }
    }

    #[test]
    fn test_two_step_local_search_improved() {
        // generate a random QUBO
        let mut prng = PRNG {
            generator: JsfLarge::default(),
        };

        let p = Qubo::make_random_qubo(50, &mut prng, 0.2);
        let selected_vars: Vec<usize> = (0..p.num_x()).collect();

        for _ in 0..100 {
            // generate a random point inside with x in {0, 1}^10 with
            let x_0 =
                crate::initial_points::generate_random_binary_point(p.num_x(), &mut prng, 0.5);

            // compute the next step
            let x_1 = one_step_local_search_improved(&p, &x_0, &selected_vars);
            let x_2 = two_step_local_search_improved(&p, &x_0);

            // compute the objective function values
            let obj_0 = p.eval_usize(&x_0);
            let obj_1 = p.eval_usize(&x_1);
            let obj_2 = p.eval_usize(&x_2);

            // ensure that two step local search is at least as good as one step local search
            assert!(obj_2 <= obj_1);

            // ensure that the objective has not increased
            assert!(obj_1 <= obj_0);
        }
    }
}
