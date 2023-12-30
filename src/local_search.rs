//! # Local Search contains all of the implemented local search algorithms
//!
//! This module contains all of the implemented local search algorithms, which are:
//! - One step local search
//! - Simple local search
//! - Simple opt criteria search
//! - Simple mixed search
//! - Multi simple local search
//! - Multi simple opt criteria search

use crate::local_search_utils;
use crate::qubo::Qubo;
use ndarray::Array1;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

/// Performs a single step of local search, which is to say that it will flip a single bit and return the best solution out of all
/// of the possible bit flips.
/// This takes O(n|Q|) + O(n) time, where |Q| is the number of non-zero elements in the QUBO matrix.
pub fn one_step_local_search_improved(
    qubo: &Qubo,
    x_0: &Array1<f64>,
    selected_vars: &Vec<usize>,
) -> Array1<f64> {
    // Do a neighborhood search of up to one bit flip and returns the best solution
    // found, this can include the original solution, out of the selected variables.

    let (_, objs) = local_search_utils::one_flip_objective(qubo, x_0);

    let best_neighbor = objs
        .iter()
        .enumerate()
        .filter(|(i, _)| selected_vars.contains(i))
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap()
        .0;

    let best_obj = objs[best_neighbor];

    let mut x_1 = x_0.clone();
    x_1[best_neighbor] = 1.0 - x_1[best_neighbor];

    match best_obj < 0.0 {
        true => x_1,
        false => x_0.clone(),
    }
}

/// Given a QUBO and an integral initial point, run simple local search until the point converges or the step limit is hit.
pub fn simple_local_search(qubo: &Qubo, x_0: &Array1<f64>, max_steps: usize) -> Array1<f64> {
    let mut x = x_0.clone();
    let variables = (0..qubo.num_x()).collect();
    let mut x_1 = local_search_utils::one_step_local_search(qubo, &x, &variables);
    let mut steps = 0;

    while x_1 != x && steps <= max_steps {
        x = x_1.clone();
        // apply the local search to the selected variables
        x_1 = one_step_local_search_improved(qubo, &x, &variables);
        steps += 1;
    }

    x_1
}

/// Given a QUBO and a vector of initial points, run local searches on each initial point and return all of the solutions.
pub fn multi_simple_local_search(qubo: &Qubo, xs: &Vec<Array1<f64>>) -> Vec<Array1<f64>> {
    // Given a vector of initial points, run simple local search on each of them
    xs.par_iter()
        .map(|x| simple_local_search(qubo, x, usize::MAX))
        .collect()
}

/// Given a QUBO and a fractional or integral initial point, run a opt search until the point converges or the step limit is hit.
pub fn simple_opt_criteria_search(qubo: &Qubo, x_0: &Array1<f64>, max_steps: usize) -> Array1<f64> {
    let mut x = x_0.clone();
    let mut x_1 = local_search_utils::get_opt_criteria(qubo, &x);
    let mut steps = 0;

    while x_1 != x && steps <= max_steps {
        x = x_1.clone();
        x_1 = local_search_utils::get_opt_criteria(qubo, &x);
        steps += 1;
    }

    x_1
}

/// Given a QUBO and a vector of initial points, run opt searches on each initial point and return all of the solutions.
pub fn multi_simple_opt_criteria_search(qubo: &Qubo, xs: &Vec<Array1<f64>>) -> Vec<Array1<f64>> {
    // Given a vector of initial points, run simple local search on each of them
    xs.par_iter()
        .map(|x| simple_opt_criteria_search(qubo, x, 1000))
        .collect()
}

/// Given a QUBO and a fractional or integral initial point, run a mixed search until the point converges or the step limit is hit.
pub fn simple_mixed_search(qubo: &Qubo, x_0: &Array1<f64>, max_steps: usize) -> Array1<f64> {
    let mut x = x_0.clone();
    let mut x_1 = local_search_utils::get_opt_criteria(qubo, &x);
    let mut steps = 0;
    let vars = (0..qubo.num_x()).collect();

    while x_1 != x && steps <= max_steps {
        x = x_1.clone();
        x_1 = one_step_local_search_improved(qubo, &x, &vars);
        x_1 = local_search_utils::get_opt_criteria(qubo, &x_1);
        steps += 1;
    }

    x_1
}
