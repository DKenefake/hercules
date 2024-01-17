//! # Local Search contains all of the implemented local search algorithms
//!
//! This module contains all of the implemented local search algorithms, which are:
//! - One step local search
//! - Simple local search
//! - Simple gain criteria search
//! - Simple mixed search
//! - Multi simple local search
//! - Multi simple gain criteria search
//! - Simple Particle Swarm Search

use crate::local_search_utils;
use crate::local_search_utils::contract_point;
use crate::qubo::Qubo;
use crate::utils::{get_best_point, make_binary_point};
use ndarray::Array1;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use smolprng::{Algorithm, PRNG};

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
/// use hercules::local_search;
///
/// // generate a random QUBO
/// let mut prng = PRNG {
///    generator: JsfLarge::default(),
/// };
/// let p = Qubo::make_random_qubo(10, &mut prng, 0.5);
///
/// // generate a random point inside with x in {0, 1}^10 with
/// let x_0 = utils::make_binary_point(p.num_x(), &mut prng);
///
/// // perform a single step of local search
/// let x_1 = local_search::one_step_local_search_improved(&p, &x_0, &vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
/// ```
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

    if best_obj < 0.0f64 {
        let mut x_1 = x_0.clone();
        x_1[best_neighbor] = 1.0 - x_1[best_neighbor];
        x_1
    } else {
        x_0.clone()
    }
}

/// Given a QUBO and an integral initial point, run simple local search until the point converges or the step limit is hit.
///
/// Example:
/// ``` rust
/// use hercules::qubo::Qubo;
/// use smolprng::{PRNG, JsfLarge};
/// use hercules::{initial_points, utils};
/// use hercules::local_search;
///
/// // generate a random QUBO
/// let mut prng = PRNG {
///   generator: JsfLarge::default(),
/// };
/// let p = Qubo::make_random_qubo(10, &mut prng, 0.5);
///
/// // generate a random point inside with x in {0, 1}^10 with
/// let x_0 = initial_points::generate_random_binary_point(&p, &mut prng, 0.5);
///
/// // perform a simple local search starting at x_0
/// let x_sol = local_search::simple_local_search(&p, &x_0, 1000);
/// ```
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
///
/// Example:
/// ``` rust
/// use hercules::qubo::Qubo;
/// use smolprng::{PRNG, JsfLarge};
/// use hercules::{initial_points, utils};
/// use hercules::local_search;
///
/// // generate a random QUBO
/// let mut prng = PRNG {
///   generator: JsfLarge::default(),
/// };
/// let p = Qubo::make_random_qubo(10, &mut prng, 0.5);
///
/// // generate a random point inside with x in {0, 1}^10
/// let x_0 = initial_points::generate_random_binary_point(&p, &mut prng, 0.5);
/// let x_1 = initial_points::generate_random_binary_point(&p, &mut prng, 0.5);
/// let x_2 = initial_points::generate_random_binary_point(&p, &mut prng, 0.5);
///
/// let xs = vec![x_0, x_1, x_2];
///
/// // perform a multiple simple local search starting at x_0
/// let x_sols = local_search::multi_simple_local_search(&p, &xs);
/// ```
pub fn multi_simple_local_search(qubo: &Qubo, xs: &Vec<Array1<f64>>) -> Vec<Array1<f64>> {
    // Given a vector of initial points, run simple local search on each of them
    xs.par_iter()
        .map(|x| simple_local_search(qubo, x, usize::MAX))
        .collect()
}

/// Given a QUBO and a fractional or integral initial point, run a gain search until the point converges or the step limit is hit.
///
/// Example:
/// ``` rust
/// use hercules::qubo::Qubo;
/// use smolprng::{PRNG, JsfLarge};
/// use hercules::{initial_points, utils};
/// use hercules::local_search;
///
/// // generate a random QUBO
/// let mut prng = PRNG {
///  generator: JsfLarge::default(),
/// };
/// let p = Qubo::make_random_qubo(10, &mut prng, 0.5);
///
/// // generate a random point inside with x in {0, 1}^10
/// let x_0 = initial_points::generate_random_binary_point(&p, &mut prng, 0.5);
///
/// // perform a simple gain criteria search starting at x_0
/// let x_sol = local_search::simple_gain_criteria_search(&p, &x_0, 1000);
/// ```
pub fn simple_gain_criteria_search(
    qubo: &Qubo,
    x_0: &Array1<f64>,
    max_steps: usize,
) -> Array1<f64> {
    let mut x = x_0.clone();
    let mut x_1 = local_search_utils::get_gain_criteria(qubo, &x);
    let mut steps = 0;

    while x_1 != x && steps <= max_steps {
        x = x_1.clone();
        x_1 = local_search_utils::get_gain_criteria(qubo, &x);
        steps += 1;
    }

    x_1
}

/// Given a QUBO and a vector of initial points, run gain searches on each initial point and return all of the solutions.
///
/// Example:
/// ``` rust
/// use hercules::qubo::Qubo;
/// use smolprng::{PRNG, JsfLarge};
/// use hercules::{initial_points, utils};
/// use hercules::local_search;
///
/// // generate a random QUBO
/// let mut prng = PRNG {
///  generator: JsfLarge::default(),
/// };
/// let p = Qubo::make_random_qubo(10, &mut prng, 0.5);
///
/// // generate a random point inside with x in {0, 1}^10
/// let x_0 = initial_points::generate_random_binary_point(&p, &mut prng, 0.5);
/// let x_1 = initial_points::generate_random_binary_point(&p, &mut prng, 0.5);
/// let x_2 = initial_points::generate_random_binary_point(&p, &mut prng, 0.5);
///
/// let xs = vec![x_0, x_1, x_2];
///
/// // perform a multiple simple gain criteria search starting at x_0
/// let x_sols = local_search::multi_simple_gain_criteria_search(&p, &xs);
/// ```
pub fn multi_simple_gain_criteria_search(qubo: &Qubo, xs: &Vec<Array1<f64>>) -> Vec<Array1<f64>> {
    // Given a vector of initial points, run simple local search on each of them
    xs.par_iter()
        .map(|x| simple_gain_criteria_search(qubo, x, 1000))
        .collect()
}

/// Given a QUBO and a fractional or integral initial point, run a mixed search until the point converges or the step limit is hit.
/// This is a combination of local search and gain criteria search, where the gain criteria search is run on the local search.
///
/// Example:
/// ``` rust
/// use hercules::qubo::Qubo;
/// use smolprng::{PRNG, JsfLarge};
/// use hercules::{initial_points, utils};
/// use hercules::local_search;
///
/// // generate a random QUBO
/// let mut prng = PRNG {
/// generator: JsfLarge::default(),
/// };
/// let p = Qubo::make_random_qubo(10, &mut prng, 0.5);
///
/// // generate a random point inside with x in {0, 1}^10
/// let x_0 = initial_points::generate_random_binary_point(&p, &mut prng, 0.5);
///
/// // perform a simple mixed search starting at x_0
/// let x_sol = local_search::simple_mixed_search(&p, &x_0, 1000);
/// ```
pub fn simple_mixed_search(qubo: &Qubo, x_0: &Array1<f64>, max_steps: usize) -> Array1<f64> {
    let mut x = x_0.clone();
    let mut x_1 = local_search_utils::get_gain_criteria(qubo, &x);
    let mut steps = 0;
    let vars = (0..qubo.num_x()).collect();

    while x_1 != x && steps <= max_steps {
        x = x_1.clone();
        x_1 = one_step_local_search_improved(qubo, &x, &vars);
        x_1 = local_search_utils::get_gain_criteria(qubo, &x_1);
        steps += 1;
    }

    x_1
}

/// Performs a particle swarm search on a QUBO.
///
/// Example:
/// ``` rust
/// use hercules::qubo::Qubo;
/// use smolprng::{PRNG, JsfLarge};
/// use hercules::{initial_points, utils};
/// use hercules::local_search;
///
/// // generate a random QUBO
/// let mut prng = PRNG {
///    generator: JsfLarge::default(),
/// };
/// let p = Qubo::make_random_qubo(10, &mut prng, 0.5);
///
/// // perform particle swarm with 150 particles for 1000 iterations
/// let x_sol = local_search::particle_swarm_search(&p, 150, 1000, &mut prng);
/// ```
pub fn particle_swarm_search<T: Algorithm>(
    qubo: &Qubo,
    num_particles: usize,
    max_steps: usize,
    prng: &mut PRNG<T>,
) -> Array1<f64> {
    // initialize the particles
    let num_dim = qubo.num_x();

    // generate random starting points
    let mut particles: Vec<_> = (0..num_particles)
        .map(|_| make_binary_point(num_dim, prng))
        .collect();

    // select all variables
    let selected_vars = (0..num_dim).collect();

    // say at each particular point that we will contract 10% of the variables
    let num_contract = qubo.num_x() / 10 + 1;

    // loop over the number of iterations
    for _ in 0..max_steps {
        // apply local search to each particle
        particles = particles
            .par_iter()
            .map(|x| one_step_local_search_improved(qubo, x, &selected_vars))
            .collect();

        // find the best particle
        let best_particle = get_best_point(qubo, &particles);

        // contract the particles towards the best particle
        particles = particles
            .iter()
            .map(|x| contract_point(&best_particle, x, num_contract))
            .collect();
    }

    // find the best particle
    get_best_point(qubo, &particles)
}
