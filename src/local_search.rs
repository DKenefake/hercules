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

use crate::initial_points::generate_random_binary_point;
use crate::local_search_utils;
use crate::qubo::Qubo;
use crate::utils::get_best_point;
use ndarray::Array1;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use smolprng::{Algorithm, PRNG};

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
/// let x_0 = initial_points::generate_random_binary_point(p.num_x(), &mut prng, 0.5);
///
/// // perform a simple local search starting at x_0
/// let x_sol = local_search::simple_local_search(&p, &x_0, 1000);
/// ```
pub fn simple_local_search(qubo: &Qubo, x_0: &Array1<usize>, max_steps: usize) -> Array1<usize> {
    let mut x = x_0.clone();
    let variables = (0..qubo.num_x()).collect();
    let mut x_1 = local_search_utils::one_step_local_search_improved(qubo, &x, &variables);
    let mut steps = 0;

    while x_1 != x && steps <= max_steps {
        x.clone_from(&x_1);
        // apply the local search to the selected variables
        x_1 = local_search_utils::one_step_local_search_improved(qubo, &x, &variables);
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
/// let x_0 = initial_points::generate_random_binary_point(p.num_x(), &mut prng, 0.5);
/// let x_1 = initial_points::generate_random_binary_point(p.num_x(), &mut prng, 0.5);
/// let x_2 = initial_points::generate_random_binary_point(p.num_x(), &mut prng, 0.5);
///
/// let xs = vec![x_0, x_1, x_2];
///
/// // perform a multiple simple local search starting at x_0
/// let x_sols = local_search::multi_simple_local_search(&p, &xs);
/// ```
pub fn multi_simple_local_search(qubo: &Qubo, xs: &Vec<Array1<usize>>) -> Vec<Array1<usize>> {
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
/// let x_0 = initial_points::generate_random_binary_point(p.num_x(), &mut prng, 0.5);
///
/// // perform a simple gain criteria search starting at x_0
/// let x_sol = local_search::simple_gain_criteria_search(&p, &x_0, 1000);
/// ```
pub fn simple_gain_criteria_search(
    qubo: &Qubo,
    x_0: &Array1<usize>,
    max_steps: usize,
) -> Array1<usize> {
    let mut x = x_0.clone();
    let mut x_1 = local_search_utils::get_gain_criteria(qubo, &x);
    let mut steps = 0;

    while x_1 != x && steps <= max_steps {
        x.clone_from(&x_1);
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
/// let x_0 = initial_points::generate_random_binary_point(p.num_x(), &mut prng, 0.5);
/// let x_1 = initial_points::generate_random_binary_point(p.num_x(), &mut prng, 0.5);
/// let x_2 = initial_points::generate_random_binary_point(p.num_x(), &mut prng, 0.5);
///
/// let xs = vec![x_0, x_1, x_2];
///
/// // perform a multiple simple gain criteria search starting at x_0
/// let x_sols = local_search::multi_simple_gain_criteria_search(&p, &xs);
/// ```
pub fn multi_simple_gain_criteria_search(
    qubo: &Qubo,
    xs: &Vec<Array1<usize>>,
) -> Vec<Array1<usize>> {
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
/// let x_0 = initial_points::generate_random_binary_point(p.num_x(), &mut prng, 0.5);
///
/// // perform a simple mixed search starting at x_0
/// let x_sol = local_search::simple_mixed_search(&p, &x_0, 1000);
/// ```
pub fn simple_mixed_search(qubo: &Qubo, x_0: &Array1<usize>, max_steps: usize) -> Array1<usize> {
    let mut x = x_0.clone();
    let mut x_1 = local_search_utils::get_gain_criteria(qubo, &x);
    let mut steps = 0;
    let vars = (0..qubo.num_x()).collect();

    while x_1 != x && steps <= max_steps {
        x.clone_from(&x_1);
        x_1 = local_search_utils::one_step_local_search_improved(qubo, &x, &vars);
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
) -> Array1<usize> {
    // initialize the particles
    let num_dim = qubo.num_x();

    // generate random starting points
    let mut particles: Vec<_> = (0..num_particles)
        .map(|_| generate_random_binary_point(num_dim, prng, 0.5))
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
            .map(|x| local_search_utils::one_step_local_search_improved(qubo, x, &selected_vars))
            .collect();

        // find the best particle
        let best_particle = get_best_point(qubo, &particles);

        // contract the particles towards the best particle
        particles = particles
            .iter()
            .map(|x| local_search_utils::contract_point(&best_particle, x, num_contract))
            .collect();
    }

    // find the best particle
    get_best_point(qubo, &particles)
}

/// Performs a random search on a QUBO, where points are randomly generated and the best point is returned. This to
/// create a baseline to compare other algorithms against just random guesses.
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
/// // perform random search with 1000 points
/// let x_sol = local_search::random_search(&p, 1000, &mut prng);
/// ```
pub fn random_search<T: Algorithm>(
    qubo: &Qubo,
    num_points: usize,
    prng: &mut PRNG<T>,
) -> Array1<usize> {
    // set up an initial best point and objective
    let mut best_point = generate_random_binary_point(qubo.num_x(), prng, 0.5);
    let mut best_objective = qubo.eval_usize(&best_point);

    // loop over the number of points
    for _ in 0..num_points {
        // generate a new point and evaluate it
        let new_point = generate_random_binary_point(qubo.num_x(), prng, 0.5);
        let new_obj = qubo.eval_usize(&new_point);

        // if the new point is better, update the best point
        if new_obj <= best_objective {
            best_point = new_point;
            best_objective = new_obj;
        }
    }

    best_point
}

#[cfg(test)]
mod tests {
    use crate::local_search::*;
    use crate::qubo::Qubo;
    use crate::tests::{make_solver_qubo, make_test_prng};
    use crate::{initial_points, local_search_utils};
    use ndarray::Array1;
    use sprs::CsMat;

    #[test]
    fn test_opt_criteria() {
        let p = make_solver_qubo();
        let mut prng = make_test_prng();

        let mut x_0 = initial_points::generate_random_binary_point(p.num_x(), &mut prng, 0.5);
        x_0 = local_search_utils::get_gain_criteria(&p, &x_0);
        println!("{:?}", p.eval_usize(&x_0));
    }

    #[test]
    fn test_multi_opt_heuristics() {
        let p = make_solver_qubo();
        let mut prng = make_test_prng();

        let mut xs = initial_points::generate_random_binary_points(p.num_x(), 10, &mut prng);

        xs = multi_simple_gain_criteria_search(&p, &xs);

        let min_obj = crate::tests::get_min_obj(&p, &xs);
        println!("{:?}", min_obj);
    }

    #[test]
    fn test_mixed_search() {
        let p = make_solver_qubo();
        let mut prng = make_test_prng();

        let mut xs = initial_points::generate_random_binary_points(p.num_x(), 10, &mut prng);

        xs = xs
            .par_iter()
            .map(|x| simple_mixed_search(&p, &x, 1000))
            .collect();

        let min_obj = crate::tests::get_min_obj(&p, &xs);
        println!("{min_obj:?}");
    }

    #[test]
    fn test_particle_swarm() {
        let p = make_solver_qubo();
        let mut prng = make_test_prng();

        let x = particle_swarm_search(&p, 5, 100, &mut prng);

        println!("{:?}", p.eval_usize(&x));
    }

    #[test]
    fn compare_methods() {
        let p = make_solver_qubo();
        let mut prng = make_test_prng();

        let x_0 = initial_points::generate_random_binary_point(p.num_x(), &mut prng, 0.5);
        let max_iter = p.num_x();

        let x_pso = particle_swarm_search(&p, 100, max_iter, &mut prng);
        let x_mixed = simple_mixed_search(&p, &x_0, max_iter);
        let x_gain = simple_gain_criteria_search(&p, &x_0, max_iter);
        let x_opt = simple_local_search(&p, &x_0, max_iter);
        let x_rand = random_search(&p, 100, &mut prng);

        println!(
            "PSO: {:?}, MIXED: {:?}, GAIN: {:?}, 1OPT: {:?}, Rand: {:?}",
            p.eval_usize(&x_pso),
            p.eval_usize(&x_mixed),
            p.eval_usize(&x_gain),
            p.eval_usize(&x_opt),
            p.eval_usize(&x_rand)
        );
    }

    #[test]
    fn qubo_heuristics() {
        let eye = CsMat::eye(3);
        let p = Qubo::new(eye);
        let x_0 = Array1::from_vec(vec![1, 0, 1]);

        let x_1 = simple_local_search(&p, &x_0, 10);
        print!("{:?}", x_1);
    }

    #[test]
    fn qubo_multi_heuristics() {
        let eye = CsMat::eye(3);
        let p = Qubo::new(eye);
        let x_0 = Array1::from_vec(vec![1, 0, 1]);
        let x_1 = Array1::from_vec(vec![0, 1, 0]);
        let x_2 = Array1::from_vec(vec![1, 1, 1]);
        let xs = vec![x_0, x_1, x_2];

        let x_3 = multi_simple_local_search(&p, &xs);
        print!("{:?}", x_3);
    }

    // #[test]
    // fn test_goemans_williamson_rounding() {
    //     let p = make_solver_qubo();
    //     let x = goemans_williamson_rounding(&p, &mut make_test_prng());
    //     println!("{:?}", x);
    // }
}
