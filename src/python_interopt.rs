//! Acts as the interface to rust code from python. Currently only supports reading the QUBO from a file, and running the one of the search algorithms.

use crate::initial_points;
use crate::local_search_utils;
use crate::qubo::Qubo;
use crate::utils;
use pyo3::prelude::*;

use crate::local_search;
use smolprng::{JsfLarge, PRNG};

/// Formats the sum of two numbers as string.
#[pyfunction]
pub fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// This reads in the QUBO from a file, and solves the QUBO using random search, returns the best solution found.
///
/// Example
/// ``` python
/// import hercules
///
/// # read in the QUBO from a file
/// x_soln = hercules.rand_from_file("file.qubo", 0, 10)
/// ```
pub fn rand_from_file(filename: String, seed: usize, num_points: usize) -> PyResult<Vec<f64>> {
    let p = Qubo::read_qubo(filename.as_str());

    let mut prng = PRNG {
        generator: JsfLarge::from(seed as u64),
    };

    let x_soln = local_search::random_search(&p, num_points, &mut prng);

    Ok(x_soln.to_vec())
}

/// This reads in the QUBO from a file, and solves the QUBO using particle swarm optimization, returns the best solution found.
///
/// Example
/// ``` python
/// import hercules
///
/// # read in the QUBO from a file
/// x_s = hercules.pso_from_file("file.qubo", 0, 10, 100)
/// ```
#[pyfunction]
pub fn pso_from_file(
    filename: String,
    seed: usize,
    num_particles: usize,
    max_steps: usize,
) -> PyResult<Vec<f64>> {
    let p = Qubo::read_qubo(filename.as_str());

    let mut prng = PRNG {
        generator: JsfLarge::from(seed as u64),
    };

    let x_s = local_search::particle_swarm_search(&p, num_particles, max_steps, &mut prng);

    Ok(x_s.to_vec())
}

/// This reads in the QUBO from a file, and solves the QUBO using gain guided local search, returns the best solution found.
///
/// Example
/// ``` python
/// import hercules
///
/// # read in the QUBO from a file
/// x_soln = hercules.gls_from_file("file.qubo", 0, 10)
/// ```
#[pyfunction]
pub fn gls_from_file(filename: String, seed: usize, max_steps: usize) -> PyResult<Vec<f64>> {
    let p = Qubo::read_qubo(filename.as_str());

    let mut prng = PRNG {
        generator: JsfLarge::from(seed as u64),
    };

    let x_0 = initial_points::generate_random_binary_point(&p, &mut prng, 0.5);

    let x_soln = local_search::simple_gain_criteria_search(&p, &x_0, max_steps);

    Ok(x_soln.to_vec())
}
