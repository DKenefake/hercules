//! Acts as the interface to rust code from python. Currently only supports reading the QUBO from a file, and running the one of the search algorithms.

use crate::initial_points;
use crate::local_search_utils;
use crate::qubo::Qubo;
use crate::utils;
use ndarray::Array1;
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
) -> PyResult<(Vec<f64>, f64)> {
    let p = Qubo::read_qubo(filename.as_str());

    let mut prng = PRNG {
        generator: JsfLarge::from(seed as u64),
    };

    let x_soln = local_search::particle_swarm_search(&p, num_particles, max_steps, &mut prng);

    Ok((x_soln.to_vec(), p.eval(&x_soln)))
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
pub fn gls_from_file(
    filename: String,
    x_0: Vec<f64>,
    max_steps: usize,
) -> PyResult<(Vec<f64>, f64)> {
    let p = Qubo::read_qubo(filename.as_str());

    let x_array = Array1::<f64>::from(x_0);

    let x_soln = local_search::simple_gain_criteria_search(&p, &x_array, max_steps);

    Ok((x_soln.to_vec(), p.eval(&x_soln)))
}

/// This reads in the QUBO from a file, and solves the QUBO using mixed local search, returns the best solution found.
///
/// Example
/// ``` python
/// import hercules
///
/// # read in the QUBO from a file
/// x_soln = hercules.mls_from_file("file.qubo", 0, 10)
/// ```
#[pyfunction]
pub fn mls_from_file(
    filename: String,
    x_0: Vec<f64>,
    max_steps: usize,
) -> PyResult<(Vec<f64>, f64)> {
    let p = Qubo::read_qubo(filename.as_str());

    let x_array = Array1::<f64>::from(x_0);

    let x_soln = local_search::simple_mixed_search(&p, &x_array, max_steps);

    Ok((x_soln.to_vec(), p.eval(&x_soln)))
}
