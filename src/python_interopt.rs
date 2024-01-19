//! Acts as the interface to rust code from python. Currently only supports reading the QUBO from a file, and running the one of the search algorithms.

use crate::qubo::Qubo;

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
    // read in the QUBO from file
    let p = Qubo::read_qubo(filename.as_str());

    // set up the prng
    let mut prng = PRNG {
        generator: JsfLarge::from(seed as u64),
    };

    // run the random search
    let x_soln = local_search::random_search(&p, num_points, &mut prng);

    // return the solution
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
    // read in the QUBO from file
    let p = Qubo::read_qubo(filename.as_str());

    // set up the prng
    let mut prng = PRNG {
        generator: JsfLarge::from(seed as u64),
    };

    // run the particle swarm search
    let x_soln = local_search::particle_swarm_search(&p, num_particles, max_steps, &mut prng);

    // return the solution and its objective
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
    // read in the QUBO from file
    let p = Qubo::read_qubo(filename.as_str());

    // convert the input to the correct type
    let x_array = Array1::<f64>::from(x_0);

    // run the gain guided local search
    let x_soln = local_search::simple_gain_criteria_search(&p, &x_array, max_steps);

    // return the solution and its objective
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
    // read in the QUBO from file
    let p = Qubo::read_qubo(filename.as_str());

    // convert the input to the correct type
    let x_array = Array1::<f64>::from(x_0);

    // run the multi-start local search
    let x_soln = local_search::simple_mixed_search(&p, &x_array, max_steps);

    // return the solution and its objective
    Ok((x_soln.to_vec(), p.eval(&x_soln)))
}

/// This reads in the QUBO from a file, and solves the QUBO using multi-start local search, returns the best solution found.
///
/// Example
/// ``` python
/// import hercules
///
/// # read in the QUBO from a file
/// x_soln = hercules.msls_from_file("file.qubo", 0, 10, 100)
/// ```
#[pyfunction]
pub fn msls_from_file(filename: String, xs: Vec<Vec<f64>>) -> PyResult<(Vec<Vec<f64>>, Vec<f64>)> {
    // read in the QUBO from file
    let p = Qubo::read_qubo(filename.as_str());

    // convert the input to the correct type
    let xs = xs.iter().map(|x| Array1::<f64>::from(x.clone())).collect();

    // run the multi-start local search
    let x_solns = local_search::multi_simple_local_search(&p, &xs);

    // convert the output to the correct type
    let x_solns_vec: Vec<Vec<_>> = x_solns.iter().map(|x| x.to_vec()).collect();

    // calculate the objective of each solution
    let objs = x_solns_vec
        .iter()
        .map(|x| p.eval(&Array1::<f64>::from(x.clone())))
        .collect();

    // return the solutions and their objectives
    Ok((x_solns_vec, objs))
}
