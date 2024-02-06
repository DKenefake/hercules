//! Acts as the interface to rust code from python. Currently only supports reading the QUBO from a file, and running the one of the search algorithms.

use crate::qubo::Qubo;
use std::collections::HashMap;
use std::time;

use ndarray::Array1;
use pyo3::exceptions::socket::timeout;
use pyo3::prelude::*;

use crate::local_search;
use crate::persistence::compute_iterative_persistence;
use smolprng::{JsfLarge, PRNG};

use crate::branchbound::*;

// type alias for the qubo data object from python
type QuboData = (Vec<usize>, Vec<usize>, Vec<f64>, Vec<f64>, usize);

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

/// This reads in the QUBO from a file, and solves the QUBO using random search, returns the best solution found.
///
/// Example
/// ``` python
/// import hercules
///
/// # read in the QUBO from a file
/// problem = hercules.read_qubo("file.qubo")
///
/// # read in the QUBO from a file
/// x_soln = hercules.rand(problem, 0, 10)
/// ```
pub fn rand(problem: QuboData, seed: usize, num_points: usize) -> PyResult<Vec<f64>> {
    // read in the QUBO from file
    let p = Qubo::from_vec(problem.0, problem.1, problem.2, problem.3, problem.4);

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

/// This reads in the QUBO from a file, and solves the QUBO using particle swarm optimization, returns the best solution found.
///
/// Example
/// ``` python
/// import hercules
///
/// # read in the QUBO from a file
/// problem = hercules.read_qubo("file.qubo")
///
/// # solve via PSO
/// x_s = hercules.pso(problem, 0, 10, 100)
/// ```
#[pyfunction]
pub fn pso(
    problem: QuboData,
    seed: usize,
    num_particles: usize,
    max_steps: usize,
) -> PyResult<(Vec<f64>, f64)> {
    // read in the QUBO from file
    let p = Qubo::from_vec(problem.0, problem.1, problem.2, problem.3, problem.4);

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

/// This reads in the QUBO from a file, and solves the QUBO using gain guided local search, returns the best solution found.
///
/// Example
/// ``` python
/// import hercules
/// import random
///
/// # read in the QUBO from a file
/// problem = hercules.read_qubo("test.qubo")
/// num_x = problem[-1] // the number of x variables is the last varaible of the problem
/// x_0 = [random.randint(0,1) for _ in range(num_x)]
///
/// # read in the QUBO from a file
/// x_soln, obj = hercules.gls(problem, x_0, 10)
/// ```
#[pyfunction]
pub fn gls(problem: QuboData, x_0: Vec<f64>, max_steps: usize) -> PyResult<(Vec<f64>, f64)> {
    // read in the QUBO from file
    let p = Qubo::from_vec(problem.0, problem.1, problem.2, problem.3, problem.4);

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
/// import random
///
/// problem = hercules.read_qubo("test.qubo")
/// num_x = problem[-1] // the number of x variables is the last varaible of the problem
/// x_0 = [random.randint(0,1) for _ in range(num_x)]
///
/// # read in the QUBO from a file
/// x_soln, obj = hercules.mls_from_file("file.qubo", x_0, 10)
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

/// This solves the QUBO using mixed local search, returns the best solution found.
///
/// Example
/// ``` python
/// import hercules
/// import random
///
/// # read in the qubo problem
/// problem = hercules.read_qubo("test.qubo")
/// num_x = problem[-1] // the number of x variables is the last variable of the problem
/// x_0 = [random.randint(0,1) for _ in range(num_x)]
///
/// # solve via multi start local search
/// x_soln, obj = hercules.mls(problem, x_0, 10)
/// ```
#[pyfunction]
pub fn mls(problem: QuboData, x_0: Vec<f64>, max_steps: usize) -> PyResult<(Vec<f64>, f64)> {
    // read in the QUBO from file
    let p = Qubo::from_vec(problem.0, problem.1, problem.2, problem.3, problem.4);

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
/// import random
///
/// # read in the qubo problem
/// problem = hercules.read_qubo("test.qubo")
/// num_x = problem[-1] // the number of x variables is the last varaible of the problem
/// num_starts = 10
///
/// # initial starts of each search
/// xs = [[random.randint(0,1) for _ in range(num_x)] for _ in range(num_starts)]
///
/// # read in the QUBO from a file
/// x_solns, objs = hercules.msls_from_file("file.qubo", xs)
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

/// This solves the QUBO using multi-start local search, returns the best solution found.
///
/// Example
/// ``` python
/// import hercules
/// import random
///
/// # read in the initial problem
/// problem = hercules.read_qubo("test.qubo")
/// num_x = problem[-1] // the number of x variables is the last varaible of the problem
/// num_starts = 10;
///
/// # generate the initial starts of each search
/// xs = [[random.randint(0,1) for _ in range(num_x)] for i in range(num_starts)]
///
/// # read in the QUBO from a file
/// x_solns, objs = hercules.msls(problem, xs)
/// ```
#[pyfunction]
pub fn msls(problem: QuboData, xs: Vec<Vec<f64>>) -> PyResult<(Vec<Vec<f64>>, Vec<f64>)> {
    // read in the QUBO from file
    let p = Qubo::from_vec(problem.0, problem.1, problem.2, problem.3, problem.4);

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

/// This reads in the QUBO from a .qubo file
///
/// Example
/// ``` python
/// import hercules
///
/// # read in the QUBO from a file
/// p = hercules.read_qubo("file.qubo")
/// ```
#[pyfunction]
pub fn read_qubo(filename: String) -> PyResult<QuboData> {
    // read in the QUBO from file
    let p = Qubo::read_qubo(filename.as_str());

    let mut i = Vec::new();
    let mut j = Vec::new();
    let mut q = Vec::new();
    let c = p.c.to_vec();
    let num_x = p.num_x();

    for (k, v) in p.q.iter() {
        i.push(v.0);
        j.push(v.1);
        q.push(*k);
    }

    Ok((i, j, q, c, num_x))
}

/// This reads in the QUBO from a .qubo file
///
/// Example
/// ``` python
/// import hercules
///
/// # read in the QUBO from a file
/// p = hercules.read_qubo("file.qubo")
/// ```
#[pyfunction]
pub fn write_qubo(problem: QuboData, filename: String) -> PyResult<()> {
    // read in the QUBO from file
    let p = Qubo::from_vec(problem.0, problem.1, problem.2, problem.3, problem.4);

    // write the QUBO to file
    Qubo::write_qubo(&p, filename.as_str());
    Ok(())
}

#[pyfunction]
pub fn get_persistence(
    problem: QuboData,
    fixed: HashMap<usize, f64>,
) -> PyResult<HashMap<usize, f64>> {
    // read in the QUBO from file
    let p = Qubo::from_vec(problem.0, problem.1, problem.2, problem.3, problem.4);

    let new_fixed = compute_iterative_persistence(&p, &fixed, p.num_x());

    Ok(new_fixed)
}

#[pyfunction]
pub fn solve_branch_bound(
    problem: QuboData,
    timeout: f64,
    warm_start: Option<Vec<f64>>
) -> PyResult<(Vec<f64>, f64, f64, usize, usize)> {
    // read in the QUBO from file
    let p = Qubo::from_vec(problem.0, problem.1, problem.2, problem.3, problem.4);
    let options = SolverOptions{max_time: timeout};
    let mut solver = BBSolver::new(p, options);

    match warm_start {
        Some(x) => {
            solver.warm_start(Array1::<f64>::from(x));
        },
        None => {}
    };

    let (x, obj) = solver.solve();

    let time_elapse = time::SystemTime::now()
        .duration_since(time::SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_secs_f64() - solver.time_start;

    Ok((x.to_vec(), obj, time_elapse, solver.nodes_visited, solver.nodes_processed))
}