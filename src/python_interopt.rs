//! Acts as the interface to rust code from python. Currently only supports reading the QUBO from a file, and running one of the search algorithms.
use crate::qubo::Qubo;
use crate::FixedVarMap;
use std::collections::HashMap;

use ndarray::Array1;
use pyo3::prelude::*;

use crate::{kopt, local_search};
use smolprng::{JsfLarge, PRNG};

use crate::branchbound::BBSolver;
use crate::branchbound_utils::get_current_time;
use crate::graph_utils::get_all_disconnected_graphs;
use crate::preprocess::preprocess_qubo;
use crate::solver_options::SolverOptions;

// type alias for the qubo data object from python
type QuboData = (Vec<usize>, Vec<usize>, Vec<f64>, Vec<f64>, usize);

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
///
/// # Errors
///
/// This function should never error, but if it does, it will abort.
pub fn rand(problem: QuboData, seed: usize, num_points: usize) -> PyResult<Vec<usize>> {
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
/// problem = hercules.read_qubo("file.qubo")
///
/// # solve via PSO
/// x_s = hercules.pso(problem, 0, 10, 100)
/// ```
/// # Errors
///
/// This function should never error, but if it does, it will abort.
#[pyfunction]
pub fn pso(
    problem: QuboData,
    seed: usize,
    num_particles: usize,
    max_steps: usize,
) -> PyResult<(Vec<usize>, f64)> {
    // read in the QUBO from file
    let p = Qubo::from_vec(problem.0, problem.1, problem.2, problem.3, problem.4);

    // set up the prng
    let mut prng = PRNG {
        generator: JsfLarge::from(seed as u64),
    };

    // run the particle swarm search
    let x_soln = local_search::particle_swarm_search(&p, num_particles, max_steps, &mut prng);

    // return the solution and its objective
    Ok((x_soln.to_vec(), p.eval_usize(&x_soln)))
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
/// # Errors
///
/// This function should never error, but if it does, it will abort.
#[pyfunction]
pub fn gls(problem: QuboData, x_0: Vec<usize>, max_steps: usize) -> PyResult<(Vec<usize>, f64)> {
    // read in the QUBO from file
    let p = Qubo::from_vec(problem.0, problem.1, problem.2, problem.3, problem.4);

    // convert the input to the correct type
    let x_array = Array1::from(x_0);

    // run the gain guided local search
    let x_soln = local_search::simple_gain_criteria_search(&p, &x_array, max_steps);

    // return the solution and its objective
    Ok((x_soln.to_vec(), p.eval_usize(&x_soln)))
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
/// # Errors
///
/// This function should never error, but if it does, it will abort.
#[pyfunction]
pub fn mls(problem: QuboData, x_0: Vec<usize>, max_steps: usize) -> PyResult<(Vec<usize>, f64)> {
    // read in the QUBO from file
    let p = Qubo::from_vec(problem.0, problem.1, problem.2, problem.3, problem.4);

    // convert the input to the correct type
    let x_array = Array1::from(x_0);

    // run the multi-start local search
    let x_soln = local_search::simple_mixed_search(&p, &x_array, max_steps);

    // return the solution and its objective
    Ok((x_soln.to_vec(), p.eval_usize(&x_soln)))
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
/// # Errors
///
/// This function should never error, but if it does, it will abort.
#[pyfunction]
pub fn msls(problem: QuboData, xs: Vec<Vec<usize>>) -> PyResult<(Vec<Vec<usize>>, Vec<f64>)> {
    // read in the QUBO from file
    let p = Qubo::from_vec(problem.0, problem.1, problem.2, problem.3, problem.4);

    // convert the input to the correct type
    let xs = xs.into_iter().map(Array1::<usize>::from).collect();

    // run the multi-start local search
    let x_solns = local_search::multi_simple_local_search(&p, &xs);

    // calculate the objective of each solution before converting back to Python lists
    let objs = x_solns.iter().map(|x| p.eval_usize(x)).collect();

    // convert the output to the correct type
    let x_solns_vec: Vec<Vec<_>> = x_solns.iter().map(|x| x.to_vec()).collect();

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
/// # Errors
///
/// if the file does not exist, then it will abort
#[pyfunction]
#[pyo3(signature = (filename))]
pub fn read_qubo(filename: String) -> PyResult<QuboData> {
    // read in the QUBO from file
    let p = Qubo::read_qubo(filename.as_str());

    let mut i_indexs = Vec::with_capacity(p.q.nnz());
    let mut j_indexs = Vec::with_capacity(p.q.nnz());
    let mut q_values = Vec::with_capacity(p.q.nnz());
    let c_values = p.c.to_vec();
    let num_x = p.num_x();

    for (k, v) in &p.q {
        i_indexs.push(v.0);
        j_indexs.push(v.1);
        q_values.push(*k);
    }

    Ok((i_indexs, j_indexs, q_values, c_values, num_x))
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
/// # Errors
///
/// if the location does not exist, then it will abort
#[pyfunction]
#[pyo3(signature = (problem, filename,))]
pub fn write_qubo(problem: QuboData, filename: String) -> PyResult<()> {
    // read in the QUBO from file
    let p = Qubo::from_vec(problem.0, problem.1, problem.2, problem.3, problem.4);

    // write the QUBO to file
    Qubo::write_qubo(&p, filename.as_str());
    Ok(())
}

/// This function computes the persistence of the QUBO, e.g. an initial set of variables that can be fixed
/// to reduce the size of the problem.
///
/// Example
/// ``` python
/// import hercules
///
/// # read in the QUBO from a file
/// problem = hercules.read_qubo("file.qubo")
///
/// # compute the persistence
/// fixed = hercules.get_persistence(problem, {})
/// ```
///
/// # Errors
///
/// if the file does not exist, then it will abort, but the persistence calculation should never fail
#[pyfunction]
#[pyo3(signature = (problem, fixed,))]
pub fn get_persistence(
    problem: QuboData,
    fixed: HashMap<usize, usize>,
) -> PyResult<HashMap<usize, usize>> {
    // read in the QUBO from file
    let p = Qubo::from_vec(problem.0, problem.1, problem.2, problem.3, problem.4);
    let p_symm = p.make_symmetric();

    let fixed: FixedVarMap = fixed.into_iter().collect();
    let new_fixed = preprocess_qubo(&p_symm, &fixed, false);

    Ok(new_fixed.into_iter().collect())
}

/// This function computes the optimal diagonal shift the maximizes the relaxed solution of the QUBO
///
/// Example
/// ``` python
/// import hercules
///
/// # read in the QUBO from a file
/// problem = hercules.read_qubo("file.qubo")
///
/// # compute the persistence
/// diag_shift = hercules.get_sdp_shift(problem)
/// ```
///
/// # Errors
///
/// if the file does not exist, then it will abort, nut the sdp_shift calculation should never fail
#[pyfunction]
#[pyo3(signature = (problem, stat_tolerance=None))]
pub fn get_sdp_shift(problem: QuboData, stat_tolerance: Option<f64>) -> PyResult<Vec<f64>> {
    // read in the QUBO from file
    let p = Qubo::from_vec(problem.0, problem.1, problem.2, problem.3, problem.4);

    // make the QUBO symmetric
    let p_symm = p.make_symmetric();

    // compute the diag shift via mixing cut
    let rank = (2 * p_symm.num_x()).isqrt() + 1;
    let diag_shift = -mixingcut::sdp_solver::compute_approx_perturbation(
        &p_symm.q,
        Some(rank),
        None,
        None,
        stat_tolerance,
        None,
        false,
    );

    Ok(diag_shift.to_vec())
}

/// This function finds the disconnected components of the QUBO, and returns them as a list of usize vectors
/// where each vector is a list of variables in that component.
///
/// Example
/// ``` python
/// import hercules
///
/// # read in the QUBO from a file
/// problem = hercules.read_qubo("file.qubo")
///
/// # find the disconnected components
/// components = hercules.get_disconnected_components(problem)
/// ```
/// # Errors
/// This function should never error, but if it does, it will abort.

#[pyfunction]
#[pyo3(signature = (problem, fixed_vars, ))]
pub fn get_qubo_components(
    problem: QuboData,
    fixed_vars: HashMap<usize, usize>,
) -> PyResult<Vec<Vec<usize>>> {
    // read in the QUBO from file
    let p = Qubo::from_vec(problem.0, problem.1, problem.2, problem.3, problem.4);
    // make the QUBO symmetric
    let p_symm = p.make_symmetric();

    let fixed_vars: FixedVarMap = fixed_vars.into_iter().collect();
    Ok(get_all_disconnected_graphs(&p_symm, &fixed_vars))
}

/// Solves the QUBO using branch and bound, returns the best solution found.
/// Node probing is controlled by the keyword-only node_probe_candidates,
/// node_probe_max_free and node_probe_max_seconds options. Zero candidates disables it.
/// roof_dual_weak_persistencies enables optional, joint SCC-based roof reductions.
/// roof_dual_relation_penalties reuses strict root relations in the roof-dual bound.
/// root_pair_dominance adds a time-capped root-only joint two-variable test.
/// root_roof_probe_candidates and root_roof_probe_max_seconds control root
/// roof probing, using compatible weak choices to preserve one global optimum.
/// root_complement_symmetry chooses one orientation per symmetric component;
/// it defaults to true and is verified on the original input coefficients.
///
/// Example
/// ``` python
/// import hercules
///
/// # read in the QUBO from a file
/// problem = hercules.read_qubo("file.qubo")
///
/// # calculate a warm start
/// x_0, _ = hercules.pso(problem, 0, 10, 100)
///
/// # solve the QUBO using branch and bound
/// x, obj, time, nodes_visited, nodes_processed = hercules.solve_branch_bound(problem, timeout = 10.0, warm_start = x_0, seed = 12345, branch_strategy = "LargestEdges", sub_problem_solver="Clarabel", cheap_lower_bound_problem="roof_dual", threads=32, verbose=1)
/// ```
///
/// # Errors
///
/// This shouldn't error, but if it does, it will abort.
#[pyfunction]
#[pyo3(signature = (problem, timeout, warm_start=None, seed=None, branch_strategy=None, sub_problem_solver=None, cheap_lower_bound_problem=None, heuristic_selection = None, threads=None, verbose=None, *, node_probe_candidates=None, node_probe_max_free=None, node_probe_max_seconds=None, roof_dual_weak_persistencies=None, roof_dual_relation_penalties=None, root_pair_dominance=None, root_roof_probe_candidates=None, root_roof_probe_max_seconds=None, root_complement_symmetry=None, root_low_degree_elimination=None, root_dominant_edge_contraction=None, root_degree_three_elimination=None, component_decomposition=None, node_structural_reductions=None, small_block_elimination=None, component_cutoff_propagation=None, root_cut_dominance=None, sdp_dual_fixing=None))]
pub fn solve_branch_bound(
    problem: QuboData,
    timeout: f64,
    warm_start: Option<Vec<usize>>,
    seed: Option<usize>,
    branch_strategy: Option<String>,
    sub_problem_solver: Option<String>,
    cheap_lower_bound_problem: Option<String>,
    heuristic_selection: Option<String>,
    threads: Option<usize>,
    verbose: Option<usize>,
    node_probe_candidates: Option<usize>,
    node_probe_max_free: Option<usize>,
    node_probe_max_seconds: Option<f64>,
    roof_dual_weak_persistencies: Option<bool>,
    roof_dual_relation_penalties: Option<bool>,
    root_pair_dominance: Option<bool>,
    root_roof_probe_candidates: Option<usize>,
    root_roof_probe_max_seconds: Option<f64>,
    root_complement_symmetry: Option<bool>,
    root_low_degree_elimination: Option<bool>,
    root_dominant_edge_contraction: Option<bool>,
    root_degree_three_elimination: Option<bool>,
    component_decomposition: Option<bool>,
    node_structural_reductions: Option<bool>,
    small_block_elimination: Option<bool>,
    component_cutoff_propagation: Option<bool>,
    root_cut_dominance: Option<bool>,
    sdp_dual_fixing: Option<bool>,
) -> PyResult<(Vec<usize>, f64, f64, usize, usize)> {
    // read in the QUBO
    let p_input = Qubo::from_vec(problem.0, problem.1, problem.2, problem.3, problem.4);

    let mut options = SolverOptions::new();

    options.seed = seed.unwrap_or(12_345_679usize);

    options.set_branch_strategy(branch_strategy);

    options.set_sub_problem_strategy(sub_problem_solver);

    options.set_node_lower_bound_strategy(cheap_lower_bound_problem);

    options.set_heuristic_strategy(heuristic_selection);

    options.threads = threads.unwrap_or(256);

    options.verbose = verbose.unwrap_or(1);

    options.max_time = timeout;

    options.node_probe_candidates = node_probe_candidates.unwrap_or(options.node_probe_candidates);
    options.node_probe_max_free = node_probe_max_free.unwrap_or(options.node_probe_max_free);
    options.roof_dual_weak_persistencies =
        roof_dual_weak_persistencies.unwrap_or(options.roof_dual_weak_persistencies);
    options.roof_dual_relation_penalties =
        roof_dual_relation_penalties.unwrap_or(options.roof_dual_relation_penalties);
    options.root_pair_dominance = root_pair_dominance.unwrap_or(options.root_pair_dominance);
    options.root_roof_probe_candidates =
        root_roof_probe_candidates.unwrap_or(options.root_roof_probe_candidates);
    options.root_complement_symmetry =
        root_complement_symmetry.unwrap_or(options.root_complement_symmetry);
    options.root_low_degree_elimination =
        root_low_degree_elimination.unwrap_or(options.root_low_degree_elimination);
    options.root_dominant_edge_contraction =
        root_dominant_edge_contraction.unwrap_or(options.root_dominant_edge_contraction);
    options.root_degree_three_elimination =
        root_degree_three_elimination.unwrap_or(options.root_degree_three_elimination);
    options.component_decomposition =
        component_decomposition.unwrap_or(options.component_decomposition);
    options.node_structural_reductions =
        node_structural_reductions.unwrap_or(options.node_structural_reductions);
    options.small_block_elimination =
        small_block_elimination.unwrap_or(options.small_block_elimination);
    options.component_cutoff_propagation =
        component_cutoff_propagation.unwrap_or(options.component_cutoff_propagation);
    options.root_cut_dominance = root_cut_dominance.unwrap_or(options.root_cut_dominance);
    options.sdp_dual_fixing = sdp_dual_fixing.unwrap_or(options.sdp_dual_fixing);
    if let Some(seconds) = root_roof_probe_max_seconds {
        if !seconds.is_finite() || seconds < 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "root_roof_probe_max_seconds must be finite and nonnegative",
            ));
        }
        options.root_roof_probe_max_seconds = seconds;
    }
    if let Some(seconds) = node_probe_max_seconds {
        if !seconds.is_finite() || seconds < 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "node_probe_max_seconds must be finite and nonnegative",
            ));
        }
        options.node_probe_max_seconds = seconds;
    }

    let mut solver = BBSolver::new(p_input, options);

    // if we have a warm start, use it
    if let Some(x) = warm_start {
        solver.warm_start(Array1::<usize>::from(x));
    }

    let (x, obj) = solver.solve();

    let time_elapse = get_current_time() - solver.time_start;

    Ok((
        x.to_vec(),
        obj,
        time_elapse,
        solver.nodes_visited,
        solver.nodes_processed,
    ))
}

/// This function converts the QUBO to a convex symmetric form
/// and returns the QUBO in vec form
///
/// Example
/// ``` python
/// import hercules
///
/// # read in the QUBO from a file
/// problem = hercules.read_qubo("file.qubo")
///
/// # convert the QUBO to a convex symmetric form
/// new_problem = hercules.convex_symmetric_form(problem)
/// ```
///
/// # Errors
/// If the eigenvalue calculation fails, then it will abort
#[pyfunction]
#[pyo3(signature = (problem,))]
pub fn convex_symmetric_form(problem: QuboData) -> PyResult<QuboData> {
    // read in the QUBO from file
    let p = Qubo::from_vec(problem.0, problem.1, problem.2, problem.3, problem.4);
    let symm_p = p.make_symmetric();
    let eigs = symm_p.hess_eigenvalues();

    // get the lowest eigenvalue
    let min_eig = eigs.iter().fold(f64::INFINITY, |a, &b| a.min(b));

    // if the problem is already convex we don't have to convex-ify
    if min_eig > 0.0 {
        return Ok(symm_p.to_vec());
    }

    Ok(p.make_symmetric()
        .make_diag_transform(min_eig.abs() * 1.1)
        .to_vec())
}

/// This function solves the QUBO using the k-opt algorithm
/// and returns the best solution found.
///
/// Example
/// ``` python
/// import hercules
///
/// # read in the QUBO from a file
/// problem = hercules.read_qubo("file.qubo")
///
/// # solve the QUBO using k-opt
/// x, obj = hercules.k_opt(problem, {}, initial_guess = None)
/// ```
///
/// # Errors
/// This shouldn't error, but if it does, it will abort.
#[pyfunction]
#[pyo3(signature = (problem, fixed, initial_guess=None))]
pub fn k_opt(
    problem: QuboData,
    fixed: HashMap<usize, usize>,
    initial_guess: Option<Vec<usize>>,
) -> PyResult<Vec<usize>> {
    // read in the QUBO from vec form
    let p = Qubo::from_vec(problem.0, problem.1, problem.2, problem.3, problem.4);
    let persistent: FixedVarMap = fixed.into_iter().collect();

    let warm_start = initial_guess.map(Array1::<usize>::from);

    Ok(kopt::solve_kopt(&p, &persistent, warm_start).to_vec())
}
