// Main Backend for running K-Opt via the branch and bound algorithm

use crate::branchbound::BBSolver;
use crate::persistence::compute_iterative_persistence;
use crate::qubo::Qubo;
use crate::solver_options::SolverOptions;
use ndarray::Array1;
use std::collections::HashMap;

/// run k-opt on a given QUBO
pub fn solve_kopt(
    qubo: &Qubo,
    fixed_variables: &HashMap<usize, usize>,
    initial_guess: Option<Array1<usize>>,
) -> Array1<usize> {
    // create a hashmap to store the persistent variables
    let mut persistent = fixed_variables.clone();

    // compute the persistent variables
    persistent = compute_iterative_persistence(qubo, &persistent, 100);

    // create a new QUBO to store the reduced QUBO
    let reduced_qubo = qubo.clone();

    let mut options = SolverOptions::new();

    options.fixed_variables = persistent;

    // use branch and bound to solve the problem
    let mut solver = BBSolver::new(reduced_qubo, options);

    // warm start the solver, if we are provided a guess

    if let Some(x_0) = initial_guess {
        solver.warm_start(x_0);
    }

    // solve the problem
    solver.solve();

    solver.best_solution
}
