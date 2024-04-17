// Main Backend for running K-Opt via the branch and bound algorithm

use crate::branch_stratagy::BranchStrategySelection;
use crate::branch_subproblem::SubProblemSelection;
use crate::branchbound::BBSolver;
use crate::persistence::compute_iterative_persistence;
use crate::qubo::Qubo;
use crate::solver_options::SolverOptions;
use ndarray::Array1;
use std::collections::HashMap;

/// run k-opt on a given QUBO
pub fn solve_kopt(
    qubo: &Qubo,
    fixed_variables: &HashMap<usize, f64>,
    initial_guess: Option<Array1<f64>>,
) -> Array1<f64> {
    // create a hashmap to store the persistent variables
    let mut persistent = fixed_variables.clone();

    // compute the persistent variables
    persistent = compute_iterative_persistence(qubo, &persistent, 100);

    // create a new QUBO to store the reduced QUBO
    let reduced_qubo = qubo.clone();

    let options = SolverOptions {
        fixed_variables: persistent,
        branch_strategy: BranchStrategySelection::MostViolated,
        sub_problem_solver: SubProblemSelection::Clarabel,
        max_time: 100.0,
        seed: 0,
        verbose: 1,
        threads: 1,
    };

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
