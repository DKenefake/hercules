use crate::branch_node::QuboBBNode;
use crate::branchbound::BBSolver;
use crate::{local_search_utils, utils};
use ndarray::Array1;

pub enum HeuristicSelection {
    SimpleRounding,
    LocalSearch,
}

impl HeuristicSelection {
    pub fn make_heuristic(&self, solver: &BBSolver, node: &QuboBBNode) -> (Array1<usize>, f64) {
        match self {
            Self::SimpleRounding => Self::simple_rounding(solver, node),
            Self::LocalSearch => Self::local_search(solver, node),
        }
    }

    pub fn simple_rounding(solver: &BBSolver, node: &QuboBBNode) -> (Array1<usize>, f64) {
        // round the solution to the nearest integer
        let rounded_solution = utils::rounded_vector(&node.solution);
        let objective = solver.qubo.eval_usize(&rounded_solution);

        (rounded_solution, objective)
    }

    pub fn local_search(solver: &BBSolver, node: &QuboBBNode) -> (Array1<usize>, f64) {
        // round the solution to the nearest integer
        let rounded_solution = utils::rounded_vector(&node.solution);

        let mut x = rounded_solution;

        // make a vector of all the variables not in node.fixed_variables
        let variables: Vec<usize> = (0..solver.qubo.num_x())
            .filter(|i| !node.fixed_variables.contains_key(i))
            .collect();

        let mut x_1 =
            local_search_utils::one_step_local_search_improved(&solver.qubo, &x, &variables);
        let mut steps = 0;

        while x_1 != x && steps <= usize::min(variables.len(), 50) {
            x = x_1.clone();
            x_1 = local_search_utils::one_step_local_search_improved(&solver.qubo, &x, &variables);
            steps += 1;
        }

        let rounded_solution = x_1;

        let objective = solver.qubo.eval_usize(&rounded_solution);

        (rounded_solution, objective)
    }
}
