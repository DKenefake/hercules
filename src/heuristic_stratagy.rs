use ndarray::Array1;
use crate::branch_node::QuboBBNode;
use crate::branchbound::BBSolver;

pub enum HeuristicSelection {
    SimpleRounding,
}

impl HeuristicSelection {
    pub fn make_heuristic(&self, solver: &BBSolver, node: &QuboBBNode) -> (Array1<usize>,f64) {
        match self {
            Self::SimpleRounding => Self::simple_rounding(solver,node),
        }
    }

    pub fn simple_rounding(solver: &BBSolver, node: &QuboBBNode) -> (Array1<usize>,f64) {
        // round the solution to the nearest integer
        let mut rounded_solution = Array1::zeros(solver.qubo.num_x());

        for (i, &x) in node.solution.iter().enumerate() {
            if x > 0.5 {
                rounded_solution[i] = 1;
            } else {
                rounded_solution[i] = 0;
            }
        }

        let objective = solver.qubo.eval_usize(&rounded_solution);

        (rounded_solution, objective)
    }
}