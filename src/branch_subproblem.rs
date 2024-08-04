use crate::branch_node::QuboBBNode;
use crate::branchbound::BBSolver;
use crate::qubo::Qubo;
use crate::subproblemsolvers::clarabel_qp::ClarabelQPSolver;
use ndarray::Array1;
use crate::subproblemsolvers::clarabel_lp::ClarabelLPSolver;

pub type SubProblemResult = (f64, Array1<f64>);

pub trait SubProblemSolver {
    fn solve_lower_bound(&self, bbsolver: &BBSolver, node: &QuboBBNode) -> SubProblemResult;
}

pub enum SubProblemSelection {
    ClarabelQP,
    ClarabelLP,
}

pub fn get_sub_problem_solver(
    qubo: &Qubo,
    sub_problem_selection: &SubProblemSelection,
) -> Box<dyn SubProblemSolver + Sync> {
    match sub_problem_selection {
        SubProblemSelection::ClarabelQP => Box::new(ClarabelQPSolver::new(qubo)),
        SubProblemSelection::ClarabelLP => Box::new(ClarabelLPSolver::new(qubo)),
    }
}
