use crate::branch_node::QuboBBNode;
use crate::branchbound::BBSolver;
use crate::qubo::Qubo;
use crate::subproblemsolvers::clarabel_qp::ClarabelSubProblemSolver;
use ndarray::Array1;

pub type SubProblemResult = (f64, Array1<f64>);

pub trait SubProblemSolver {
    fn new(qubo: &Qubo) -> Self;

    fn solve_lower_bound(&self, bbsolver: &BBSolver, node: &QuboBBNode) -> SubProblemResult;
}

pub enum SubProblemSelection {
    ClarabelQP,
}

pub fn get_sub_problem_solver(
    qubo: &Qubo,
    sub_problem_selection: &SubProblemSelection,
) -> ClarabelSubProblemSolver {
    match sub_problem_selection {
        SubProblemSelection::ClarabelQP => ClarabelSubProblemSolver::new(qubo),
    }
}
