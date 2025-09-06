use crate::branch_node::QuboBBNode;
use crate::branchbound::BBSolver;
use crate::qubo::Qubo;
use crate::subproblemsolvers::clarabel_lp::ClarabelLPSolver;
use crate::subproblemsolvers::clarabel_qp::ClarabelQPSolver;
use crate::subproblemsolvers::hercules_cd_qp::HerculesCDQPSolver;
use ndarray::Array1;

pub type SubProblemResult = (f64, Array1<f64>);

#[derive(Clone, Copy)]
pub struct SubProblemOptions {
    pub max_iterations: Option<usize>,
}

impl SubProblemOptions {
    pub const fn new(max_iterations: Option<usize>) -> Self {
        Self { max_iterations }
    }
}

pub trait SubProblemSolver {
    fn solve_lower_bound(
        &self,
        bbsolver: &BBSolver,
        node: &QuboBBNode,
        sub_problem_options: Option<SubProblemOptions>,
    ) -> SubProblemResult;
}

#[derive(Clone, Copy)]
pub enum SubProblemSelection {
    ClarabelQP,
    ClarabelLP,
    HerculesCDQP,
}

pub fn get_sub_problem_solver(
    qubo: &Qubo,
    sub_problem_selection: &SubProblemSelection,
) -> Box<dyn SubProblemSolver + Sync> {
    match sub_problem_selection {
        SubProblemSelection::ClarabelQP => Box::new(ClarabelQPSolver::new(qubo)),
        SubProblemSelection::ClarabelLP => Box::new(ClarabelLPSolver::new(qubo)),
        SubProblemSelection::HerculesCDQP => Box::new(HerculesCDQPSolver::new(qubo)),
    }
}
