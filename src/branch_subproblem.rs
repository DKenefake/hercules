use crate::branch_node::QuboBBNode;
use crate::branch_subproblem::SubProblemSelection::HerculesPGDQP;
use crate::branchbound::BBSolver;
use crate::qubo::Qubo;
use crate::subproblemsolvers::clarabel_lp::ClarabelLPSolver;
use crate::subproblemsolvers::clarabel_qp::ClarabelQPSolver;
use crate::subproblemsolvers::hercules_cd_qp::HerculesCDQPSolver;
use crate::subproblemsolvers::hercules_pgd_qp::HerculesQPSolver;
use ndarray::Array1;

pub type SubProblemResult = (f64, Array1<f64>);

#[derive(Clone, Copy)]
pub struct SubProblemOptions{
    pub max_iterations: Option<usize>
}

impl SubProblemOptions {
    pub fn new(max_iterations: Option<usize>) -> Self {
        Self { max_iterations }
    }
}


pub trait SubProblemSolver {
    fn solve_lower_bound(&self, bbsolver: &BBSolver, node: &QuboBBNode, sub_problem_options: Option<SubProblemOptions>) -> SubProblemResult;
}

pub enum SubProblemSelection {
    ClarabelQP,
    ClarabelLP,
    HerculesPGDQP,
    HerculesCDQP,
}

pub fn get_sub_problem_solver(
    qubo: &Qubo,
    sub_problem_selection: &SubProblemSelection,
) -> Box<dyn SubProblemSolver + Sync> {
    match sub_problem_selection {
        SubProblemSelection::ClarabelQP => Box::new(ClarabelQPSolver::new(qubo)),
        SubProblemSelection::ClarabelLP => Box::new(ClarabelLPSolver::new(qubo)),
        HerculesPGDQP => Box::new(HerculesQPSolver::new(qubo)),
        SubProblemSelection::HerculesCDQP => Box::new(HerculesCDQPSolver::new(qubo)),
    }
}
