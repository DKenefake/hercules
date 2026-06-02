use crate::branch_node::{QuboBBNode, SubProblemNodeState};
use crate::branchbound::BBSolver;
use crate::qubo::Qubo;
use crate::subproblemsolvers::clarabel_lp::ClarabelLPSolver;
use crate::subproblemsolvers::clarabel_qp::ClarabelQPSolver;
use crate::subproblemsolvers::hercules_abqp::HerculesABQPSolver;
use crate::subproblemsolvers::hercules_cd_qp::HerculesCDQPSolver;
use crate::subproblemsolvers::mixingcut_sdp::MixingCutSDPSolver;
use crate::subproblemsolvers::roofdual::RoofDualSolver;
use ndarray::Array1;

pub trait SubProblemResult {
    fn lower_bound(&self) -> f64;
    fn relaxed_solution(&self) -> Option<&Array1<f64>>;
    fn candidate_primal_solution(&self) -> Option<&Array1<usize>>;
    fn subproblem_state(&self) -> Option<&SubProblemNodeState>;
    fn into_parts(
        self: Box<Self>,
    ) -> (
        f64,
        Option<Array1<f64>>,
        Option<Array1<usize>>,
        Option<SubProblemNodeState>,
    );
}

pub struct BasicSubProblemResult {
    pub lower_bound: f64,
    pub relaxed_solution: Array1<f64>,
}

impl SubProblemResult for BasicSubProblemResult {
    fn lower_bound(&self) -> f64 {
        self.lower_bound
    }

    fn relaxed_solution(&self) -> Option<&Array1<f64>> {
        Some(&self.relaxed_solution)
    }

    fn candidate_primal_solution(&self) -> Option<&Array1<usize>> {
        None
    }

    fn subproblem_state(&self) -> Option<&SubProblemNodeState> {
        None
    }

    fn into_parts(
        self: Box<Self>,
    ) -> (
        f64,
        Option<Array1<f64>>,
        Option<Array1<usize>>,
        Option<SubProblemNodeState>,
    ) {
        (self.lower_bound, Some(self.relaxed_solution), None, None)
    }
}

#[derive(Clone, Copy)]
pub struct SubProblemOptions {
    pub max_iterations: Option<usize>,
}

/// Options for the sub-problem solver
/// - max_iterations: maximum number of iterations to run the solver for
///   If None, the solver will run until convergence or a default maximum number of iterations
impl SubProblemOptions {
    pub const fn new(max_iterations: Option<usize>) -> Self {
        Self { max_iterations }
    }
}

/// Trait for solving sub-problems in branch and bound
/// The sub-problem solver takes in a branch and bound solver, a node, and options
/// and returns a lower bound and a solution
pub trait SubProblemSolver {
    fn solve_lower_bound(
        &self,
        bbsolver: &BBSolver,
        node: &QuboBBNode,
        sub_problem_options: Option<SubProblemOptions>,
    ) -> Box<dyn SubProblemResult>;
}

#[derive(Clone, Copy)]
pub enum SubProblemSelection {
    ClarabelQP,
    ClarabelLP,
    HerculesABQP,
    HerculesCDQP,
    MixingCutSDP,
    RoofDualQPBO,
}

pub fn get_sub_problem_solver(
    qubo: &Qubo,
    sub_problem_selection: &SubProblemSelection,
) -> Box<dyn SubProblemSolver + Sync> {
    match sub_problem_selection {
        SubProblemSelection::ClarabelQP => Box::new(ClarabelQPSolver::new(qubo)),
        SubProblemSelection::ClarabelLP => Box::new(ClarabelLPSolver::new(qubo)),
        SubProblemSelection::HerculesABQP => Box::new(HerculesABQPSolver::new(qubo)),
        SubProblemSelection::HerculesCDQP => Box::new(HerculesCDQPSolver::new(qubo)),
        SubProblemSelection::MixingCutSDP => Box::new(MixingCutSDPSolver::new(qubo)),
        SubProblemSelection::RoofDualQPBO => Box::new(RoofDualSolver::new(qubo)),
    }
}
