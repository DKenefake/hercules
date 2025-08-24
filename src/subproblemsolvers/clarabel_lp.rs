use crate::branch_node::QuboBBNode;
use crate::branch_subproblem::{SubProblemOptions, SubProblemResult};
use crate::branch_subproblem::SubProblemSolver;
use crate::branchbound::BBSolver;
use crate::qubo::Qubo;
use clarabel::algebra::CscMatrix;
use ndarray::Array1;
use sprs::CsMat;

#[derive(Clone)]
pub struct ClarabelLPSolver {
    pub q: CscMatrix,
    pub c: Array1<f64>,
}

impl ClarabelLPSolver {
    pub fn make_cb_form(p0: &CsMat<f64>) -> CscMatrix {
        let (t, y, u) = p0.to_csc().into_raw_storage();
        CscMatrix::new(p0.rows(), p0.cols(), t, y, u)
    }

    pub fn new(qubo: &Qubo) -> Self {
        let q_new = Self::make_cb_form(&(qubo.q));
        Self {
            q: q_new,
            c: qubo.c.clone(),
        }
    }
}
impl SubProblemSolver for ClarabelLPSolver {
    fn solve_lower_bound(&self, _bbsolver: &BBSolver, _node: &QuboBBNode, _: Option<SubProblemOptions>) -> SubProblemResult {
        !unimplemented!("This function is not implemented yet")
    }
}
