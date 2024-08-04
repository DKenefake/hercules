use clarabel::algebra::CscMatrix;
use clarabel::solver::DefaultSettings;
use ndarray::Array1;
use sprs::CsMat;
use crate::branch_node::QuboBBNode;
use crate::branch_subproblem::SubProblemSolver;
use crate::branchbound::BBSolver;
use crate::qubo::Qubo;
use crate::branch_subproblem::SubProblemResult;

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


    fn solve_lower_bound(&self, bbsolver: &BBSolver, node: &QuboBBNode) -> SubProblemResult {

        let mut settings = DefaultSettings {
            verbose: false,
            ..Default::default()
        };

        // this solves the glover's reformulation of the problem, e.g. the QUBO is transformed into a linear problem

        settings.presolve_enable = true;

        // first find the number of edges that we are dealing with in this qubo
        let edges = &bbsolver.qubo.q.iter().filter(|(_, (i,j))| i < j).collect::<Vec<_>>();
        let edge_weights = Array1::from(edges.iter().map(|&v| v.0));
        let num_aux_vars = edge_weights.count();

        let num_unfixed = bbsolver.qubo.num_x() - node.fixed_variables.len();

        let P = CscMatrix::zeros((num_unfixed + num_aux_vars, num_unfixed + num_aux_vars));

        !unimplemented!("This function is not implemented yet")
    }
}
