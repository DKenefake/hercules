use crate::branch_node::QuboBBNode;
use crate::branchbound::BBSolver;
use crate::qubo::Qubo;
use clarabel::algebra::CscMatrix;
use clarabel::solver::{DefaultSettings, DefaultSolver, IPSolver, NonnegativeConeT, ZeroConeT};
use ndarray::Array1;
use sprs::{CsMat, TriMat};

type SubProblemResult = (f64, Array1<f64>);

pub trait SubProblemSolver {
    fn new(qubo: &Qubo) -> Self;

    fn solve(&self, bbsolver: &BBSolver, node: &QuboBBNode) -> SubProblemResult;
}

pub enum SubProblemSelection {
    Clarabel,
}

pub fn get_sub_problem_solver(
    qubo: &Qubo,
    sub_problem_selection: &SubProblemSelection,
) -> ClarabelSubProblemSolver {
    match sub_problem_selection {
        SubProblemSelection::Clarabel => ClarabelSubProblemSolver::new(&qubo),
    }
}

#[derive(Clone)]
pub struct ClarabelSubProblemSolver {
    q: CscMatrix,
    c: Array1<f64>,
}

impl SubProblemSolver for ClarabelSubProblemSolver {
    fn new(qubo: &Qubo) -> Self {
        let q_new = Self::make_cb_form(&(qubo.q));
        Self {
            q: q_new,
            c: qubo.c.clone(),
        }
    }

    fn solve(&self, bbsolver: &BBSolver, node: &QuboBBNode) -> SubProblemResult {
        // solve QP associated with the node
        // generate default settings
        let settings = DefaultSettings {
            verbose: false,
            ..Default::default()
        };

        // generate the constraint matrix
        let A_size = 2 * bbsolver.qubo.num_x() + node.fixed_variables.len();
        let mut A = TriMat::new((A_size, bbsolver.qubo.num_x()));
        let mut b = Array1::<f64>::zeros(A_size);

        // add the equality constraints
        for (index, (&key, &value)) in node.fixed_variables.iter().enumerate() {
            A.add_triplet(index, key, 1.0);
            b[index] = value as f64;
        }

        // add the inequality constraints
        for (index, i) in (0..bbsolver.qubo.num_x()).enumerate() {
            let offset = node.fixed_variables.len() + index * 2;
            A.add_triplet(offset, i, 1.0);
            A.add_triplet(offset + 1, i, -1.0);
            b[offset] = 1.0;
            b[offset + 1] = 0.0;
        }

        // convert the matrix to CSC format and then Clarabel format
        let A_csc = A.to_csc();
        let A_clara = Self::make_cb_form(&A_csc);

        // generate the cones for the solver
        let cones = [
            ZeroConeT(node.fixed_variables.len()),
            NonnegativeConeT(2 * bbsolver.qubo.num_x()),
        ];

        // set up the solver with the matrices
        let mut solver = DefaultSolver::new(
            &self.q,
            self.c.as_slice().unwrap(),
            &A_clara,
            b.as_slice().unwrap(),
            &cones,
            settings,
        );

        // solve the optimization problem
        solver.solve();

        (solver.solution.obj_val, Array1::from(solver.solution.x))
    }
}

impl ClarabelSubProblemSolver {
    pub fn make_cb_form(p0: &CsMat<f64>) -> CscMatrix {
        let (t, y, u) = p0.to_csc().into_raw_storage();
        CscMatrix::new(p0.rows(), p0.cols(), t, y, u)
    }
}
