use crate::branch_node::QuboBBNode;
use crate::branch_subproblem::SubProblemSolver;
use crate::branch_subproblem::{SubProblemOptions, SubProblemResult};
use crate::branchbound::BBSolver;
use crate::qubo::Qubo;
use clarabel::algebra::CscMatrix;
use clarabel::solver::{DefaultSettings, DefaultSolver, IPSolver, NonnegativeConeT, ZeroConeT};
use ndarray::Array1;
use sprs::{CsMat, TriMat};

#[derive(Clone)]
pub struct ClarabelLPSolver {}

impl ClarabelLPSolver {
    pub fn make_cb_form(p0: &CsMat<f64>) -> CscMatrix {
        let (t, y, u) = p0.to_csc().into_raw_storage();
        CscMatrix::new(p0.rows(), p0.cols(), t, y, u)
    }

    pub fn new(_: &Qubo) -> Self {
        Self {}
    }
}
impl SubProblemSolver for ClarabelLPSolver {
    fn solve_lower_bound(
        &self,
        bbsolver: &BBSolver,
        node: &QuboBBNode,
        _: Option<SubProblemOptions>,
    ) -> SubProblemResult {
        // Uses the Glover Relaxation to solve the LP associated with the node
        let settings = DefaultSettings {
            verbose: false,
            ..Default::default()
        };

        // we only care about edges : i > j
        let mut reduced_edges = Vec::new();
        for (&Q_ij, (i, j)) in &bbsolver.qubo.q {
            if i > j && Q_ij != 0.0 {
                reduced_edges.push((i, j, Q_ij));
            }
        }

        // set up the constraint matrix A = |E| + |V| by 4|E| + 2|V| + |F|
        let num_constraints = 2 * bbsolver.qubo.num_x() + 4 * reduced_edges.len() + node.fixed_variables.len();
        let num_variables = bbsolver.qubo.num_x() + reduced_edges.len();

        let mut A = TriMat::new((num_constraints,num_variables));
        let mut b = Array1::<f64>::zeros(num_constraints);

        let x_start = reduced_edges.len();
        // add the envelope reformulation for the y_ij variables
        for (index, (i, j, _)) in reduced_edges.iter().enumerate() {
            let offset = index * 4;
            // y_ij <= x_i
            A.add_triplet(offset, index, 1.0);
            A.add_triplet(offset, x_start + i, -1.0);
            b[offset] = 0.0;
            // y_ij <= x_j
            A.add_triplet(offset + 1, index, 1.0);
            A.add_triplet(offset + 1, x_start + j, -1.0);
            b[offset + 1] = 0.0;
            // y_ij >= x_i + x_j - 1 -> -y_ij + x_i + x_j <= 1
            A.add_triplet(offset + 2, index, -1.0);
            A.add_triplet(offset + 2, x_start + i, 1.0);
            A.add_triplet(offset + 2, x_start + j, 1.0);
            b[offset + 2] = 1.0;
            // y_ij >= 0 -> -y_ij <= 0
            A.add_triplet(offset + 3, index, -1.0);
            b[offset + 3] = 0.0;
        }
        // add the box constraints for the x_i variables
        for i in 0..bbsolver.qubo.num_x() {
            let offset = 4 * reduced_edges.len() + i * 2;
            // x_i <= 1
            A.add_triplet(offset, x_start + i, 1.0);
            b[offset] = 1.0;
            // x_i >= 0 -> -x_i <= 0
            A.add_triplet(offset + 1, x_start + i, -1.0);
            b[offset + 1] = 0.0;
        }

        // add the fixed variable constraints
        let fixed_offset = 4 * reduced_edges.len() + 2 * bbsolver.qubo.num_x();
        for (index, (&i, &val)) in node.fixed_variables.iter().enumerate() {
            A.add_triplet(fixed_offset + index, x_start + i, 1.0);
            b[fixed_offset + index] = val as f64;
        }

        // create the linear term with Q_ij for y_ij and 0.5Q_ii + c_i for x_i
        let mut c = Array1::<f64>::zeros(num_variables);
        for (index, (_i, _j, Q_ij)) in reduced_edges.iter().enumerate() {
            c[index] = *Q_ij;
        }

        for i in 0..bbsolver.qubo.num_x() {
            c[x_start + i] = 0.5 * bbsolver.qubo.q[[i,i]] + bbsolver.qubo.c[i];
        }

        // convert the matrix to CSC format and then Clarabel format
        let A_csc = A.to_csc();
        let A_clara = Self::make_cb_form(&A_csc);

        let cones = [NonnegativeConeT(2 * bbsolver.qubo.num_x() + 4 * reduced_edges.len()), ZeroConeT(node.fixed_variables.len())];

        // set up the solver with the matrices
        let mut solver = DefaultSolver::new(
            &Self::make_cb_form(&CsMat::zero((num_variables, num_variables))), // zero matrix for Q
            c.as_slice().unwrap(), // unwrap is safe because Array1 is stored in contiguous memory
            &A_clara,
            b.as_slice().unwrap(), // the same with b
            &cones,
            settings,
        );

        // solve the optimization problem
        solver.solve();

        // extract the solution
        let x_sol_only = &solver.solution.x[x_start..];
        let x_sol = Array1::from_vec(x_sol_only.to_vec());

        let lower_bound = solver.solution.obj_val;

        // output a dummy value for now
        (lower_bound, x_sol)
    }
}
