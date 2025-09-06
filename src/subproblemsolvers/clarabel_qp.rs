use crate::branch_node::QuboBBNode;
use crate::branch_subproblem::SubProblemSolver;
use crate::branch_subproblem::{SubProblemOptions, SubProblemResult};
use crate::branchbound::BBSolver;
use crate::preprocess::make_sub_problem;
use crate::qubo::Qubo;
use clarabel::algebra::CscMatrix;
use clarabel::solver::{DefaultSettings, DefaultSolver, IPSolver, NonnegativeConeT};
use ndarray::Array1;
use sprs::{CsMat, TriMat};

#[derive(Clone)]
pub struct ClarabelQPSolver {
    pub q: CscMatrix,
    pub c: Array1<f64>,
}

impl SubProblemSolver for ClarabelQPSolver {
    fn solve_lower_bound(
        &self,
        bbsolver: &BBSolver,
        node: &QuboBBNode,
        _: Option<SubProblemOptions>,
    ) -> SubProblemResult {
        // solve QP associated with the node
        // generate default settings
        let settings = DefaultSettings {
            verbose: false,
            ..Default::default()
        };

        // if we are fully fixed, then just return the solution
        if node.fixed_variables.len() == bbsolver.qubo.num_x() {
            let mut sol = Array1::zeros(bbsolver.qubo.num_x());
            for (&i, &val) in &node.fixed_variables {
                sol[i] = val as f64;
            }
            let obj = bbsolver.qubo.eval(&sol);
            return (obj, sol);
        }

        // find projected subproblem
        let (sub_qubo, unfixed_map, _constant) =
            make_sub_problem(&bbsolver.qubo, &node.fixed_variables);

        // generate the constraint matrix
        let A_size = 2 * sub_qubo.num_x();
        let mut A = TriMat::new((A_size, sub_qubo.num_x()));
        let mut b = Array1::<f64>::zeros(A_size);

        // add the inequality constraints
        for (index, i) in (0..sub_qubo.num_x()).enumerate() {
            let offset = index * 2;
            A.add_triplet(offset, i, 1.0);
            A.add_triplet(offset + 1, i, -1.0);
            b[offset] = 1.0;
            b[offset + 1] = 0.0;
        }

        // convert the matrix to CSC format and then Clarabel format
        let A_csc = A.to_csc();
        let A_clara = Self::make_cb_form(&A_csc);

        // generate the cones for the solver
        let cones = [NonnegativeConeT(2 * sub_qubo.num_x())];

        // set up the solver with the matrices
        let mut solver = DefaultSolver::new(
            &Self::make_cb_form(&sub_qubo.q),
            sub_qubo.c.as_slice().unwrap(), // unwrap is safe because Array1 is stored in contiguous memory
            &A_clara,
            b.as_slice().unwrap(),
            &cones,
            settings,
        );

        // solve the optimization problem
        solver.solve();

        // convert the solution back to the original space
        let mut x = Array1::<f64>::zeros(bbsolver.qubo.num_x());

        // map out the unfixed variables
        for (&original, &new) in &unfixed_map {
            x[original] = solver.solution.x[new];
        }

        // map out the fixed variables
        for (&i, &val) in &node.fixed_variables {
            x[i] = val as f64;
        }

        let obj = bbsolver.qubo.eval(&x);
        (obj, x)
    }
}

impl ClarabelQPSolver {
    pub fn new(qubo: &Qubo) -> Self {
        let q_new = Self::make_cb_form(&(qubo.q));
        Self {
            q: q_new,
            c: qubo.c.clone(),
        }
    }
    pub fn make_cb_form(p0: &CsMat<f64>) -> CscMatrix {
        let (t, y, u) = p0.to_csc().into_raw_storage();
        CscMatrix::new(p0.rows(), p0.cols(), t, y, u)
    }
}

#[cfg(test)]
mod tests {
    use crate::subproblemsolvers::clarabel_qp::ClarabelQPSolver;
    use crate::tests::make_solver_qubo;

    #[test]
    fn ensure_matrix_equivlence() {
        // the idea of this test is to make sure that the conversion from sprs to clarabel is correct

        let qubo = make_solver_qubo();

        let clarabel_matrix = ClarabelQPSolver::make_cb_form(&qubo.q);

        // check the number of non-zero elements
        assert_eq!(qubo.q.nnz(), clarabel_matrix.nnz());

        // check the values of the non-zero elements
        for (&val, (i, j)) in qubo.q.iter() {
            assert_eq!(val, clarabel_matrix.get_entry((i, j)).unwrap());
        }
    }
}
