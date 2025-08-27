use crate::branch_node::QuboBBNode;
use crate::branch_subproblem::SubProblemSolver;
use crate::branch_subproblem::{SubProblemOptions, SubProblemResult};
use crate::branchbound::BBSolver;
use crate::qubo::Qubo;
use crate::subproblemsolvers::enumerate_qubo::enumerate_solve;
use clarabel::algebra::CscMatrix;
use clarabel::solver::{DefaultSettings, DefaultSolver, IPSolver, NonnegativeConeT};
use ndarray::Array1;
use sprs::{CsMat, TriMat};
use std::collections::HashMap;

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

        // find projected subproblem
        let (sub_qubo, unfixed_map, _constant) =
            make_sub_problem(&bbsolver.qubo, node.fixed_variables.clone());

        // if the sub_qubo is small enough, we can solve it directly
        if sub_qubo.num_x() <= 15 {
            let (_, sub_sol) = enumerate_solve(&sub_qubo);

            // convert the solution back to the original space
            let mut x = Array1::<f64>::zeros(bbsolver.qubo.num_x());

            // map out the unfixed variables
            for (&original, &new) in &unfixed_map {
                x[original] = sub_sol[new] as f64;
            }

            // map out the fixed variables
            for (&i, &val) in &node.fixed_variables {
                x[i] = val as f64;
            }

            let obj = bbsolver.qubo.eval(&x);
            return (obj, x);
        }

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

fn make_sub_problem(
    qubo: &Qubo,
    fixed_vars: HashMap<usize, usize>,
) -> (Qubo, HashMap<usize, usize>, f64) {
    // do some accounting
    let num_fixed = fixed_vars.len();
    let num_unfixed = qubo.num_x() - num_fixed;

    // create a new matrix and vector to store the results
    let mut Q_tri = TriMat::new((num_unfixed, num_unfixed));
    let mut c_new = Array1::<f64>::zeros(num_unfixed);
    let mut constant = 0.0;

    // make a map between the unfixed variables and the new index
    let mut unfixed_map = HashMap::new();

    for i in 0..qubo.num_x() {
        if fixed_vars.contains_key(&i) {
            continue;
        }
        let new_index = unfixed_map.len();
        unfixed_map.insert(i, new_index);
    }

    for (&q_ij, (i, j)) in &qubo.q {
        let i_fixed = fixed_vars.contains_key(&i);
        let j_fixed = fixed_vars.contains_key(&j);

        // if both variables are fixed, then we can ignore this
        if i_fixed && j_fixed {
            constant += 0.5
                * q_ij
                * (*fixed_vars.get(&i).unwrap() as f64)
                * (*fixed_vars.get(&j).unwrap() as f64);
        } else if i_fixed && !j_fixed {
            // we know that i is fixed and j is not
            let j_new = *unfixed_map.get(&j).unwrap();

            c_new[j_new] += 0.5 * q_ij * (*fixed_vars.get(&i).unwrap() as f64);
        } else if !i_fixed && j_fixed {
            // we know that j is fixed and i is not
            let i_new = *unfixed_map.get(&i).unwrap();

            c_new[i_new] += 0.5 * q_ij * (*fixed_vars.get(&j).unwrap() as f64);
        } else {
            // both variables are unfixed
            let i_new = *unfixed_map.get(&i).unwrap();
            let j_new = *unfixed_map.get(&j).unwrap();

            Q_tri.add_triplet(i_new, j_new, q_ij);
        }
    }

    for (i, &c_i) in qubo.c.iter().enumerate() {
        if fixed_vars.contains_key(&i) {
            continue;
        }
        let i_new = *unfixed_map.get(&i).unwrap();
        c_new[i_new] += c_i;
    }

    (
        Qubo::new_with_c(Q_tri.to_csc(), c_new),
        unfixed_map,
        constant,
    )
}

#[cfg(test)]
mod tests {
    use crate::qubo::Qubo;
    use crate::subproblemsolvers::clarabel_qp::ClarabelQPSolver;
    use crate::tests::make_solver_qubo;
    use ndarray::Array1;
    use sprs::{CsMat, TriMat};
    use std::collections::HashMap;

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

    #[test]
    fn test_generate_sub_problem_1() {
        // the idea of this test is, given a QUBO & some fixed variables, generate an equivalent problem

        // f(x) = 0.5<x,x>
        let q = CsMat::<f64>::eye(3);
        let c = Array1::<f64>::zeros(3);
        let p = Qubo::new_with_c(q, c);

        // we have variable x_0 fixed to 1
        let mut fixed_variables = HashMap::new();
        fixed_variables.insert(0, 1);

        // this should generate a subproblem with the following matrix
        // [1 0]  [0]
        // [0 1]  [0]

        let (sub_p, _, constant) = super::make_sub_problem(&p, fixed_variables);

        // fix the expected matrix
        let q_target = CsMat::<f64>::eye(2);
        let c_target = Array1::<f64>::zeros(2);

        // check the linear term
        for i in 0..2 {
            assert_eq!(c_target[i], sub_p.c[i]);
        }

        // check the quadratic term
        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(
                    q_target.get(i, j).unwrap_or(&0.0),
                    sub_p.q.get(i, j).unwrap_or(&0.0)
                );
            }
        }

        // check the constant term
        assert_eq!(0.5, constant);
    }

    #[test]
    fn test_generate_sub_problem_2() {
        // the idea of this test is, given a QUBO & some fixed variables, generate an equivalent problem

        // f(x) = 0.5<x,x>
        let mut q = TriMat::<f64>::new((3, 3));

        q.add_triplet(0, 0, 1.0);
        q.add_triplet(0, 1, 2.0);
        q.add_triplet(0, 2, 3.0);

        q.add_triplet(1, 0, 5.0);
        q.add_triplet(1, 1, 0.0);
        q.add_triplet(1, 2, 1.0);

        q.add_triplet(2, 0, 1.0);
        q.add_triplet(2, 1, 5.0);
        q.add_triplet(2, 2, 6.0);

        let c = Array1::<f64>::from_vec(vec![0.0, 1.0, 3.0]);
        let p = Qubo::new_with_c(q.to_csr(), c);

        // we have variable x_0 fixed to 1
        let mut fixed_variables = HashMap::new();
        fixed_variables.insert(0, 1);

        // this should generate a subproblem with the following matrix
        // [0 1]  [4.5]
        // [5 6]  [5]

        let (sub_p, _, constant) = super::make_sub_problem(&p, fixed_variables);

        // fix the expected matrix
        let mut q_target_tri = TriMat::<f64>::new((2, 2));

        q_target_tri.add_triplet(0, 0, 0.0);
        q_target_tri.add_triplet(0, 1, 1.0);
        q_target_tri.add_triplet(1, 0, 5.0);
        q_target_tri.add_triplet(1, 1, 6.0);

        let q_target: CsMat<f64> = q_target_tri.to_csr();

        let c_target = Array1::<f64>::from_vec(vec![4.5, 5.0]);

        // check the linear term
        for i in 0..2 {
            assert_eq!(c_target[i], sub_p.c[i]);
        }

        // check the quadratic term
        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(
                    q_target.get(i, j).unwrap_or(&0.0),
                    sub_p.q.get(i, j).unwrap_or(&0.0)
                );
            }
        }

        // check the constant term
        assert_eq!(0.5, constant);
    }
}
