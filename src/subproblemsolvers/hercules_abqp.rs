use crate::branch_node::QuboBBNode;
use crate::branch_subproblem::{
    BasicSubProblemResult, SubProblemOptions, SubProblemResult, SubProblemSolver,
};
use crate::branchbound::BBSolver;
use crate::qubo::Qubo;
use herculesabqp::matrix::QuadraticMatrix;
use herculesabqp::solver::{PreparedSolver, SolverOptions as ABQPSolverOptions};
use ndarray::Array1;

#[derive(Clone, Debug)]
pub struct HerculesABQPSolver {
    prepared_solver: PreparedSolver,
    base_options: ABQPSolverOptions,
}

impl HerculesABQPSolver {
    pub fn new(qubo: &Qubo) -> Self {
        let q = QuadraticMatrix::sparse(qubo.q.clone());
        let mut options = ABQPSolverOptions {
            assume_symmetric: true,
            ..Default::default()
        };
        options.logging.verbose = false;
        options.stopping.dual_certification = true;

        let prepared_solver = PreparedSolver::new(&q, qubo.c.as_slice().unwrap(), &options)
            .expect("Failed to prepare HerculesABQP solver");

        Self {
            prepared_solver,
            base_options: options,
        }
    }
}

impl SubProblemSolver for HerculesABQPSolver {
    fn solve_lower_bound(
        &self,
        bbsolver: &BBSolver,
        node: &QuboBBNode,
        sub_problem_options: Option<SubProblemOptions>,
    ) -> Box<dyn SubProblemResult> {
        let n = bbsolver.qubo.num_x();
        let mut lb = vec![0.0; n];
        let mut ub = vec![1.0; n];

        for (&index, &value) in &node.fixed_variables {
            let fixed_value = value as f64;
            lb[index] = fixed_value;
            ub[index] = fixed_value;
        }

        let mut options = self.base_options.clone();
        options.x0 = Some(node.solution.to_vec());
        if let Some(max_iterations) = sub_problem_options.and_then(|opts| opts.max_iterations) {
            options.stopping.max_iter = max_iterations;
        }

        let result = self
            .prepared_solver
            .solve(&lb, &ub, &options)
            .expect("HerculesABQP subproblem solve failed");

        Box::new(BasicSubProblemResult {
            lower_bound: result.quality.certified_lower_bound,
            relaxed_solution: Array1::from_vec(result.x),
        })
    }
}

