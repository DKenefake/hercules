use crate::branch_node::QuboBBNode;
use crate::branch_subproblem::{SubProblemOptions, SubProblemResult, SubProblemSolver};
use crate::branchbound::BBSolver;
use crate::preprocess::make_sub_problem;
use crate::qubo::Qubo;
use ndarray::Array1;
use std::collections::HashMap;

#[derive(Clone)]
pub struct HerculesCDQPSolver {}

impl HerculesCDQPSolver {
    pub const fn new(_: &Qubo) -> Self {
        Self {}
    }
}

impl SubProblemSolver for HerculesCDQPSolver {
    fn solve_lower_bound(
        &self,
        bbsolver: &BBSolver,
        node: &QuboBBNode,
        sub_problem_options: Option<SubProblemOptions>,
    ) -> SubProblemResult {
        let max_iterations = sub_problem_options
            .and_then(|opts| opts.max_iterations)
            .unwrap_or(100_000);

        let x = cd_main_loop(
            node.solution.clone(),
            &bbsolver.qubo,
            &node.fixed_variables,
            max_iterations,
        );
        let obj = bbsolver.qubo.eval(&x);

        (obj, x)
    }
}

fn cd_main_loop(
    mut x_0: Array1<f64>,
    qubo: &Qubo,
    fixed_variables: &HashMap<usize, usize>,
    max_iterations: usize,
) -> Array1<f64> {
    // project to a smaller problem space
    let (sub_qubo, var_mapping, _) = make_sub_problem(qubo, fixed_variables);

    let mut x = Array1::<f64>::zeros(sub_qubo.num_x());

    // make sure that x_0 aggrees with the fixed variables
    for (&i, &val) in fixed_variables {
        x_0[i] = val as f64;
    }

    // we now need to map back to the original space
    for (&original, &new) in &var_mapping {
        x[new] = x_0[original];
    }

    fn project(mut x: Array1<f64>) -> Array1<f64> {
        // project into a box by clamping every value between 0 and 1, this is done in place (paranoia)
        for i in 0..x.len() {
            x[i] = x[i].clamp(0.0, 1.0);
        }
        x
    }

    // for safety and paranoia, project the initial solution to the box
    let mut x = project(x);

    let mut i = 0;

    while i <= max_iterations {
        // take a step in the direction of the gradient
        let diff = cd_step(&mut x, &sub_qubo);

        // check if the solution has converged, if so exit
        if diff < 1E-12 {
            break;
        }

        // iterate the iteration counter
        i += 1;
    }

    // we now need to map back to the original space
    for (&original, &new) in &var_mapping {
        x_0[original] = x[new];
    }

    x_0
}

pub fn cd_step(x: &mut Array1<f64>, qubo: &Qubo) -> f64 {
    let mut shift: f64 = 0.0;

    for i in 0..x.len() {
        let Q_i = qubo.q.outer_view(i).unwrap();

        // compute the linear term of the cd expression
        let l_i = Q_i.dot(&x) + qubo.c[i];
        let q_ii = qubo.q[[i, i]];

        // if Q_ii is zero, we have the case of a linear term only
        let d_i = -if q_ii == 0.0 {
            l_i.signum()
        } else {
            l_i / q_ii
        };

        // update the variable
        let x_moved = (x[i] + d_i).clamp(0.0, 1.0);
        shift += (x_moved - x[i]).abs();
        x[i] = x_moved;
    }

    shift
}
