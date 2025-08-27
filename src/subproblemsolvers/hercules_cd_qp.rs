use crate::branch_node::QuboBBNode;
use crate::branch_subproblem::{SubProblemOptions, SubProblemResult, SubProblemSolver};
use crate::branchbound::BBSolver;
use crate::qubo::Qubo;
use ndarray::Array1;

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

        let x = cd_main_loop(node.solution.clone(), &bbsolver.qubo, node, max_iterations);
        let obj = bbsolver.qubo.eval(&x);

        (obj, x)
    }
}

fn project(mut x: Array1<f64>, node: &QuboBBNode) -> Array1<f64> {
    // project into a box by clamping every value between 0 and 1
    for i in 0..x.len() {
        x[i] = x[i].clamp(0.0, 1.0);
    }

    // set the fixed variables
    for (key, value) in &node.fixed_variables {
        x[*key] = *value as f64;
    }

    x
}

fn cd_main_loop(
    x_0: Array1<f64>,
    qubo: &Qubo,
    node: &QuboBBNode,
    max_iterations: usize,
) -> Array1<f64> {
    let mut x = project(x_0, node);

    let mut i = 0;

    while i <= max_iterations {
        // take a step in the direction of the gradient
        let diff = cd_step(&mut x, qubo, node);

        // check if the solution has converged, if so exit
        if diff < 1E-12 {
            break;
        }

        // iterate the iteration counter
        i += 1;
    }

    x
}

pub fn cd_step(x: &mut Array1<f64>, qubo: &Qubo, node: &QuboBBNode) -> f64 {
    let mut shift: f64 = 0.0;

    for i in 0..x.len() {
        if node.fixed_variables.contains_key(&i) {
            continue; // Skip fixed variables
        }

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
