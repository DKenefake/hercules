use crate::branch_node::QuboBBNode;
use crate::branch_subproblem::SubProblemResult;
use crate::branch_subproblem::SubProblemSolver;
use crate::branchbound::BBSolver;
use crate::qubo::Qubo;
use ndarray::linalg::Dot;
use ndarray::Array1;
use ndarray_linalg::Norm;

#[derive(Clone)]
pub struct HerculesQPSolver {}

impl HerculesQPSolver {
    pub const fn new(_: &Qubo) -> Self {
        Self {}
    }
}
impl SubProblemSolver for HerculesQPSolver {
    fn solve_lower_bound(&self, bbsolver: &BBSolver, node: &QuboBBNode) -> SubProblemResult {
        let x = pgd_main_loop(node.solution.clone(), &bbsolver.qubo, node);
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

fn step(x: Array1<f64>, qubo: &Qubo, node: &QuboBBNode) -> Array1<f64> {
    let Qx = qubo.q.dot(&x);

    let dx = Qx + qubo.c.clone();

    let grad_norm = dx.dot(&dx);

    let dxQdx = dx.dot(&qubo.q.dot(&dx));

    let alpha = -grad_norm / dxQdx;

    let y = x + alpha * dx;

    project(y, node)
}

fn pgd_main_loop(x_0: Array1<f64>, qubo: &Qubo, node: &QuboBBNode) -> Array1<f64> {
    // start with projecting the initial guess to the feasible set of the node
    let mut x = project(x_0, node);
    let mut x_next = x.clone();

    let mut i = 0;
    let iteration_max = 100_000;

    while i <= iteration_max {
        // iterate the iteration counter
        i += 1;

        // take a step in the direction of the gradient
        x_next = step(x.clone(), qubo, node);

        if (x_next.clone() - x.clone()).norm_l2() < 1E-12 {
            break;
        }

        x = x_next;
    }

    x
}

#[cfg(test)]
mod tests {}
