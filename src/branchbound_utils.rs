use crate::branch_node::QuboBBNode;
use ndarray::Array1;

/// Bare bones implementation of B&B. Currently requires the QUBO to be symmetrical and convex.

pub fn check_integer_feasibility(node: &QuboBBNode) -> (bool, Array1<f64>) {
    let mut sum = 0;
    let num_x = node.solution.len();
    let mut buffer = Array1::zeros(num_x);

    let epsilon = 1E-10;

    for i in 0..num_x {
        match node.fixed_variables.get(&i) {
            Some(val) => {
                sum += 1;
                buffer[i] = *val;
            }
            None => {
                if node.solution[i].abs() <= epsilon {
                    sum += 1;
                    buffer[i] = 0.0;
                }

                if (node.solution[i] - 1.0).abs() <= epsilon {
                    sum += 1;
                    buffer[i] = 1.0;
                }
            }
        }
    }
    (sum == num_x, buffer)
}
