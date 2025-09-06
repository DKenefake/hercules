use crate::branch_node::QuboBBNode;
use ndarray::Array1;
use std::time;

/// Utility function to check if a node has an integer solution, and if so, returns the rounded solution
pub fn check_integer_feasibility(node: &QuboBBNode) -> (bool, Array1<usize>) {
    let mut sum = 0;
    let num_x = node.solution.len();
    let mut buffer = Array1::zeros(num_x);

    let epsilon = 1E-10;

    for i in 0..num_x {
        if let Some(val) = node.fixed_variables.get(&i) {
            sum += 1;
            buffer[i] = *val;
        } else {
            if node.solution[i] <= epsilon {
                sum += 1;
                buffer[i] = 0;
            }

            if node.solution[i] >= 1.0 - epsilon {
                sum += 1;
                buffer[i] = 1;
            }
        }
    }
    (sum == num_x, buffer)
}

/// Utility function to get current time in seconds
///
/// # Panics
///
/// If there is an error in getting the current time
pub fn get_current_time() -> f64 {
    time::SystemTime::now()
        .duration_since(time::SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_secs_f64()
}
