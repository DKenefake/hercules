use ndarray::Array1;
use crate::qubo::Qubo;

pub fn get_opt_criteria(qubo: &Qubo, x: &Array1<f64>) -> Array1<f64> {
    // Given a point, x, calculate what the
    let mut fixed = Array1::zeros(qubo.num_x());
    let grad = qubo.eval_grad(x);

    for i in 0..qubo.num_x() {
        if grad[i] <= 0.0 {
            fixed[i] = 1.0;
        }else {
            fixed[i] = 0.0;
        }
    }

    fixed
}
