use crate::qubo::Qubo;
use ndarray::Array1;

pub fn get_opt_criteria(qubo: &Qubo, x: &Array1<f64>) -> Array1<f64> {
    // calculate the optimal criteria for each variable, given the point x
    // if the gradient is negative, then the optimal criteria is 1.0 for x_1
    // if the gradient is positive, then the optimal criteria is 0.0 for x_1

    let mut fixed = Array1::zeros(qubo.num_x());
    let grad = qubo.eval_grad(x);

    for i in 0..qubo.num_x() {
        if grad[i] <= 0.0 {
            fixed[i] = 1.0;
        } else {
            fixed[i] = 0.0;
        }
    }

    fixed
}

pub fn compute_d(x_0: &Array1<f64>, grad: &Array1<f64>) -> Array1<f64> {
    // compute the variable importance function
    let mut d = Array1::<f64>::zeros(x_0.len());
    for i in 0..x_0.len() {
        // find the max of the two terms
        d[i] = f64::max(-x_0[i] * grad[i], (1.0 - x_0[i]) * grad[i]);
    }
    d
}

pub fn compute_I(d: &Array1<f64>) -> Vec<usize> {
    // compute the variable selection function
    d.iter()
        .filter(|x| **x > 0.0)
        .enumerate()
        .map(|(i, _)| i)
        .collect()
}

pub fn one_flip_objective(qubo: &Qubo, x_0: &Array1<f64>) -> (f64, Array1<f64>) {
    // Efficient calculation of the delta of the objective function for a single bit flip for each variable
    // more or less this is a helper function that allows for selecting the best bit to flip option without
    // having to calculate the objective function for each bit flip, independently.
    //
    // Run time is O(|Q|) + O(|x|)
    //

    let mut objs = Array1::<f64>::zeros(qubo.num_x());

    let x_q = 0.5 * (&qubo.q * x_0);
    let q_x = 0.5 * (&qubo.q.transpose_view() * x_0);
    let q_jj = 0.5 * qubo.q.diag().to_dense();
    let delta = 1.0 - 2.0 * x_0;

    for i in 0..qubo.num_x() {
        objs[i] = q_jj[i] + delta[i] * (x_q[i] + q_x[i] + qubo.c[i]);
    }

    let obj_0 = x_0.dot(&x_q) + qubo.c.dot(x_0);
    (obj_0, objs)
}
