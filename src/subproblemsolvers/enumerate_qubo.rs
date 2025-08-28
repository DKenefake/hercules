use crate::qubo::Qubo;
use ndarray::Array1;

pub fn enumerate_solve(qubo: &Qubo) -> (f64, Array1<usize>) {
    // Enumerate all possible binary solutions solutions
    let num_vars = qubo.num_x();
    let mut best_obj = f64::INFINITY;
    let mut best_solution = Array1::<usize>::zeros(num_vars);
    let mut solution = Array1::<usize>::zeros(num_vars);

    for i in 0..=(1 << num_vars) {
        for j in 0..num_vars {
            solution[j] = (i >> j) & 1;
        }

        let obj = qubo.eval_usize(&solution);
        if obj < best_obj {
            best_obj = obj;
            best_solution.clone_from(&solution);
        }
    }

    (best_obj, best_solution)
}
