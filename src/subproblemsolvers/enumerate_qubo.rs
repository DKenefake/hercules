use crate::qubo::Qubo;
use ndarray::Array1;

pub fn enumerate_solve(qubo: &Qubo) -> (f64, Array1<usize>) {
    let num_vars = qubo.num_x();
    let total_masks = 1usize
        .checked_shl(num_vars as u32)
        .expect("enumerate_solve only supports component sizes below the machine word width");

    let q_terms: Vec<_> = qubo.q.iter().map(|(&value, (i, j))| (i, j, value)).collect();
    let c_terms = qubo.c.to_vec();

    let mut best_obj = f64::INFINITY;
    let mut best_mask = 0usize;

    for mask in 0..total_masks {
        let mut obj = 0.0;

        for (i, &c_i) in c_terms.iter().enumerate() {
            if (mask >> i) & 1 == 1 {
                obj += c_i;
            }
        }

        for &(i, j, q_ij) in &q_terms {
            if ((mask >> i) & 1 == 1) && ((mask >> j) & 1 == 1) {
                obj += 0.5 * q_ij;
            }
        }

        if obj < best_obj {
            best_obj = obj;
            best_mask = mask;
        }
    }

    let mut best_solution = Array1::<usize>::zeros(num_vars);
    for i in 0..num_vars {
        best_solution[i] = (best_mask >> i) & 1;
    }

    (best_obj, best_solution)
}
