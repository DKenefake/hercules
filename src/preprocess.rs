use crate::persistence::compute_iterative_persistence;
/// This file is the main module that defines the preprocessing functions
///
/// Currently the following features are implemented:
/// - Iterative persistence
use crate::qubo::Qubo;
use ndarray::Array1;
use std::collections::HashMap;

/// This is the main entry point for preprocessing
pub fn preprocess_qubo(
    qubo: &Qubo,
    fixed_variables: &HashMap<usize, usize>,
) -> HashMap<usize, usize> {
    let initial_fixed = fixed_variables.clone();

    // start with an initial persistence check
    let fixed_variables = compute_iterative_persistence(&qubo, &initial_fixed, qubo.num_x());

    fixed_variables
}

/// This function is used to get the effect of the fixed variables on the linear term, we want to
/// avoid generating copies of the Qubo object
///
/// This is generally as expensive as a function evaluation
pub fn get_fixed_c(qubo: &Qubo, fixed_variables: &HashMap<usize, usize>) -> Array1<f64> {
    let mut new_c = qubo.c.clone();

    // there is likely a better way to do this, but for now we are looping through the Hessian
    for (&value, (i, j)) in &qubo.q {
        // diagonal elements will never be extracted
        if i == j {
            continue;
        }

        // check we have fixed variables
        let x_i_fixed = fixed_variables.contains_key(&i);
        let x_j_fixed = fixed_variables.contains_key(&j);

        // if both are fixed, then the term is a constant and it doesn't matter
        if x_i_fixed && x_j_fixed {
            continue;
        }

        if x_i_fixed {
            let x_i = fixed_variables[&i] as f64;
            new_c[j] += value * x_i;
        } else if x_j_fixed {
            let x_j = fixed_variables[&j] as f64;
            new_c[i] += (value * x_j).min(0.0);
        }

        // if neither is fixed, then we don't need to do anything
    }

    new_c
}

#[cfg(test)]
mod tests {
    use crate::preprocess::preprocess_qubo;
    use crate::qubo::Qubo;
    use ndarray::Array1;
    use sprs::CsMat;
    use std::collections::HashMap;

    #[test]
    fn test_preprocess_qubo_1() {
        let eye = CsMat::eye(3);
        let c = Array1::from_vec(vec![1.1, 2.0, 3.0]);
        let p = Qubo::new_with_c(eye, c);
        let fixed_variables = HashMap::new();
        let fixed_variables = preprocess_qubo(&p, &fixed_variables);
        assert_eq!(fixed_variables.len(), 3);
    }
}
