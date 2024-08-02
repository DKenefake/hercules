use crate::persistence::compute_iterative_persistence;
/// This file is the main module that defines the preprocessing functions
///
/// Currently the following features are implemented:
/// - Iterative persistence
use crate::qubo::Qubo;
use ndarray::Array1;
use sprs::TriMat;
use std::collections::HashMap;

/// This is the main entry point for preprocessing
pub fn preprocess_qubo(
    qubo: &Qubo,
    fixed_variables: &HashMap<usize, usize>,
    in_standard_form: bool,
) -> HashMap<usize, usize> {
    // copy the fixed variables
    let mut initial_fixed = fixed_variables.clone();

    // find variables that have no effect in the QUBO
    let no_effect_vars = fix_no_effect_variables(qubo);

    // combine the fixed variables with the no effect variables
    for (key, value) in no_effect_vars {
        initial_fixed.insert(key, value);
    }

    // create an auxiliary QUBO were we have zeroed out the diagonal elements
    if in_standard_form {
        return compute_iterative_persistence(qubo, &initial_fixed, qubo.num_x());
    }

    let qubo_shift = shift_qubo(qubo);

    // start with an initial persistence check against the zero diagonal QUBO
    // This is provably the tightest bound we can get for this calculation
    let fixed_variables =
        compute_iterative_persistence(&qubo_shift, &initial_fixed, qubo_shift.num_x());

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

/// Find variables that have no effect in the QUBO, where the linear term is zero and the quadratic
/// terms are zero. This is useful for reducing the size of the QUBO.
pub fn find_no_effect_variables(qubo: &Qubo) -> Vec<usize> {
    let mut is_no_effect_var = Array1::from_elem(qubo.num_x(), true);

    // check the quadratic terms
    for (&_value, (i, j)) in &qubo.q {
        is_no_effect_var[i] = false;
        is_no_effect_var[j] = false;
    }

    // check the linear terms
    for i in 0..qubo.num_x() {
        if qubo.c[i] != 0.0 {
            is_no_effect_var[i] = false;
        }
    }

    is_no_effect_var
        .indexed_iter()
        .filter(|(_, &value)| value)
        .map(|(i, _)| i)
        .collect()
}

/// Fixes variables that have no effect in the QUBO, where the linear term is zero and the quadratic
/// terms are zero. This is useful for reducing the size of the QUBO.
pub fn fix_no_effect_variables(qubo: &Qubo) -> HashMap<usize, usize> {
    let no_effect_vars = find_no_effect_variables(qubo);

    no_effect_vars.iter().map(|&i| (i, 0)).collect()
}

/// Creates a new QUBO where the diagonal elements are zeroed out and the linear term is adjusted
/// accordingly
pub fn shift_qubo(qubo: &Qubo) -> Qubo {
    let mut new_q = TriMat::new((qubo.num_x(), qubo.num_x()));
    let mut new_c = qubo.c.clone();

    for (&value, (i, j)) in &qubo.q {
        if i == j {
            new_c[i] += 0.5 * value;
        } else {
            new_q.add_triplet(i, j, value);
        }
    }

    Qubo::new_with_c(new_q.to_csr(), new_c)
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
        let fixed_variables = preprocess_qubo(&p, &fixed_variables, false);
        assert_eq!(fixed_variables.len(), 3);
    }
}
