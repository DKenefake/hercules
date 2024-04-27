//! This module contains lower-bounding functions for the branch and bound algorithm
//!
//! These include:
//! - Lower Bound Function Proposed in pardalos1990
//! - Lower Bound Function Proposed in Li2012 (Initial)

use crate::qubo::Qubo;
use ndarray::Array1;
use std::collections::HashMap;

/// Calculates the initial lower bound for a qubo, based on pardalos1990
///
/// Is roughly as expensive as an objective evaluation
pub fn pardalos_rodgers_lower_bound(qubo: &Qubo, fixed_variables: &HashMap<usize, usize>) -> f64 {
    // calculate the lower bound
    let mut lower_bound = 0.0;

    // calculate the quadratic terms
    for (value, (i, j)) in &qubo.q {
        if i != j {
            // if it is inside the fixed variables, then add the fixed value else finds the minimum
            if fixed_variables.contains_key(&i) && fixed_variables.contains_key(&j) {
                lower_bound += value * fixed_variables[&i] as f64 * fixed_variables[&j] as f64;
            } else if fixed_variables.contains_key(&i) {
                let x_i = fixed_variables[&i] as f64;
                lower_bound += (value * x_i).min(0.0);
            } else if fixed_variables.contains_key(&j) {
                let x_j = fixed_variables[&j] as f64;
                lower_bound += (value * x_j).min(0.0);
            } else {
                lower_bound += value.min(0.0);
            }
        }
    }

    // scale the quadratic terms
    lower_bound *= 0.5;

    // calculate the linear terms
    for i in 0..qubo.num_x() {
        let q_ii = qubo.q.get(i, i).unwrap_or(&0.0);
        // if it is inside the fixed variables, then add the fixed value else finds the minimum
        if fixed_variables.contains_key(&i) {
            let x_i = fixed_variables[&i] as f64;
            lower_bound += (qubo.c[i] + 0.5 * q_ii) * x_i;
        } else {
            lower_bound += (qubo.c[i] + 0.5 * q_ii).min(0.0);
        }
    }

    lower_bound
}

/// Calculates an initial lower bound for a qubo, based on equation 15 of li2012
///
/// Is roughly as expensive as an objective evaluation, it has been shown that it is a tighter bound
/// than the one generated in pardalos1990, and it is roughly the same computational cost
pub fn li_lower_bound(qubo: &Qubo, fixed_variables: &HashMap<usize, usize>) -> f64 {
    // tracking variable for the lower bound
    let mut lower_bound = 0.0;
    let mut a = Array1::<f64>::zeros(qubo.num_x());

    // calculate a term
    for (value, (i, j)) in &qubo.q {
        // make sure we are not on the diagonal
        if i != j {
            // check if the variables are fixed
            let x_i_fixed = fixed_variables.contains_key(&i);
            let x_j_fixed = fixed_variables.contains_key(&j);

            // if both are fixed, then the term is a constant
            if x_i_fixed && x_j_fixed {
                let x_i = fixed_variables[&i] as f64;
                let x_j = fixed_variables[&j] as f64;
                a[i] += value * x_j;
                a[j] += value * x_i;
            } else if x_i_fixed {
                let x_i = fixed_variables[&i] as f64;
                a[j] += (value * x_i).min(0.0);
                a[i] += value.min(0.0);
            } else if x_j_fixed {
                let x_j = fixed_variables[&j] as f64;
                a[i] += (value * x_j).min(0.0);
                a[j] += value.min(0.0);
            } else {
                a[i] += value.min(0.0);
                a[j] += value.min(0.0);
            }
        }
    }

    // rescale a term
    a *= 0.5;

    // calculate the lower bound
    for i in 0..qubo.num_x() {
        let q_ii = qubo.q.get(i, i).unwrap_or(&0.0);
        if fixed_variables.contains_key(&i) {
            let x_i = fixed_variables[&i] as f64;
            lower_bound += (qubo.c[i] + 0.5 * q_ii + 0.5 * a[i]) * x_i;
        } else {
            lower_bound += (qubo.c[i] + 0.5 * q_ii + 0.5 * a[i]).min(0.0);
        }
    }

    lower_bound
}

#[cfg(test)]
mod tests {
    use crate::lower_bound::{li_lower_bound, pardalos_rodgers_lower_bound};
    use crate::qubo::Qubo;
    use crate::tests::make_solver_qubo;
    use ndarray::Array1;
    use sprs::TriMat;
    use std::collections::HashMap;

    /// This is based on the first example problem in the li2012 paper
    #[test]
    fn test_lower_bounds_1() {
        let mut q = TriMat::new((3, 3));

        q.add_triplet(0, 0, 10.0);
        q.add_triplet(0, 1, 4.0);
        q.add_triplet(0, 2, 3.0);

        q.add_triplet(1, 0, 4.0);
        q.add_triplet(1, 1, 10.0);
        q.add_triplet(1, 2, 3.0);

        q.add_triplet(2, 0, 3.0);
        q.add_triplet(2, 1, 3.0);
        q.add_triplet(2, 2, 10.0);

        let c = Array1::from_vec(vec![7.0, 11.0, -7.0]);
        let qubo = Qubo::new_with_c(q.to_csr(), c);
        let fixed_vars = HashMap::new();

        let pardalos_lb = pardalos_rodgers_lower_bound(&qubo, &fixed_vars);
        let li_lb = li_lower_bound(&qubo, &fixed_vars);

        assert_eq!(pardalos_lb, -2.0);
        assert_eq!(li_lb, -2.0);
    }

    #[test]
    fn test_lower_bound_2() {
        let mut q = TriMat::new((2, 2));

        q.add_triplet(0, 0, 6.0);
        q.add_triplet(0, 1, -3.0);
        q.add_triplet(1, 0, -3.0);
        q.add_triplet(1, 1, 4.0);

        let c = Array1::from_vec(vec![-4.0, 3.0]);
        let qubo = Qubo::new_with_c(q.to_csr(), c);
        let mut fixed_vars = HashMap::new();

        fixed_vars.insert(0, 1);
        fixed_vars.insert(1, 1);

        let pardalos_lb = pardalos_rodgers_lower_bound(&qubo, &fixed_vars);
        let li_lb = li_lower_bound(&qubo, &fixed_vars);

        assert_eq!(pardalos_lb, 1.0);
        assert_eq!(li_lb, 1.0);
    }

    #[test]
    fn test_lower_bound_3() {
        let mut q = TriMat::new((2, 2));

        q.add_triplet(0, 0, 6.0);
        q.add_triplet(0, 1, -3.0);
        q.add_triplet(1, 0, -3.0);
        q.add_triplet(1, 1, 4.0);

        let c = Array1::from_vec(vec![-4.0, 3.0]);
        let qubo = Qubo::new_with_c(q.to_csr(), c);
        let mut fixed_vars = HashMap::new();

        fixed_vars.insert(0, 0);

        let pardalos_lb = pardalos_rodgers_lower_bound(&qubo, &fixed_vars);
        let li_lb = li_lower_bound(&qubo, &fixed_vars);

        assert_eq!(pardalos_lb, 0.0);
        assert_eq!(li_lb, 0.0);
    }

    #[test]
    fn test_lower_bound_4() {
        let mut q = TriMat::new((2, 2));

        q.add_triplet(0, 0, 6.0);
        q.add_triplet(0, 1, -3.0);
        q.add_triplet(1, 0, -3.0);
        q.add_triplet(1, 1, 4.0);

        let c = Array1::from_vec(vec![-4.0, 3.0]);
        let qubo = Qubo::new_with_c(q.to_csr(), c);
        let mut fixed_vars = HashMap::new();

        fixed_vars.insert(0, 1);

        let pardalos_lb = pardalos_rodgers_lower_bound(&qubo, &fixed_vars);
        let li_lb = li_lower_bound(&qubo, &fixed_vars);

        assert_eq!(pardalos_lb, -4.0);
        assert_eq!(li_lb, -2.5);
    }

    #[test]
    fn test_lower_bound_qubo_problem() {
        let p = make_solver_qubo();

        let fixed_vars = HashMap::new();

        let pardalos_lb = pardalos_rodgers_lower_bound(&p, &fixed_vars);
        let li_lb = li_lower_bound(&p, &fixed_vars);

        println!("Pardalos Lower Bound: {}", pardalos_lb);
        println!("Li Lower Bound: {}", li_lb);
    }
}
