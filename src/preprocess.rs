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

/// solve small unconnected components optimally via enumerition if the number of variables is small
/// enough
pub fn solve_small_components(
    qubo: &Qubo,
    fixed_vars: &HashMap<usize, usize>,
    max_size: usize,
) -> HashMap<usize, usize> {
    // copy the fixed variables
    let mut new_fixed_variables = fixed_vars.clone();

    // get all the disconnected graphs
    let components = crate::graph_utils::get_all_disconnected_graphs(qubo, &new_fixed_variables);

    for component in components {
        // if the component is too large, skip it same with it being empty
        if component.len() > max_size || component.is_empty() {
            continue;
        }

        // remove the fixed variables from the component
        let (sub_qubo, mapping) = make_component_qubo(qubo, &component, &new_fixed_variables);

        // solve the subproblem via enumeration
        let (_, solution) = crate::subproblemsolvers::enumerate_qubo::enumerate_solve(&sub_qubo);

        // map the solution back to the original variables
        for (key, &value) in &mapping {
            new_fixed_variables.insert(*key, solution[value]);
        }
    }

    new_fixed_variables
}

pub fn make_component_qubo(
    qubo: &Qubo,
    component: &Vec<usize>,
    fixed_vars: &HashMap<usize, usize>,
) -> (Qubo, HashMap<usize, usize>) {
    let mut Q_tri = TriMat::new((component.len(), component.len()));
    let mut c_new = Array1::<f64>::zeros(component.len());

    let mut index_map = HashMap::new();

    for (new_index, &old_index) in component.iter().enumerate() {
        index_map.insert(old_index, new_index);
        c_new[new_index] = qubo.c[old_index];
    }

    for (&value, (i, j)) in &qubo.q {
        if index_map.contains_key(&i) && index_map.contains_key(&j) {
            let i_new = index_map[&i];
            let j_new = index_map[&j];
            Q_tri.add_triplet(i_new, j_new, value);
        } else if index_map.contains_key(&i) && fixed_vars.contains_key(&j) {
            let i_new = index_map[&i];
            c_new[i_new] += 0.5 * value * (fixed_vars[&j] as f64);
        } else if index_map.contains_key(&j) && fixed_vars.contains_key(&i) {
            let j_new = index_map[&j];
            c_new[j_new] += 0.5 * value * (fixed_vars[&i] as f64);
        }
    }

    (Qubo::new_with_c(Q_tri.to_csc(), c_new), index_map)
}

pub fn make_sub_problem(
    qubo: &Qubo,
    fixed_vars: &HashMap<usize, usize>,
) -> (Qubo, HashMap<usize, usize>, f64) {
    // do some accounting
    let num_fixed = fixed_vars.len();
    let num_unfixed = qubo.num_x() - num_fixed;

    // create a new matrix and vector to store the results
    let mut Q_tri = TriMat::new((num_unfixed, num_unfixed));
    let mut c_new = Array1::<f64>::zeros(num_unfixed);
    let mut constant = 0.0;

    // make a map between the unfixed variables and the new index
    let mut unfixed_map = HashMap::new();

    for i in 0..qubo.num_x() {
        if fixed_vars.contains_key(&i) {
            continue;
        }
        let new_index = unfixed_map.len();
        unfixed_map.insert(i, new_index);
    }

    for (&q_ij, (i, j)) in &qubo.q {
        let i_fixed = fixed_vars.contains_key(&i);
        let j_fixed = fixed_vars.contains_key(&j);

        // if both variables are fixed, then we can ignore this
        if i_fixed && j_fixed {
            constant += 0.5
                * q_ij
                * (*fixed_vars.get(&i).unwrap() as f64)
                * (*fixed_vars.get(&j).unwrap() as f64);
        } else if i_fixed && !j_fixed {
            // we know that i is fixed and j is not
            let j_new = *unfixed_map.get(&j).unwrap();

            c_new[j_new] += 0.5 * q_ij * (*fixed_vars.get(&i).unwrap() as f64);
        } else if !i_fixed && j_fixed {
            // we know that j is fixed and i is not
            let i_new = *unfixed_map.get(&i).unwrap();

            c_new[i_new] += 0.5 * q_ij * (*fixed_vars.get(&j).unwrap() as f64);
        } else {
            // both variables are unfixed
            let i_new = *unfixed_map.get(&i).unwrap();
            let j_new = *unfixed_map.get(&j).unwrap();

            Q_tri.add_triplet(i_new, j_new, q_ij);
        }
    }

    for (i, &c_i) in qubo.c.iter().enumerate() {
        if fixed_vars.contains_key(&i) {
            continue;
        }
        let i_new = *unfixed_map.get(&i).unwrap();
        c_new[i_new] += c_i;
    }

    (
        Qubo::new_with_c(Q_tri.to_csc(), c_new),
        unfixed_map,
        constant,
    )
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
    use sprs::{CsMat, TriMat};
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

    #[test]
    fn test_generate_sub_problem_1() {
        // the idea of this test is, given a QUBO & some fixed variables, generate an equivalent problem

        // f(x) = 0.5<x,x>
        let q = CsMat::<f64>::eye(3);
        let c = Array1::<f64>::zeros(3);
        let p = Qubo::new_with_c(q, c);

        // we have variable x_0 fixed to 1
        let mut fixed_variables = HashMap::new();
        fixed_variables.insert(0, 1);

        // this should generate a subproblem with the following matrix
        // [1 0]  [0]
        // [0 1]  [0]

        let (sub_p, _, constant) = super::make_sub_problem(&p, &fixed_variables);

        // fix the expected matrix
        let q_target = CsMat::<f64>::eye(2);
        let c_target = Array1::<f64>::zeros(2);

        // check the linear term
        for i in 0..2 {
            assert_eq!(c_target[i], sub_p.c[i]);
        }

        // check the quadratic term
        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(
                    q_target.get(i, j).unwrap_or(&0.0),
                    sub_p.q.get(i, j).unwrap_or(&0.0)
                );
            }
        }

        // check the constant term
        assert_eq!(0.5, constant);
    }

    #[test]
    fn test_generate_sub_problem_2() {
        // the idea of this test is, given a QUBO & some fixed variables, generate an equivalent problem

        // f(x) = 0.5<x,x>
        let mut q = TriMat::<f64>::new((3, 3));

        q.add_triplet(0, 0, 1.0);
        q.add_triplet(0, 1, 2.0);
        q.add_triplet(0, 2, 3.0);

        q.add_triplet(1, 0, 5.0);
        q.add_triplet(1, 1, 0.0);
        q.add_triplet(1, 2, 1.0);

        q.add_triplet(2, 0, 1.0);
        q.add_triplet(2, 1, 5.0);
        q.add_triplet(2, 2, 6.0);

        let c = Array1::<f64>::from_vec(vec![0.0, 1.0, 3.0]);
        let p = Qubo::new_with_c(q.to_csr(), c);

        // we have variable x_0 fixed to 1
        let mut fixed_variables = HashMap::new();
        fixed_variables.insert(0, 1);

        // this should generate a subproblem with the following matrix
        // [0 1]  [4.5]
        // [5 6]  [5]

        let (sub_p, _, constant) = super::make_sub_problem(&p, &fixed_variables);

        // fix the expected matrix
        let mut q_target_tri = TriMat::<f64>::new((2, 2));

        q_target_tri.add_triplet(0, 0, 0.0);
        q_target_tri.add_triplet(0, 1, 1.0);
        q_target_tri.add_triplet(1, 0, 5.0);
        q_target_tri.add_triplet(1, 1, 6.0);

        let q_target: CsMat<f64> = q_target_tri.to_csr();

        let c_target = Array1::<f64>::from_vec(vec![4.5, 5.0]);

        // check the linear term
        for i in 0..2 {
            assert_eq!(c_target[i], sub_p.c[i]);
        }

        // check the quadratic term
        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(
                    q_target.get(i, j).unwrap_or(&0.0),
                    sub_p.q.get(i, j).unwrap_or(&0.0)
                );
            }
        }

        // check the constant term
        assert_eq!(0.5, constant);
    }
}
