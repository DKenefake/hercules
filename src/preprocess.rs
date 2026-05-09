use crate::persistence::compute_iterative_persistence;
use crate::subproblemsolvers::roofdual::roof_duality_presolve;
/// This file is the main module that defines the preprocessing functions

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
        let fixed_variables = compute_iterative_persistence(qubo, &initial_fixed, qubo.num_x());
        return apply_roof_duality_closure(qubo, fixed_variables);
    }

    let qubo_shift = shift_qubo(qubo);

    // start with an initial persistence check against the zero diagonal QUBO
    // This is provably the tightest bound we can get for this calculation
    let fixed_variables =
        compute_iterative_persistence(&qubo_shift, &initial_fixed, qubo_shift.num_x());

    apply_roof_duality_closure(&qubo_shift, fixed_variables)
}

fn apply_roof_duality_closure(
    qubo: &Qubo,
    mut fixed_variables: HashMap<usize, usize>,
) -> HashMap<usize, usize> {
    let roof_dual = roof_duality_presolve(qubo, &fixed_variables);
    let mut changed = false;

    for (key, value) in roof_dual.fixed_variables {
        if fixed_variables.insert(key, value) != Some(value) {
            changed = true;
        }
    }

    if changed {
        compute_iterative_persistence(qubo, &fixed_variables, qubo.num_x())
    } else {
        fixed_variables
    }
}

/// This is the heavy entry point for preprocessing + variable probing
pub fn preprocess_qubo_heavy(
    qubo: &Qubo,
    fixed_variables: &HashMap<usize, usize>,
    in_standard_form: bool,
) -> HashMap<usize, usize> {
    // copy the fixed variables
    let mut new_persistent = fixed_variables.clone();

    // the number of required iterations is always below the number of variables
    let iters = qubo.num_x();

    // loop over the number of iters
    for _ in 0..iters {
        let mut incoming_persistent = preprocess_qubo(&qubo, &new_persistent, in_standard_form);

        let (_, probe_fixes) = crate::variable_reduction::probe(&qubo, &new_persistent, in_standard_form);

        // add the probe fixes to the incoming persistent
        for (key, value) in probe_fixes {
            incoming_persistent.insert(key, value);
        }

        if new_persistent == incoming_persistent {
            return new_persistent;
        }
        new_persistent = incoming_persistent;
    }

    new_persistent
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
    component: &[usize],
    fixed_vars: &HashMap<usize, usize>,
) -> (Qubo, HashMap<usize, usize>) {
    let mut Q_tri = TriMat::new((component.len(), component.len()));
    let mut c_new = Array1::<f64>::zeros(component.len());

    let mut index_map = HashMap::with_capacity(component.len());

    for (new_index, &old_index) in component.iter().enumerate() {
        index_map.insert(old_index, new_index);
        c_new[new_index] = qubo.c[old_index];
    }

    for (&value, (i, j)) in &qubo.q {
        match (
            index_map.get(&i),
            index_map.get(&j),
            fixed_vars.get(&i),
            fixed_vars.get(&j),
        ) {
            (Some(&i_new), Some(&j_new), _, _) => {
                Q_tri.add_triplet(i_new, j_new, value);
            }
            (Some(&i_new), None, _, Some(&fixed_j)) => {
                c_new[i_new] += 0.5 * value * fixed_j as f64;
            }
            (None, Some(&j_new), Some(&fixed_i), _) => {
                c_new[j_new] += 0.5 * value * fixed_i as f64;
            }
            _ => {}
        }
    }

    (Qubo::new_with_c(Q_tri.to_csc(), c_new), index_map)
}

/// Given a QUBO and a set of fixed variables, create a new QUBO where the fixed variables are
/// removed and the linear term is adjusted accordingly. Also return a mapping between the new
/// variable indices and the old variable indices, as well as the constant term that was added to
/// the objective function due to the fixed variables.
///
/// # Panics
/// This function will not panic if there are no free variables left removing unfixed varaibles.
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
    let mut unfixed_map = HashMap::with_capacity(num_unfixed);

    for i in 0..qubo.num_x() {
        if fixed_vars.contains_key(&i) {
            continue;
        }
        let new_index = unfixed_map.len();
        unfixed_map.insert(i, new_index);
    }

    for (&q_ij, (i, j)) in &qubo.q {
        match (
            fixed_vars.get(&i),
            fixed_vars.get(&j),
            unfixed_map.get(&i),
            unfixed_map.get(&j),
        ) {
            (Some(&fixed_i), Some(&fixed_j), _, _) => {
                constant += 0.5 * q_ij * fixed_i as f64 * fixed_j as f64;
            }
            (Some(&fixed_i), None, _, Some(&j_new)) => {
                c_new[j_new] += 0.5 * q_ij * fixed_i as f64;
            }
            (None, Some(&fixed_j), Some(&i_new), _) => {
                c_new[i_new] += 0.5 * q_ij * fixed_j as f64;
            }
            (None, None, Some(&i_new), Some(&j_new)) => {
                Q_tri.add_triplet(i_new, j_new, q_ij);
            }
            _ => {}
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
