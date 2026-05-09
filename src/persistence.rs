use crate::preprocess::solve_small_components;
use crate::qubo::Qubo;
use std::cmp::min;
use std::collections::{HashMap, VecDeque};

/// This function takes a QUBO and a set of persistent variables and returns a new set of persistent variables by repeatedly
/// recomputing the persistent variables until.
pub fn compute_iterative_persistence(
    qubo: &Qubo,
    persistent: &HashMap<usize, usize>,
    iter_lim: usize,
) -> HashMap<usize, usize> {
    // make a copy of the passed persistence variable
    let mut new_persistent = persistent.clone();

    // the number of required iterations is always below the number of variables
    let iters = min(iter_lim, qubo.num_x());
    let adjacency = build_gradient_adjacency(qubo);

    // loop over the number of iters
    for _ in 0..iters {
        let incoming_persistent = propagate_persistent(qubo, &new_persistent, &adjacency);
        let incoming_persistent = solve_small_components(qubo, &incoming_persistent, 10);

        if new_persistent == incoming_persistent {
            break;
        }
        new_persistent = incoming_persistent;
    }

    new_persistent
}

fn propagate_persistent(
    qubo: &Qubo,
    persistent: &HashMap<usize, usize>,
    adjacency: &[Vec<(usize, f64)>],
) -> HashMap<usize, usize> {
    let num_x = qubo.num_x();
    let mut new_persistent = persistent.clone();
    let mut fixed_values = vec![None; num_x];

    for (&index, &value) in persistent {
        fixed_values[index] = Some(value as u8);
    }

    let mut lower = qubo.c.to_vec();
    let mut upper = qubo.c.to_vec();

    for i in 0..num_x {
        for &(neighbor, coeff) in &adjacency[i] {
            if let Some(value) = fixed_values[neighbor] {
                apply_fixed_term(&mut lower[i], &mut upper[i], coeff, value);
            } else {
                apply_free_term(&mut lower[i], &mut upper[i], coeff);
            }
        }
    }

    let mut queue = VecDeque::new();

    for i in 0..num_x {
        if fixed_values[i].is_some() {
            continue;
        }

        if lower[i] > 0.0 {
            fixed_values[i] = Some(0);
            new_persistent.insert(i, 0);
            queue.push_back((i, 0u8));
        } else if upper[i] < 0.0 {
            fixed_values[i] = Some(1);
            new_persistent.insert(i, 1);
            queue.push_back((i, 1u8));
        }
    }

    while let Some((fixed_variable, value)) = queue.pop_front() {
        for &(target, coeff) in &adjacency[fixed_variable] {

            if fixed_values[target].is_some() {
                continue;
            }

            remove_free_term(&mut lower[target], &mut upper[target], coeff);
            apply_fixed_term(&mut lower[target], &mut upper[target], coeff, value);

            if lower[target] > 0.0 {
                fixed_values[target] = Some(0);
                new_persistent.insert(target, 0);
                queue.push_back((target, 0u8));
            } else if upper[target] < 0.0 {
                fixed_values[target] = Some(1);
                new_persistent.insert(target, 1);
                queue.push_back((target, 1u8));
            }
        }
    }

    new_persistent
}

fn apply_free_term(lower: &mut f64, upper: &mut f64, coeff: f64) {
    if coeff <= 0.0 {
        *lower += coeff;
    }
    if coeff >= 0.0 {
        *upper += coeff;
    }
}

fn remove_free_term(lower: &mut f64, upper: &mut f64, coeff: f64) {
    if coeff <= 0.0 {
        *lower -= coeff;
    }
    if coeff >= 0.0 {
        *upper -= coeff;
    }
}

fn apply_fixed_term(lower: &mut f64, upper: &mut f64, coeff: f64, value: u8) {
    let contribution = coeff * f64::from(value);
    *lower += contribution;
    *upper += contribution;
}

fn build_gradient_adjacency(qubo: &Qubo) -> Vec<Vec<(usize, f64)>> {
    let num_x = qubo.num_x();
    let mut counts = vec![0usize; num_x];

    for (&_value, (i, j)) in &qubo.q {
        counts[i] += 1;
        counts[j] += 1;
    }

    let mut adjacency = counts
        .into_iter()
        .map(Vec::with_capacity)
        .collect::<Vec<_>>();

    for (&value, (i, j)) in &qubo.q {
        let coeff = 0.5 * value;

        adjacency[i].push((j, coeff));
        adjacency[j].push((i, coeff));
    }

    adjacency
}

/// This function takes a QUBO and a set of persistent variables and returns a new set of persistent variables by computing the
/// persistent variables once.
pub fn compute_persistent(
    qubo: &Qubo,
    persistent: &HashMap<usize, usize>,
) -> HashMap<usize, usize> {
    // create a new hashmap to store the new persistent variables
    let mut new_persistent = persistent.clone();

    // iterate over all the variables in the QUBO
    for i in 0..qubo.num_x() {
        if persistent.contains_key(&i) {
            continue;
        }

        // find the bounds of the gradient in each direction
        let (lower, upper) = grad_bounds(qubo, i, persistent);

        // if the lower bound it positive, then we can set the variable to 0
        if lower > 0.0 {
            new_persistent.insert(i, 0);
        }

        // if the upper bound is below 0, then we can set the variable to 1
        if upper < 0.0 {
            new_persistent.insert(i, 1);
        }
    }

    new_persistent
}

/// Finds bounds of the i-th index of the gradients of the QUBO function
///
/// # Panics
/// This function should not panic as the unwraps are bounded on the size of the QUBO matrix
pub fn grad_bounds(qubo: &Qubo, i: usize, persistent: &HashMap<usize, usize>) -> (f64, f64) {
    // set up tracking variables for each bound
    let mut lower = 0.0;
    let mut upper = 0.0;

    // get the i-th row of the Q matrix, this is safe as we are bounded by the size of the Q matrix
    let x = qubo.q.outer_view(i).unwrap();

    // get the i-th column of the Q matrix, this is safe for the same reason
    let binding = qubo.q.transpose_view();
    let y = binding.outer_view(i).unwrap();

    accumulate_grad_terms(x.iter(), persistent, &mut lower, &mut upper);
    accumulate_grad_terms(y.iter(), persistent, &mut lower, &mut upper);

    // add the contribution from the constant term
    lower += qubo.c[i];
    upper += qubo.c[i];

    (lower, upper)
}

fn accumulate_grad_terms<'a, I>(
    terms: I,
    persistent: &HashMap<usize, usize>,
    lower: &mut f64,
    upper: &mut f64,
) where
    I: Iterator<Item = (usize, &'a f64)>,
{
    for (index, x_j) in terms {
        let value = 0.5 * *x_j;

        // if it is a fixed variable, we have effectively removed this variable from the QUBO
        if let Some(&fixed) = persistent.get(&index) {
            let fixed = fixed as f64;
            *lower += value * fixed;
            *upper += value * fixed;
        } else {
            // if it is not in the persistent set, then we can choose the best value

            // if the value is negative, we would set it to 1 to minimize the gradient else 0
            if value <= 0.0 {
                *lower += value;
            }

            // if the value is positive, we would set it to 1 to maximize the gradient else 0
            if value >= 0.0 {
                *upper += value;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qubo::Qubo;
    use ndarray::Array1;
    use sprs::CsMat;
    use std::collections::HashMap;

    #[test]
    fn test_persistence() {
        //build the problem
        let eye = CsMat::eye(3);
        let c = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let p = Qubo::new_with_c(eye, c);
        let persist = compute_iterative_persistence(&p, &HashMap::new(), 3);

        assert!(persist.contains_key(&0));
        assert!(persist.contains_key(&1));
        assert!(persist.contains_key(&2));

        assert!(persist[&0].eq(&0));
        assert!(persist[&1].eq(&0));
        assert!(persist[&2].eq(&0));
    }
    #[test]
    fn test_grad_bounds_1() {
        let eye = CsMat::eye(3);
        let c = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let p = Qubo::new_with_c(eye, c);
        assert_eq!(grad_bounds(&p, 0, &HashMap::new()), (1.0, 2.0));
        assert_eq!(grad_bounds(&p, 1, &HashMap::new()), (2.0, 3.0));
        assert_eq!(grad_bounds(&p, 2, &HashMap::new()), (3.0, 4.0));
    }

    #[test]
    fn test_grad_bounds_2() {
        let eye = CsMat::eye(3);
        let c = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let p = Qubo::new_with_c(eye, c);

        let mut fixed_vars = HashMap::new();
        fixed_vars.insert(0, 1);
        fixed_vars.insert(2, 1);

        assert_eq!(grad_bounds(&p, 0, &fixed_vars), (2.0, 2.0));
        assert_eq!(grad_bounds(&p, 1, &fixed_vars), (2.0, 3.0));
        assert_eq!(grad_bounds(&p, 2, &fixed_vars), (4.0, 4.0));
    }

    #[test]
    fn test_grad_bounds_3() {
        let zero = CsMat::zero((3, 3));
        let c = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let p = Qubo::new_with_c(zero, c);

        assert_eq!(grad_bounds(&p, 0, &HashMap::new()), (1.0, 1.0));
        assert_eq!(grad_bounds(&p, 1, &HashMap::new()), (2.0, 2.0));
        assert_eq!(grad_bounds(&p, 2, &HashMap::new()), (3.0, 3.0));
    }

    // old test, not really relevant anymore as the presolve will solve this entirely
    // #[test]
    // fn test_alternating_persistence() {
    //     let p = make_solver_qubo();
    //     let p_symm = p.make_symmetric();
    //     let persist = compute_iterative_persistence(&p_symm, &HashMap::new(), p.num_x());
    //
    //     assert_eq!(persist.len(), 50);
    // }
}
