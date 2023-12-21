use crate::local_search_utils;
use crate::qubo::Qubo;
use ndarray::Array1;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

pub fn one_step_local_search(qubo: &Qubo, x_0: &Array1<f64>, subset: &Vec<usize>) -> Array1<f64> {
    // Do a neighborhood search of up to one bit flip and returns the best solution
    // found, this can include the original solution.

    let current_obj = qubo.eval(&x_0);

    let y = 1.0f64 - x_0;
    let mut objs = Array1::<f64>::zeros(qubo.num_x());

    for i in subset.iter() {
        let mut x = x_0.clone();
        x[*i] = y[*i];
        objs[*i] = qubo.eval(&x);
    }

    let best_neighbor = objs
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap()
        .0;
    let best_obj = objs[best_neighbor];

    let mut x_1 = x_0.clone();
    x_1[best_neighbor] = 1.0 - x_1[best_neighbor];

    match best_obj < current_obj {
        true => x_1,
        false => x_0.clone(),
    }
}

pub fn one_step_local_search_improved(
    qubo: &Qubo,
    x_0: &Array1<f64>,
    selected_vars: &Vec<usize>,
) -> Array1<f64> {
    // Do a neighborhood search of up to one bit flip and returns the best solution
    // found, this can include the original solution, out of the selected variables.

    let (_, objs) = local_search_utils::one_flip_objective(qubo, &x_0);

    let best_neighbor = objs
        .iter()
        .enumerate()
        .filter(|(i, _)| selected_vars.contains(i))
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap()
        .0;

    let best_obj = objs[best_neighbor];

    let mut x_1 = x_0.clone();
    x_1[best_neighbor] = 1.0 - x_1[best_neighbor];

    match best_obj < 0.0 {
        true => x_1,
        false => x_0.clone(),
    }
}

pub fn simple_local_search(qubo: &Qubo, x_0: &Array1<f64>, max_steps: usize) -> Array1<f64> {
    let mut x = x_0.clone();
    let variables = (0..qubo.num_x()).collect();
    let mut x_1 = one_step_local_search(qubo, &x, &variables);
    let mut steps = 0;

    while x_1 != x && steps <= max_steps {
        x = x_1.clone();
        x_1 = one_step_local_search_improved(qubo, &x, &variables);
        steps += 1;
    }

    x_1
}

pub fn multi_simple_local_search(qubo: &Qubo, xs: &Vec<Array1<f64>>) -> Vec<Array1<f64>> {
    // Given a vector of initial points, run simple local search on each of them
    xs.par_iter()
        .map(|x| simple_local_search(qubo, x, usize::MAX))
        .collect()
}

pub fn simple_opt_criteria_search(qubo: &Qubo, x_0: &Array1<f64>, max_steps: usize) -> Array1<f64> {
    let mut x = x_0.clone();
    let mut x_1 = local_search_utils::get_opt_criteria(qubo, &x);
    let mut steps = 0;

    while x_1 != x && steps <= max_steps {
        x = x_1.clone();
        x_1 = local_search_utils::get_opt_criteria(qubo, &x);
        steps += 1;
    }

    x_1
}

pub fn multi_simple_opt_criteria_search(qubo: &Qubo, xs: &Vec<Array1<f64>>) -> Vec<Array1<f64>> {
    // Given a vector of initial points, run simple local search on each of them
    xs.par_iter()
        .map(|x| simple_opt_criteria_search(qubo, x, 1000))
        .collect()
}

pub fn simple_mixed_search(qubo: &Qubo, x_0: &Array1<f64>, max_steps: usize) -> Array1<f64> {
    let mut x = x_0.clone();
    let mut x_1 = local_search_utils::get_opt_criteria(qubo, &x);
    let mut steps = 0;
    let vars = (0..qubo.num_x()).collect();

    while x_1 != x && steps <= max_steps {
        x = x_1.clone();
        x_1 = one_step_local_search_improved(qubo, &x, &vars);
        x_1 = local_search_utils::get_opt_criteria(qubo, &x_1);
        steps += 1;
    }

    x_1
}
