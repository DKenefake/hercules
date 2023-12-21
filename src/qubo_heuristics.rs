
use ndarray::Array1;
use crate::qubo::Qubo;
use smolprng::PRNG as PRNG;
use smolprng::Algorithm as Algorithm;
use rayon::prelude::*;
pub fn generate_random_starting_points<T: Algorithm>(qubo: &Qubo, num_points: usize, prng: &mut PRNG<T>) -> Vec<Array1<f64>> {
    // Generate num_points number of random starting points

    let mut xs = Vec::<Array1<f64>>::new();

    for _ in 0..num_points {
        let mut x = Array1::<f64>::zeros(qubo.num_x());
        for i in 0..x.len() {
            x[i] = prng.gen_f64();
        }
        xs.push(x);
    }

    xs
}

pub fn generate_central_starting_points(qubo: &Qubo) -> Array1<f64> {
    // generate a point that is the center of the hypercube
    Array1::<f64>::zeros(qubo.num_x()) + 0.5f64
}

pub fn generate_alpha_starting_point(qubo: &Qubo) -> Array1<f64> {
    // generate a point that is the center of the hypercube
    Array1::<f64>::zeros(qubo.num_x()) + qubo.alpha()
}

pub fn one_step_local_search(qubo: &Qubo, x_0: &Array1<f64>, subset: &Vec<usize>) -> Array1<f64> {
    // Do a neighborhood search of up to one bit flip and returns the best solution
    // found, this can include the original solution.

    let current_obj = qubo.eval(&x_0);

    let y = 1.0f64 - x_0;
    let mut objs = Array1::<f64>::zeros(qubo.num_x());

    for i in subset.iter(){

        let mut x = x_0.clone();
        x[*i] = y[*i];
        objs[*i] = qubo.eval(&x);
    }

    let best_neighbor = objs.iter().enumerate().min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap().0;
    let best_obj = objs[best_neighbor];

    let mut x_1 = x_0.clone();
    x_1[best_neighbor] = 1.0 - x_1[best_neighbor];

    match best_obj < current_obj {
        true => x_1,
        false => x_0.clone(),
    }
}

pub fn simple_local_search(qubo: &Qubo, x_0: &Array1<f64>, max_steps:usize) -> Array1<f64>{

    let mut x = x_0.clone();
    let variables = (0..qubo.num_x()).collect();
    let mut x_1 = one_step_local_search(qubo, &x, &variables);
    let mut steps = 0;

    while x_1 != x && steps <= max_steps {
        x = x_1.clone();
        x_1 = one_step_local_search(qubo, &x, &variables);
        steps += 1;
    }

    x_1
}

pub fn multi_simple_local_search(qubo: &Qubo, xs: &Vec<Array1<f64>>) -> Vec<Array1<f64>>{
    // Given a vector of initial points, run simple local search on each of them
    xs.par_iter().map(|x| simple_local_search(qubo, x, usize::MAX)).collect()
}

pub fn mutate_solution<T: Algorithm>(x: &Array1<f64>, sites:usize, prng: &mut PRNG<T>) -> Array1<f64> {
    // Given a point, x, flip sites number of bits and return the new point, this can include a bit that is already flipped.
    let mut x_1 = x.clone();

    for _ in 0..sites {
        let i = prng.gen_u64() as usize % x_1.len();
        x_1[i] = 1.0 - x_1[i];
    }

    x_1
}

pub fn invert(x: &Array1<f64>) -> Array1<f64> {
    // Given a point, x, flip all bits and return the new point.
    1.0f64 - x
}

pub fn get_opt_criteria(qubo: &Qubo, x: &Array1<f64>) -> Array1<f64> {
    // Given a point, x, calculate what the
    let mut fixed = Array1::zeros(qubo.num_x());
    let grad = qubo.eval_grad(x);

    for i in 0..qubo.num_x() {
        if grad[i] <= 0.0 {
            fixed[i] = 1.0;
        }else {
            fixed[i] = 0.0;
        }
    }

    fixed
}

pub fn simple_opt_criteria_search(qubo: &Qubo, x_0: &Array1<f64>, max_steps:usize) -> Array1<f64>{

    let mut x = x_0.clone();
    let mut x_1 = get_opt_criteria(qubo, &x);
    let mut steps = 0;

    while x_1 != x && steps <= max_steps {
        x = x_1.clone();
        x_1 = get_opt_criteria(qubo, &x);
        steps += 1;
    }

    x_1
}

pub fn multi_simple_opt_criteria_search(qubo: &Qubo, xs: &Vec<Array1<f64>>) -> Vec<Array1<f64>>{
    // Given a vector of initial points, run simple local search on each of them
    xs.par_iter().map(|x| simple_opt_criteria_search(qubo, x, 1000)).collect()
}

pub fn simple_mixed_search(qubo: &Qubo, x_0: &Array1<f64>, max_steps:usize) -> Array1<f64>{

    let mut x = x_0.clone();
    let mut x_1 = get_opt_criteria(qubo, &x);
    let mut steps = 0;

    while x_1 != x && steps <= max_steps {
        x = x_1.clone();
        x_1 = one_step_local_search(qubo, &x, &(0..qubo.num_x()).collect());
        x_1 = get_opt_criteria(qubo, &x_1);
        steps += 1;
    }

    x_1
}

pub fn compute_d(x_0: &Array1<f64>, grad: &Array1<f64>) -> Array1<f64> {
    // compute the variable importance function
    let mut d = Array1::<f64>::zeros(x_0.len());
    for i in 0..x_0.len() {
        // find the max of the two terms
        d[i] = f64::max(-x_0[i]*grad[i], (1.0-x_0[i])*grad[i]);
    }
    d
}

pub fn compute_I(d:&Array1<f64>) -> Vec<usize> {
    // compute the variable selection function
    d.iter().filter(|x| **x > 0.0).enumerate().map(|(i, _)| i).collect()
}
