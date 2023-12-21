use ndarray::Array1;
use smolprng::{Algorithm, PRNG};

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

pub fn calculate_hamming_distance(x_0: &Array1<f64>, x_1: &Array1<f64>) -> usize {
    // Given two points, x_0 and x_1, calculate the hamming distance between them.
    let mut distance = 0;
    for i in 0..x_0.len() {
        if x_0[i] != x_1[i] {
            distance += 1;
        }
    }
    distance
}