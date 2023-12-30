//! This is the general Utils module, which contains functions that are used by multiple algorithms and there is not a
//! better place to put them as of yet.

use ndarray::Array1;
use smolprng::{Algorithm, PRNG};

/// Given a point, x, flip sites number of bits and return the new point, this can include a bit that is already flipped.
pub fn mutate_solution<T: Algorithm>(
    x: &Array1<f64>,
    sites: usize,
    prng: &mut PRNG<T>,
) -> Array1<f64> {
    let mut x_1 = x.clone();

    for _ in 0..sites {
        let i = prng.gen_u64() as usize % x_1.len();
        x_1[i] = 1.0 - x_1[i];
    }

    x_1
}

/// Flips every bit in a point.
pub fn invert(x: &Array1<f64>) -> Array1<f64> {
    1.0f64 - x
}

/// Computes the hamming distance two points.
pub fn calculate_hamming_distance(x_0: &Array1<f64>, x_1: &Array1<f64>) -> usize {
    let mut distance = 0;
    for i in 0..x_0.len() {
        if x_0[i] != x_1[i] {
            distance += 1;
        }
    }
    distance
}

/// Given a point, x, determine if it is fractional e.g. not just 0.0f64 or 1.0f64
pub fn is_fractional(x: &Array1<f64>) -> bool {
    for i in 0..x.len() {
        if x[i] != 0.0 && x[i] != 1.0 {
            return true;
        }
    }
    false
}
