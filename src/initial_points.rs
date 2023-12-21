use ndarray::Array1;
use smolprng::{Algorithm, PRNG};
use crate::qubo::Qubo;

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
