//! Initial points, is there general code that can be used to create starting points for the algorithms you write.
//!
//! Provides a set of initial points that can be used to start the algorithms, these include:
//! - Random points
//! - Central points
//! - Alpha points
//! - Rho points
//! - Random binary points

use crate::qubo::Qubo;
use ndarray::Array1;
use smolprng::{Algorithm, PRNG};

/// Generates a vector of random starting points, that is fractional, meaning that it will return a vector of arrays
/// that are not just 0.0 or 1.0, but also numbers between.
///
/// Example:
/// ``` rust
/// use hercules::qubo::Qubo;
/// use smolprng::{PRNG, JsfLarge};
/// use hercules::initial_points;
///
/// let mut prng = PRNG {
///   generator: JsfLarge::default(),
/// };
/// let p = Qubo::make_random_qubo(10, &mut prng, 0.5);
/// let x_0 = initial_points::generate_random_binary_points(p.num_x(), 10, &mut prng);
/// ```
pub fn generate_random_binary_points<T: Algorithm>(
    n: usize,
    num_points: usize,
    prng: &mut PRNG<T>,
) -> Vec<Array1<usize>> {
    (0..num_points)
        .map(|_| generate_random_binary_point(n, prng, 0.5))
        .collect()
}

/// Generates a starting point of exactly 0.5 for each variable.
///
/// Example:
/// ``` rust
/// use hercules::qubo::Qubo;
/// use smolprng::{PRNG, JsfLarge};
/// use hercules::initial_points;
///
/// let mut prng = PRNG {
///    generator: JsfLarge::default(),
/// };
/// let p = Qubo::make_random_qubo(10, &mut prng, 0.5);
/// let x_0 = initial_points::generate_central_starting_points(p.num_x());
/// ```
pub fn generate_central_starting_points(n: usize) -> Array1<f64> {
    Array1::<f64>::zeros(n) + 0.5
}

/// Generates a starting point based on the alpha heuristic, from the paper boros2007, which is the optimal solution to
/// the relaxed QUBO, given x_i = x_j for all i, j.
///
/// Example:
/// ``` rust
/// use hercules::qubo::Qubo;
/// use smolprng::{PRNG, JsfLarge};
/// use hercules::initial_points;
///
/// let mut prng = PRNG {
///    generator: JsfLarge::default(),
/// };
/// let p = Qubo::make_random_qubo(10, &mut prng, 0.5);
/// let x_0 = initial_points::generate_alpha_starting_point(&p);
/// ```
pub fn generate_alpha_starting_point(qubo: &Qubo) -> Array1<f64> {
    Array1::<f64>::zeros(qubo.num_x()) + qubo.alpha()
}

/// Generates a starting point based on the rho heuristic, from the paper boros2007, which is the optimal solution to
/// the relaxed QUBO, given x_i = x_j for all i, j.
///
/// Example:
/// ``` rust
/// use hercules::qubo::Qubo;
/// use smolprng::{PRNG, JsfLarge};
/// use hercules::initial_points;
///
/// let mut prng = PRNG {
///    generator: JsfLarge::default(),
/// };
/// let p = Qubo::make_random_qubo(10, &mut prng, 0.5);
/// let x_0 = initial_points::generate_rho_starting_point(&p);
/// ```
pub fn generate_rho_starting_point(qubo: &Qubo) -> Array1<f64> {
    Array1::<f64>::zeros(qubo.num_x()) + qubo.rho()
}

/// Generates a random binary point, where each variable has a probability of being 1.0 equal to sparsity.
///
/// Example:
/// ``` rust
/// use hercules::qubo::Qubo;
/// use smolprng::{PRNG, JsfLarge};
/// use hercules::initial_points;
///
/// let mut prng = PRNG {
///    generator: JsfLarge::default(),
/// };
/// let p = Qubo::make_random_qubo(10, &mut prng, 0.5);
/// let x_0 = initial_points::generate_random_binary_point(p.num_x(), &mut prng, 0.5);
/// ```
pub fn generate_random_binary_point<T: Algorithm>(
    n:usize,
    prng: &mut PRNG<T>,
    sparsity: f64,
) -> Array1<usize> {
    // set up a zeroed buffer
    let mut x = Array1::<usize>::zeros(n);

    // for each variable, if the random number is less than sparsity, set it to 1.0
    for i in 0..x.len() {
        if prng.gen_f64() < sparsity {
            x[i] = 1;
        }
    }
    x
}

/// Given a vector of probabilities of each bit being 1, generate a binary point.
///
/// Example:
/// ``` rust
/// use hercules::qubo::Qubo;
/// use smolprng::{PRNG, JsfLarge};
/// use hercules::{initial_points, utils};
///
/// // generate a random QUBO
/// let mut prng = PRNG {
///  generator: JsfLarge::default(),
/// };
/// let p = Qubo::make_random_qubo(10, &mut prng, 0.5);
///
/// // starting point with 50% chance of each bit being 1
/// let x_0 = initial_points::generate_central_starting_points(p.num_x());
///
/// let x_rand = initial_points::gen_binary_point_from_dist(&mut prng, &x_0);
/// ```
pub fn gen_binary_point_from_dist<T: Algorithm>(
    prng: &mut PRNG<T>,
    dist: &Array1<f64>,
) -> Array1<f64> {
    // generate a zeroed buffer
    let mut x = Array1::<f64>::zeros(dist.len());

    // for each variable, set it to either 0 or 1, based on the dist array
    for i in 0..x.len() {
        if prng.gen_f64() < dist[i] {
            x[i] = 1.0;
        }
    }

    x
}