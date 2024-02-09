//! This is the general Utils module, which contains functions that are used by multiple algorithms and there is not a
//! better place to put them as of yet.

use crate::qubo::Qubo;
use ndarray::Array1;
use smolprng::{Algorithm, PRNG};

/// Given a point, x, flip sites number of bits and return the new point, this can include a bit that is already flipped.
/// This is used in the local search algorithms.
///
/// Example:
/// ``` rust
/// use hercules::qubo::Qubo;
/// use smolprng::{PRNG, JsfLarge};
/// use hercules::{initial_points, utils};
///
/// // generate a random QUBO
/// let mut prng = PRNG {
///    generator: JsfLarge::default(),
/// };
/// let p = Qubo::make_random_qubo(10, &mut prng, 0.5);
///
/// // generate a random point inside with x in {0, 1}^10 with
/// let x_0 = utils::make_binary_point(p.num_x(), &mut prng);
///
/// // mutate 1 bit
/// let x_1 = utils::mutate_solution(&x_0, 1, &mut prng);
///
/// // mutate 2 bits
/// let x_2 = utils::mutate_solution(&x_0, 2, &mut prng);
/// ```
pub fn mutate_solution<T: Algorithm>(
    x: &Array1<f64>,
    sites: usize,
    prng: &mut PRNG<T>,
) -> Array1<f64> {
    let mut x_1 = x.clone();

    for _ in 0..sites {
        #[allow(clippy::cast_possible_truncation)]
        // the max value that sparse matrices are addressable is usize::MAX
        let i = prng.gen_u64() as usize % x_1.len();
        x_1[i] = 1.0 - x_1[i];
    }

    x_1
}

/// Flips every bit in a point.
///
/// Example:
/// ``` rust
/// use hercules::qubo::Qubo;
/// use smolprng::{PRNG, JsfLarge};
/// use hercules::{initial_points, utils};
///
/// // generate a random QUBO
/// let mut prng = PRNG {
///     generator: JsfLarge::default(),
/// };
/// let p = Qubo::make_random_qubo(10, &mut prng, 0.5);
///
/// // generate a random point inside with x in {0, 1}^10
/// let x_0 = utils::make_binary_point(p.num_x(), &mut prng);
///
/// // flip all of the bits
/// let x_1 = utils::invert(&x_0);
/// ```
pub fn invert(x: &Array1<f64>) -> Array1<f64> {
    1.0f64 - x
}

/// Computes the hamming distance two points.
/// This is the number of bits that are different between the two points.
///
/// Example:
/// ``` rust
/// use hercules::qubo::Qubo;
/// use smolprng::{PRNG, JsfLarge};
/// use hercules::{initial_points, utils};
///
/// // generate a random QUBO
/// let mut prng = PRNG {
///    generator: JsfLarge::default(),
/// };
/// let p = Qubo::make_random_qubo(10, &mut prng, 0.5);
///
/// // generate random points inside with x in {0, 1}^10 with
/// let x_0 = utils::make_binary_point(p.num_x(), &mut prng);
/// let x_1 = utils::make_binary_point(p.num_x(), &mut prng);
///
/// // calculate the hamming distance
/// let distance = utils::calculate_hamming_distance(&x_0, &x_1);
/// ```
pub fn calculate_hamming_distance(x_0: &Array1<f64>, x_1: &Array1<f64>) -> usize {
    let mut distance = 0;
    for i in 0..x_0.len() {
        #[allow(clippy::float_cmp)]
        // This is a false positive, as we are checking for fractional values
        if x_0[i] != x_1[i] {
            distance += 1;
        }
    }
    distance
}

/// Given a point, x, determine if it is fractional e.g. not just 0.0f64 or 1.0f64
///
/// Example:
/// ``` rust
/// use hercules::qubo::Qubo;
/// use smolprng::{PRNG, JsfLarge};
/// use hercules::{initial_points, utils};
///
/// // generate a random QUBO
/// let mut prng = PRNG {
/// generator: JsfLarge::default(),
/// };
/// let p = Qubo::make_random_qubo(10, &mut prng, 0.5);
///
/// // generate a random point inside with x in {0, 1}^10 with
/// let x_0 = utils::make_binary_point(p.num_x(), &mut prng);
///
/// // check if the point is fractional
/// let is_fractional = utils::is_fractional(&x_0);
/// ```
pub fn is_fractional(x: &Array1<f64>) -> bool {
    for i in 0..x.len() {
        #[allow(clippy::float_cmp)]
        // This is a false positive, as we are checking for fractional values
        if x[i] != 0.0f64 && x[i] != 1.0f64 {
            return true;
        }
    }
    false
}

/// Given a problem size and a prng, generate a random binary point.
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
/// // generate a random point inside with x in {0, 1}^10 with
/// let x_0 = utils::make_binary_point(p.num_x(), &mut prng);
/// ```
pub fn make_binary_point<T: Algorithm>(num_dim: usize, prng: &mut PRNG<T>) -> Array1<f64> {
    let mut x = Array1::<f64>::zeros(num_dim);
    for i in 0..x.len() {
        if prng.gen_f64() > 0.5 {
            x[i] = 1.0;
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
/// let x_0 = initial_points::generate_central_starting_points(&p);
///
/// let x_rand = utils::gen_binary_point_from_dist(&mut prng, &x_0);
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

/// Given a vector of points, return the best point, based on objective.
///
/// Example:
/// ``` rust
/// use hercules::qubo::Qubo;
/// use smolprng::{PRNG, JsfLarge};
/// use hercules::{initial_points, utils};
///
/// // generate a random QUBO
/// let mut prng = PRNG {
/// generator: JsfLarge::default(),
/// };
/// let p = Qubo::make_random_qubo(10, &mut prng, 0.5);
///
/// // generate a random point inside with x in {0, 1}^10 with
/// let x_0 = utils::make_binary_point(p.num_x(), &mut prng);
/// let x_1 = utils::make_binary_point(p.num_x(), &mut prng);
/// let x_2 = utils::make_binary_point(p.num_x(), &mut prng);
///
/// // form them into a vector
/// let points = vec![x_0, x_1, x_2];
///
/// // select the best point based on the objective out of this vector
/// let best_point = utils::get_best_point(&p, &points);
/// ```
pub fn get_best_point(qubo: &Qubo, points: &Vec<Array1<f64>>) -> Array1<f64> {
    // set the first point as the best point
    let mut best_point = points[0].clone();
    let mut best_obj = qubo.eval(&best_point);

    // search thru the points and find the best one
    for point in points {
        let obj = qubo.eval(point);
        if obj < best_obj {
            best_obj = obj;
            best_point = point.clone();
        }
    }

    best_point
}

#[cfg(test)]
mod tests {
    use crate::tests::make_test_prng;
    use crate::utils::*;
    use ndarray::Array1;

    #[test]
    fn test_flip() {
        let x_0 = Array1::from_vec(vec![1.0, 0.0, 1.0]);
        let target = Array1::from_vec(vec![0.0, 1.0, 0.0]);
        let x_1 = invert(&x_0);

        assert_eq!(x_1, target);
    }

    #[test]
    fn test_mutate() {
        let mut prng = make_test_prng();
        let x_0 = Array1::from_vec(vec![1.0, 0.0, 1.0]);
        let target = Array1::from_vec(vec![0.0, 0.0, 1.0]);
        let x_1 = mutate_solution(&x_0, 1, &mut prng);

        assert_eq!(x_1, target);
    }
}
