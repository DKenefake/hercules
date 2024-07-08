//! This is the general Utils module, which contains functions that are used by multiple algorithms and there is not a
//! better place to put them as of yet.

use crate::qubo::Qubo;
use ndarray::Array1;
use ndarray_linalg::Norm;
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
/// let x_0 = initial_points::generate_random_binary_point(p.num_x(), &mut prng, 0.5);
///
/// // mutate 1 bit
/// let x_1 = utils::mutate_solution(&x_0, 1, &mut prng);
///
/// // mutate 2 bits
/// let x_2 = utils::mutate_solution(&x_0, 2, &mut prng);
/// ```
pub fn mutate_solution<T: Algorithm>(
    x: &Array1<usize>,
    sites: usize,
    prng: &mut PRNG<T>,
) -> Array1<usize> {
    let mut x_1 = x.clone();

    for _ in 0..sites {
        #[allow(clippy::cast_possible_truncation)]
        // the max value that sparse matrices are addressable is usize::MAX
        let i = prng.gen_u64() as usize % x_1.len();
        x_1[i] = 1 - x_1[i];
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
/// let x_0 = initial_points::generate_random_binary_point(p.num_x(), &mut prng, 0.5);
///
/// // flip all of the bits
/// let x_1 = utils::invert(&x_0);
/// ```
pub fn invert(x: &Array1<usize>) -> Array1<usize> {
    1 - x
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
/// let x_0 = initial_points::generate_random_binary_point(p.num_x(), &mut prng, 0.5);
/// let x_1 = initial_points::generate_random_binary_point(p.num_x(), &mut prng, 0.5);
///
/// // calculate the hamming distance
/// let distance = utils::calculate_hamming_distance(&x_0, &x_1);
/// ```
pub fn calculate_hamming_distance(x_0: &Array1<usize>, x_1: &Array1<usize>) -> usize {
    let mut distance = 0;
    for i in 0..x_0.len() {
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
/// let x_0 = initial_points::generate_central_starting_points(p.num_x());
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

/// Generate a random point on a circle of radius 1 in n dimensions.
pub fn sample_circle<T: Algorithm>(n: usize, prng: &mut PRNG<T>) -> Array1<f64> {
    let mut x = Array1::zeros(n);
    for i in 0..n {
        x[i] = prng.normal();
    }
    let x_norm = x.norm_l2();

    x / x_norm
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
/// let x_0 = initial_points::generate_random_binary_point(p.num_x(), &mut prng, 0.5);
/// let x_1 = initial_points::generate_random_binary_point(p.num_x(), &mut prng, 0.5);
/// let x_2 = initial_points::generate_random_binary_point(p.num_x(), &mut prng, 0.5);
///
/// // form them into a vector
/// let points = vec![x_0, x_1, x_2];
///
/// // select the best point based on the objective out of this vector
/// let best_point = utils::get_best_point(&p, &points);
/// ```
pub fn get_best_point(qubo: &Qubo, points: &Vec<Array1<usize>>) -> Array1<usize> {
    // set the first point as the best point
    let mut best_point = points[0].clone();
    let mut best_obj = qubo.eval_usize(&best_point);

    // search thru the points and find the best one
    for point in points {
        let obj = qubo.eval_usize(point);
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
        let x_0 = Array1::from_vec(vec![1, 0, 1]);
        let target = Array1::from_vec(vec![0, 1, 0]);
        let x_1 = invert(&x_0);

        assert_eq!(x_1, target);
    }

    #[test]
    fn test_mutate() {
        let mut prng = make_test_prng();
        let x_0 = Array1::from_vec(vec![1, 0, 1]);
        let target = Array1::from_vec(vec![0, 0, 1]);
        let x_1 = mutate_solution(&x_0, 1, &mut prng);

        assert_eq!(x_1, target);
    }
}
