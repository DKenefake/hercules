//! This module contains the QUBO struct and associated methods
//!
//! The QUBO struct uses a sparse representation of the QUBO matrix, and is stored in CSR order, it is not assumed to be symmetrical.

use ndarray::Array1;
use ndarray_linalg::{Eigh, UPLO};

use sprs::{CsMat, TriMat};
use std::io::BufRead;
use std::io::Write;

use smolprng::Algorithm;
use smolprng::PRNG;

/// The QUBO struct, which contains the QUBO matrix and the linear coefficients. With the following form:
///
/// $$ \min_{x\in \{0,1\}^n} 0.5 x^T Q x + c^Tx $$
#[derive(Clone)]
pub struct Qubo {
    /// The Hessian of the QUBO problem
    pub q: CsMat<f64>,
    /// The linear term of the QUBO problem
    pub c: Array1<f64>,
}

impl Qubo {
    /// Generate a new QUBO struct from a sparse matrix, assumed that the linear coefficients are zero
    ///
    /// Example to create a QUBO from a sparse Q matrix:
    /// ```rust
    /// use hercules::qubo::Qubo;
    /// use sprs::CsMat;
    ///
    /// let q = CsMat::<f64>::eye(10);
    /// let p = Qubo::new(q);
    /// ```
    pub fn new(q: CsMat<f64>) -> Self {
        let num_vars = q.cols();
        Self {
            q,
            c: Array1::<f64>::zeros(num_vars),
        }
    }

    /// Generate a new QUBO struct from a sparse matrix and a dense vector of linear coefficients
    ///
    /// Example to create a QUBO from a sparse Q matrix and a dense c vector:
    /// ```rust
    /// use hercules::qubo::Qubo;
    /// use sprs::CsMat;
    /// use ndarray::Array1;
    ///
    /// let q = CsMat::<f64>::eye(10);
    /// let c = Array1::<f64>::zeros(10);
    /// let p = Qubo::new_with_c(q, c);
    /// ```
    pub const fn new_with_c(q: CsMat<f64>, c: Array1<f64>) -> Self {
        Self { q, c }
    }

    /// Generate a QUBO struct from the list format
    ///
    /// Example to create a QUBO from a list of tuples:
    /// ```rust
    /// use hercules::qubo::Qubo;
    /// use ndarray::Array1;
    ///
    /// let x = vec![0,1,2];
    /// let y = vec![0,1,2];
    /// let q = vec![1.0,1.0,1.0];
    /// let c = vec![0.0,0.0,0.0];
    /// let num_x = 3;
    /// let p = Qubo::from_vec(x, y, q, c, 3);
    /// ```
    pub fn from_vec(i: Vec<usize>, j: Vec<usize>, q: Vec<f64>, c: Vec<f64>, num_x: usize) -> Self {
        // set up the sparse matrix and dense vector
        let mut q_mat = TriMat::<f64>::new((num_x, num_x));
        let mut c_vec = Array1::<f64>::zeros(num_x);

        // read the file
        for (k, v) in q.iter().enumerate() {
            q_mat.add_triplet(i[k], j[k], *v);
        }

        // set up the linear component
        for (k, v) in c.iter().enumerate() {
            c_vec[k] = *v;
        }

        Self::new_with_c(q_mat.to_csr(), c_vec)
    }

    pub fn to_vec(&self) -> (Vec<usize>, Vec<usize>, Vec<f64>, Vec<f64>, usize) {
        let mut i = Vec::new();
        let mut j = Vec::new();
        let mut q = Vec::new();
        let mut c = Vec::new();

        for (&value, (row, col)) in &self.q {
            i.push(row);
            j.push(col);
            q.push(value);
        }

        for &value in &self.c {
            c.push(value);
        }

        (i, j, q, c, self.num_x())
    }

    /// Generate a random QUBO struct with a given number of variables, sparsity, and PRNG. This function is deterministic.
    ///
    /// Example to create a random QUBO with 10 variables and a sparsity of 0.5:
    /// ```rust
    /// use hercules::qubo::Qubo;
    /// use smolprng::*;
    ///
    /// let mut prng = PRNG {
    ///    generator: JsfLarge::default(),
    /// };
    ///
    /// let n = 10;
    /// let sparsity = 0.5;
    /// let p = Qubo::make_random_qubo(n, &mut prng, sparsity);
    /// ```
    pub fn make_random_qubo<T: Algorithm>(num_x: usize, prng: &mut PRNG<T>, sparsity: f64) -> Self {
        // generate an empty sparse matrix in Triplet format
        let mut q = TriMat::<f64>::new((num_x, num_x));

        // given a probability of sparsity, add a random uniform variable [-.5, .5] to the sparse matrix at element (i,j)
        for i in 0..num_x {
            for j in i..num_x {
                if prng.gen_f64() < sparsity {
                    q.add_triplet(i, j, prng.gen_f64() - 0.5f64);
                }
            }
        }

        // generate a dense vector of random uniform variables [-.5, .5] for the linear coefficients
        let mut c = Array1::<f64>::zeros(num_x);
        for i in 0..num_x {
            c[i] = prng.gen_f64() - 0.5f64;
        }

        Self::new_with_c(q.to_csr(), c)
    }

    /// Given an initial point, x, calculate the objective function value of the QUBO
    ///
    /// Example of calculating the objective function value of a QUBO:
    /// ```rust
    /// use hercules::qubo::Qubo;
    /// use ndarray::Array1;
    /// use sprs::CsMat;
    ///
    /// let q = CsMat::<f64>::eye(3);
    /// let c = Array1::<f64>::zeros(3);
    /// let p = Qubo::new_with_c(q, c);
    /// let x_0 = Array1::from_vec(vec![1.0, 0.0, 1.0]);
    ///
    /// let obj = p.eval(&x_0);
    /// ```
    pub fn eval(&self, x: &Array1<f64>) -> f64 {
        let temp = &self.q * x;
        0.5 * x.dot(&temp) + self.c.dot(x)
    }

    pub fn eval_usize(&self, x: &Array1<usize>) -> f64 {
        let x_f64 = x.mapv(|x| x as f64);
        self.eval(&x_f64)
    }

    /// Return the number of variables in the QUBO
    ///
    /// Example of getting the number of variables in a QUBO:
    /// ```rust
    /// use hercules::qubo::Qubo;
    /// use ndarray::Array1;
    /// use sprs::CsMat;
    ///
    /// let q = CsMat::<f64>::eye(3);
    /// let c = Array1::<f64>::zeros(3);
    /// let p = Qubo::new_with_c(q, c);
    ///
    /// let num_x = p.num_x();
    /// ```
    pub fn num_x(&self) -> usize {
        self.q.cols()
    }

    /// Given an initial point, x, calculate the gradient of the QUBO (at that point)
    ///
    /// Example of calculating the gradient of a QUBO:
    /// ```rust
    /// use hercules::qubo::Qubo;
    /// use ndarray::Array1;
    /// use sprs::CsMat;
    ///
    /// let q = CsMat::<f64>::eye(3);
    /// let c = Array1::<f64>::zeros(3);
    /// let p = Qubo::new_with_c(q, c);
    ///
    /// let x_0 = Array1::from_vec(vec![1.0, 0.0, 1.0]);
    /// let grad = p.eval_grad(&x_0);
    /// ```
    pub fn eval_grad(&self, x: &Array1<f64>) -> Array1<f64> {
        // takes the gradient of the QUBO at x, does not assume that the QUBO is symmetric
        0.5 * (&self.q * x + &self.q.transpose_view() * x) + &self.c
    }

    pub fn eval_grad_usize(&self, x: &Array1<usize>) -> Array1<f64> {
        // takes the gradient of the QUBO at x, does not assume that the QUBO is symmetric
        let x_f64 = x.mapv(|x| x as f64);
        self.eval_grad(&x_f64)
    }

    /// Computes the optimal solution of the relaxed QUBO problem where x* = \alpha. From Boros2007.
    ///
    /// Assuming all variables take the same value, find the minimizing value of alpha
    /// \alpha = \argmax_{\lambda \in [0, 1]} \lambda(\sum_i c_i + \lambda \sum_i \sum_j q_{ij}
    ///
    /// Example of calculating the optimal solution of the relaxed QUBO problem:
    /// ```rust
    /// use hercules::qubo::Qubo;
    /// use ndarray::Array1;
    /// use sprs::CsMat;
    ///
    /// let q = CsMat::<f64>::eye(3);
    /// let c = Array1::<f64>::zeros(3);
    /// let p = Qubo::new_with_c(q, c);
    ///
    /// let alpha = p.alpha();
    /// ```
    pub fn alpha(&self) -> f64 {
        // find sum of all elements of q
        let q_sum = self.q.data().iter().sum::<f64>();
        let c_sum = self.c.sum();

        // in the case of q_sum == 0
        if q_sum == 0.0 {
            return match c_sum > 0.0 {
                true => 1.0,
                false => 0.0,
            };
        }

        // solve for the optimal solution of the 1D relaxed QUBO problem
        let alpha = -0.5 * c_sum / q_sum;

        // return the optimal solution within bounds
        alpha.clamp(0.0, 1.0)
    }

    /// Computes computes rho, the starting point heuristic from Boros2007
    ///
    /// Example of calculating rho:
    /// ```rust
    /// use hercules::qubo::Qubo;
    /// use ndarray::Array1;
    /// use sprs::CsMat;
    ///
    /// let q = CsMat::<f64>::eye(3);
    /// let c = Array1::<f64>::zeros(3);
    /// let p = Qubo::new_with_c(q, c);
    ///
    /// let rho = p.rho();
    /// ```
    pub fn rho(&self) -> f64 {
        // calculate the terms q+,q-,c+,and c- from boros2007
        let q_plus: f64 = self.q.data().iter().filter(|x| **x > 0.0).sum();
        let q_minus: f64 = self.q.data().iter().filter(|x| **x < 0.0).sum();
        let c_plus: f64 = self.c.iter().filter(|x| **x > 0.0).sum();
        let c_minus: f64 = self.c.iter().filter(|x| **x < 0.0).sum();

        let p = q_plus + c_plus;
        let n = q_minus + c_minus;

        let rho = p / (p - n);
        rho
    }

    /// Computes the complexity of the QUBO problem.
    ///
    /// The complexity is defined as the number of non-zero elements in the QUBO matrix plus the number of non-zero elements in the linear coefficients.
    ///
    /// Example of calculating the complexity of a QUBO:
    /// ```rust
    /// use hercules::qubo::Qubo;
    /// use ndarray::Array1;
    /// use sprs::CsMat;
    ///
    /// let q = CsMat::<f64>::eye(3);
    /// let c = Array1::<f64>::zeros(3);
    /// let p = Qubo::new_with_c(q, c);
    ///
    /// let complexity = p.complexity(); //(Array of size 3)
    /// ```
    pub fn complexity(&self) -> Array1<usize> {
        // set up a zeroed buffer
        let mut w = Array1::<usize>::zeros(self.num_x());

        // for each nonzero Q_ij, increment the corresponding w_i and w_j
        for (value, (i, j)) in &self.q {
            if *value != 0.0f64 {
                w[i] += 1;
                w[j] += 1;
            }
        }

        // for each nonzero c_i, increment the corresponding w_i
        for (i, value) in self.c.iter().enumerate() {
            if *value != 0.0f64 {
                w[i] += 1;
            }
        }

        w
    }

    /// Writes the QUBO to a file in the ORL problem format
    ///
    /// Example of writing a QUBO to a file:
    /// ```rust
    /// use hercules::qubo::Qubo;
    /// use smolprng::{PRNG, JsfLarge};
    ///
    /// let mut prng = PRNG {
    ///   generator: JsfLarge::default(),
    /// };
    /// let p = Qubo::make_random_qubo(50, &mut prng, 0.01);
    /// p.write_qubo("test.qubo");
    /// ```
    ///
    /// # Panics
    ///
    /// Will panics if it is not possible to write to the file.
    pub fn write_qubo(&self, filename: &str) {
        // open the file, create file writer
        let file = std::fs::File::create(filename).unwrap();
        let mut writer = std::io::BufWriter::new(file);

        // write the number of variables
        writeln!(writer, "{}", self.num_x()).unwrap();

        // for every nonzero Q_ij, write the indices and the value
        for (value, (i, j)) in &self.q {
            writeln!(writer, "{i} {j} {value}").unwrap();
        }

        // for every nonzero c_i, write the index and the value
        for i in 0..self.num_x() {
            let value = self.c[i];
            if value != 0.0 {
                writeln!(writer, "{i} {value}").unwrap();
            }
        }
    }

    /// Reads a QUBO from a file in the ORL problem format
    ///
    /// Example of reading a QUBO from a file:
    /// ```rust
    /// use hercules::qubo::Qubo;
    ///
    /// let p = Qubo::read_qubo("test.qubo");
    /// ```
    ///
    /// # Panics
    ///
    /// Will panic if there is not a file at the given filename in the .qubo format.
    #[must_use]
    pub fn read_qubo(filename: &str) -> Self {
        // open the file
        let file = std::fs::File::open(filename).unwrap();
        let mut reader = std::io::BufReader::new(file);

        // get the number of variables
        let mut line = String::new();
        reader.read_line(&mut line).unwrap();
        let num_x = line.trim().parse::<usize>().unwrap();

        line = String::new();
        // set up the sparse matrix and dense vector
        let mut q = TriMat::<f64>::new((num_x, num_x));
        let mut c = Array1::<f64>::zeros(num_x);

        // read the file
        while reader.read_line(&mut line).unwrap() > 0 {
            let row_data: Vec<_> = line.split_whitespace().collect();

            // we add to the column vector if there are only two elements
            if row_data.len() == 2 {
                let i = row_data[0].parse::<usize>().unwrap();
                let value = row_data[1].parse::<f64>().unwrap();
                c[i] = value;
            }

            // otherwise, we add to the sparse matrix
            if row_data.len() == 3 {
                let i = row_data[0].parse::<usize>().unwrap();
                let j = row_data[1].parse::<usize>().unwrap();
                let value = row_data[2].parse::<f64>().unwrap();
                q.add_triplet(i, j, value);
            }

            // reset the line
            line = String::new();
        }

        Self::new_with_c(q.to_csr(), c)
    }

    /// Generates a Symmetric QUBO from the current QUBO
    ///
    /// Example of making a QUBO symmetric:
    /// ```rust
    /// use hercules::qubo::Qubo;
    /// use smolprng::{PRNG, JsfLarge};
    ///
    /// // make a random number generator
    /// let mut prng = PRNG {
    ///     generator: JsfLarge::default(),
    /// };
    ///
    /// // make a random QUBO of 50 variables with an approximate sparsity of 0.1
    /// let p = Qubo::make_random_qubo(50, &mut prng, 0.1);
    ///
    /// // make a symmetric QUBO from this QUBO
    /// let p_sym = p.make_symmetric();
    /// ```
    #[must_use]
    pub fn make_symmetric(&self) -> Self {
        let mut tri_q = TriMat::<f64>::new((self.num_x(), self.num_x()));

        let c = self.c.clone();

        for (&value, (i, j)) in &self.q {
            if i == j {
                tri_q.add_triplet(i, j, value);
            } else {
                tri_q.add_triplet(j, i, 0.5 * value);
                tri_q.add_triplet(i, j, 0.5 * value);
            }
        }

        Self::new_with_c(tri_q.to_csr(), c)
    }

    /// Convexifies the QUBO problem by modifying the Hessian and linear coefficients,rendering a convex problem.
    ///
    /// Currently, assume that the required factor,'s' is known.
    ///
    /// $$\frac{1}{2}x^TQx + c^Tx = \frac{1}{2}x^T(Q + sI)x + c^Tx - 0.5s^Tx$$
    #[must_use]
    pub fn make_diag_transform(&self, s: f64) -> Self {
        // generate the scaled (sparse) identity matrix
        let mut s_eye_tri = TriMat::<f64>::new((self.num_x(), self.num_x()));
        for i in 0..self.num_x() {
            s_eye_tri.add_triplet(i, i, s);
        }
        let s_eye = s_eye_tri.to_csr();

        Self::new_with_c(&self.q + &s_eye, self.c.clone() - 0.5 * s)
    }

    #[must_use]
    pub fn make_diag_array_transform(&self, s: &Array1<f64>) -> Self {
        // generate the scaled (dense) identity matrix
        let mut s_eye_tri = TriMat::<f64>::new((self.num_x(), self.num_x()));
        for i in 0..self.num_x() {
            s_eye_tri.add_triplet(i, i, s[i]);
        }
        let s_eye = s_eye_tri.to_csr();

        Self::new_with_c(&self.q + &s_eye, self.c.clone() - 0.5 * s)
    }

    /// Calculates the eigenvalues of the QUBO Hessian matrix this is a somewhat expensive operation.
    /// Converts the QUBO to a dense matrix and then calculates the eigenvalues. Assume that the QUBO is symmetric.
    ///
    /// Example of calculating the eigenvalues of a QUBO:
    /// ```rust
    /// use hercules::qubo::Qubo;
    /// use ndarray::Array1;
    /// use sprs::CsMat;
    ///
    /// let q = CsMat::<f64>::eye(3);
    /// let c = Array1::<f64>::zeros(3);
    /// let p = Qubo::new_with_c(q, c);
    ///
    /// let eigs = p.hess_eigenvalues();
    /// ```
    ///
    /// # Panics
    ///  If the eigenvalue calculation fails, The only panic annotation from the ndarray_linalg crate is when the matrix is non-square
    pub fn hess_eigenvalues(&self) -> Array1<f64> {
        let q_dense = self.q.to_dense();
        let (eigs, _) = q_dense.eigh(UPLO::Upper).unwrap();
        eigs
    }

    /// Creates the convex symmetric form of the QUBO problem. This is an exact operation, and results in a convex symmetric matrix that is equivalent to the original QUBO.
    #[must_use]
    pub fn convex_symmetric_form(&self) -> Self {
        // make the QUBO symmetric
        let p_sym = self.make_symmetric();

        // compute the diag shift via mixing cut
        let rank = (2 * self.num_x()).isqrt() + 1;
        let diag_shift = -mixingcut::sdp_solver::compute_approx_perturbation(
            &self.q,
            Some(rank),
            None,
            None,
            None,
            None,
            false,
        );

        let p_sym = p_sym.make_diag_array_transform(&diag_shift);

        // calculate the eigenvalues of the QUBO
        let eigs = p_sym.hess_eigenvalues();

        // find the minimum eigenvalue, create a factor that scales the minimum eigenvalue to 1
        let min_eig = eigs.iter().fold(f64::INFINITY, |acc, &x| x.min(acc));
        let s = 0.01 - min_eig;

        // make the QUBO convex
        p_sym.make_diag_transform(s)
    }

    /// Creates the Hessian only equivalent form of the QUBO. Where the linear term is adsorbed into
    /// The Hessian matrix. This is an exact operation, and generates an equivalent form.
    ///
    /// $$ 0.5 x^T Q x + c^Tx = 0.5 x^T (Q + 2diag(c)) x $$
    /// Checks if the QUBO is symmetric
    pub fn is_symmetric(&self) -> bool {
        let error_margin = f64::EPSILON;

        for (&q_ij, (i, j)) in &self.q {
            let q_ji = self.q.get(j, i);
            match q_ji {
                Some(&q_ji) => {
                    if (q_ij - q_ji).abs() > error_margin {
                        return false;
                    }
                }
                None => {
                    return false;
                }
            }
        }

        true
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::initial_points::generate_random_binary_points;
    use crate::tests::{make_solver_qubo, make_test_prng};
    use ndarray::Array1;
    use sprs::CsMat;

    #[test]
    fn test_qubo_new() {
        let q = CsMat::<f64>::eye(10);
        let p = Qubo::new(q);

        assert_eq!(p.num_x(), 10);
        assert_eq!(p.q.nnz(), 10);
        assert_eq!(p.c.len(), 10);
    }

    #[test]
    fn test_qubo_new_with_c() {
        let q = CsMat::<f64>::eye(10);
        let c = Array1::<f64>::zeros(10);
        let p = Qubo::new_with_c(q, c);

        assert_eq!(p.num_x(), 10);
        assert_eq!(p.q.nnz(), 10);
        assert_eq!(p.c.len(), 10);
    }

    #[test]
    fn test_qubo_from_vec() {
        // Create a QUBO from a list of tuples:
        let x = vec![0, 1, 2];
        let y = vec![0, 1, 2];
        let q = vec![1.0, 1.0, 1.0];
        let c = vec![0.0, 0.0, 0.0];
        let num_x = 3;

        // actually create the QUBO
        let p = Qubo::from_vec(x, y, q, c, num_x);

        // check that the QUBO was created correctly
        assert_eq!(p.num_x(), 3);
        assert_eq!(p.q.nnz(), 3);
        assert_eq!(p.c.len(), 3);
        assert_eq!(p.q, CsMat::<f64>::eye(3));
    }

    #[test]
    fn test_qubo_eval() {
        let q = CsMat::<f64>::eye(3);
        let c = Array1::<f64>::zeros(3);
        let p = Qubo::new_with_c(q, c);
        let x_0 = Array1::from_vec(vec![1.0, 0.0, 1.0]);
        let obj = p.eval(&x_0);

        assert_eq!(obj, 1.0);
    }

    #[test]
    fn test_qubo_eval_grad() {
        let q = CsMat::<f64>::eye(3);
        let c = Array1::<f64>::zeros(3);
        let p = Qubo::new_with_c(q, c);
        let x_0 = Array1::from_vec(vec![1.0, 1.0, 1.0]);
        let grad = p.eval_grad(&x_0);

        assert_eq!(grad, Array1::from_vec(vec![1.0, 1.0, 1.0]));
    }

    #[test]
    fn test_qubo_eval_grad_non_symmetric() {
        // Create a QUBO from a list of tuples:
        let x = vec![0];
        let y = vec![2];
        let q = vec![1.0];
        let c = vec![0.0, 0.0, 0.0];
        let num_x = 3;

        // actually create the QUBO
        let p = Qubo::from_vec(x, y, q, c, num_x);
        let x_0 = Array1::from_vec(vec![1.0, 1.0, 1.0]);
        let grad = p.eval_grad(&x_0);

        assert_eq!(grad, Array1::from_vec(vec![0.5, 0.0, 0.5]));
    }

    #[test]
    fn test_make_symmetric_from_symmetric() {
        let q = CsMat::<f64>::eye(3);
        let c = Array1::<f64>::zeros(3);
        let p = Qubo::new_with_c(q, c);
        let p_sym = p.make_symmetric();

        assert_eq!(p_sym.q.nnz(), 3);
        assert_eq!(p_sym.q, CsMat::<f64>::eye(3));
    }

    #[test]
    fn test_make_symmetric_from_non_symmetric() {
        // Create a QUBO from a list of tuples:
        let x = vec![0, 0];
        let y = vec![0, 2];
        let q = vec![1.0, 1.0];
        let c = vec![0.0, 0.0, 0.0];
        let num_x = 3;
        let p = Qubo::from_vec(x, y, q, c, num_x);

        // make a symmetric QUBO from this QUBO
        let p_sym = p.make_symmetric();

        assert_eq!(p_sym.q.nnz(), 3);
        assert_eq!(p_sym.q.get(0, 0), Some(&1.0f64));
        assert_eq!(p_sym.q.get(0, 2), Some(&0.5f64));
        assert_eq!(p_sym.q.get(2, 0), Some(&0.5f64));
    }

    #[test]
    fn test_make_symmetric_from_non_symmetric_off_diagonal() {
        // Create a QUBO from a list of tuples:
        let x = vec![0, 0, 2];
        let y = vec![0, 2, 0];
        let q = vec![1.0, 1.5, 0.5];
        let c = vec![0.0, 0.0, 0.0];
        let num_x = 3;
        let p = Qubo::from_vec(x, y, q, c, num_x);

        // make a symmetric QUBO from this QUBO
        let p_sym = p.make_symmetric();

        assert_eq!(p_sym.q.nnz(), 3);
        assert_eq!(p_sym.q.get(0, 0), Some(&1.0f64));
        assert_eq!(p_sym.q.get(0, 2), Some(&1.0f64));
        assert_eq!(p_sym.q.get(2, 0), Some(&1.0f64));
    }

    #[test]
    fn read_write_consistency() {
        // make a qubo and write it to a file
        let mut prng = crate::tests::make_test_prng();
        let p = Qubo::make_random_qubo(10, &mut prng, 0.1);
        Qubo::write_qubo(&p, "test.qubo");

        // now read it back in
        let q = Qubo::read_qubo("test.qubo");

        // check that the two are the same
        assert_eq!(p.q, q.q);
        assert_eq!(p.c, q.c);
        assert_eq!(p.num_x(), q.num_x());
        assert_eq!(p.q.nnz(), q.q.nnz())
    }

    #[test]
    fn large_scale_write_qubo() {
        // make a large qubo instance and write it to file
        let mut prng = crate::tests::make_test_prng();
        let p = Qubo::make_random_qubo(1000, &mut prng, 0.01);
        Qubo::write_qubo(&p, "test_large.qubo");

        // read it back in
        let q = Qubo::read_qubo("test_large.qubo");

        // check that the two are the same
        assert_eq!(p.q, q.q);
        assert_eq!(p.c, q.c);
        assert_eq!(p.num_x(), q.num_x());
        assert_eq!(p.q.nnz(), q.q.nnz())
    }

    #[test]
    fn test_is_symmetric_on_symmetric() {
        let q = CsMat::<f64>::eye(3);
        let c = Array1::<f64>::zeros(3);
        let p = Qubo::new_with_c(q, c);
        assert_eq!(p.is_symmetric(), true);
    }

    #[test]
    fn test_is_symmetric_on_not_symmetric() {
        // Create a QUBO from a list of tuples:
        let x = vec![0, 0];
        let y = vec![0, 2];
        let q = vec![1.0, 1.0];
        let c = vec![0.0, 0.0, 0.0];
        let num_x = 3;
        let p = Qubo::from_vec(x, y, q, c, num_x);

        // make a symmetric QUBO from this QUBO
        let p_sym = p.make_symmetric();

        assert_eq!(p_sym.is_symmetric(), true);
        assert_eq!(p.is_symmetric(), false);
    }

    #[test]
    fn test_is_symmetric_on_random_qubo() {
        let mut prng = crate::tests::make_test_prng();
        for _ in 0..100 {
            // make a random qubo
            let p = Qubo::make_random_qubo(5, &mut prng, 1.0);
            // now that we have rendered it, it should be symmetric
            let p_sym = p.make_symmetric();
            assert_eq!(p_sym.is_symmetric(), true);
        }
    }

    #[test]
    fn test_convex_symmetric_form() {
        let p = make_solver_qubo();
        let p_convex = p.convex_symmetric_form();

        let eigs = p_convex.hess_eigenvalues();

        // check that the QUBO is symmetric
        assert_eq!(p_convex.is_symmetric(), true);

        // check that the QUBO is convex
        for eig in eigs.iter() {
            assert!(*eig >= 0.00099);
        }

        // check that the QUBO is equivalent to the original QUBO for binary vectors
        // make a bunch of random binary points, and check that the objective function is the same

        let mut prng = make_test_prng();
        let xs = generate_random_binary_points(p.num_x(), 50, &mut prng);

        for x in xs.iter() {
            let obj = p.eval_usize(x);
            let obj_convex = p_convex.eval_usize(x);
            assert!((obj - obj_convex).abs() < 1e-5);
        }
    }
}
