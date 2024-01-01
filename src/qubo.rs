//! This module contains the QUBO struct and associated methods
//!
//! The QUBO struct uses a sparse representation of the QUBO matrix, and is stored in CSR order, it is not assumed to be symmetrical.

use ndarray::Array1;
use sprs::{CsMat, TriMat};
use std::io::BufRead;
use std::io::Write;

use smolprng::Algorithm;
use smolprng::PRNG;

// TODO: Figure out how to render the math expressions in the documentation

/// The QUBO struct, which contains the QUBO matrix and the linear coefficients.
///
/// \min_x 0.5 x^T Q x + c^T x

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
    /// use hurricane::qubo::Qubo;
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
    /// use hurricane::qubo::Qubo;
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

    /// Generate a random QUBO struct with a given number of variables, sparsity, and PRNG. This function is deterministic.
    ///
    /// Example to create a random QUBO with 10 variables and a sparsity of 0.5:
    /// ```rust
    /// use hurricane::qubo::Qubo;
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
    /// use hurricane::qubo::Qubo;
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

    /// Return the number of variables in the QUBO
    ///
    /// Example of getting the number of variables in a QUBO:
    /// ```rust
    /// use hurricane::qubo::Qubo;
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
    /// use hurricane::qubo::Qubo;
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

    /// Computes the optimal solution of the relaxed QUBO problem where x* = \alpha. From Boros2007.
    ///
    /// Assuming all variables take the same value, find the minimizing value of alpha
    /// \alpha = \argmax_{\lambda \in [0, 1]} \lambda(\sum_i c_i + \lambda \sum_i \sum_j q_{ij}
    ///
    /// Example of calculating the optimal solution of the relaxed QUBO problem:
    /// ```rust
    /// use hurricane::qubo::Qubo;
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
        // find sum of all elements of q, get index of non zero elements
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
    /// use hurricane::qubo::Qubo;
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
    /// use hurricane::qubo::Qubo;
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
        let mut w = Array1::<usize>::zeros(self.num_x());

        for (value, (i, j)) in &self.q {
            if *value != 0.0f64 {
                w[i] += 1;
                w[j] += 1;
            }
        }

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
    /// use hurricane::qubo::Qubo;
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
        let file = std::fs::File::create(filename).unwrap();
        let mut writer = std::io::BufWriter::new(file);

        writeln!(writer, "{}", self.num_x()).unwrap();
        for (value, (i, j)) in &self.q {
            writeln!(writer, "{i} {j} {value}").unwrap();
        }

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
    /// use hurricane::qubo::Qubo;
    ///
    /// let p = Qubo::read_qubo("test.qubo");
    /// ```
    ///
    /// # Panics
    ///
    /// Will panic if there is not a file at the given filename in the .qubo format.
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

            // otherwise we add to the sparse matrix
            if row_data.len() == 3 {
                let i = row_data[0].parse::<usize>().unwrap();
                let j = row_data[1].parse::<usize>().unwrap();
                let value = row_data[2].parse::<f64>().unwrap();
                q.add_triplet(i, j, value);
            }

            line = String::new();
        }

        Self::new_with_c(q.to_csr(), c)
    }
}
