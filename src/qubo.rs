use ndarray::Array1;
use sprs::{CsMat, TriMat};

use smolprng::Algorithm;
use smolprng::PRNG;

pub struct Qubo {
    pub q: CsMat<f64>,
    pub c: Array1<f64>,
}

impl Qubo {
    pub fn new(q: CsMat<f64>) -> Self {
        let num_vars = q.cols();
        Self {
            q,
            c: Array1::<f64>::zeros(num_vars),
        }
    }

    pub fn new_with_c(q: CsMat<f64>, c: Array1<f64>) -> Self {
        Self { q, c }
    }

    pub fn make_random_qubo<T: Algorithm>(num_x: usize, prng: &mut PRNG<T>, sparsity: f64) -> Self {
        let mut q = TriMat::<f64>::new((num_x, num_x));
        for i in 0..num_x {
            for j in i..num_x {
                if prng.gen_f64() < sparsity {
                    q.add_triplet(i, j, prng.gen_f64() - 0.5f64);
                }
            }
        }

        // generate random c
        let mut c = Array1::<f64>::zeros(num_x);
        for i in 0..num_x {
            c[i] = prng.gen_f64() - 0.5f64;
        }

        Self::new_with_c(q.to_csr(), c)
    }

    pub fn eval(&self, x: &Array1<f64>) -> f64 {
        let temp = &self.q * x;
        return 0.5 * x.dot(&temp) + self.c.dot(x);
    }

    pub fn num_x(&self) -> usize {
        self.q.cols()
    }

    pub fn eval_grad(&self, x: &Array1<f64>) -> Array1<f64> {
        // takes the gradient of the QUBO at x, does not assume that the QUBO is symmetric
        0.5 * (&self.q * x + &self.q.transpose_view() * x) + &self.c
    }

    pub fn alpha(&self) -> f64 {
        // Assuming all variables take the same value, find the minimizing value of alpha
        // \alpha = \argmax_{\lambda \in [0, 1]} \lambda(\sum_i c_i + \lambda \sum_i \sum_j q_{ij}

        // find sum of all elements of q, get index of non zero elements
        let q_sum = self.q.data().iter().sum::<f64>();
        let c_sum = self.c.sum();

        // in the cas of q_sum == 0
        if q_sum == 0.0 {
            return match c_sum > 0.0 {
                true => 1.0,
                false => 0.0,
            };
        }

        let alpha = -0.5 * c_sum / q_sum;

        alpha.clamp(0.0, 1.0)
    }

    pub fn rho(&self) -> f64 {
        /// rho expression from boros2007
        let q_plus: f64 = self.q.data().iter().filter(|x| **x > 0.0).sum();
        let q_minus: f64 = self.q.data().iter().filter(|x| **x < 0.0).sum();
        let c_plus: f64 = self.c.iter().filter(|x| **x > 0.0).sum();
        let c_minus: f64 = self.c.iter().filter(|x| **x < 0.0).sum();

        let p = q_plus + c_plus;
        let n = q_minus + c_minus;

        let rho = p / (p - n);
        rho
    }

    pub fn complexity(&self) -> Array1<usize> {
        /// complexity expression from boros2007
        let mut w = Array1::<usize>::zeros(self.num_x());

        for (value, (i, j)) in self.q.iter() {
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
}
