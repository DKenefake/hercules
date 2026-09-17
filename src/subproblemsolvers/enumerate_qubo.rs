use crate::qubo::Qubo;
use ndarray::Array1;

pub fn enumerate_solve(qubo: &Qubo) -> (f64, Array1<usize>) {
    let num_vars = qubo.num_x();
    let total_masks = 1usize
        .checked_shl(num_vars as u32)
        .expect("enumerate_solve only supports component sizes below the machine word width");

    let (best_obj, best_mask) = if num_vars <= 10 {
        enumerate_small(qubo)
    } else {
        enumerate_masks(qubo, total_masks)
    };

    let mut best_solution = Array1::<usize>::zeros(num_vars);
    for i in 0..num_vars {
        best_solution[i] = (best_mask >> i) & 1;
    }

    (best_obj, best_solution)
}

fn enumerate_small(qubo: &Qubo) -> (f64, usize) {
    let terms = qubo
        .c
        .iter()
        .enumerate()
        .map(|(i, &value)| (1 << i, value))
        .chain(
            qubo.q
                .iter()
                .map(|(&value, (i, j))| ((1 << i) | (1 << j), 0.5 * value)),
        );

    enumerate_small_terms(qubo.num_x(), terms)
}

/// Terms encode the set of bits that must be one for their coefficient to apply.
pub(crate) fn enumerate_small_terms(
    num_vars: usize,
    terms: impl IntoIterator<Item = (usize, f64)>,
) -> (f64, usize) {
    assert!(num_vars <= 10);
    let total_masks = 1 << num_vars;
    let mut objectives = [0.0; 1 << 10];
    let objectives = &mut objectives[..total_masks];

    // Visit only assignments containing each term. Each objective receives the
    // same additions in the same order as direct evaluation (unlike Gray-code
    // delta updates), including for nonsymmetric matrices and diagonal terms.
    for (required, value) in terms {
        let free = (total_masks - 1) ^ required;
        let mut subset = free;
        loop {
            objectives[required | subset] += value;
            if subset == 0 {
                break;
            }
            subset = (subset - 1) & free;
        }
    }

    let mut best = (f64::INFINITY, 0);
    for (mask, &objective) in objectives.iter().enumerate() {
        if objective < best.0 {
            best = (objective, mask);
        }
    }
    best
}

fn enumerate_masks(qubo: &Qubo, total_masks: usize) -> (f64, usize) {
    let q_terms: Vec<_> = qubo
        .q
        .iter()
        .map(|(&value, (i, j))| ((1 << i) | (1 << j), 0.5 * value))
        .collect();
    let mut best = (f64::INFINITY, 0);
    for mask in 0..total_masks {
        let mut objective = 0.0;
        for (i, &value) in qubo.c.iter().enumerate() {
            if (mask >> i) & 1 == 1 {
                objective += value;
            }
        }
        for &(required, value) in &q_terms {
            if mask & required == required {
                objective += value;
            }
        }
        if objective < best.0 {
            best = (objective, mask);
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use smolprng::{JsfLarge, PRNG};
    use sprs::{CsMat, TriMat};

    fn reference(qubo: &Qubo) -> (f64, usize) {
        let mut best = (f64::INFINITY, 0);
        for mask in 0..(1 << qubo.num_x()) {
            let mut objective = 0.0;
            for (i, &value) in qubo.c.iter().enumerate() {
                if mask >> i & 1 == 1 {
                    objective += value;
                }
            }
            for (&value, (i, j)) in &qubo.q {
                if mask >> i & 1 == 1 && mask >> j & 1 == 1 {
                    objective += 0.5 * value;
                }
            }
            if objective < best.0 {
                best = (objective, mask);
            }
        }
        best
    }

    fn check(qubo: &Qubo) {
        let expected = reference(qubo);
        let (objective, solution) = enumerate_solve(qubo);
        let mask = solution
            .iter()
            .enumerate()
            .fold(0, |mask, (i, &bit)| mask | (bit << i));
        assert_eq!(objective.to_bits(), expected.0.to_bits());
        assert_eq!(mask, expected.1);
    }

    #[test]
    fn enumeration_preserves_exact_objectives_and_tie_breaking() {
        let mut prng = PRNG {
            generator: JsfLarge::from(312598u64),
        };
        for n in 1..=11 {
            for _ in 0..8 {
                let qubo = Qubo::make_random_qubo(n, &mut prng, 0.4);
                check(&qubo);
                check(&Qubo::new_with_c(qubo.q.to_csc(), qubo.c));
            }
            check(&Qubo::new(CsMat::zero((n, n))));
        }
        check(&Qubo::new(CsMat::zero((0, 0))));
        let mut terms = TriMat::new((4, 4));
        for (i, j, value) in [(0, 0, 1e16), (0, 1, -1e16), (1, 2, -0.1), (2, 3, 0.3)] {
            terms.add_triplet(i, j, value);
        }
        for matrix in [terms.to_csr(), terms.to_csc()] {
            check(&Qubo::new_with_c(
                matrix,
                Array1::from_vec(vec![1e-16, -0.2, 1.0, -0.0]),
            ));
        }
    }
}
