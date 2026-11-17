//! Native binary objective lattice; never infer integrality from convexified data.
use crate::qubo::Qubo;
use std::collections::BTreeMap;

pub(super) struct ObjectiveLattice {
    step: f64,
    magnitude: f64,
}

impl ObjectiveLattice {
    pub(super) fn new(qubo: &Qubo) -> Option<Self> {
        // Exact dyadic conversion only. Bound the absolute sum so all native
        // binary objective sums remain exactly representable in f64.
        let mut coefficients = BTreeMap::<(usize, usize), i128>::new();
        let mut magnitude = 0i128;
        for (key, value) in qubo
            .c
            .iter()
            .enumerate()
            .map(|(i, &v)| ((i, i), v * 16.0))
            .chain(
                qubo.q
                    .iter()
                    .map(|(&v, (i, j))| ((i.min(j), i.max(j)), v * 8.0)),
            )
        {
            if !value.is_finite() || value.fract() != 0.0 || value.abs() > (1u64 << 48) as f64 {
                return None;
            }
            let value = value as i128;
            magnitude += value.abs();
            if magnitude > 1i128 << 48 {
                return None;
            }
            *coefficients.entry(key).or_default() += value;
        }
        let mut gcd = 0;
        for value in coefficients.into_values() {
            let mut b = value.abs();
            while b != 0 {
                let remainder = gcd % b;
                gcd = b;
                b = remainder;
            }
        }
        (gcd != 0).then_some(Self {
            step: gcd as f64 / 16.0,
            magnitude: magnitude as f64 / 16.0,
        })
    }

    pub(super) fn strengthen(&self, lower: f64) -> f64 {
        self.certified_grid_point(lower)
            .map_or(lower, |grid| lower.max(grid))
    }

    pub(super) fn certified_grid_point(&self, lower: f64) -> Option<f64> {
        if !lower.is_finite() || lower.abs() > self.magnitude {
            return None;
        }
        let guard = 64.0 * f64::EPSILON * self.magnitude + 1e-10 * lower.abs().max(1.0);
        let quotient = ((lower - guard).next_down() / self.step).next_down();
        // The integer numerator of this grid point is below 2^49, so the
        // dyadic product is exact (also for non-power-of-two lattice steps).
        Some(quotient.ceil() * self.step)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn grid_point_only_advances_past_the_numerical_guard() {
        let lattice = ObjectiveLattice {
            step: 2.0,
            magnitude: 20.0,
        };
        let barely_above = -4.0 + 1e-12;
        assert_eq!(lattice.certified_grid_point(barely_above), Some(-4.0));
        assert_eq!(lattice.strengthen(barely_above), barely_above);
        assert_eq!(lattice.certified_grid_point(-4.0 + 1e-5), Some(-2.0));
        assert_eq!(lattice.certified_grid_point(-21.0), None);
    }

    #[test]
    fn effective_binary_coefficients_include_linear_diagonal_and_both_orientations() {
        let mut q = sprs::TriMat::new((2, 2));
        q.add_triplet(0, 0, 6.0);
        q.add_triplet(0, 1, 3.0);
        q.add_triplet(1, 0, 5.0);
        let q = Qubo::new_with_c(q.to_csr(), Array1::from_vec(vec![-1.0, -6.0]));
        let lattice = ObjectiveLattice::new(&q).unwrap();
        assert_eq!(lattice.step, 2.0);
        assert!(lattice.strengthen(-5.9) > -4.00000001);
        assert!(lattice.strengthen(-4.0 + 1e-12) < -3.9);
        for mask in 0..4 {
            let value = q.eval_usize(&Array1::from_iter((0..2).map(|i| (mask >> i) & 1)));
            for k in 0..100 {
                assert!(lattice.strengthen(value - k as f64 / 10.0) <= value);
            }
        }
        assert_eq!(lattice.strengthen(f64::NEG_INFINITY), f64::NEG_INFINITY);
    }

    #[test]
    fn unsupported_coefficients_decline_instead_of_rounding() {
        for value in [0.1, f64::NAN, f64::INFINITY, 1e20] {
            let q = Qubo::new_with_c(sprs::CsMat::zero((1, 1)), Array1::from_vec(vec![value]));
            assert!(ObjectiveLattice::new(&q).is_none());
        }
    }

    #[test]
    fn lattice_pruning_preserves_exhaustive_fixed_node_optima() {
        use crate::{branchbound::BBSolver, solver_options::SolverOptions, FixedVarMap};
        let mut rng = crate::tests::make_test_prng();
        for sample in 0..8 {
            let mut q = Qubo::make_random_qubo(10, &mut rng, 0.4);
            for value in q.q.data_mut() {
                *value = value.round() * 0.5;
            }
            q.c.mapv_inplace(|v| v.round() * 0.25);
            let expected = (0..1024usize)
                .filter(|mask| mask & 1 == sample % 2)
                .map(|mask| q.eval_usize(&Array1::from_iter((0..10).map(|i| (mask >> i) & 1))))
                .fold(f64::INFINITY, f64::min);
            let mut options = SolverOptions::new();
            options.verbose = 0;
            options.objective_lattice_pruning = true;
            options.fixed_variables = FixedVarMap::from_iter([(0, sample % 2)]);
            options.bounded_width_elimination = false;
            options.width_region_elimination = false;
            let mut solver = BBSolver::new(q.clone(), options);
            let (x, value) = solver.solve();
            assert_eq!(x[0], sample % 2);
            assert!((value - expected).abs() < 1e-8);
            assert_eq!(q.eval_usize(&x), expected);
            assert!(solver.nodes.is_empty() && !solver.early_stop);
        }
    }
}
