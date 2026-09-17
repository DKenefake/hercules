//! Rank-one conditional bounds extracted from a proposed dual slack factor.
//!
//! For binary spins, f(s) >= base + ||L^T s||^2. With v=e_i+t*e_anchor,
//! s_i=t*s_anchor implies (v^T s)^2=4, hence f(s)>=base+4/||L^-1 v||^2.
//! The base includes an outward-rounded bound on C-L*L^T, so correctness does
//! not rely on the proposed dual multipliers or Cholesky being accurate.
use super::*;
use crate::branch_subproblem::ConditionalLowerBound;
use std::time::Instant;

#[derive(Clone, Copy, Default)]
struct Interval {
    lo: f64,
    hi: f64,
}

impl Interval {
    fn point(x: f64) -> Self {
        Self { lo: x, hi: x }
    }
    fn add(self, b: Self) -> Self {
        Self {
            lo: (self.lo + b.lo).next_down(),
            hi: (self.hi + b.hi).next_up(),
        }
    }
    fn neg(self) -> Self {
        Self {
            lo: -self.hi,
            hi: -self.lo,
        }
    }
    fn sub(self, b: Self) -> Self {
        self.add(b.neg())
    }
    fn scale(self, x: f64) -> Self {
        let a = self.lo * x;
        let b = self.hi * x;
        Self {
            lo: a.min(b).next_down(),
            hi: a.max(b).next_up(),
        }
    }
    fn divide_positive(self, x: f64) -> Self {
        Self {
            lo: (self.lo / x).next_down(),
            hi: (self.hi / x).next_up(),
        }
    }
    fn abs_upper(self) -> f64 {
        self.lo.abs().max(self.hi.abs())
    }
    fn midpoint(self) -> f64 {
        0.5 * self.lo + 0.5 * self.hi
    }
}

// Enclose the native polynomial exactly, including asymmetric storage and c.
// Linear terms are absorbed into the diagonal to match MixingCut's convention.
fn sign_matrix(qubo: &Qubo) -> Vec<Interval> {
    let a = qubo.num_x();
    let n = a + 1;
    let mut c = vec![Interval::default(); n * n];
    for (&q, (i, j)) in &qubo.q {
        let h = Interval::point(q).scale(0.0625);
        for (r, s, v) in [
            (i, j, h),
            (j, i, h),
            (i, a, h.neg()),
            (a, i, h.neg()),
            (j, a, h.neg()),
            (a, j, h.neg()),
            (a, a, h.scale(2.0)),
        ] {
            c[r * n + s] = c[r * n + s].add(v);
        }
    }
    for i in 0..a {
        let h = Interval::point(qubo.c[i]).scale(0.25);
        for (r, s, v) in [(i, i, h), (i, a, h.neg()), (a, i, h.neg()), (a, a, h)] {
            c[r * n + s] = c[r * n + s].add(v);
        }
    }
    c
}

pub(super) fn conditional_bounds(
    qubo: &Qubo,
    dual: &Array1<f64>,
    reported_bound: f64,
    deadline: Instant,
) -> Vec<ConditionalLowerBound> {
    let n = qubo.num_x() + 1;
    let empty = Vec::new();
    if n <= 1
        || dual.len() != n
        || !reported_bound.is_finite()
        || dual.iter().any(|x| !x.is_finite())
        || Instant::now() >= deadline
    {
        return empty;
    }
    let c = sign_matrix(qubo);
    if c.iter().any(|x| !x.lo.is_finite() || !x.hi.is_finite()) {
        return empty;
    }
    let scale = c.iter().map(|x| x.abs_upper()).fold(1.0, f64::max);
    // Only a proposal: failure/roundoff is checked below, not assumed away.
    let shift = ((dual.sum() - reported_bound) / (n as f64)).max(0.0) + 1e-9 * scale;
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        if Instant::now() >= deadline {
            return empty;
        }
        for j in 0..=i {
            let mut v = c[i * n + j].midpoint();
            if i == j {
                v += shift - dual[i];
            }
            for k in 0..j {
                v -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if !v.is_finite() || v <= 0.0 {
                    return empty;
                }
                l[i * n + j] = v.sqrt();
            } else {
                l[i * n + j] = v / l[j * n + j];
                if !l[i * n + j].is_finite() {
                    return empty;
                }
            }
        }
    }
    // For s_i in {-1,1}, the residual is bounded by its diagonal sum minus
    // absolute off-diagonal entries. This also covers conversion roundoff.
    let mut base = 0.0;
    for i in 0..n {
        if Instant::now() >= deadline {
            return empty;
        }
        for j in 0..=i {
            let mut residual = c[i * n + j];
            for k in 0..=j {
                residual = residual.sub(Interval::point(l[i * n + k]).scale(l[j * n + k]));
            }
            let contribution = if i == j {
                residual.lo
            } else {
                -(2.0 * residual.abs_upper()).next_up()
            };
            base = (base + contribution).next_down();
        }
    }
    if !base.is_finite() {
        return empty;
    }
    // Outward triangular solves enclose the exact inverse of the stored L.
    let mut inv = vec![Interval::default(); n * n];
    for col in 0..n {
        if Instant::now() >= deadline {
            return empty;
        }
        for i in col..n {
            let mut v = Interval::point(f64::from(i == col));
            for k in col..i {
                v = v.sub(inv[k * n + col].scale(l[i * n + k]));
            }
            inv[i * n + col] = v.divide_positive(l[i * n + i]);
        }
    }
    let mut bounds = Vec::with_capacity(n - 1);
    for i in 0..n - 1 {
        if Instant::now() >= deadline {
            break;
        }
        let mut values = [base; 2];
        for (value, bound) in values.iter_mut().enumerate() {
            let spin = if value == 0 { 1.0 } else { -1.0 };
            let mut norm = 0.0;
            for row in i..n {
                let u = inv[row * n + i]
                    .add(inv[row * n + n - 1].scale(spin))
                    .abs_upper();
                norm = (norm + (u * u).next_up()).next_up();
            }
            if norm.is_finite() && norm > 0.0 {
                *bound = (base + (4.0 / norm).next_down()).next_down();
            }
        }
        if values.iter().all(|x| x.is_finite()) {
            bounds.push(ConditionalLowerBound {
                variable: i,
                zero: values[0],
                one: values[1],
            });
        }
    }
    bounds
}

#[cfg(test)]
mod tests {
    use super::*;

    fn deadline() -> Instant {
        Instant::now() + Duration::from_secs(10)
    }

    #[test]
    fn excludes_the_wrong_side_of_a_linear_problem() {
        let q = Qubo::new_with_c(sprs::CsMat::zero((1, 1)), Array1::from_vec(vec![4.0]));
        let b = conditional_bounds(&q, &Array1::zeros(2), 0.0, deadline());
        assert_eq!(b.len(), 1);
        assert!(b[0].zero <= 0.0);
        assert!(b[0].one > 3.99 && b[0].one <= 4.0);
    }

    #[test]
    fn all_conditional_bounds_hold_for_exhaustive_assignments() {
        let mut certified = 0;
        for sample in 0..80 {
            let n = 7;
            let mut q = sprs::TriMat::new((n, n));
            // Asymmetric and noninteger native data exercise the conversion.
            for i in 0..n {
                for j in 0..n {
                    if (i * 7 + j * 3 + sample) % 4 == 0 {
                        q.add_triplet(
                            i,
                            j,
                            ((sample * 11 + i * 5 + j * 13) % 23) as f64 / 3.0 - 4.0,
                        );
                    }
                }
            }
            let qubo = Qubo::new_with_c(
                q.to_csr(),
                Array1::from_iter((0..n).map(|i| ((i * 3 + sample) % 11) as f64 / 7.0 - 0.5)),
            );
            // Deliberately arbitrary duals: the certificate must not trust them.
            let dual = Array1::from_elem(n + 1, -20.0);
            let bounds = conditional_bounds(&qubo, &dual, -200.0, deadline());
            certified += bounds.len();
            for mask in 0..1 << n {
                let x = Array1::from_iter((0..n).map(|i| (mask >> i) & 1));
                let value = qubo.eval_usize(&x);
                for b in &bounds {
                    let lower = if x[b.variable] == 0 { b.zero } else { b.one };
                    assert!(
                        lower <= value + 1e-10,
                        "sample={sample} mask={mask} {lower} > {value}"
                    );
                }
            }
        }
        assert_eq!(certified, 80 * 7);
    }

    #[test]
    fn mixingcut_certificates_hold_with_truncated_and_momentum_solves() {
        let mut certified = 0;
        for sample in 0..24 {
            let mut rng = PRNG {
                generator: JsfLarge::from(6100 + sample as u64),
            };
            let mut q = Qubo::make_random_qubo(8, &mut rng, 0.6);
            q.q = (&q.q + &q.q.transpose_view()).to_csr();
            for beta in [0.0, 0.8] {
                let opts = MixingCutSDPSolver::with_momentum(beta)
                    .default_options(8, Some(if sample % 2 == 0 { 1 } else { 80 }));
                let result = solve_qubo_sdp_subproblem(&q.q, &q.c, &opts);
                let bounds = conditional_bounds(
                    &q,
                    &result.dual_variables,
                    result.qubo_lower_bound,
                    deadline(),
                );
                certified += bounds.len();
                for mask in 0..256 {
                    let x = Array1::from_iter((0..8).map(|i| (mask >> i) & 1));
                    let value = q.eval_usize(&x);
                    for b in &bounds {
                        assert!(if x[b.variable] == 0 { b.zero } else { b.one } <= value + 1e-9);
                    }
                }
            }
        }
        assert!(certified > 200);
    }

    #[test]
    fn malformed_and_expired_certificates_are_skipped() {
        let q = Qubo::new(sprs::CsMat::eye(3));
        assert!(conditional_bounds(&q, &Array1::zeros(4), 0.0, Instant::now()).is_empty());
        assert!(conditional_bounds(&q, &Array1::zeros(3), 0.0, deadline()).is_empty());
        assert!(
            conditional_bounds(&q, &Array1::from_elem(4, f64::NAN), 0.0, deadline()).is_empty()
        );
        // A forged high scalar bound must not turn an indefinite slack into proof.
        assert!(conditional_bounds(&q, &Array1::from_elem(4, 100.0), 1e9, deadline()).is_empty());
    }
}
