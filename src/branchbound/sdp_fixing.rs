use super::*;
use crate::branch_subproblem::ConditionalLowerBound;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Duration;

#[derive(Default)]
pub(super) struct SdpFixingCounters {
    attempts: AtomicUsize,
    certificates: AtomicUsize,
    fixed: AtomicUsize,
    pruned: AtomicUsize,
    nanoseconds: AtomicU64,
}

#[derive(Debug, Default)]
pub struct SdpFixingStatistics {
    pub attempts: usize,
    pub conditional_certificates: usize,
    pub fixed_variables: usize,
    pub pruned_nodes: usize,
    /// Summed worker time, not elapsed wall time.
    pub seconds: f64,
}

impl BBSolver {
    pub fn sdp_fixing_statistics(&self) -> SdpFixingStatistics {
        let c = &self.sdp_fixing_counters;
        SdpFixingStatistics {
            attempts: c.attempts.load(Ordering::Relaxed),
            conditional_certificates: c.certificates.load(Ordering::Relaxed),
            fixed_variables: c.fixed.load(Ordering::Relaxed),
            pruned_nodes: c.pruned.load(Ordering::Relaxed),
            seconds: c.nanoseconds.load(Ordering::Relaxed) as f64 * 1e-9,
        }
    }

    pub(crate) fn record_sdp_fixing(&self, elapsed: Duration, certificates: usize) {
        let c = &self.sdp_fixing_counters;
        c.attempts.fetch_add(1, Ordering::Relaxed);
        c.certificates.fetch_add(certificates, Ordering::Relaxed);
        c.nanoseconds.fetch_add(
            elapsed.as_nanos().min(u64::MAX as u128) as u64,
            Ordering::Relaxed,
        );
    }

    /// Returns (closed, changed). These are cutoff-local deductions, not global
    /// persistencies; retain discarded-side certificates in nested searches.
    pub(super) fn apply_sdp_bounds(
        &self,
        node: &mut QuboBBNode,
        bounds: &[ConditionalLowerBound],
        incumbent: f64,
    ) -> (bool, bool) {
        let mut changed = false;
        for b in bounds {
            if b.variable >= self.qubo.num_x()
                || node.fixed_variables.contains_key(&b.variable)
                || !b.zero.is_finite()
                || !b.one.is_finite()
            {
                continue;
            }
            node.lower_bound = node.lower_bound.max(b.zero.min(b.one));
            let zero = self.prunes_with_incumbent(b.zero, incumbent);
            let one = self.prunes_with_incumbent(b.one, incumbent);
            if zero && one {
                self.sdp_fixing_counters
                    .pruned
                    .fetch_add(1, Ordering::Relaxed);
                return (true, changed);
            }
            if zero || one {
                let value = usize::from(zero);
                node.fixed_variables.insert(b.variable, value);
                node.solution[b.variable] = value as f64;
                node.lower_bound = node.lower_bound.max(if zero { b.one } else { b.zero });
                self.sdp_fixing_counters
                    .fixed
                    .fetch_add(1, Ordering::Relaxed);
                changed = true;
            }
        }
        (false, changed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deductions_are_local_and_external_cutoffs_keep_their_proof() {
        let mut solver = BBSolver::new(Qubo::new(sprs::CsMat::eye(2)), SolverOptions::new());
        solver.best_solution_value = 20.0;
        solver.tighten_external_cutoff(10.0);
        let mut node = QuboBBNode {
            fixed_variables: FixedVarMap::default(),
            lower_bound: -10.0,
            solution: Array1::from_elem(2, 0.5),
            run_heuristic: false,
            subproblem_state: None,
        };
        let b = [ConditionalLowerBound {
            variable: 0,
            zero: 11.0,
            one: 4.0,
        }];
        assert_eq!(solver.apply_sdp_bounds(&mut node, &b, 20.0), (false, true));
        assert_eq!(node.fixed_variables[&0], 1);
        assert_eq!(node.lower_bound, 4.0);
        assert_eq!(solver.cutoff.discarded_bound(), 11.0);
        assert_eq!(solver.best_solution_value, 20.0);
        assert!(solver.root_constraints.is_empty());
        assert_eq!(
            solver.apply_sdp_bounds(
                &mut node,
                &[ConditionalLowerBound {
                    variable: 1,
                    zero: 12.0,
                    one: 13.0
                }],
                20.0
            ),
            (true, false)
        );
    }

    #[test]
    fn solving_with_fixing_preserves_optima_and_incoming_fixings() {
        use crate::branch_subproblem::SubProblemSelection;
        use smolprng::{JsfLarge, PRNG};
        let mut attempts = 0;
        for sample in 0..16 {
            let mut rng = PRNG {
                generator: JsfLarge::from(3810 + sample as u64),
            };
            let n = 13;
            let mut terms = sprs::TriMat::new((n, n));
            let mut linear = Array1::zeros(n);
            for i in 0..n {
                for j in i + 1..n {
                    let w = (rng.gen_f64() * 11.0).floor() - 5.0;
                    terms.add_triplet(i, j, w);
                    terms.add_triplet(j, i, w);
                    linear[i] -= 0.5 * w;
                    linear[j] -= 0.5 * w;
                }
                linear[i] += 0.1;
            }
            let q = Qubo::new_with_c(terms.to_csr(), linear);
            let fixed: FixedVarMap = [(3, sample % 2)].into_iter().collect();
            let expected = (0..1 << n)
                .filter(|mask| (mask >> 3) & 1 == sample % 2)
                .map(|mask| q.eval_usize(&Array1::from_iter((0..n).map(|i| (mask >> i) & 1))))
                .fold(f64::INFINITY, f64::min);
            for enabled in [false, true] {
                let mut o = SolverOptions::new();
                o.verbose = 0;
                o.sdp_dual_fixing = enabled;
                o.fixed_variables = fixed.clone();
                o.sub_problem_solver = SubProblemSelection::MixingCutSDP;
                o.root_low_degree_elimination = false;
                o.root_dominant_edge_contraction = false;
                o.node_structural_reductions = false;
                o.node_probe_candidates = 0;
                o.node_lower_bound = crate::solver_options::NodeLowerBoundSelection::Li;
                let mut solver = BBSolver::new(q.clone(), o);
                let (x, value) = solver.solve();
                assert!(solver.nodes.is_empty());
                assert_eq!(x[3], sample % 2);
                assert!((value - expected).abs() < 1e-8);
                assert!((q.eval_usize(&x) - expected).abs() < 1e-8);
                attempts += solver.sdp_fixing_statistics().attempts;
            }
        }
        assert!(attempts > 0);
    }
}
