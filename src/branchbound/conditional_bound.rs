// Shared application of certified conditional bounds, including cycle probes.
use super::*;
use crate::branch_subproblem::ConditionalLowerBound;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Duration;

#[derive(Default)]
pub(super) struct ConditionalBoundCounters {
    attempts: AtomicUsize,
    certificates: AtomicUsize,
    fixed: AtomicUsize,
    pruned: AtomicUsize,
    nanoseconds: AtomicU64,
}

#[derive(Debug, Default)]
pub struct ConditionalBoundStatistics {
    pub attempts: usize,
    pub conditional_certificates: usize,
    pub fixed_variables: usize,
    pub pruned_nodes: usize,
    /// Summed worker time, not elapsed wall time.
    pub seconds: f64,
}

impl BBSolver {
    pub fn conditional_bound_statistics(&self) -> ConditionalBoundStatistics {
        let c = &self.conditional_bound_counters;
        ConditionalBoundStatistics {
            attempts: c.attempts.load(Ordering::Relaxed),
            conditional_certificates: c.certificates.load(Ordering::Relaxed),
            fixed_variables: c.fixed.load(Ordering::Relaxed),
            pruned_nodes: c.pruned.load(Ordering::Relaxed),
            seconds: c.nanoseconds.load(Ordering::Relaxed) as f64 * 1e-9,
        }
    }

    pub(crate) fn record_conditional_bounds(&self, elapsed: Duration, certificates: usize) {
        let c = &self.conditional_bound_counters;
        c.attempts.fetch_add(1, Ordering::Relaxed);
        c.certificates.fetch_add(certificates, Ordering::Relaxed);
        c.nanoseconds.fetch_add(
            elapsed.as_nanos().min(u64::MAX as u128) as u64,
            Ordering::Relaxed,
        );
    }

    /// Returns (closed, changed). These are cutoff-local deductions, not global
    /// persistencies; retain discarded-side certificates in nested searches.
    pub(super) fn apply_conditional_bounds(
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
            Self::retain_probe_bound(node, *b);
            node.lower_bound = node.lower_bound.max(b.zero.min(b.one));
            let zero = self.prunes_with_incumbent(b.zero, incumbent);
            let one = self.prunes_with_incumbent(b.one, incumbent);
            if zero && one {
                self.conditional_bound_counters
                    .pruned
                    .fetch_add(1, Ordering::Relaxed);
                return (true, changed);
            }
            if zero || one {
                let value = usize::from(zero);
                node.fixed_variables.insert(b.variable, value);
                node.solution[b.variable] = value as f64;
                node.lower_bound = node.lower_bound.max(if zero { b.one } else { b.zero });
                self.conditional_bound_counters
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
            branch_origin: None,
            probe_bounds: None,
            probe_presolved: None,
        };
        let b = [ConditionalLowerBound {
            variable: 0,
            zero: 11.0,
            one: 4.0,
        }];
        assert_eq!(
            solver.apply_conditional_bounds(&mut node, &b, 20.0),
            (false, true)
        );
        assert_eq!(node.fixed_variables[&0], 1);
        assert_eq!(node.lower_bound, 4.0);
        assert_eq!(solver.cutoff.discarded_bound(), 11.0);
        assert_eq!(solver.best_solution_value, 20.0);
        assert!(solver.root_constraints.is_empty());
        assert_eq!(
            solver.apply_conditional_bounds(
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
}
