//! An external cutoff is not a local primal solution. Remember the certified
//! bounds of discarded regions so an exhausted subsearch cannot invent an optimum.
use super::*;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

pub(super) struct CutoffState {
    limit: f64,
    discarded_lower_bound: AtomicU64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn external_cutoff_keeps_a_certificate_without_replacing_the_incumbent() {
        let mut solver = BBSolver::new(Qubo::new(sprs::CsMat::eye(2)), SolverOptions::new());
        solver.best_solution_value = 20.0;
        solver.tighten_external_cutoff(10.0);
        assert!(solver.prunes_with_incumbent(12.0, 20.0));
        assert_eq!(solver.best_solution_value, 20.0);
        assert_eq!(solver.search_lower_bound(), 12.0);
        assert!(!solver.search_is_exact());
        solver.tighten_external_cutoff(8.0);
        assert!(solver.prunes_with_incumbent(9.0, 20.0));
        assert_eq!(solver.search_lower_bound(), 9.0);
        solver.tighten_external_cutoff(19.0);
        assert_eq!(solver.pruning_upper_bound(), 8.0);
        solver.best_solution_value = 7.0;
        assert!(solver.search_is_exact());
    }

    #[test]
    fn translated_cutoffs_do_not_use_a_child_scale_tolerance() {
        let mut solver = BBSolver::new(Qubo::new(sprs::CsMat::eye(2)), SolverOptions::new());
        solver.best_solution_value = 1e12 + 100.0;
        solver.tighten_external_cutoff(1e12);
        assert!(!solver.prunes_with_incumbent(1e12 - 1.0, 1e12 + 100.0));
        assert_eq!(solver.cutoff.discarded_bound(), f64::INFINITY);
    }

    #[test]
    fn lattice_cutoff_records_the_strengthened_certificate() {
        let q = Qubo::new_with_c(sprs::CsMat::zero((2, 2)), Array1::from_vec(vec![2.0, -4.0]));
        let mut solver = BBSolver::new(q, SolverOptions::new());
        solver.options.objective_lattice_pruning = false;
        solver.best_solution_value = 0.0;
        solver.tighten_external_cutoff(-2.0);
        assert!(!solver.prunes_with_incumbent(-3.0, 0.0));
        solver.options.objective_lattice_pruning = true;
        assert!(solver.prunes_with_incumbent(-3.0, 0.0));
        assert_eq!(solver.cutoff.discarded_bound(), -2.0);
        assert_eq!(solver.cutoff_statistics().lattice_prunes, 1);
    }

    #[test]
    fn default_lattice_rounds_the_global_certificate_in_native_units() {
        let q = Qubo::new_with_c(sprs::CsMat::zero((2, 2)), Array1::from_vec(vec![4.0, -2.0]));
        let mut solver = BBSolver::new(q, SolverOptions::new());
        solver.cutoff.record(-3.5);
        assert_eq!(solver.search_lower_bound(), -2.0);
        assert_eq!(solver.best_solution_value, 0.0);
        solver.options.objective_lattice_pruning = false;
        assert_eq!(solver.search_lower_bound(), -3.5);
    }

    #[test]
    fn default_lattice_leaves_unsupported_objectives_unrounded() {
        let q = Qubo::new_with_c(sprs::CsMat::zero((2, 2)), Array1::from_vec(vec![0.1, -2.0]));
        let solver = BBSolver::new(q, SolverOptions::new());
        solver.cutoff.record(-2.05);
        assert_eq!(solver.search_lower_bound(), -2.05);
    }
}

impl Default for CutoffState {
    fn default() -> Self {
        Self {
            limit: f64::INFINITY,
            discarded_lower_bound: AtomicU64::new(f64::INFINITY.to_bits()),
        }
    }
}

impl CutoffState {
    pub(super) fn is_tighter_than(&self, incumbent: f64) -> bool {
        self.limit < incumbent
    }
    pub(super) fn reset_proof(&self) {
        self.discarded_lower_bound
            .store(f64::INFINITY.to_bits(), Ordering::Relaxed);
    }

    pub(super) fn record(&self, bound: f64) {
        self.discarded_lower_bound
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |old| {
                (bound < f64::from_bits(old)).then_some(bound.to_bits())
            })
            .ok();
    }

    pub(super) fn discarded_bound(&self) -> f64 {
        f64::from_bits(self.discarded_lower_bound.load(Ordering::Relaxed))
    }
}

#[derive(Default)]
pub(super) struct CutoffCounters {
    lattice_prunes: AtomicUsize,
    updates: AtomicUsize,
    prunes: AtomicUsize,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CutoffStatistics {
    /// Successful cutoff comparisons newly enabled by lattice rounding (not unique nodes).
    pub lattice_prunes: usize,
    pub updates: usize,
    pub prunes: usize,
}

impl BBSolver {
    pub fn cutoff_statistics(&self) -> CutoffStatistics {
        CutoffStatistics {
            lattice_prunes: self.cutoff_counters.lattice_prunes.load(Ordering::Relaxed),
            updates: self.cutoff_counters.updates.load(Ordering::Relaxed),
            prunes: self.cutoff_counters.prunes.load(Ordering::Relaxed),
        }
    }

    pub(crate) fn pruning_upper_bound(&self) -> f64 {
        self.best_solution_value
            .min(self.batch_incumbent.value())
            .min(self.cutoff.limit)
    }

    pub(super) fn tighten_external_cutoff(&mut self, limit: f64) {
        if self.options.component_cutoff_propagation
            && limit.is_finite()
            && limit < self.cutoff.limit
        {
            self.cutoff.limit = limit;
            self.cutoff_counters.updates.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub(super) fn verified_objective_lattice(
        &self,
    ) -> Option<&super::objective_lattice::ObjectiveLattice> {
        if !self.options.objective_lattice_pruning {
            return None;
        }
        self.objective_lattice
            .get_or_init(|| super::objective_lattice::ObjectiveLattice::new(&self.input_qubo))
            .as_ref()
    }

    pub(super) fn lattice_lower_bound(&self, bound: f64) -> f64 {
        if !bound.is_finite() {
            return bound;
        }
        self.verified_objective_lattice()
            .map_or(bound, |lattice| lattice.strengthen(bound))
    }

    pub(super) fn prunes_with_incumbent(&self, bound: f64, incumbent: f64) -> bool {
        let fresh = incumbent.min(self.batch_incumbent.value());
        let bound = if self.options.objective_lattice_pruning {
            let rounded = self.lattice_lower_bound(bound);
            let closes = |b| {
                b >= self.cutoff.limit
                    || if self.cutoff.limit < fresh {
                        b >= fresh
                    } else {
                        Self::bound_closes_gap(b, fresh)
                    }
            };
            if !closes(bound) && closes(rounded) {
                self.cutoff_counters
                    .lattice_prunes
                    .fetch_add(1, Ordering::Relaxed);
            }
            rounded
        } else {
            bound
        };
        if fresh < incumbent {
            let old_prunes = bound >= self.cutoff.limit
                || if self.cutoff.limit < incumbent {
                    bound >= incumbent
                } else {
                    Self::bound_closes_gap(bound, incumbent)
                };
            let new_prunes = bound >= self.cutoff.limit
                || if self.cutoff.limit < fresh {
                    bound >= fresh
                } else {
                    Self::bound_closes_gap(bound, fresh)
                };
            if new_prunes && !old_prunes {
                self.work_counters
                    .live_prunes
                    .fetch_add(1, Ordering::Relaxed);
            }
        }
        let incumbent = fresh;
        // Use no relative tolerance on a translated external cutoff: a large
        // eliminated constant can make child and parent objective scales differ.
        if bound < incumbent && bound >= self.cutoff.limit {
            self.cutoff.record(bound);
            self.cutoff_counters.prunes.fetch_add(1, Ordering::Relaxed);
            return true;
        }
        if self.cutoff.limit < incumbent {
            bound >= incumbent
        } else {
            Self::bound_closes_gap(bound, incumbent)
        }
    }

    pub(crate) fn search_lower_bound(&self) -> f64 {
        let bound = self.nodes.iter().map(|n| n.lower_bound).fold(
            self.best_solution_value.min(self.cutoff.discarded_bound()),
            f64::min,
        );
        self.lattice_lower_bound(bound)
    }

    #[cfg(test)]
    pub(super) fn search_is_exact(&self) -> bool {
        self.nodes.is_empty() && self.cutoff.discarded_bound() >= self.best_solution_value
    }

    pub(super) fn lift_cutoff_proof(&self, child: &BBSolver, constant: f64) {
        let lower = child.cutoff.discarded_bound();
        if lower.is_finite() {
            self.cutoff.record((lower + constant).next_down());
        }
    }
}
