//! A batch-local feasible incumbent. Never share objective units across solvers.
use super::*;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Duration;

pub(super) struct BatchIncumbent {
    active: AtomicBool,
    value: AtomicU64,
    candidate: Mutex<Option<(Array1<usize>, f64)>>,
}

impl Default for BatchIncumbent {
    fn default() -> Self {
        Self {
            active: AtomicBool::new(false),
            value: AtomicU64::new(f64::INFINITY.to_bits()),
            candidate: Mutex::new(None),
        }
    }
}

impl BatchIncumbent {
    pub(super) fn value(&self) -> f64 {
        f64::from_bits(self.value.load(Ordering::Acquire))
    }

    pub(super) fn begin(&mut self, incumbent: f64, enabled: bool) {
        *self.active.get_mut() = enabled;
        *self.candidate.get_mut().unwrap() = None;
        *self.value.get_mut() = if enabled { incumbent } else { f64::INFINITY }.to_bits();
    }

    pub(super) fn finish(&mut self) -> Option<(Array1<usize>, f64)> {
        *self.active.get_mut() = false;
        *self.value.get_mut() = f64::INFINITY.to_bits();
        self.candidate.get_mut().unwrap().take()
    }
}

#[derive(Default)]
pub(super) struct WorkCounters {
    pub(super) cycle_relation_eligible: AtomicUsize,
    pub(super) cycle_relation_attempts: AtomicUsize,
    pub(super) cycle_relation_pairs: AtomicUsize,
    pub(super) cycle_relation_reductions: AtomicUsize,
    pub(super) cycle_relation_prunes: AtomicUsize,
    pub(super) cycle_relation_micros: AtomicU64,
    pub(super) weak_gradient_fixings: AtomicUsize,
    pub(super) width_attempts: AtomicUsize,
    pub(super) width_solved: AtomicUsize,
    pub(super) width_declined: [AtomicUsize; 3],
    pub(super) publications: AtomicUsize,
    pub(super) live_prunes: AtomicUsize,
    presolve_rechecks: AtomicUsize,
    presolve_changes: AtomicUsize,
    presolve_nanoseconds: AtomicU64,
    all_presolve_calls: AtomicUsize,
    all_presolve_nanoseconds: AtomicU64,
    pub(super) presolve_reuses: AtomicUsize,
}

#[derive(Debug, Default)]
pub struct SearchWorkStatistics {
    pub cycle_relation_eligible: usize,
    pub cycle_relation_attempts: usize,
    pub cycle_relation_pairs: usize,
    pub cycle_relation_reductions: usize,
    pub cycle_relation_prunes: usize,
    pub cycle_relation_seconds: f64,
    /// Includes conditional probes and repeated visits, not unique variables.
    pub weak_gradient_fixings: usize,
    pub width_attempts: usize,
    pub width_solved: usize,
    /// Width, work/time budget, and exact-arithmetic rejections.
    pub width_declined: [usize; 3],
    pub incumbent_publications: usize,
    /// Pruning checks enabled by the live incumbent, not unique nodes.
    pub live_cutoff_prunes: usize,
    pub probe_presolve_rechecks: usize,
    pub probe_presolve_changes: usize,
    /// Summed worker time, not wall time; only collected when auditing is enabled.
    pub probe_presolve_seconds: f64,
    pub all_presolve_calls: usize,
    pub all_presolve_seconds: f64,
    pub probe_presolve_reuses: usize,
}

impl BBSolver {
    pub fn search_work_statistics(&self) -> SearchWorkStatistics {
        let c = &self.work_counters;
        SearchWorkStatistics {
            cycle_relation_eligible: c.cycle_relation_eligible.load(Ordering::Relaxed),
            cycle_relation_attempts: c.cycle_relation_attempts.load(Ordering::Relaxed),
            cycle_relation_pairs: c.cycle_relation_pairs.load(Ordering::Relaxed),
            cycle_relation_reductions: c.cycle_relation_reductions.load(Ordering::Relaxed),
            cycle_relation_prunes: c.cycle_relation_prunes.load(Ordering::Relaxed),
            cycle_relation_seconds: c.cycle_relation_micros.load(Ordering::Relaxed) as f64 * 1e-6,
            weak_gradient_fixings: c.weak_gradient_fixings.load(Ordering::Relaxed),
            width_attempts: c.width_attempts.load(Ordering::Relaxed),
            width_solved: c.width_solved.load(Ordering::Relaxed),
            width_declined: std::array::from_fn(|i| c.width_declined[i].load(Ordering::Relaxed)),
            incumbent_publications: c.publications.load(Ordering::Relaxed),
            live_cutoff_prunes: c.live_prunes.load(Ordering::Relaxed),
            probe_presolve_rechecks: c.presolve_rechecks.load(Ordering::Relaxed),
            probe_presolve_changes: c.presolve_changes.load(Ordering::Relaxed),
            probe_presolve_seconds: c.presolve_nanoseconds.load(Ordering::Relaxed) as f64 * 1e-9,
            all_presolve_calls: c.all_presolve_calls.load(Ordering::Relaxed),
            all_presolve_seconds: c.all_presolve_nanoseconds.load(Ordering::Relaxed) as f64 * 1e-9,
            probe_presolve_reuses: c.presolve_reuses.load(Ordering::Relaxed),
        }
    }

    pub(super) fn publish_batch_candidate(&self, candidate: &Option<(Array1<usize>, f64)>) {
        let Some((x, reported)) = candidate else {
            return;
        };
        let current = self.batch_incumbent.value();
        // Re-evaluate only potential improvements in an active batch.
        if !self.batch_incumbent.active.load(Ordering::Relaxed)
            || !reported.is_finite()
            || *reported >= current
            || x.len() != self.qubo.num_x()
            || x.iter().any(|&v| v > 1)
        {
            return;
        }
        let value = self.qubo.eval_usize(x);
        if !value.is_finite() || value >= current {
            return;
        }
        let mut saved = self.batch_incumbent.candidate.lock().unwrap();
        if value < self.batch_incumbent.value() {
            // Save the witness before making its objective available to other workers.
            *saved = Some((x.clone(), value));
            self.batch_incumbent
                .value
                .store(value.to_bits(), Ordering::Release);
            self.work_counters
                .publications
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    pub(super) fn finish_batch_incumbent(&mut self) {
        if let Some((x, value)) = self.batch_incumbent.finish() {
            self.update_solution_if_better(&x, value);
        }
    }

    pub(super) fn record_presolve(&self, duration: Duration) {
        self.work_counters
            .all_presolve_calls
            .fetch_add(1, Ordering::Relaxed);
        self.work_counters.all_presolve_nanoseconds.fetch_add(
            duration.as_nanos().min(u64::MAX as u128) as u64,
            Ordering::Relaxed,
        );
    }

    pub(super) fn record_probe_presolve(&self, duration: Duration, changed: bool) {
        self.work_counters
            .presolve_rechecks
            .fetch_add(1, Ordering::Relaxed);
        self.work_counters
            .presolve_changes
            .fetch_add(usize::from(changed), Ordering::Relaxed);
        self.work_counters.presolve_nanoseconds.fetch_add(
            duration.as_nanos().min(u64::MAX as u128) as u64,
            Ordering::Relaxed,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Barrier};

    fn linear(cost: f64) -> BBSolver {
        BBSolver::new(
            Qubo::new_with_c(sprs::CsMat::zero((1, 1)), Array1::from_vec(vec![cost])),
            SolverOptions::new(),
        )
    }

    fn cached_node(solver: &BBSolver) -> QuboBBNode {
        let (fixed, bound) = solver.presolve_node(FixedVarMap::default()).unwrap();
        let mut node = QuboBBNode {
            lower_bound: bound,
            fixed_variables: fixed,
            solution: Array1::from_elem(solver.qubo.num_x(), 0.5),
            run_heuristic: false,
            subproblem_state: None,
            branch_origin: None,
            probe_bounds: None,
            probe_presolved: None,
        };
        solver.cache_probe_presolve(&mut node);
        node
    }

    #[test]
    fn probe_presolve_reuse_consumes_cache_and_keeps_fresh_cutoff_checks() {
        let mut solver = linear(-2.0);
        solver.options.audit_probe_presolve = true;
        let mut node = cached_node(&solver);
        let expected = (node.fixed_variables.clone(), node.lower_bound);
        let calls = solver.search_work_statistics().all_presolve_calls;
        assert_eq!(
            solver.presolve_or_reuse_probe(&mut node),
            Some(expected.clone())
        );
        assert!(node.probe_presolved.is_none());
        assert_eq!(solver.search_work_statistics().all_presolve_calls, calls);
        assert_eq!(solver.search_work_statistics().probe_presolve_reuses, 1);
        node.fixed_variables = expected.0;
        solver.presolve_or_reuse_probe(&mut node).unwrap();
        assert_eq!(
            solver.search_work_statistics().all_presolve_calls,
            calls + 1
        );

        let node = cached_node(&solver);
        solver.batch_incumbent.begin(0.0, true);
        solver.publish_batch_candidate(&Some((Array1::from_vec(vec![1]), -2.0)));
        assert!(solver.prunes_with_incumbent(node.lower_bound, 0.0));
    }

    #[test]
    fn probe_presolve_rejects_changed_fixings_context_and_options() {
        for change in 0..6 {
            let mut solver = linear(-2.0);
            solver.options.audit_probe_presolve = true;
            let mut node = cached_node(&solver);
            match change {
                0 => {
                    node.fixed_variables.insert(0, 0);
                }
                1 => {
                    node.fixed_variables.clear();
                }
                2 => {
                    solver.presolve_context = std::sync::Arc::new(());
                }
                3 => {
                    solver.options.roof_dual_weak_persistencies =
                        !solver.options.roof_dual_weak_persistencies;
                }
                4 => {
                    solver.options.node_lower_bound = NodeLowerBoundSelection::Li;
                }
                _ => {
                    solver.options.reuse_probe_presolve = false;
                }
            }
            let calls = solver.search_work_statistics().all_presolve_calls;
            solver.presolve_or_reuse_probe(&mut node).unwrap();
            assert_eq!(
                solver.search_work_statistics().all_presolve_calls,
                calls + 1
            );
            assert_eq!(solver.search_work_statistics().probe_presolve_reuses, 0);
        }
        let source = linear(-2.0);
        let mut other = linear(-3.0);
        other.options.audit_probe_presolve = true;
        let mut node = cached_node(&source);
        other.presolve_or_reuse_probe(&mut node).unwrap();
        assert_eq!(other.search_work_statistics().all_presolve_calls, 1);
        assert_eq!(other.search_work_statistics().probe_presolve_reuses, 0);
    }

    #[test]
    fn workers_see_improvement_before_commit_and_witness_is_not_lost() {
        let mut solver = linear(-2.0);
        solver.options.objective_lattice_pruning = false;
        solver.batch_incumbent.begin(0.0, true);
        let barrier = Arc::new(Barrier::new(2));
        std::thread::scope(|scope| {
            let solver = &solver;
            let ready = Arc::clone(&barrier);
            scope.spawn(move || {
                solver.publish_batch_candidate(&Some((Array1::ones(1), -2.0)));
                ready.wait();
            });
            scope.spawn(move || {
                barrier.wait();
                assert_eq!(solver.best_solution_value, 0.0);
                assert!(solver.prunes_with_incumbent(-1.0, 0.0));
            });
        });
        solver.finish_batch_incumbent();
        assert_eq!(solver.best_solution_value, -2.0);
        assert_eq!(solver.best_solution, Array1::<usize>::ones(1));
        assert_eq!(solver.batch_incumbent.value(), f64::INFINITY);
        assert_eq!(solver.search_work_statistics().incumbent_publications, 1);
        assert_eq!(solver.search_work_statistics().live_cutoff_prunes, 1);
        solver.batch_incumbent.begin(-2.0, true);
        assert!(solver.batch_incumbent.finish().is_none());
    }

    #[test]
    fn independent_units_disabled_batches_and_external_cutoffs_stay_separate() {
        let mut parent = linear(-100.0);
        let mut child = linear(-1.0);
        parent.batch_incumbent.begin(0.0, true);
        child.batch_incumbent.begin(0.0, true);
        child.publish_batch_candidate(&Some((Array1::ones(1), -1.0)));
        assert_eq!(parent.pruning_upper_bound(), 0.0);
        parent.publish_batch_candidate(&Some((Array1::ones(1), -100.0)));
        assert!((child.pruning_upper_bound() + 1.0).abs() < 1e-12);
        child.tighten_external_cutoff(-2.0);
        assert!(child.prunes_with_incumbent(-1.5, 0.0));
        assert_eq!(child.cutoff.discarded_bound(), -1.5);
        child.finish_batch_incumbent();
        assert!((child.best_solution_value + 1.0).abs() < 1e-12);
        assert!(!child.search_is_exact());
        parent.finish_batch_incumbent();
        let mut disabled = linear(-10.0);
        disabled.options.objective_lattice_pruning = false;
        disabled.batch_incumbent.begin(0.0, false);
        disabled.publish_batch_candidate(&Some((Array1::ones(1), -10.0)));
        assert!(!disabled.prunes_with_incumbent(-1.0, 0.0));
        disabled.finish_batch_incumbent();
        assert_eq!(disabled.best_solution_value, 0.0);
    }

    #[test]
    fn competing_publications_keep_the_best_recomputed_binary_witness() {
        let mut solver = BBSolver::new(
            Qubo::new_with_c(
                sprs::CsMat::zero((3, 3)),
                Array1::from_vec(vec![-1.0, -2.0, -4.0]),
            ),
            SolverOptions::new(),
        );
        solver.batch_incumbent.begin(0.0, true);
        std::thread::scope(|scope| {
            for bits in 1..8 {
                let solver = &solver;
                scope.spawn(move || {
                    let x = Array1::from_iter((0..3).map(|i| (bits >> i) & 1));
                    solver.publish_batch_candidate(&Some((x, -100.0))); // Recompute, do not trust this value.
                });
            }
        });
        solver.publish_batch_candidate(&Some((Array1::from_vec(vec![2, 2, 2]), -1000.0)));
        solver.finish_batch_incumbent();
        assert_eq!(solver.best_solution_value, -7.0);
        assert_eq!(solver.qubo.eval_usize(&solver.best_solution), -7.0);
        assert!(solver.best_solution.iter().all(|&v| v == 1));
    }

    #[test]
    fn live_batch_skips_a_backend_call_and_audit_keeps_presolve_enabled() {
        use crate::branch_subproblem::{SubProblemOptions, SubProblemResult};
        use crate::subproblemsolvers::mixingcut_sdp::MixingCutSubProblemResult;
        struct Candidate;
        impl SubProblemSolver for Candidate {
            fn solve_lower_bound(
                &self,
                _: &BBSolver,
                node: &QuboBBNode,
                _: Option<SubProblemOptions>,
            ) -> Box<dyn SubProblemResult> {
                let flip = node.fixed_variables[&0];
                Box::new(MixingCutSubProblemResult {
                    lower_bound: -20.0,
                    relaxed_solution: node.solution.clone(),
                    candidate_primal_solution: Some(Array1::from_iter(
                        (0..13).map(|i| (i % 2) ^ flip),
                    )),
                    subproblem_state: None,
                    conditional_bounds: Vec::new(),
                })
            }
        }
        let mut terms = sprs::TriMat::new((13, 13));
        for i in 0..13 {
            terms.add_triplet(i, (i + 1) % 13, 2.0);
            terms.add_triplet((i + 1) % 13, i, 2.0);
        }
        let mut c = Array1::from_elem(13, -2.0);
        c[0] += 1.0;
        for enabled in [false, true] {
            let mut options = SolverOptions::new();
            options.live_batch_incumbent = enabled;
            options.audit_probe_presolve = true;
            options.reuse_probe_presolve = false;
            // Count backend calls saved by incumbent sharing, not exact elimination.
            options.bounded_width_elimination = false;
            options.node_probe_candidates = 0;
            options.node_structural_reductions = false;
            let mut solver = BBSolver::new(Qubo::new_with_c(terms.to_csr(), c.clone()), options);
            solver.subproblem_solver = Box::new(Candidate);
            for value in 0..2 {
                let mut node = QuboBBNode {
                    fixed_variables: [(0, value)].into_iter().collect(),
                    lower_bound: -12.0 + value as f64,
                    solution: Array1::from_elem(13, 0.5),
                    run_heuristic: false,
                    subproblem_state: None,
                    branch_origin: None,
                    probe_bounds: None,
                    probe_presolved: None,
                };
                solver.cache_probe_presolve(&mut node);
                solver.nodes.push(node);
            }
            // Deterministic task ordering; publication still occurs inside the Rayon batch.
            rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .unwrap()
                .install(|| solver.search_batch(2));
            assert_eq!(solver.best_solution_value, -12.0);
            assert_eq!(solver.qubo.eval_usize(&solver.best_solution), -12.0);
            assert!(solver.nodes.is_empty());
            let expected = if enabled { 1 } else { 2 };
            assert_eq!(solver.node_probing_statistics().subproblem_calls, expected);
            assert_eq!(
                solver.search_work_statistics().probe_presolve_rechecks,
                expected
            );
            assert_eq!(solver.search_work_statistics().probe_presolve_changes, 0);
        }
    }
}
