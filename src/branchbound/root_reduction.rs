use super::*;
use crate::preprocess::low_degree::reduce_root_with_statistics;
use std::time::Duration;

impl BBSolver {
    fn certified_native_root_objective(&self) -> Option<f64> {
        if self.component_depth != 0
            || self.structural_root_reduced
            || !self.root_cycle_floor.is_finite()
        {
            return None;
        }
        let bound = self
            .verified_objective_lattice()?
            .certified_grid_point(self.root_cycle_floor)?;
        let objective = self.native_feasible_incumbent()?;
        (objective == bound).then_some(objective)
    }

    fn native_feasible_incumbent(&self) -> Option<f64> {
        let x = &self.best_solution;
        if x.len() != self.input_qubo.num_x()
            || x.iter().any(|&value| value > 1)
            || self
                .options
                .fixed_variables
                .iter()
                .any(|(&i, &value)| x.get(i) != Some(&value))
        {
            return None;
        }
        // On the verified native lattice, this objective sum is exact.
        Some(self.input_qubo.eval_usize(x))
    }

    pub(super) fn try_layered_root_certificate(&mut self) -> bool {
        if self.component_depth != 0 || self.structural_root_reduced {
            return false;
        }
        let seconds = self.options.max_time - (get_current_time() - self.time_start);
        let budget = (seconds / 4.0).min(4.0);
        if seconds.is_nan() || budget < 0.02 {
            return false;
        }
        let start = std::time::Instant::now();
        let Some(certificate) = self.prepared_cycle_bound.as_ref().and_then(|model| {
            model.layered_root_certificate(Duration::from_secs_f64(budget))
        }) else {
            return false;
        };
        if self.verified_objective_lattice().is_none()
            || !self.native_feasible_incumbent().is_some_and(|value| certificate.lower <= value)
        {
            return false;
        }
        self.root_cycle_floor = certificate.lower;
        if self.options.verbose >= 2 {
            println!(
                "Root layered certificate: {:.9}, width {}, entries {}, {:.6} s",
                certificate.lower,
                certificate.width,
                certificate.entries,
                start.elapsed().as_secs_f64(),
            );
        }
        // This is the exact unconstrained objective, so cycle packing cannot improve it.
        true
    }

    pub(super) fn try_close_native_cycle_root(
        &mut self,
        remaining: usize,
    ) -> Option<(Array1<usize>, f64)> {
        let objective = self.certified_native_root_objective()?;
        self.best_solution_value = objective;
        self.root_reduction_statistics.remaining = remaining;
        self.apply_logging_action(NodeLoggingAction::Visited);
        self.apply_logging_action(NodeLoggingAction::Processed);
        self.solver_logger.output_header(self);
        if !self.suppress_exit_log {
            self.solver_logger.generate_exit_line(self);
        }
        Some((self.best_solution.clone(), objective))
    }

    pub(super) fn try_reduced_root(
        &mut self,
        fixed: &FixedVarMap,
        root_bound: f64,
    ) -> Option<(Array1<usize>, f64)> {
        let seconds = self.options.max_time - (get_current_time() - self.time_start);
        if seconds <= 0.0 || seconds.is_nan() {
            return None;
        }
        let mut cut_stats = Default::default();
        let mut region_stats = Default::default();
        let mut root_options = self.options.clone();
        root_options.root_region_persistency &=
            self.component_depth == 0 && !self.structural_root_reduced;
        let reduction = reduce_root_with_statistics(
            &self.input_qubo,
            fixed,
            &root_options,
            Duration::from_secs_f64(seconds.min(1.0)),
            &mut cut_stats,
            &mut region_stats,
        );
        self.root_reduction_statistics.cut_dominance = cut_stats;
        self.root_reduction_statistics.region_persistency = region_stats;
        let reduction = reduction?;
        let mut options = self.options.clone();
        options.fixed_variables.clear();
        options.root_low_degree_elimination = false;
        options.root_dominant_edge_contraction = false;
        options.root_cut_dominance = false;
        options.root_region_persistency = false;
        options.branch_strategy = self.branch_strategy;
        options.max_time =
            (self.options.max_time - (get_current_time() - self.time_start)).max(0.0);

        let (solution, value) = if reduction.remaining.is_empty() {
            self.root_reduction_statistics = reduction.statistics;
            self.nodes.clear();
            self.nodes_visited += 1;
            self.nodes_processed += 1;
            let solution = reduction.reconstruct(&Array1::zeros(0));
            let value = self.input_qubo.eval_usize(&solution);
            (solution, value)
        } else {
            let mut inner = BBSolver::new(reduction.qubo.clone(), options);
            inner.inherit_relation_history(self, &reduction.remaining);
            // Do not substitute a default backend for a caller's custom solver,
            // or reuse an ABQP preparation containing the old matrix.
            inner.subproblem_solver = self.subproblem_solver.for_reduced_qubo(&inner.qubo)?;
            inner.options.max_time =
                (self.options.max_time - (get_current_time() - self.time_start)).max(0.0);
            self.root_reduction_statistics = reduction.statistics;
            inner.objective_offset = self.objective_offset + reduction.constant;
            inner.suppress_exit_log = true;
            inner.structural_root_reduced = true;
            inner.root_cycle_floor = (self.root_cycle_floor - reduction.constant).next_down();
            inner.warm_start(reduction.project(&self.best_solution));
            inner.tighten_external_cutoff(
                (self.pruning_upper_bound() - reduction.constant).next_up(),
            );
            if self.options.verbose > 0 {
                println!(
                    "Root reduction: original={} fixed={} eliminated={} degree_three={} degree_three_skipped={} contracted={} weak_contractions={} remaining={} passes={} blocks={} block_variables={} cut_dominance={:?} offset={}",
                    self.input_qubo.num_x(), reduction.fixed.len(),
                    reduction.statistics.eliminated, reduction.statistics.degree_three_eliminated,
                    reduction.statistics.degree_three_skipped, reduction.statistics.contracted,
                    reduction.statistics.weak_contractions, reduction.remaining.len(),
                    reduction.statistics.passes, reduction.statistics.blocks_eliminated,
                    reduction.statistics.block_variables, reduction.statistics.cut_dominance, reduction.constant,
                );
            }
            let (small_solution, _) = inner.solve();
            self.lift_cutoff_proof(&inner, reduction.constant);
            let solution = reduction.reconstruct(&small_solution);
            let value = self.input_qubo.eval_usize(&solution);
            self.nodes_solved += inner.nodes_solved;
            self.nodes_processed += inner.nodes_processed;
            self.nodes_visited += inner.nodes_visited;
            self.probing_counters.merge(inner.node_probing_statistics());
            self.component_counters = std::sync::Arc::clone(&inner.component_counters);
            self.reduction_counters = std::sync::Arc::clone(&inner.reduction_counters);
            self.cutoff_counters = std::sync::Arc::clone(&inner.cutoff_counters);
            self.conditional_bound_counters = std::sync::Arc::clone(&inner.conditional_bound_counters);
            self.history_counters = std::sync::Arc::clone(&inner.history_counters);
            self.work_counters = std::sync::Arc::clone(&inner.work_counters);
            self.conflicts.share_statistics(&inner.conflicts);
            self.relation_history = inner.relation_history.clone();
            self.early_stop = inner.early_stop;
            // Expose unfinished nodes in original coordinates and objective units.
            // Eliminated variables are not unconditional fixed values.
            let reconstruction = std::sync::Arc::new(reduction.clone());
            self.nodes = inner
                .nodes
                .into_iter()
                .map(|node| {
                    let component_state = match node.subproblem_state {
                        Some(crate::branch_node::SubProblemNodeState::Components(handle)) => {
                            handle
                                .lock()
                                .expect("component search poisoned")
                                .lift(std::sync::Arc::clone(&reconstruction));
                            Some(crate::branch_node::SubProblemNodeState::Components(handle))
                        }
                        _ => None,
                    };
                    let mut fixed_variables = reduction.fixed.clone();
                    for (i, value) in node.fixed_variables {
                        fixed_variables.insert(reduction.remaining[i], value);
                    }
                    let mut relaxed = Array1::from_elem(self.input_qubo.num_x(), 0.5);
                    for (i, &original) in reduction.remaining.iter().enumerate() {
                        relaxed[original] = node.solution[i];
                    }
                    for (&i, &value) in &fixed_variables {
                        relaxed[i] = value as f64;
                    }
                    QuboBBNode {
                        lower_bound: self.lattice_lower_bound(
                            root_bound.max((node.lower_bound + reduction.constant).next_down()),
                        ),
                        solution: relaxed,
                        fixed_variables,
                        run_heuristic: node.run_heuristic,
                        subproblem_state: component_state,
                        branch_origin: None,
                        probe_bounds: None,
                        probe_presolved: None,
                    }
                })
                .collect();
            (solution, value)
        };
        // A warm incumbent need not respect presolve's weak choices, but must
        // respect the caller's actual fixings. Never return the infeasible zero
        // initializer when the caller fixed a positive-cost variable to one.
        let old_is_feasible = self
            .options
            .fixed_variables
            .iter()
            .all(|(&i, &v)| self.best_solution[i] == v);
        if !old_is_feasible || value <= self.best_solution_value {
            self.best_solution = solution;
            self.best_solution_value = value;
        }
        self.options.fixed_variables = reduction.fixed;
        if reduction.remaining.is_empty() {
            self.solver_logger.output_header(self);
            self.solver_logger.generate_output_line(self);
        }
        if !self.suppress_exit_log {
            self.solver_logger.generate_exit_line(self);
        }
        Some((self.best_solution.clone(), self.best_solution_value))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::branch_subproblem::{BasicSubProblemResult, SubProblemOptions, SubProblemSelection};
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    fn fixture() -> Qubo {
        let n = 16;
        let mut q = sprs::TriMat::new((n, n));
        let mut c = Array1::zeros(n);
        let mut edge = |i, j, weight| {
            q.add_triplet(i, j, 2.0 * weight);
            c[i] -= weight * 0.5;
            c[j] -= weight * 0.5;
        };
        for i in 0..13 {
            for j in i + 1..13 {
                edge(i, j, 2.0);
            }
        }
        for (i, j, weight) in [
            (0, 13, 2.0),
            (13, 14, 2.0),
            (14, 1, 2.0),
            (2, 15, -4.0),
            (15, 3, 2.0),
        ] {
            edge(i, j, weight);
        }
        Qubo::new_with_c(q.to_csr(), c)
    }

    fn exact(qubo: &Qubo, fixed: &FixedVarMap) -> f64 {
        (0..1 << qubo.num_x())
            .filter(|mask| fixed.iter().all(|(&i, &v)| (mask >> i) & 1 == v))
            .map(|mask| {
                qubo.eval_usize(&Array1::from_iter(
                    (0..qubo.num_x()).map(|i| (mask >> i) & 1),
                ))
            })
            .fold(f64::INFINITY, f64::min)
    }

    #[test]
    fn native_cycle_certificate_closes_root_without_backend() {
        let n = 36;
        let mut q = sprs::TriMat::new((n, n));
        for i in (0..n).step_by(3) {
            q.add_triplet(i, i + 1, 8.0);
            q.add_triplet(i, i + 2, 8.0);
            q.add_triplet(i + 1, i + 2, 8.0);
        }
        let qubo = Qubo::new_with_c(q.to_csr(), Array1::from_elem(n, -4.0));
        let start = Array1::from_iter((0..n).map(|i| usize::from(i % 3 == 0)));
        let mut options = SolverOptions::new();
        options.verbose = 0;
        let mut solver = BBSolver::new(qubo.clone(), options);
        solver.warm_start(start.clone());
        let stats = crate::subproblemsolvers::cycle_bound::PreparedCycleBound::new(&qubo)
            .unwrap()
            .short_cycle_root_bound(Duration::from_secs(1), Duration::ZERO, |_| None)
            .unwrap();
        solver.root_cycle_floor = stats.lower_bound();
        let (solution, value) = solver.try_close_native_cycle_root(n).unwrap();
        assert_eq!(solution, start);
        assert!((value + 48.0).abs() < 1e-9);
        assert_eq!(solver.root_reduction_statistics.eliminated, 0);
        assert_eq!(solver.nodes_processed, 1);
        assert!(solver.nodes.is_empty() && !solver.early_stop);
        assert_eq!(solver.node_probing_statistics().subproblem_calls, 0);
    }

    #[test]
    fn native_certificate_requires_exact_match_and_feasible_incumbent() {
        let qubo = Qubo::new_with_c(sprs::CsMat::zero((2, 2)), Array1::from_vec(vec![-2.0, 2.0]));
        let mut solver = BBSolver::new(qubo, SolverOptions::new());
        solver.warm_start(Array1::from_vec(vec![1, 0]));
        solver.best_solution_value = (-2.0_f64).next_up();
        solver.root_cycle_floor = -3.0;
        assert_eq!(solver.certified_native_root_objective(), Some(-2.0));

        solver.root_cycle_floor = -4.0;
        assert!(solver.certified_native_root_objective().is_none());
        solver.root_cycle_floor = -1.0;
        assert!(solver.certified_native_root_objective().is_none());

        solver.root_cycle_floor = -3.0;
        solver.options.fixed_variables.insert(0, 0);
        assert!(solver.certified_native_root_objective().is_none());
        solver.options.fixed_variables.clear();
        solver.best_solution[0] = 2;
        assert!(solver.certified_native_root_objective().is_none());
        solver.best_solution[0] = 1;
        solver.options.objective_lattice_pruning = false;
        assert!(solver.certified_native_root_objective().is_none());
        solver.options.objective_lattice_pruning = true;
        solver.structural_root_reduced = true;
        assert!(solver.certified_native_root_objective().is_none());
    }

    #[test]
    fn reduced_root_backends_reconstruct_optimum_and_accept_warm_start() {
        let qubo = fixture();
        for backend in [
            SubProblemSelection::HerculesABQP,
            SubProblemSelection::MixingCutSDP,
            SubProblemSelection::RoofDualQPBO,
        ] {
            for enabled in [false, true] {
                let mut options = SolverOptions::new();
                options.verbose = 0;
                options.root_low_degree_elimination = enabled;
                options.root_complement_symmetry = false;
                options.sub_problem_solver = backend;
                options.fixed_variables.insert(4, 1);
                options.threads = 4;
                let expected = exact(&qubo, &options.fixed_variables);
                let mut solver = BBSolver::new(qubo.clone(), options);
                let mut start = Array1::zeros(16);
                start[4] = 1;
                solver.warm_start(start);
                for _ in 0..2 {
                    let (x, value) = solver.solve();
                    assert_eq!(x.len(), 16);
                    assert_eq!(x[4], 1);
                    assert!((value - expected).abs() < 1e-7, "{value} != {expected}");
                    assert!((qubo.eval_usize(&x) - expected).abs() < 1e-7);
                    assert!(solver.nodes.is_empty());
                    assert_eq!(solver.qubo.num_x(), 16);
                    if enabled {
                        assert!(solver.root_reduction_statistics.eliminated > 0);
                    }
                }
            }
        }
    }

    #[test]
    fn degree_three_root_reduction_reconstructs_backend_solutions() {
        let n = 16;
        let mut q = sprs::TriMat::new((n, n));
        let mut c = Array1::zeros(n);
        let mut edge = |i, j| {
            q.add_triplet(i, j, 4.0);
            c[i] -= 1.0;
            c[j] -= 1.0;
        };
        for i in 0..13 {
            for j in i + 1..13 {
                edge(i, j);
            }
        }
        for (leaf, anchor) in [(13, 0), (14, 1), (15, 2)] {
            for neighbor in [anchor, anchor + 3, anchor + 4] {
                edge(leaf, neighbor);
            }
        }
        let qubo = Qubo::new_with_c(q.to_csc(), c);
        let fixed = [(12, 1)].into_iter().collect();
        let expected = exact(&qubo, &fixed);
        for backend in [
            SubProblemSelection::HerculesABQP,
            SubProblemSelection::MixingCutSDP,
        ] {
            for enabled in [false, true] {
                let mut options = SolverOptions::new();
                options.verbose = 0;
                options.root_complement_symmetry = false;
                options.root_dominant_edge_contraction = false;
                options.root_degree_three_elimination = enabled;
                options.fixed_variables = fixed.clone();
                options.sub_problem_solver = backend;
                let mut solver = BBSolver::new(qubo.clone(), options);
                solver.warm_start(Array1::ones(n));
                let (x, value) = solver.solve();
                assert_eq!(x.len(), n);
                assert_eq!(x[12], 1);
                assert!(x.iter().all(|&v| v <= 1));
                assert!((value - expected).abs() < 1e-7);
                assert!((qubo.eval_usize(&x) - expected).abs() < 1e-7);
                assert!(solver.nodes.is_empty());
                assert_eq!(
                    solver.root_reduction_statistics.degree_three_eliminated > 0,
                    enabled
                );
            }
        }
    }

    #[test]
    fn dominant_root_contractions_reprepare_backends_and_preserve_fixed_optimum() {
        let n = 16;
        let mut q = sprs::TriMat::new((n, n));
        let mut c = Array1::zeros(n);
        let mut edge = |i, j, weight| {
            q.add_triplet(i, j, 2.0 * weight);
            c[i] -= 0.5 * weight;
            c[j] -= 0.5 * weight;
        };
        for i in 0..13 {
            for j in i + 1..13 {
                edge(i, j, 2.0);
            }
        }
        for (leaf, anchor) in [(13, 0), (14, 1), (15, 2)] {
            edge(leaf, anchor, 32.0);
            edge(leaf, anchor + 3, 2.0);
            edge(leaf, anchor + 4, -2.0);
        }
        let qubo = Qubo::new_with_c(q.to_csc(), c);
        let fixed = [(12, 1)].into_iter().collect();
        let expected = exact(&qubo, &fixed);
        for backend in [
            SubProblemSelection::HerculesABQP,
            SubProblemSelection::MixingCutSDP,
        ] {
            for enabled in [false, true] {
                let mut options = SolverOptions::new();
                options.verbose = 0;
                options.root_complement_symmetry = false;
                options.root_low_degree_elimination = false;
                options.root_dominant_edge_contraction = enabled;
                options.fixed_variables = fixed.clone();
                options.sub_problem_solver = backend;
                let mut solver = BBSolver::new(qubo.clone(), options);
                solver.warm_start(Array1::ones(n));
                let (x, value) = solver.solve();
                assert_eq!(x.len(), n);
                assert_eq!(x[12], 1);
                assert!(x.iter().all(|&v| v <= 1));
                assert!((value - expected).abs() < 1e-7);
                assert_eq!(qubo.eval_usize(&x), expected);
                assert!(solver.nodes.is_empty());
                assert_eq!(solver.root_reduction_statistics.eliminated, 0);
                assert_eq!(solver.root_reduction_statistics.contracted > 0, enabled);
            }
        }
    }

    #[test]
    fn reduced_root_can_finish_without_a_backend_and_respects_positive_fixed_cost() {
        let mut q = sprs::TriMat::new((14, 14));
        let mut c = Array1::from_elem(14, -2.0);
        for i in 0..13 {
            q.add_triplet(i, (i + 1) % 13, 4.0);
        }
        c[13] = 20.0;
        let qubo = Qubo::new_with_c(q.to_csr(), c);
        let mut options = SolverOptions::new();
        options.verbose = 0;
        options.root_complement_symmetry = false;
        options.fixed_variables.insert(13, 1);
        let mut solver = BBSolver::new(qubo.clone(), options);
        let (x, value) = solver.solve();
        assert_eq!(value, 8.0);
        assert_eq!(qubo.eval_usize(&x), value);
        assert_eq!(x[13], 1);
        assert_eq!(solver.root_reduction_statistics.eliminated, 13);
        assert_eq!(solver.root_reduction_statistics.remaining, 0);
        assert_eq!(solver.node_probing_statistics().subproblem_calls, 0);
        assert!(solver.nodes.is_empty());
    }

    #[test]
    fn whole_cut_root_reduction_reconstructs_a_complete_solve() {
        // Two K6 graphs connected by a heavier bridge. No endpoint dominates,
        // but flipping a whole clique always satisfies the bridge at no cost.
        let mut q = sprs::TriMat::new((12, 12));
        let mut c = Array1::zeros(12);
        let mut edge = |i, j, b| {
            q.add_triplet(i, j, 2.0 * b);
            c[i] -= 0.5 * b;
            c[j] -= 0.5 * b;
        };
        for group in [0..6, 6..12] {
            for i in group.clone() {
                for j in i + 1..group.end {
                    edge(i, j, 2.0);
                }
            }
        }
        edge(0, 6, 3.0);
        let qubo = Qubo::new_with_c(q.to_csc(), c);
        for backend in [
            SubProblemSelection::RoofDualQPBO,
            SubProblemSelection::MixingCutSDP,
        ] {
            let mut options = SolverOptions::new();
            options.verbose = 0;
            options.root_low_degree_elimination = false;
            options.root_dominant_edge_contraction = false;
            options.root_cut_dominance = true;
            options.root_complement_symmetry = false;
            options.sub_problem_solver = backend;
            let mut solver = BBSolver::new(qubo.clone(), options);
            let (x, value) = solver
                .try_reduced_root(&FixedVarMap::default(), -30.0)
                .unwrap();
            assert!(solver.root_reduction_statistics.cut_dominance.contracted > 0);
            assert!(solver.nodes.is_empty());
            assert!((value + 19.5).abs() < 1e-7);
            assert_eq!(qubo.eval_usize(&x), -19.5);
        }
    }

    #[test]
    fn reduced_root_timeout_keeps_original_units_and_unfinished_nodes() {
        struct SlowBound(Arc<AtomicUsize>);
        impl SubProblemSolver for SlowBound {
            fn for_reduced_qubo(&self, _: &Qubo) -> Option<Box<dyn SubProblemSolver + Sync>> {
                Some(Box::new(Self(self.0.clone())))
            }
            fn solve_lower_bound(
                &self,
                solver: &BBSolver,
                node: &QuboBBNode,
                _: Option<SubProblemOptions>,
            ) -> Box<dyn SubProblemResult> {
                self.0.store(solver.qubo.num_x(), Ordering::Relaxed);
                std::thread::sleep(Duration::from_millis(250));
                Box::new(BasicSubProblemResult {
                    lower_bound: -1e6,
                    relaxed_solution: node.solution.clone(),
                })
            }
        }
        let qubo = fixture();
        let mut options = SolverOptions::new();
        options.verbose = 0;
        options.root_complement_symmetry = false;
        options.node_probe_candidates = 0;
        options.max_time = 0.2;
        options.threads = 1;
        let seen_size = Arc::new(AtomicUsize::new(0));
        let mut solver = BBSolver::new(qubo.clone(), options);
        solver.subproblem_solver = Box::new(SlowBound(seen_size.clone()));
        let (x, value) = solver.solve();
        assert!(seen_size.load(Ordering::Relaxed) < 16);
        assert_eq!(x.len(), 16);
        assert_eq!(qubo.eval_usize(&x), value);
        assert!(!solver.nodes.is_empty());
        let expected = exact(&qubo, &FixedVarMap::default());
        for node in &solver.nodes {
            assert_eq!(node.solution.len(), 16);
            assert!(node.lower_bound <= exact(&qubo, &node.fixed_variables) + 1e-7);
        }
        assert!(
            solver
                .nodes
                .iter()
                .map(|node| node.lower_bound)
                .fold(value, f64::min)
                <= expected + 1e-7
        );
    }
}
