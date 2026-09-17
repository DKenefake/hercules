use super::*;
use crate::preprocess::low_degree::reduce_root_with_statistics;
use std::time::Duration;

impl BBSolver {
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
        let reduction = reduce_root_with_statistics(
            &self.input_qubo,
            fixed,
            &self.options,
            Duration::from_secs_f64(seconds.min(1.0)),
            &mut cut_stats,
        );
        self.root_reduction_statistics.cut_dominance = cut_stats;
        let reduction = reduction?;
        let mut options = self.options.clone();
        options.fixed_variables.clear();
        options.root_low_degree_elimination = false;
        options.root_dominant_edge_contraction = false;
        options.root_cut_dominance = false;
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
            // Do not substitute a default backend for a caller's custom solver,
            // or reuse an ABQP preparation containing the old matrix.
            inner.subproblem_solver = self.subproblem_solver.for_reduced_qubo(&inner.qubo)?;
            inner.options.max_time =
                (self.options.max_time - (get_current_time() - self.time_start)).max(0.0);
            self.root_reduction_statistics = reduction.statistics;
            inner.objective_offset = self.objective_offset + reduction.constant;
            inner.suppress_exit_log = true;
            inner.structural_root_reduced = true;
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
            self.sdp_fixing_counters = std::sync::Arc::clone(&inner.sdp_fixing_counters);
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
                        lower_bound: root_bound
                            .max((node.lower_bound + reduction.constant).next_down()),
                        solution: relaxed,
                        fixed_variables,
                        run_heuristic: node.run_heuristic,
                        subproblem_state: component_state,
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
            assert!(!solver.options.small_block_elimination);
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
