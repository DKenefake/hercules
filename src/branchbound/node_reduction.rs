//! Structural node compaction uses the same bounded frontier as AND nodes,
//! with a single residual problem and a reverse reconstruction map.
use super::*;
use crate::branch_node::SubProblemNodeState;
use crate::preprocess::low_degree::reduce_root;
use components::{Component, ComponentSearch};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

#[derive(Clone, Copy, Debug, Default)]
pub struct NodeReductionStatistics {
    pub checks: usize,
    pub compacted: usize,
    pub eliminated: usize,
    pub contracted: usize,
    pub additional_fixed: usize,
    pub small_gain_skips: usize,
    pub backend_skips: usize,
    pub blocks_eliminated: usize,
    pub block_variables: usize,
}

#[derive(Default)]
pub(super) struct NodeReductionCounters {
    checks: AtomicUsize,
    compacted: AtomicUsize,
    eliminated: AtomicUsize,
    contracted: AtomicUsize,
    additional_fixed: AtomicUsize,
    small_gain_skips: AtomicUsize,
    backend_skips: AtomicUsize,
    blocks_eliminated: AtomicUsize,
    block_variables: AtomicUsize,
}

impl BBSolver {
    pub fn node_reduction_statistics(&self) -> NodeReductionStatistics {
        let c = &self.reduction_counters;
        NodeReductionStatistics {
            checks: c.checks.load(Ordering::Relaxed),
            compacted: c.compacted.load(Ordering::Relaxed),
            eliminated: c.eliminated.load(Ordering::Relaxed),
            contracted: c.contracted.load(Ordering::Relaxed),
            additional_fixed: c.additional_fixed.load(Ordering::Relaxed),
            small_gain_skips: c.small_gain_skips.load(Ordering::Relaxed),
            backend_skips: c.backend_skips.load(Ordering::Relaxed),
            blocks_eliminated: c.blocks_eliminated.load(Ordering::Relaxed),
            block_variables: c.block_variables.load(Ordering::Relaxed),
        }
    }

    pub(super) fn try_reduced_node(
        &self,
        node: &QuboBBNode,
        candidate: Option<&(Array1<usize>, f64)>,
    ) -> Option<ProcessNodeState> {
        let seconds = self.time_start + self.options.max_time - get_current_time();
        if !self.options.node_structural_reductions
            || seconds <= 0.0
            || seconds.is_nan()
            || (self.structural_root_reduced && node.fixed_variables.is_empty())
        {
            return None;
        }
        let counters = &self.reduction_counters;
        counters.checks.fetch_add(1, Ordering::Relaxed);
        let mut options = self.options.clone();
        // These root flags are disabled in nested contexts to prevent the old
        // root solve-to-completion path. Node compaction has its own switch.
        options.root_low_degree_elimination = true;
        options.root_dominant_edge_contraction = true;
        options.root_cut_dominance = false;
        let reduction = reduce_root(
            &self.input_qubo,
            &node.fixed_variables,
            &options,
            Duration::from_secs_f64(seconds.min(0.005)),
        )?;
        let free = self.input_qubo.num_x() - node.fixed_variables.len();
        let n = reduction.remaining.len();
        if n != 0 && free - n < (free / 8).max(4) {
            counters.small_gain_skips.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        let stats = reduction.statistics;
        let projected = reduction.project(&self.best_solution);
        let mut components = Vec::new();
        let small_solution = if n == 0 {
            Array1::zeros(0)
        } else {
            options.fixed_variables.clear();
            options.verbose = 0;
            options.root_low_degree_elimination = false;
            options.root_dominant_edge_contraction = false;
            options.branch_strategy = self.branch_strategy;
            let mut solver = BBSolver::new(reduction.qubo.clone(), options);
            let Some(backend) = self.subproblem_solver.for_reduced_qubo(&solver.qubo) else {
                counters.backend_skips.fetch_add(1, Ordering::Relaxed);
                return None;
            };
            solver.subproblem_solver = backend;
            solver.component_counters = Arc::clone(&self.component_counters);
            solver.reduction_counters = Arc::clone(counters);
            solver.cutoff_counters = Arc::clone(&self.cutoff_counters);
            solver.sdp_fixing_counters = Arc::clone(&self.sdp_fixing_counters);
            solver.component_depth = self.component_depth + 1;
            solver.structural_root_reduced = true;
            solver.time_start = self.time_start;
            solver.warm_start(projected);
            solver.nodes.push(QuboBBNode {
                lower_bound: (node.lower_bound - reduction.constant).next_down(),
                fixed_variables: FixedVarMap::default(),
                solution: Array1::from_iter(reduction.remaining.iter().map(|&i| node.solution[i])),
                run_heuristic: node.run_heuristic,
                subproblem_state: None,
            });
            let solution = solver.best_solution.clone();
            components.push(Component {
                variables: (0..n).collect(),
                solver,
                // Ordinary node presolve runs on the first slice. Avoid a new
                // root probing/relaxation phase for every compacted child.
                initialized: true,
                certified_lower_bound: f64::NEG_INFINITY,
            });
            solution
        };
        counters.compacted.fetch_add(1, Ordering::Relaxed);
        counters
            .eliminated
            .fetch_add(stats.eliminated, Ordering::Relaxed);
        counters
            .contracted
            .fetch_add(stats.contracted, Ordering::Relaxed);
        counters
            .additional_fixed
            .fetch_add(stats.additional_fixed, Ordering::Relaxed);
        counters
            .blocks_eliminated
            .fetch_add(stats.blocks_eliminated, Ordering::Relaxed);
        counters
            .block_variables
            .fetch_add(stats.block_variables, Ordering::Relaxed);
        let full = reduction.reconstruct(&small_solution);
        let value = self.qubo.eval_usize(&full);
        let best_update = Self::better_candidate(candidate.cloned(), Some((full, value)));
        let continuation = if n == 0 {
            None
        } else {
            let search = ComponentSearch {
                components,
                fixed: FixedVarMap::default(),
                size: n,
                constant: 0.0,
                next_component: 0,
                postsolve: vec![Arc::new(reduction)],
                structural: true,
                cutoff_certificate: f64::NEG_INFINITY,
            };
            let mut node = node.clone();
            node.subproblem_state = Some(SubProblemNodeState::Components(Arc::new(Mutex::new(
                search,
            ))));
            Some(node)
        };
        Some(ProcessNodeState {
            best_update,
            branches: None,
            logging: if n == 0 {
                NodeLoggingAction::Solved
            } else {
                NodeLoggingAction::Processed
            },
            continuation,
            component_work: (0, 0, 0),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::branch_subproblem::SubProblemSelection;

    fn fixture() -> Qubo {
        // K13 has maximum cut 42. Four pendant edges contribute four more.
        let mut q = sprs::TriMat::new((17, 17));
        let mut c = Array1::zeros(17);
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
        for i in 13..17 {
            edge(0, i);
        }
        Qubo::new_with_c(q.to_csc(), c)
    }

    fn setup() -> (BBSolver, QuboBBNode) {
        let mut options = SolverOptions::new();
        options.verbose = 0;
        options.threads = 1;
        options.node_probe_candidates = 0;
        options.small_block_elimination = false;
        options.sub_problem_solver = SubProblemSelection::MixingCutSDP;
        let solver = BBSolver::new(fixture(), options);
        let node = QuboBBNode {
            lower_bound: -50.0,
            fixed_variables: FixedVarMap::default(),
            solution: Array1::from_elem(17, 0.5),
            run_heuristic: true,
            subproblem_state: None,
        };
        (solver, node)
    }

    #[test]
    fn node_reduction_is_resumable_and_preserves_bound_and_postsolve() {
        let (mut solver, node) = setup();
        let state = solver.try_reduced_node(&node, None).unwrap();
        assert!(state.continuation.is_some());
        assert_eq!(state.component_work, (0, 0, 0));
        assert_eq!(solver.node_reduction_statistics().compacted, 1);
        solver.options.max_time = -1.0;
        let paused = solver.advance_component_node(state.continuation.unwrap(), state.best_update);
        assert_eq!(paused.component_work, (0, 0, 0));
        assert!(paused.continuation.is_some());
        solver.apply_process_result(paused);
        solver.options.max_time = 100.0;
        for _ in 0..1000 {
            if solver.nodes.is_empty() {
                break;
            }
            assert!(solver.nodes.iter().all(|n| n.lower_bound <= -46.0 + 1e-7));
            solver.search_batch(1);
        }
        assert!(solver.nodes.is_empty());
        assert!((solver.best_solution_value + 46.0).abs() < 1e-7);
        assert_eq!(solver.input_qubo.eval_usize(&solver.best_solution), -46.0);
    }

    #[test]
    fn node_reduction_respects_switch_deadline_and_complete_elimination() {
        let (mut solver, node) = setup();
        solver.options.node_structural_reductions = false;
        assert!(solver.try_reduced_node(&node, None).is_none());
        solver.options.node_structural_reductions = true;
        solver.options.max_time = -1.0;
        assert!(solver.try_reduced_node(&node, None).is_none());
        solver.options.max_time = 100.0;
        let mut node = node;
        node.fixed_variables
            .extend((0..13).map(|i| (i, usize::from(i < 6))));
        let state = solver.try_reduced_node(&node, None).unwrap();
        assert!(state.continuation.is_none());
        let (x, value) = state.best_update.unwrap();
        assert!((value + 46.0).abs() < 1e-7);
        assert!(node.fixed_variables.iter().all(|(&i, &v)| x[i] == v));
    }

    #[test]
    fn block_postsolve_composes_with_nested_node_searches_for_all_backends() {
        for boundary in 1..=2 {
            // A K(5+boundary) attached to K13 at the boundary vertices. Every
            // vertex has degree >= 5, so ordinary low-degree rules cannot start.
            let mut q = sprs::TriMat::new((18, 18));
            let mut c = Array1::zeros(18);
            for i in 0..18 {
                for j in i + 1..18 {
                    if i >= 5 || j < 5 + boundary {
                        q.add_triplet(i, j, 4.0);
                        c[i] -= 1.0;
                        c[j] -= 1.0;
                    }
                }
            }
            let qubo = Qubo::new_with_c(q.to_csc(), c);
            let expected = if boundary == 1 { -51.0 } else { -54.0 };
            for backend in [
                SubProblemSelection::RoofDualQPBO,
                SubProblemSelection::HerculesABQP,
                SubProblemSelection::MixingCutSDP,
            ] {
                let mut options = SolverOptions::new();
                options.verbose = 0;
                options.threads = 1;
                options.node_probe_candidates = 0;
                options.sub_problem_solver = backend;
                options.small_block_elimination = true;
                let mut solver = BBSolver::new(qubo.clone(), options);
                let node = QuboBBNode {
                    lower_bound: f64::NEG_INFINITY,
                    fixed_variables: FixedVarMap::default(),
                    solution: Array1::from_elem(18, 0.5),
                    run_heuristic: true,
                    subproblem_state: None,
                };
                let state = solver.try_reduced_node(&node, None).unwrap();
                assert!(solver.node_reduction_statistics().blocks_eliminated > 0);
                solver.apply_process_result(state);
                for _ in 0..1000 {
                    if solver.nodes.is_empty() {
                        break;
                    }
                    let bound = solver
                        .nodes
                        .iter()
                        .map(|n| n.lower_bound)
                        .fold(solver.best_solution_value, f64::min);
                    assert!(bound <= expected + 1e-7);
                    solver.search_batch(1);
                }
                assert!(solver.nodes.is_empty());
                assert!((solver.best_solution_value - expected).abs() < 1e-7);
                assert_eq!(qubo.eval_usize(&solver.best_solution), expected);
            }
        }
    }
}
