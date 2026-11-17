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
    pub relation_branches: usize,
    pub relation_materializations: usize,
    pub checks: usize,
    pub compacted: usize,
    pub eliminated: usize,
    pub contracted: usize,
    pub subgraph_contracted: usize,
    pub additional_fixed: usize,
    pub small_gain_skips: usize,
    pub backend_skips: usize,
    pub blocks_eliminated: usize,
    pub block_variables: usize,
}

#[derive(Default)]
pub(super) struct NodeReductionCounters {
    relation_branches: AtomicUsize,
    relation_materializations: AtomicUsize,
    checks: AtomicUsize,
    compacted: AtomicUsize,
    eliminated: AtomicUsize,
    contracted: AtomicUsize,
    subgraph_contracted: AtomicUsize,
    additional_fixed: AtomicUsize,
    small_gain_skips: AtomicUsize,
    backend_skips: AtomicUsize,
    blocks_eliminated: AtomicUsize,
    block_variables: AtomicUsize,
}

/// A queued relation needs no prepared presolver, roof-dual graph, or solver heap.
pub(super) struct DeferredComponent {
    reduction: Arc<crate::preprocess::low_degree::RootReduction>,
    convex: Qubo,
    backend: Box<dyn crate::branch_subproblem::SubProblemSolver + Sync>,
    options: SolverOptions,
    relation_ids: Vec<usize>,
    structurally_reduced: bool,
    pub(super) candidate: Array1<usize>,
    pub(super) node: QuboBBNode,
}

impl DeferredComponent {
    pub(super) fn materialize(self, parent: &BBSolver) -> Component {
        let started = std::time::Instant::now();
        let mut solver = BBSolver::from_convex_qubo(
            self.reduction.qubo.clone(),
            self.convex,
            self.options,
            self.backend,
        );
        solver.relation_history = Arc::clone(&parent.relation_history);
        solver.relation_ids = self.relation_ids;
        solver.component_counters = Arc::clone(&parent.component_counters);
        solver.reduction_counters = Arc::clone(&parent.reduction_counters);
        solver.cutoff_counters = Arc::clone(&parent.cutoff_counters);
        solver.conditional_bound_counters = Arc::clone(&parent.conditional_bound_counters);
        solver.history_counters = Arc::clone(&parent.history_counters);
        solver.work_counters = Arc::clone(&parent.work_counters);
        solver.conflicts.share_statistics(&parent.conflicts);
        solver.component_depth = parent.component_depth + 1;
        solver.structural_root_reduced = self.structurally_reduced;
        solver.time_start = parent.time_start;
        solver.warm_start(self.candidate);
        solver.nodes.push(self.node);
        if !self.structurally_reduced {
            parent
                .reduction_counters
                .relation_materializations
                .fetch_add(1, Ordering::Relaxed);
            parent.relation_history.lock().unwrap().construction_seconds +=
                started.elapsed().as_secs_f64();
        }
        Component {
            variables: (0..solver.qubo.num_x()).collect(),
            solver,
            initialized: true,
            certified_lower_bound: f64::NEG_INFINITY,
        }
    }
}

impl BBSolver {
    pub(super) fn try_cycle_relation_reduction(
        &self, node: &mut QuboBBNode, candidate: Option<&(Array1<usize>, f64)>,
    ) -> Option<ProcessNodeState> {
        if !self.options.selective_cycle_relations { return None; }
        let cutoff = candidate.map_or(self.pruning_upper_bound(), |(_, v)| self.pruning_upper_bound().min(*v));
        let free = self.qubo.num_x() - node.fixed_variables.len();
        let gap = cutoff - node.lower_bound;
        if !(64..=512).contains(&free) || !cutoff.is_finite() || gap <= 0.0
            || gap > 0.005 * cutoff.abs().max(1.0) { return None; }
        let prepared = self.prepared_cycle_bound.as_ref()?;
        let c = &self.work_counters;
        if c.cycle_relation_eligible.fetch_add(1, Ordering::Relaxed) % 31 != 0 { return None; }
        let seconds = (self.time_start + self.options.max_time - get_current_time()).min(0.025);
        if seconds <= 0.0 { return None; }
        c.cycle_relation_attempts.fetch_add(1, Ordering::Relaxed);
        let started = std::time::Instant::now();
        let report = crate::subproblemsolvers::cycle_bound::relation_probe::screen_limited(
            prepared, &node.fixed_variables, cutoff, node.lower_bound,
            Duration::from_secs_f64(seconds), Duration::from_millis(2), 8, true,
        );
        c.cycle_relation_micros.fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
        let report = report?;
        c.cycle_relation_pairs.fetch_add(report.pairs, Ordering::Relaxed);
        node.lower_bound = node.lower_bound.max(report.union_bound);
        if report.closed || report.baseline_closed || node.lower_bound >= cutoff {
            // A parity contradiction also proves the cutoff even if min(L0,L1)
            // alone is weaker. Keep that certificate in the parent context.
            self.cutoff.record(cutoff);
            c.cycle_relation_prunes.fetch_add(1, Ordering::Relaxed);
            return Some(ProcessNodeState::finished(candidate.cloned(), NodeLoggingAction::Solved));
        }
        let &(i, j, complement, rejected, retained) = report.certified.first()?;
        self.apply_certified_cycle_relation(node, candidate, i, j, complement, rejected, retained)
    }

    fn apply_certified_cycle_relation(
        &self, node: &QuboBBNode, candidate: Option<&(Array1<usize>, f64)>,
        i: usize, j: usize, complement: bool, rejected: f64, retained: f64,
    ) -> Option<ProcessNodeState> {
        let reduction = crate::preprocess::low_degree::restrict_relation(
            &self.input_qubo, &node.fixed_variables, i, j, complement)?;
        let convex = self.relation_convex_qubo(&reduction, i, j, complement);
        let mut options = self.options.clone();
        // Do not recursively repeat this experimental pass in its restricted child.
        options.selective_cycle_relations = false;
        let mut child = node.clone();
        child.lower_bound = child.lower_bound.max(retained);
        let state = self.wrap_reduction(&child, candidate, reduction, options, false, Some(convex))?;
        // The removed parity is not infeasible. Its proof must remain available
        // when a component subsearch reports a lower bound or exhausts its queue.
        self.cutoff.record(rejected);
        self.work_counters.cycle_relation_reductions.fetch_add(1, Ordering::Relaxed);
        Some(state)
    }

    pub fn node_reduction_statistics(&self) -> NodeReductionStatistics {
        let c = &self.reduction_counters;
        NodeReductionStatistics {
            relation_branches: c.relation_branches.load(Ordering::Relaxed),
            relation_materializations: c.relation_materializations.load(Ordering::Relaxed),
            checks: c.checks.load(Ordering::Relaxed),
            compacted: c.compacted.load(Ordering::Relaxed),
            eliminated: c.eliminated.load(Ordering::Relaxed),
            contracted: c.contracted.load(Ordering::Relaxed),
            subgraph_contracted: c.subgraph_contracted.load(Ordering::Relaxed),
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
        options.root_region_persistency = false;
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
        self.wrap_reduction(node, candidate, reduction, options, true, None)
    }

    fn wrap_reduction(
        &self,
        node: &QuboBBNode,
        candidate: Option<&(Array1<usize>, f64)>,
        reduction: crate::preprocess::low_degree::RootReduction,
        mut options: SolverOptions,
        structurally_reduced: bool,
        inherited_convex: Option<Qubo>,
    ) -> Option<ProcessNodeState> {
        let counters = &self.reduction_counters;
        let n = reduction.remaining.len();
        let stats = reduction.statistics;
        counters
            .subgraph_contracted
            .fetch_add(stats.subgraph_contracted, Ordering::Relaxed);
        let projected = reduction.project(&self.best_solution);
        let reduction = Arc::new(reduction);
        let mut components = Vec::new();
        let mut deferred = None;
        let small_solution = if n == 0 {
            Array1::zeros(0)
        } else {
            options.fixed_variables.clear();
            options.verbose = 0;
            options.root_low_degree_elimination = false;
            options.root_dominant_edge_contraction = false;
            options.branch_strategy = self.branch_strategy;
            let convex = inherited_convex.unwrap_or_else(|| reduction.qubo.convex_symmetric_form());
            let Some(backend) = self.subproblem_solver.for_reduced_qubo(&convex) else {
                counters.backend_skips.fetch_add(1, Ordering::Relaxed);
                return None;
            };
            let initial_node = QuboBBNode {
                lower_bound: (node.lower_bound - reduction.constant).next_down(),
                fixed_variables: FixedVarMap::default(),
                solution: Array1::from_iter(reduction.remaining.iter().map(|&i| node.solution[i])),
                run_heuristic: node.run_heuristic,
                subproblem_state: None,
                branch_origin: None,
                probe_bounds: None,
                probe_presolved: None,
            };
            let solution = if convex.eval_usize(&projected) < 0.0 {
                projected
            } else {
                Array1::zeros(n)
            };
            let prepared = DeferredComponent {
                reduction: Arc::clone(&reduction),
                convex,
                backend,
                options,
                relation_ids: reduction
                    .remaining
                    .iter()
                    .map(|&i| self.relation_ids[i])
                    .collect(),
                structurally_reduced,
                candidate: solution.clone(),
                node: initial_node,
            };
            if !structurally_reduced {
                deferred = Some(prepared);
            } else {
                components.push(prepared.materialize(self));
            }
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
                deferred,
                relation_origin: None,
                components,
                fixed: FixedVarMap::default(),
                size: n,
                constant: 0.0,
                next_component: 0,
                postsolve: vec![reduction],
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

    pub(super) fn try_relation_branches(
        &self,
        node: &QuboBBNode,
        candidate: Option<&(Array1<usize>, f64)>,
    ) -> Option<ProcessNodeState> {
        if !self.options.relation_branching || self.options.relation_branching_budget == 0 {
            return None;
        }
        let Some(SubProblemNodeState::RelationCandidate(i, j, rho)) =
            node.subproblem_state.as_ref()
        else {
            return None;
        };
        let mut children = Vec::with_capacity(2);
        let mut best = candidate.cloned();
        let started = std::time::Instant::now();
        for complement in [false, true] {
            let reduction = crate::preprocess::low_degree::restrict_relation(
                &self.input_qubo,
                &node.fixed_variables,
                *i,
                *j,
                complement,
            )?;
            let mut options = self.options.clone();
            if options.relation_branching_budget != usize::MAX {
                options.relation_branching_budget -= 1;
            }
            options.root_cut_dominance = false;
            options.root_region_persistency = false;
            // A parity substitution can create low-degree vertices. Unlike a
            // completed presolve reduction, it must not skip structural presolve.
            let convex = self.relation_convex_qubo(&reduction, *i, *j, complement);
            let state =
                self.wrap_reduction(node, candidate, reduction, options, false, Some(convex))?;
            best = Self::better_candidate(best, state.best_update);
            let child = state.continuation?;
            if self.options.relation_pseudocosts {
                if let Some(SubProblemNodeState::Components(handle)) = &child.subproblem_state {
                    handle.lock().unwrap().relation_origin =
                        Some(self.relation_origin(*i, *j, *rho, complement, node.lower_bound));
                }
            }
            children.push(child);
        }
        self.relation_history.lock().unwrap().construction_seconds +=
            started.elapsed().as_secs_f64();
        let one = children.pop().unwrap();
        let zero = children.pop().unwrap();
        self.reduction_counters
            .relation_branches
            .fetch_add(1, Ordering::Relaxed);
        Some(ProcessNodeState {
            best_update: best,
            branches: Some((zero, one)),
            continuation: None,
            logging: NodeLoggingAction::Solved,
            component_work: (0, 0, 0),
        })
    }

    fn relation_convex_qubo(
        &self,
        reduction: &crate::preprocess::low_degree::RootReduction,
        i: usize,
        j: usize,
        complement: bool,
    ) -> Qubo {
        // The native restriction folds all diagonals into c. Restore the diagonal
        // of T'QT for x_i=x_j or x_i=1-x_j; its off-diagonals already match.
        // This is an affine restriction of the parent's PSD Hessian, so no new
        // convexification is needed. The binary objective and constant stay put.
        let q = &self.qubo.q;
        let sign = if complement { -1.0 } else { 1.0 };
        let diagonal = Array1::from_iter(reduction.remaining.iter().map(|&k| {
            let value = q.get(k, k).copied().unwrap_or(0.0);
            if k == j {
                value
                    + q.get(i, i).copied().unwrap_or(0.0)
                    + sign
                        * (q.get(i, j).copied().unwrap_or(0.0)
                            + q.get(j, i).copied().unwrap_or(0.0))
            } else {
                value
            }
        }));
        reduction.qubo.make_diag_array_transform(&diagonal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::branch_subproblem::SubProblemSelection;

    #[test]
    fn certified_relation_keeps_discarded_bound_and_reconstructs() {
        let q = Qubo::new_with_c(sprs::CsMat::zero((4,4)), ndarray::array![1.0,2.0,-3.0,4.0]);
        let mut options = SolverOptions::new();
        options.verbose = 0;
        options.selective_cycle_relations = true;
        let mut parent = BBSolver::new(q.clone(), options);
        parent.best_solution_value = 0.0;
        parent.best_solution = Array1::zeros(4);
        parent.tighten_external_cutoff(-2.5);
        // Equality has minimum -3; complement has minimum -2 and cannot beat -2.5.
        let node = QuboBBNode { lower_bound: -4.0, solution: Array1::from_elem(4,0.5),
            fixed_variables: FixedVarMap::default(), run_heuristic: false,
            subproblem_state: None, branch_origin: None, probe_bounds: None, probe_presolved: None };
        let state = parent.apply_certified_cycle_relation(&node,None,0,1,false,-2.0,-3.0).unwrap();
        assert_eq!(parent.cutoff.discarded_bound(), -2.0);
        assert!(!parent.search_is_exact());
        let continuation = state.continuation.unwrap();
        let Some(SubProblemNodeState::Components(handle)) = continuation.subproblem_state else { panic!("missing restricted search"); };
        let search = handle.lock().unwrap();
        assert!(!search.deferred.as_ref().unwrap().options.selective_cycle_relations);
        let reduction = &search.postsolve[0];
        let mut best = f64::INFINITY;
        for mask in 0..8 {
            let x = Array1::from_iter((0..3).map(|i| (mask>>i)&1));
            let full = reduction.reconstruct(&x);
            assert_eq!(full[0],full[1]);
            assert_eq!(q.eval_usize(&full), reduction.qubo.eval_usize(&x)+reduction.constant);
            best = best.min(q.eval_usize(&full));
        }
        assert_eq!(best,-3.0);
    }

    #[test]
    fn relation_restrictions_partition_and_reconstruct_every_assignment() {
        use crate::preprocess::low_degree::restrict_relation;
        let mut rng = crate::tests::make_test_prng();
        for _ in 0..12 {
            let mut qubo = Qubo::make_random_qubo(6, &mut rng, 0.6);
            for v in qubo.q.data_mut() {
                *v = 2.0 * v.round();
            }
            qubo.c.mapv_inplace(f64::round);
            let mut options = SolverOptions::new();
            options.verbose = 0;
            let parent = BBSolver::new(qubo.clone(), options);
            for fixed in [
                FixedVarMap::default(),
                [(2, 0), (5, 1)].into_iter().collect(),
            ] {
                let mut seen = std::collections::BTreeSet::new();
                for complement in [false, true] {
                    let r = restrict_relation(&qubo, &fixed, 0, 3, complement).unwrap();
                    let convex = parent.relation_convex_qubo(&r, 0, 3, complement);
                    assert!(convex.hess_eigenvalues().iter().all(|&v| v >= -1e-8));
                    assert_eq!(r.remaining.len(), 6 - fixed.len() - 1);
                    for mask in 0..1usize << r.remaining.len() {
                        let x = Array1::from_iter((0..r.remaining.len()).map(|i| (mask >> i) & 1));
                        let full = r.reconstruct(&x);
                        assert_eq!(full[0] ^ full[3], usize::from(complement));
                        assert!(fixed.iter().all(|(&i, &v)| full[i] == v));
                        assert!(
                            (qubo.eval_usize(&full) - r.qubo.eval_usize(&x) - r.constant).abs()
                                < 1e-8
                        );
                        assert!(seen.insert(full.to_vec()));
                        assert!((convex.eval_usize(&x) - r.qubo.eval_usize(&x)).abs() < 1e-8);
                        // Also check the full fractional affine restriction, not
                        // just binary equivalence (which alone would not imply PSD).
                        let y = x.mapv(|v| 0.2 + 0.6 * v as f64);
                        let mut lifted = full.mapv(|v| v as f64);
                        for (&k, &v) in r.remaining.iter().zip(&y) {
                            lifted[k] = v;
                        }
                        lifted[0] = if complement {
                            1.0 - lifted[3]
                        } else {
                            lifted[3]
                        };
                        assert!(
                            (parent.qubo.eval(&lifted) - convex.eval(&y) - r.constant).abs() < 1e-8
                        );
                    }
                }
                assert_eq!(seen.len(), 1usize << (6 - fixed.len()));
            }
        }
    }

    #[test]
    fn relation_branching_closes_odd_clique_with_a_valid_proof() {
        let n = 9;
        let mut matrix = sprs::TriMat::new((n, n));
        for i in 0..n {
            for j in i + 1..n {
                matrix.add_triplet(i, j, 4.0);
            }
        }
        let qubo = Qubo::new_with_c(matrix.to_csr(), Array1::from_elem(n, -8.0));
        let mut options = SolverOptions::new();
        options.verbose = 0;
        options.relation_branching = true;
        options.root_roof_probe_candidates = 0;
        options.node_probe_candidates = 0;
        options.root_low_degree_elimination = false;
        options.root_dominant_edge_contraction = false;
        options.root_complement_symmetry = false;
        options.node_structural_reductions = false;
        options.component_decomposition = false;
        options.bounded_width_elimination = false;
        options.width_region_elimination = false;
        let mut solver = BBSolver::new(qubo.clone(), options);
        let (x, value) = solver.solve();
        assert_eq!(qubo.eval_usize(&x), -20.0);
        assert!((value + 20.0).abs() < 1e-8);
        assert!(solver.nodes.is_empty());
        assert!(!solver.early_stop);
        // An ordinary bound may certify this instance before branching; the
        // independent-context test below explicitly exercises both parities.
    }

    #[test]
    fn relation_children_solve_both_parities_in_independent_contexts() {
        let mut rng = crate::tests::make_test_prng();
        for _ in 0..8 {
            let mut qubo = Qubo::make_random_qubo(7, &mut rng, 0.5);
            for v in qubo.q.data_mut() {
                *v = 2.0 * v.round();
            }
            qubo.c.mapv_inplace(f64::round);
            let mut options = SolverOptions::new();
            options.verbose = 0;
            options.relation_branching = true;
            options.relation_pseudocosts = true;
            options.relation_branching_budget = 1;
            let parent = BBSolver::new(qubo.clone(), options);
            let node = QuboBBNode {
                lower_bound: f64::NEG_INFINITY,
                solution: Array1::from_elem(7, 0.5),
                fixed_variables: [(6, 1)].into_iter().collect(),
                run_heuristic: true,
                subproblem_state: Some(SubProblemNodeState::RelationCandidate(0, 3, 0.0)),
                branch_origin: None,
                probe_bounds: None,
                probe_presolved: None,
            };
            let state = parent.try_relation_branches(&node, None).unwrap();
            let (zero, one) = state.branches.unwrap();
            let handles: Vec<_> = [zero, one]
                .into_iter()
                .map(|n| {
                    let Some(SubProblemNodeState::Components(h)) = n.subproblem_state else {
                        panic!("missing child");
                    };
                    h
                })
                .collect();
            assert!(!Arc::ptr_eq(&handles[0], &handles[1]));
            for (parity, handle) in handles.iter().enumerate() {
                let mut search = handle.lock().unwrap();
                assert!(search.components.is_empty());
                assert!(search.deferred.is_some());
                search.materialize(&parent);
                search.materialize(&parent);
                assert_eq!(search.components.len(), 1);
                assert!(!search.components[0].solver.structural_root_reduced);
                assert_eq!(
                    search.components[0]
                        .solver
                        .options
                        .relation_branching_budget,
                    0
                );
                assert_eq!(
                    std::mem::discriminant(&search.components[0].solver.branch_strategy),
                    std::mem::discriminant(&parent.branch_strategy)
                );
                assert!(search.components[0]
                    .solver
                    .try_relation_branches(&node, None)
                    .is_none());
                let r = Arc::clone(&search.postsolve[0]);
                let (x, _) = search.components[0].solver.solve();
                let full = r.reconstruct(&x);
                assert_eq!(full[0] ^ full[3], parity);
                assert_eq!(full[6], 1);
                let best = (0..128usize)
                    .filter(|mask| (mask >> 6) & 1 == 1 && ((mask ^ (mask >> 3)) & 1) == parity)
                    .map(|mask| {
                        qubo.eval_usize(&Array1::from_iter((0..7).map(|i| (mask >> i) & 1)))
                    })
                    .fold(f64::INFINITY, f64::min);
                assert!((qubo.eval_usize(&full) - best).abs() < 1e-8);
            }
        }
    }

    #[test]
    fn queued_relation_can_pause_and_prune_without_materialization() {
        let (mut parent, mut node) = setup();
        parent.options.relation_branching = true;
        node.subproblem_state = Some(SubProblemNodeState::RelationCandidate(0, 1, 0.0));
        let state = parent.try_relation_branches(&node, None).unwrap();
        let (child, _) = state.branches.unwrap();
        let Some(SubProblemNodeState::Components(handle)) = &child.subproblem_state else {
            panic!("missing relation search");
        };
        let handle = Arc::clone(handle);
        assert_eq!(
            parent.node_reduction_statistics().relation_materializations,
            0
        );
        parent.options.max_time = -1.0;
        let paused = parent.process_node(&child);
        assert_eq!(paused.component_work, (0, 0, 0));
        let mut child = paused.continuation.unwrap();
        assert_eq!(child.lower_bound, node.lower_bound);
        let (x, value) = paused.best_update.unwrap();
        assert!((parent.input_qubo.eval_usize(&x) - value).abs() < 1e-8);
        child.lower_bound = parent.best_solution_value;
        assert!(parent.process_node(&child).continuation.is_none());
        let search = handle.lock().unwrap();
        assert!(search.deferred.is_some());
        assert!(search.components.is_empty());
        assert_eq!(
            parent.node_reduction_statistics().relation_materializations,
            0
        );
    }

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
        options.sub_problem_solver = SubProblemSelection::MixingCutSDP;
        let solver = BBSolver::new(fixture(), options);
        let node = QuboBBNode {
            lower_bound: -50.0,
            fixed_variables: FixedVarMap::default(),
            solution: Array1::from_elem(17, 0.5),
            run_heuristic: true,
            subproblem_state: None,
            branch_origin: None,
            probe_bounds: None,
            probe_presolved: None,
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
    fn region_postsolve_composes_with_nested_node_searches_for_all_backends() {
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
                options.width_region_elimination = true;
                let mut solver = BBSolver::new(qubo.clone(), options);
                let node = QuboBBNode {
                    lower_bound: f64::NEG_INFINITY,
                    fixed_variables: FixedVarMap::default(),
                    solution: Array1::from_elem(18, 0.5),
                    run_heuristic: true,
                    subproblem_state: None,
                    branch_origin: None,
                    probe_bounds: None,
                    probe_presolved: None,
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
