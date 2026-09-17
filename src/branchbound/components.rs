//! Dynamic AND nodes. Component objectives add; each component's OR frontier
//! takes a minimum. Never enqueue component-local bounds in the parent heap.
use super::*;
use crate::branch_node::SubProblemNodeState;
use crate::preprocess::low_degree::RootReduction;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

#[derive(Clone, Copy, Debug, Default)]
pub struct ComponentStatistics {
    pub checks: usize,
    pub splits: usize,
    pub components_created: usize,
    pub components_solved: usize,
    pub batches: usize,
    pub cross_relation_skips: usize,
    pub backend_skips: usize,
    pub projection_skips: usize,
    pub max_depth: usize,
}

#[derive(Default)]
pub(super) struct ComponentCounters {
    checks: AtomicUsize,
    splits: AtomicUsize,
    components_created: AtomicUsize,
    components_solved: AtomicUsize,
    batches: AtomicUsize,
    cross_relation_skips: AtomicUsize,
    backend_skips: AtomicUsize,
    projection_skips: AtomicUsize,
    max_depth: AtomicUsize,
}

impl ComponentCounters {
    fn snapshot(&self) -> ComponentStatistics {
        ComponentStatistics {
            checks: self.checks.load(Ordering::Relaxed),
            splits: self.splits.load(Ordering::Relaxed),
            components_created: self.components_created.load(Ordering::Relaxed),
            components_solved: self.components_solved.load(Ordering::Relaxed),
            batches: self.batches.load(Ordering::Relaxed),
            cross_relation_skips: self.cross_relation_skips.load(Ordering::Relaxed),
            backend_skips: self.backend_skips.load(Ordering::Relaxed),
            projection_skips: self.projection_skips.load(Ordering::Relaxed),
            max_depth: self.max_depth.load(Ordering::Relaxed),
        }
    }
}

pub(super) struct Component {
    pub(super) variables: Vec<usize>,
    pub(super) solver: BBSolver,
    pub(super) initialized: bool,
    pub(super) certified_lower_bound: f64,
}

impl Component {
    fn solved(&self) -> bool {
        self.finished() && self.lower_bound() >= self.solver.best_solution_value
    }

    fn finished(&self) -> bool {
        self.initialized && self.solver.nodes.is_empty()
    }

    fn lower_bound(&self) -> f64 {
        if !self.initialized {
            return f64::NEG_INFINITY;
        }
        self.solver
            .search_lower_bound()
            .max(self.certified_lower_bound)
            .min(self.solver.best_solution_value)
    }
}

/// Persistent component frontiers. Cloned AND-node handles share this state;
/// they must not be used as ordinary binary branches with different fixings.
pub struct ComponentSearch {
    pub(super) components: Vec<Component>,
    pub(super) fixed: FixedVarMap,
    pub(super) size: usize,
    pub(super) constant: f64,
    pub(super) next_component: usize,
    pub(super) postsolve: Vec<Arc<RootReduction>>,
    pub(super) structural: bool,
    pub(super) cutoff_certificate: f64,
}

impl ComponentSearch {
    pub(super) fn lift(&mut self, reduction: Arc<RootReduction>) {
        self.cutoff_certificate = (self.cutoff_certificate + reduction.constant).next_down();
        self.postsolve.push(reduction);
    }

    fn candidate(&self) -> Array1<usize> {
        let mut x = Array1::zeros(self.size);
        for (&i, &v) in &self.fixed {
            x[i] = v;
        }
        for component in &self.components {
            for (&i, &v) in component
                .variables
                .iter()
                .zip(&component.solver.best_solution)
            {
                x[i] = v;
            }
        }
        for reduction in &self.postsolve {
            x = reduction.reconstruct(&x);
        }
        x
    }

    fn lower_bound(&self) -> f64 {
        let mut bound = self.constant;
        for component in &self.components {
            let lower = component.lower_bound();
            if lower == f64::NEG_INFINITY {
                return lower;
            }
            // Round sums outwards rather than accidentally strengthen a certificate.
            bound = (bound + lower).next_down();
        }
        for reduction in &self.postsolve {
            bound = (bound + reduction.constant).next_down();
        }
        bound.max(self.cutoff_certificate)
    }

    fn solved(&self) -> bool {
        self.components.iter().all(Component::solved)
    }

    fn component_cutoff(&self, parent: &BBSolver, index: usize) -> f64 {
        if !parent.options.component_cutoff_propagation {
            return f64::INFINITY;
        }
        // Subtract LOWER bounds of the other components, never their incumbents.
        // Round upwards so the translated cutoff cannot prune too aggressively.
        let mut limit = parent.pruning_upper_bound();
        for reduction in self.postsolve.iter().rev() {
            limit = (limit - reduction.constant).next_up();
        }
        limit = (limit - self.constant).next_up();
        for (other, component) in self.components.iter().enumerate() {
            if other != index {
                limit = (limit - component.lower_bound()).next_up();
            }
        }
        limit
    }

    fn advance(&mut self, parent: &BBSolver) -> (usize, usize, usize) {
        if get_current_time() >= parent.time_start + parent.options.max_time {
            return (0, 0, 0);
        }
        // Fair bounded slices: completed components are never scheduled again.
        let Some(index) = (0..self.components.len())
            .map(|offset| (self.next_component + offset) % self.components.len())
            .find(|&index| !self.components[index].finished())
        else {
            return (0, 0, 0);
        };
        self.next_component = (index + 1) % self.components.len();
        let cutoff = self.component_cutoff(parent, index);
        let component = &mut self.components[index];
        let solver = &mut component.solver;
        solver.tighten_external_cutoff(cutoff);
        solver.time_start = parent.time_start;
        solver.options.max_time = parent.options.max_time;
        let before = (
            solver.nodes_visited,
            solver.nodes_processed,
            solver.nodes_solved,
        );
        if !component.initialized {
            // Structural root compaction is disabled here: it calls solve to
            // completion. Normal presolve, probing and dynamic splits remain on.
            assert!(solver.initialize_search().is_none());
            component.initialized = true;
        }
        // Preserve the strongest bound for the whole context. A probing side
        // can have a weaker certificate than its inherited node bound; recording
        // that discarded region must not erase a previously valid global bound.
        component.certified_lower_bound = component
            .certified_lower_bound
            .max(solver.search_lower_bound())
            .min(solver.best_solution_value);
        if !solver.termination_condition() {
            solver.search_batch(parent.options.threads.clamp(1, 16));
        }
        component.certified_lower_bound = component
            .certified_lower_bound
            .max(solver.search_lower_bound())
            .min(solver.best_solution_value);
        if cutoff.is_finite() && component.certified_lower_bound >= cutoff {
            // The outward-rounded translation itself proves this threshold.
            // Keep it explicitly: summing the certificate back downwards could
            // otherwise strand a finished child a few ulps below its cutoff.
            self.cutoff_certificate = self.cutoff_certificate.max(parent.pruning_upper_bound());
        }
        let work = (
            solver.nodes_visited - before.0,
            solver.nodes_processed - before.1,
            solver.nodes_solved - before.2,
        );
        parent
            .probing_counters
            .merge(solver.probing_counters.take());
        if !self.structural {
            parent
                .component_counters
                .batches
                .fetch_add(1, Ordering::Relaxed);
        }
        if !self.structural && component.solved() {
            parent
                .component_counters
                .components_solved
                .fetch_add(1, Ordering::Relaxed);
        }
        work
    }
}

impl BBSolver {
    pub fn component_statistics(&self) -> ComponentStatistics {
        self.component_counters.snapshot()
    }

    fn component_constant(&self, fixed: &FixedVarMap, labels: &[usize]) -> Option<f64> {
        use crate::preprocess::low_degree::add;
        let mut linear = self.input_qubo.c.to_vec();
        let mut constant = 0.0;
        for (&value, (i, j)) in &self.input_qubo.q {
            if !value.is_finite() {
                return None;
            }
            match (fixed.get(&i), fixed.get(&j)) {
                (None, None) => {
                    if labels[i] != labels[j] {
                        // A cross-component entry is harmless only if its
                        // opposite orientation cancels it exactly.
                        let opposite = self.input_qubo.q.get(j, i).copied().unwrap_or(0.0);
                        if add(value, opposite)? != 0.0 {
                            return None;
                        }
                    }
                }
                (Some(&0), _) | (_, Some(&0)) => {}
                (a, b) => {
                    let half = value * 0.5;
                    if half * 2.0 != value {
                        return None;
                    }
                    match (a, b) {
                        (Some(_), Some(_)) => constant = add(constant, half)?,
                        (Some(_), None) => linear[j] = add(linear[j], half)?,
                        (None, Some(_)) => linear[i] = add(linear[i], half)?,
                        _ => unreachable!(),
                    }
                }
            }
        }
        for (i, &value) in self.input_qubo.c.iter().enumerate() {
            if !value.is_finite() {
                return None;
            }
            if fixed.get(&i) == Some(&1) {
                constant = add(constant, value)?;
            }
        }
        Some(constant)
    }

    pub(super) fn make_component_search(
        &self,
        node: &QuboBBNode,
    ) -> Option<Arc<Mutex<ComponentSearch>>> {
        if !self.options.component_decomposition
            || self.qubo.num_x() - node.fixed_variables.len() < 2
            || get_current_time() >= self.time_start + self.options.max_time
        {
            return None;
        }
        self.component_counters
            .checks
            .fetch_add(1, Ordering::Relaxed);
        // The prepared adjacency is undirected and ignores cancelled/zero terms,
        // unlike walking outer sparse rows of potentially asymmetric input Q.
        let parts = crate::graph_utils::small_components_with_adjacency(
            self.prepared_preprocess.adjacency(),
            &node.fixed_variables,
            None,
            usize::MAX,
        );
        if parts.len() < 2 {
            return None;
        }
        let mut labels = vec![usize::MAX; self.qubo.num_x()];
        for (index, part) in parts.iter().enumerate() {
            for &i in part {
                labels[i] = index;
            }
        }
        if self.root_constraints.iter().any(|relation| {
            let (a, b) = (labels[relation.x_i], labels[relation.x_j]);
            a != usize::MAX && b != usize::MAX && a != b
        }) {
            self.component_counters
                .cross_relation_skips
                .fetch_add(1, Ordering::Relaxed);
            return None;
        }

        // Count fixed-only terms once, never once per component.
        let Some(constant) = self.component_constant(&node.fixed_variables, &labels) else {
            self.component_counters
                .projection_skips
                .fetch_add(1, Ordering::Relaxed);
            return None;
        };
        let mut components = Vec::with_capacity(parts.len());
        for variables in parts {
            if get_current_time() >= self.time_start + self.options.max_time {
                return None;
            }
            let (qubo, _) = preprocess::make_component_qubo(
                &self.input_qubo,
                &variables,
                &node.fixed_variables,
            );
            let mut options = self.options.clone();
            options.fixed_variables.clear();
            options.verbose = 0;
            options.root_low_degree_elimination = false;
            options.root_dominant_edge_contraction = false;
            options.root_cut_dominance = false;
            options.branch_strategy = self.branch_strategy;
            let mut solver = BBSolver::new(qubo, options);
            let Some(backend) = self.subproblem_solver.for_reduced_qubo(&solver.qubo) else {
                self.component_counters
                    .backend_skips
                    .fetch_add(1, Ordering::Relaxed);
                return None;
            };
            solver.subproblem_solver = backend;
            solver.component_counters = Arc::clone(&self.component_counters);
            solver.reduction_counters = Arc::clone(&self.reduction_counters);
            solver.cutoff_counters = Arc::clone(&self.cutoff_counters);
            solver.sdp_fixing_counters = Arc::clone(&self.sdp_fixing_counters);
            solver.component_depth = self.component_depth + 1;
            solver.time_start = self.time_start;
            solver.warm_start(Array1::from_iter(
                variables.iter().map(|&i| self.best_solution[i]),
            ));
            components.push(Component {
                variables,
                solver,
                initialized: false,
                certified_lower_bound: f64::NEG_INFINITY,
            });
        }
        self.component_counters
            .splits
            .fetch_add(1, Ordering::Relaxed);
        self.component_counters
            .components_created
            .fetch_add(components.len(), Ordering::Relaxed);
        self.component_counters
            .max_depth
            .fetch_max(self.component_depth + 1, Ordering::Relaxed);
        Some(Arc::new(Mutex::new(ComponentSearch {
            components,
            fixed: node.fixed_variables.clone(),
            size: self.qubo.num_x(),
            constant,
            next_component: 0,
            postsolve: Vec::new(),
            structural: false,
            cutoff_certificate: f64::NEG_INFINITY,
        })))
    }

    pub(super) fn advance_component_node(
        &self,
        mut node: QuboBBNode,
        candidate: Option<(Array1<usize>, f64)>,
    ) -> ProcessNodeState {
        let Some(SubProblemNodeState::Components(handle)) = &node.subproblem_state else {
            unreachable!("only AND nodes have component continuations");
        };
        let mut search = handle.lock().expect("component search poisoned");
        let component_work = search.advance(self);
        let solution = search.candidate();
        let value = self.qubo.eval_usize(&solution);
        node.lower_bound = node.lower_bound.max(search.lower_bound());
        let solved = search.solved();
        drop(search);
        let best_update = Self::better_candidate(candidate, Some((solution, value)));
        let incumbent = best_update
            .as_ref()
            .map_or(self.best_solution_value, |(_, value)| {
                value.min(self.best_solution_value)
            });
        let closed = solved || self.prunes_with_incumbent(node.lower_bound, incumbent);
        ProcessNodeState {
            best_update,
            branches: None,
            component_work,
            continuation: (!closed).then_some(node),
            logging: if closed {
                NodeLoggingAction::Solved
            } else {
                NodeLoggingAction::Processed
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::branch_subproblem::{BasicSubProblemResult, SubProblemOptions, SubProblemSelection};
    use crate::constraint::{Constraint, ConstraintType};
    use sprs::TriMat;

    fn options() -> SolverOptions {
        let mut o = SolverOptions::new();
        o.verbose = 0;
        o.threads = 1;
        o.root_low_degree_elimination = false;
        o.root_dominant_edge_contraction = false;
        o.root_complement_symmetry = false;
        o.node_probe_candidates = 0;
        o.node_structural_reductions = false;
        o.small_block_elimination = false;
        o.component_cutoff_propagation = false;
        o.branch_strategy = BranchStrategy::FirstNotFixed;
        o.sub_problem_solver = SubProblemSelection::RoofDualQPBO;
        o
    }

    fn cycles(count: usize, hub: bool) -> Qubo {
        let n = count * 11 + usize::from(hub);
        let mut q = TriMat::new((n, n));
        let mut c = Array1::zeros(n);
        for component in 0..count {
            let start = usize::from(hub) + component * 11;
            for bit in 0..11 {
                let (i, j) = (start + bit, start + (bit + 1) % 11);
                q.add_triplet(i, j, 4.0);
                c[i] -= 1.0;
                c[j] -= 1.0;
            }
            if hub {
                q.add_triplet(0, start, 4.0);
                c[start] -= 2.0; // Conditioning hub=1 restores the plain cycle.
            }
        }
        if hub {
            c[0] = 27.0;
        }
        Qubo::new_with_c(q.to_csc(), c)
    }

    fn node(n: usize, fixed_variables: FixedVarMap) -> QuboBBNode {
        QuboBBNode {
            lower_bound: f64::NEG_INFINITY,
            solution: Array1::from_elem(n, 0.5),
            fixed_variables,
            run_heuristic: false,
            subproblem_state: None,
        }
    }

    #[test]
    fn external_cutoffs_use_other_lower_bounds_and_do_not_invent_component_optima() {
        let mut parent = BBSolver::new(cycles(2, false), options());
        parent.options.component_cutoff_propagation = true;
        // Simulate a better incumbent from outside this AND node.
        parent.best_solution_value = -22.0;
        let mut n = node(22, FixedVarMap::default());
        let handle = parent.make_component_search(&n).unwrap();
        {
            let mut group = handle.lock().unwrap();
            assert_eq!(group.component_cutoff(&parent, 0), f64::INFINITY);
            for component in &mut group.components {
                component.initialized = true;
                component.solver.best_solution_value = -8.0;
                let mut child = node(11, FixedVarMap::default());
                child.lower_bound = -11.0; // Valid: each odd cycle's optimum is -10.
                component.solver.nodes.push(child);
            }
            assert!((group.component_cutoff(&parent, 0) + 11.0).abs() < 1e-12);
            // Using the other incumbent (-8) would incorrectly give -14.
            group.components[1].solver.nodes.clear();
            group.components[1].solver.best_solution_value = -10.0;
            assert!((group.component_cutoff(&parent, 0) + 12.0).abs() < 1e-12);
        }
        n.subproblem_state = Some(SubProblemNodeState::Components(Arc::clone(&handle)));
        let result = parent.advance_component_node(n, None);
        assert!(result.continuation.is_none());
        let group = handle.lock().unwrap();
        let component = &group.components[0];
        assert!(component.finished());
        assert!(!component.solved());
        assert!(!group.solved());
        assert_eq!(component.solver.best_solution_value, -8.0);
        assert_eq!(component.lower_bound(), -11.0);
        assert!((group.lower_bound() + 21.0).abs() < 1e-12);
        assert!(parent.cutoff_statistics().prunes > 0);
    }

    #[test]
    fn cutoff_propagation_preserves_complete_component_solutions() {
        for enabled in [false, true] {
            let mut o = options();
            o.component_cutoff_propagation = enabled;
            o.sub_problem_solver = SubProblemSelection::MixingCutSDP;
            let qubo = cycles(3, true);
            o.fixed_variables.insert(0, 1);
            let mut solver = BBSolver::new(qubo.clone(), o);
            let (solution, value) = solver.solve();
            assert_eq!(solution[0], 1);
            assert!((value + 3.0).abs() < 1e-7);
            assert_eq!(qubo.eval_usize(&solution), -3.0);
            assert!(solver.search_is_exact());
        }
    }

    #[test]
    fn inherited_bounds_survive_weaker_discarded_certificates() {
        let parent = BBSolver::new(cycles(2, false), options());
        let handle = parent
            .make_component_search(&node(22, FixedVarMap::default()))
            .unwrap();
        let mut group = handle.lock().unwrap();
        let component = &mut group.components[0];
        component.initialized = true;
        component.certified_lower_bound = -10.0;
        component.solver.best_solution_value = -8.0;
        component.solver.options.component_cutoff_propagation = true;
        component.solver.tighten_external_cutoff(-12.0);
        assert!(component.solver.prunes_with_incumbent(-11.0, -8.0));
        assert_eq!(component.solver.search_lower_bound(), -11.0);
        assert_eq!(component.lower_bound(), -10.0);
        assert!(component.finished());
        assert!(!component.solved());
    }

    #[test]
    fn rounded_cutoff_translation_can_close_a_nested_context_without_stalling() {
        let mut o = options();
        o.component_cutoff_propagation = true;
        let mut parent = BBSolver::new(cycles(2, false), o);
        parent.best_solution_value = -18.0;
        parent.tighten_external_cutoff(-21.0);
        let mut n = node(22, FixedVarMap::default());
        let handle = parent.make_component_search(&n).unwrap();
        {
            let mut group = handle.lock().unwrap();
            for component in &mut group.components {
                component.initialized = true;
            }
            group.components[1].solver.best_solution_value = -10.0;
            group.components[0].solver.best_solution_value = -8.0;
            let mut child = node(11, FixedVarMap::default());
            child.lower_bound = group.component_cutoff(&parent, 0);
            assert!(child.lower_bound >= -11.0 && child.lower_bound < -10.0);
            group.components[0].solver.nodes.push(child);
        }
        n.subproblem_state = Some(SubProblemNodeState::Components(Arc::clone(&handle)));
        let result = parent.advance_component_node(n, None);
        assert!(result.continuation.is_none());
        assert!(!handle.lock().unwrap().solved());
        assert!(parent.cutoff.discarded_bound() >= -21.0);
        assert!(parent.cutoff.discarded_bound() < -18.0);
    }

    fn finish(solver: &mut BBSolver, expected: f64) {
        for _ in 0..1000 {
            if solver.nodes.is_empty() {
                break;
            }
            assert!(
                solver
                    .nodes
                    .iter()
                    .map(|n| n.lower_bound)
                    .fold(f64::INFINITY, f64::min)
                    <= expected + 1e-7
            );
            solver.search_batch(1);
        }
        assert!(solver.nodes.is_empty(), "search did not finish");
        assert!((solver.best_solution_value - expected).abs() < 1e-7);
        assert!((solver.input_qubo.eval_usize(&solver.best_solution) - expected).abs() < 1e-7);
    }

    #[test]
    fn component_solves_match_independent_optima_with_fixed_constant_and_backends() {
        for backend in [
            SubProblemSelection::RoofDualQPBO,
            SubProblemSelection::HerculesABQP,
            SubProblemSelection::MixingCutSDP,
        ] {
            for hub in [false, true] {
                for enabled in [false, true] {
                    let mut o = options();
                    o.sub_problem_solver = backend;
                    o.component_decomposition = enabled;
                    if hub {
                        o.fixed_variables.insert(0, 1);
                    }
                    let qubo = cycles(2, hub);
                    // An odd 11-cycle has maximum cut 10; native energy is -cut.
                    let expected = if hub { 7.0 } else { -20.0 };
                    let mut solver = BBSolver::new(qubo.clone(), o);
                    let (x, value) = solver.solve();
                    assert!((value - expected).abs() < 1e-7, "{value} != {expected}");
                    assert!((qubo.eval_usize(&x) - expected).abs() < 1e-7);
                    assert!(solver.nodes.is_empty());
                    if hub {
                        assert_eq!(x[0], 1);
                    }
                    assert_eq!(solver.component_statistics().splits > 0, enabled);
                }
            }
        }
    }

    #[test]
    fn fixing_an_articulation_creates_a_node_level_split() {
        let mut solver = BBSolver::new(cycles(2, true), options());
        assert!(solver
            .make_component_search(&node(23, FixedVarMap::default()))
            .is_none());
        solver.best_solution[0] = 1;
        solver.best_solution_value = solver.qubo.eval_usize(&solver.best_solution);
        let state = solver.process_node(&node(23, [(0, 1)].into_iter().collect()));
        assert!(state.branches.is_none());
        assert!(state.continuation.is_some());
        assert_eq!(solver.component_statistics().splits, 1);
        solver.apply_process_result(state);
        finish(&mut solver, 7.0);
        assert_eq!(solver.best_solution[0], 1);
    }

    #[test]
    fn and_bounds_sum_or_frontier_minima_and_keep_unsolved_components_open() {
        let solver = BBSolver::new(cycles(2, false), options());
        let handle = solver
            .make_component_search(&node(22, FixedVarMap::default()))
            .unwrap();
        let mut group = handle.lock().unwrap();
        assert_eq!(group.lower_bound(), f64::NEG_INFINITY);
        assert!(!group.solved());
        for c in &mut group.components {
            c.initialized = true;
            c.solver.best_solution_value = -10.0;
            let mut a = node(11, FixedVarMap::default());
            a.lower_bound = -12.0;
            let mut b = a.clone();
            b.lower_bound = -11.0;
            c.solver.nodes.extend([a, b]);
        }
        assert!((group.lower_bound() + 24.0).abs() < 1e-12);
        group.components[0].solver.nodes.clear();
        assert!(!group.solved());
        assert!((group.lower_bound() + 22.0).abs() < 1e-12);
        group.components[1].solver.nodes.clear();
        assert!(group.solved());
        assert!((group.lower_bound() + 20.0).abs() < 1e-12);
    }

    #[test]
    fn component_deadline_retains_frontiers_and_finished_components_are_not_restarted() {
        let mut solver = BBSolver::new(cycles(2, false), options());
        let mut n = node(22, FixedVarMap::default());
        let handle = solver.make_component_search(&n).unwrap();
        n.subproblem_state = Some(SubProblemNodeState::Components(Arc::clone(&handle)));
        solver.options.max_time = -1.0;
        let state = solver.advance_component_node(n, None);
        assert_eq!(solver.component_statistics().batches, 0);
        let n = state.continuation.unwrap();
        assert!(!handle.lock().unwrap().solved());
        assert!(handle
            .lock()
            .unwrap()
            .components
            .iter()
            .all(|c| !c.initialized));
        solver.options.max_time = 100.0;
        let state = solver.advance_component_node(n, None);
        let n = state.continuation.unwrap();
        let before = handle.lock().unwrap().components[0].solver.nodes.len();
        assert!(before > 0);
        solver.options.max_time = -1.0;
        let state = solver.advance_component_node(n, None);
        assert_eq!(
            handle.lock().unwrap().components[0].solver.nodes.len(),
            before
        );
        solver.options.max_time = 100.0;
        solver.apply_process_result(state);
        finish(&mut solver, -20.0);
        assert_eq!(solver.component_statistics().components_solved, 2);
        let group = handle.lock().unwrap();
        assert!(group.solved());
        assert_eq!(
            group.components[0].solver.nodes_visited,
            group.components[1].solver.nodes_visited
        );
    }

    #[test]
    fn component_splitting_declines_cross_relations_and_unsupported_backends() {
        struct NoReprepare;
        impl SubProblemSolver for NoReprepare {
            fn solve_lower_bound(
                &self,
                _: &BBSolver,
                _: &QuboBBNode,
                _: Option<SubProblemOptions>,
            ) -> Box<dyn SubProblemResult> {
                Box::new(BasicSubProblemResult {
                    lower_bound: -100.0,
                    relaxed_solution: Array1::from_elem(22, 0.5),
                })
            }
        }
        let mut solver = BBSolver::new(cycles(2, false), options());
        let n = node(22, FixedVarMap::default());
        solver
            .root_constraints
            .push(Constraint::new(0, 11, ConstraintType::Equal));
        assert!(solver.make_component_search(&n).is_none());
        assert_eq!(solver.component_statistics().cross_relation_skips, 1);
        solver.root_constraints.clear();
        solver.subproblem_solver = Box::new(NoReprepare);
        assert!(solver.make_component_search(&n).is_none());
        assert_eq!(solver.component_statistics().backend_skips, 1);
    }

    #[test]
    fn binary_branching_never_shares_a_component_frontier_between_different_fixings() {
        let solver = BBSolver::new(cycles(2, false), options());
        let mut n = node(22, FixedVarMap::default());
        n.subproblem_state = Some(SubProblemNodeState::Components(
            solver.make_component_search(&n).unwrap(),
        ));
        let (zero, one) = BBSolver::branch(n, 0, -22.0, Array1::from_elem(22, 0.5));
        assert!(zero.subproblem_state.is_none() && one.subproblem_state.is_none());
        assert_eq!(zero.fixed_variables[&0], 0);
        assert_eq!(one.fixed_variables[&0], 1);
        assert_eq!(zero.lower_bound, -22.0);
        assert_eq!(one.lower_bound, -22.0);
    }

    #[test]
    fn component_projection_declines_inexact_fixed_constants_and_linear_updates() {
        for fixed_constant in [false, true] {
            let mut q = TriMat::new((5, 5));
            let mut c = Array1::from_elem(5, -1.0);
            let fixed = if fixed_constant {
                c[0] = 1e20;
                c[1] = 1.0;
                [(0, 1), (1, 1)].into_iter().collect()
            } else {
                c[2] = 1e20;
                q.add_triplet(0, 2, 2.0);
                [(0, 1)].into_iter().collect()
            };
            let solver = BBSolver::new(Qubo::new_with_c(q.to_csr(), c), options());
            assert!(solver.make_component_search(&node(5, fixed)).is_none());
            assert_eq!(solver.component_statistics().projection_skips, 1);
        }
    }

    #[test]
    fn component_projection_matches_every_assignment_of_generated_asymmetric_qubos() {
        for sample in 0..32 {
            let n = 10;
            let mut q = TriMat::new((n, n));
            let c = Array1::from_iter((0..n).map(|i| ((sample + i * 3) % 9) as f64 - 4.0));
            for i in 0..n {
                q.add_triplet(i, i, (sample % 3) as f64);
                for j in i + 1..n {
                    if i == 0 || (i - 1) / 3 == (j - 1) / 3 {
                        q.add_triplet(j, i, ((sample + i * 3 + j) % 9) as f64 - 4.0);
                    }
                }
            }
            let qubo = Qubo::new_with_c(
                if sample % 2 == 0 {
                    q.to_csr()
                } else {
                    q.to_csc()
                },
                c,
            );
            let mut solver = BBSolver::new(qubo.clone(), options());
            let fixed = [(0, sample % 2)].into_iter().collect();
            let mut n = node(10, fixed);
            let handle = solver.make_component_search(&n).unwrap();
            let mut expected = f64::INFINITY;
            {
                let group = handle.lock().unwrap();
                for mask in 0..1024 {
                    let x = Array1::from_iter((0..10).map(|bit| (mask >> bit) & 1));
                    if x[0] != sample % 2 {
                        continue;
                    }
                    let sum = group
                        .components
                        .iter()
                        .fold(group.constant, |sum, component| {
                            let small =
                                Array1::from_iter(component.variables.iter().map(|&i| x[i]));
                            sum + component.solver.input_qubo.eval_usize(&small)
                        });
                    assert_eq!(sum, qubo.eval_usize(&x));
                    expected = expected.min(sum);
                }
            }
            solver.best_solution[0] = sample % 2;
            solver.best_solution_value = qubo.eval_usize(&solver.best_solution);
            n.subproblem_state = Some(SubProblemNodeState::Components(handle));
            solver.nodes.push(n);
            finish(&mut solver, expected);
        }
    }

    #[test]
    fn component_search_recursively_splits_and_shares_a_single_deadline() {
        // Initially two components. In the first, presolve fixes a nonnegative
        // hub to zero, separating two 11-cycles. Neither cycle is enumerated.
        let mut qubo = cycles(3, true);
        let mut q = TriMat::new(qubo.q.shape());
        for (&v, (i, j)) in &qubo.q {
            if (i, j) != (0, 23) {
                q.add_triplet(i, j, v);
            }
        }
        qubo.q = q.to_csc();
        qubo.c[0] = 1.0;
        for i in [1, 12, 23] {
            qubo.c[i] += 2.0;
        }
        let mut solver = BBSolver::new(qubo, options());
        let mut n = node(34, FixedVarMap::default());
        let handle = solver.make_component_search(&n).unwrap();
        n.subproblem_state = Some(SubProblemNodeState::Components(Arc::clone(&handle)));
        solver.nodes.push(n);
        finish(&mut solver, -30.0);
        let stats = solver.component_statistics();
        assert_eq!(stats.splits, 2);
        assert_eq!(stats.max_depth, 2);
        assert_eq!(stats.components_created, 4);
        assert_eq!(stats.components_solved, 4);
        let group = handle.lock().unwrap();
        for component in &group.components {
            assert_eq!(component.solver.time_start, solver.time_start);
            assert_eq!(component.solver.options.max_time, solver.options.max_time);
        }
    }

    #[test]
    fn component_timeout_postsolve_restores_original_variables_and_bound_units() {
        let n = 23;
        let mut q = TriMat::new((n, n));
        let mut c = Array1::zeros(n);
        for start in [0, 11] {
            for i in start..start + 11 {
                for j in i + 1..start + 11 {
                    q.add_triplet(i, j, 4.0);
                    c[i] -= 1.0;
                    c[j] -= 1.0;
                }
            }
        }
        q.add_triplet(0, 22, 4.0);
        c[0] -= 1.0;
        c[22] = -1.0;
        let qubo = Qubo::new_with_c(q.to_csc(), c);
        let mut o = options();
        o.root_low_degree_elimination = true;
        let reduction = Arc::new(
            preprocess::low_degree::reduce_root(
                &qubo,
                &FixedVarMap::default(),
                &o,
                std::time::Duration::from_secs(5),
            )
            .unwrap(),
        );
        assert_eq!(reduction.remaining.len(), 22);
        let inner = BBSolver::new(reduction.qubo.clone(), options());
        let handle = inner
            .make_component_search(&node(22, FixedVarMap::default()))
            .unwrap();
        handle.lock().unwrap().lift(reduction);
        let mut outer = BBSolver::new(qubo, options());
        outer.component_counters = Arc::clone(&inner.component_counters);
        let mut n = node(23, FixedVarMap::default());
        n.lower_bound = -100.0;
        n.subproblem_state = Some(SubProblemNodeState::Components(handle));
        outer.options.max_time = -1.0;
        let state = outer.advance_component_node(n, None);
        assert!(state.continuation.is_some());
        let (x, value) = state.best_update.as_ref().unwrap();
        assert_eq!(x.len(), 23);
        assert!((outer.input_qubo.eval_usize(x) - value).abs() < 1e-7);
        outer.apply_process_result(state);
        outer.options.max_time = 100.0;
        finish(&mut outer, -61.0); // Two K11 cuts plus the eliminated leaf edge.
    }
}
