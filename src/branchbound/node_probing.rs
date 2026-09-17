use super::{BBSolver, QuboBBNode};
use crate::variable_reduction::select_probe_candidates;
use crate::FixedVarMap;
use ndarray::Array1;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

#[derive(Clone, Copy, Debug, Default)]
pub struct NodeProbingStatistics {
    /// Conditional-presolve entries, including repeated root passes.
    pub nodes: usize,
    /// Conditional presolve calls, not main subproblem solves.
    pub assignments: usize,
    pub failed_literals: usize,
    /// Additional local fixings summed across nodes, not distinct original variables.
    pub fixed_variables: usize,
    pub pruned_nodes: usize,
    pub bound_improvements: usize,
    pub children_reused: usize,
    /// Main backend calls, including the initial root and strong-branching trials.
    pub subproblem_calls: usize,
    /// Weak roof reductions across all presolve calls, including probe trials.
    /// Counts repeated local fixings, not distinct original variables.
    pub weak_roof_fixings: usize,
    /// Root work is also included in the totals above.
    pub root_rounds: usize,
    pub root_assignments: usize,
    pub root_fixed_variables: usize,
    pub root_symmetry_fixings: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{qubo::Qubo, solver_options::SolverOptions};

    fn cycle_solver(n: usize, bias: f64) -> BBSolver {
        let mut terms = sprs::TriMat::new((n, n));
        for i in 0..n {
            let j = (i + 1) % n;
            terms.add_triplet(i, j, 2.0);
            terms.add_triplet(j, i, 2.0);
        }
        let mut linear = Array1::from_elem(n, -2.0);
        linear[0] += bias;
        let mut options = SolverOptions::new();
        options.verbose = 0;
        options.root_complement_symmetry = false;
        options.root_low_degree_elimination = false;
        options.root_dominant_edge_contraction = false;
        options.node_structural_reductions = false;
        options.small_block_elimination = false;
        options.node_probe_candidates = 1;
        options.node_probe_max_seconds = 10.0;
        BBSolver::new(Qubo::new_with_c(terms.to_csr(), linear), options)
    }

    fn presolved_node(solver: &BBSolver, fixed: FixedVarMap) -> QuboBBNode {
        let (fixed, bound) = solver.presolve_node(fixed).unwrap();
        QuboBBNode {
            fixed_variables: fixed,
            lower_bound: bound,
            solution: Array1::from_elem(solver.qubo.num_x(), 0.5),
            run_heuristic: false,
            subproblem_state: None,
        }
    }

    #[test]
    fn weak_roof_closes_large_balanced_component_without_exporting_relations() {
        let mut solver = cycle_solver(12, 0.0);
        solver.options.node_probe_candidates = 0;
        solver.options.roof_dual_weak_persistencies = false;
        let (strong, _) = solver.presolve_node(FixedVarMap::default()).unwrap();
        assert!(strong.is_empty());
        solver.options.roof_dual_weak_persistencies = true;
        let (fixed, bound) = solver.presolve_node(FixedVarMap::default()).unwrap();
        assert_eq!(fixed.len(), 12);
        assert!((bound + 12.0).abs() < 1e-9);
        let (solution, objective) = solver.solve();
        assert!((objective + 12.0).abs() < 1e-8);
        assert!((solver.qubo.eval_usize(&solution) - objective).abs() < 1e-8);
        assert!(solver.nodes.is_empty());
        assert!(solver.root_constraints.is_empty());
        assert_eq!(solver.node_probing_statistics().weak_roof_fixings, 12);
    }

    fn minimum(qubo: &Qubo, fixed: &FixedVarMap) -> f64 {
        (0..(1 << qubo.num_x()))
            .filter(|mask| fixed.iter().all(|(&i, &v)| (mask >> i) & 1 == v))
            .map(|mask| {
                qubo.eval_usize(&Array1::from_iter(
                    (0..qubo.num_x()).map(|i| (mask >> i) & 1),
                ))
            })
            .fold(f64::INFINITY, f64::min)
    }

    #[test]
    fn root_roof_probe_keeps_one_tied_optimum_without_a_backend_solve() {
        let mut solver = cycle_solver(13, 0.0);
        solver.options.node_probe_candidates = 0;
        solver.options.node_probe_max_free = 1;
        solver.options.root_roof_probe_candidates = 2;
        solver.options.root_roof_probe_max_seconds = 10.0;
        let mut node = presolved_node(&solver, FixedVarMap::default());
        assert!(solver.probe_root(&mut node));
        assert!(solver.root_constraints.is_empty());
        let (solution, objective) = solver.solve();
        assert!((objective + 12.0).abs() < 1e-8);
        assert!((solver.qubo.eval_usize(&solution) - objective).abs() < 1e-8);
        assert!(solver.nodes.is_empty());
        let stats = solver.node_probing_statistics();
        assert_eq!(stats.subproblem_calls, 0);
        assert_eq!(stats.root_assignments, 2);
        assert_eq!(stats.root_rounds, 1);
        assert!(stats.weak_roof_fixings > 0);
    }

    #[test]
    fn root_roof_probe_respects_disabled_and_expired_budgets() {
        for mode in 0..6 {
            let mut solver = cycle_solver(13, 0.0);
            let mut node = presolved_node(&solver, FixedVarMap::default());
            solver.options.root_roof_probe_candidates = if mode == 0 { 0 } else { 2 };
            solver.options.root_roof_probe_max_seconds = match mode {
                1 => 0.0,
                2 => f64::NAN,
                3 => f64::INFINITY,
                _ => 10.0,
            };
            if mode == 4 {
                solver.options.max_time = -1.0;
            }
            if mode == 5 {
                solver.options.node_lower_bound =
                    crate::solver_options::NodeLowerBoundSelection::Li;
            }
            let fixed = node.fixed_variables.clone();
            let bound = node.lower_bound;
            assert!(!solver.probe_root(&mut node));
            assert_eq!(node.fixed_variables, fixed);
            assert_eq!(node.lower_bound, bound);
            assert_eq!(solver.node_probing_statistics().root_assignments, 0);
        }
    }

    #[test]
    fn complement_symmetry_keeps_one_global_optimum_and_respects_input_fixings() {
        use smolprng::{JsfLarge, PRNG};
        let mut rng = PRNG {
            generator: JsfLarge::from(99215u64),
        };
        for sample in 0..16 {
            let mut q = sprs::TriMat::new((12, 12));
            let mut c = Array1::zeros(12);
            for i in 0..12 {
                for j in i + 1..12 {
                    if j == i + 1 || rng.gen_f64() < 0.3 {
                        let weight = if rng.gen_f64() < 0.5 { -2.0 } else { 2.0 };
                        q.add_triplet(i, j, 2.0 * weight);
                        c[i] -= 0.5 * weight;
                        c[j] -= 0.5 * weight;
                    }
                }
            }
            let qubo = Qubo::new_with_c(q.to_csr(), c);
            for incoming in [FixedVarMap::default(), [(3, 1)].into_iter().collect()] {
                let expected = minimum(&qubo, &incoming);
                for weak in [false, true] {
                    let mut options = SolverOptions::new();
                    options.verbose = 0;
                    options.root_complement_symmetry = true;
                    options.fixed_variables = incoming.clone();
                    options.roof_dual_weak_persistencies = weak;
                    options.root_roof_probe_candidates = if sample % 2 == 0 { 2 } else { 0 };
                    options.sub_problem_solver =
                        crate::branch_subproblem::SubProblemSelection::MixingCutSDP;
                    let mut solver = BBSolver::new(qubo.clone(), options);
                    let (x, value) = solver.solve();
                    assert!((value - expected).abs() < 1e-8);
                    assert!((qubo.eval_usize(&x) - expected).abs() < 1e-8);
                    assert!(solver.nodes.is_empty());
                    assert!(incoming.iter().all(|(&i, &v)| x[i] == v));
                    assert_eq!(
                        solver.node_probing_statistics().root_symmetry_fixings,
                        usize::from(incoming.is_empty())
                    );
                }
            }
        }
    }

    #[test]
    fn root_roof_probing_preserves_a_conditional_optimum_on_generated_qubos() {
        use smolprng::{JsfLarge, PRNG};
        let mut rng = PRNG {
            generator: JsfLarge::from(53791u64),
        };
        for sample in 0..64 {
            let mut qubo = Qubo::make_random_qubo(12, &mut rng, 0.4);
            qubo.q
                .data_mut()
                .iter_mut()
                .for_each(|v| *v = (*v * 8.0).round());
            qubo.c.mapv_inplace(|v| (v * 4.0).round());
            let mut options = SolverOptions::new();
            options.verbose = 0;
            options.root_roof_probe_candidates = 1;
            options.root_roof_probe_max_seconds = 10.0;
            let mut solver = BBSolver::new(qubo, options);
            // A feasible incumbent for the conditional problem, not an
            // independently chosen tie fixing that could conflict with it.
            let candidate = Array1::from_iter((0..12).map(|i| (i + sample) % 2));
            solver.best_solution_value = solver.qubo.eval_usize(&candidate);
            solver.best_solution = candidate;
            let incoming: FixedVarMap = [(0, sample % 2)].into_iter().collect();
            let expected = minimum(&solver.qubo, &incoming);
            let mut node = presolved_node(&solver, incoming);
            let closed = solver.probe_root(&mut node);
            let remaining = if closed {
                f64::INFINITY
            } else {
                minimum(&solver.qubo, &node.fixed_variables)
            };
            assert!(
                (remaining.min(solver.best_solution_value) - expected).abs() < 1e-8,
                "case {sample}"
            );
            if !closed {
                assert!(node.lower_bound <= remaining + 1e-8);
            }
            assert!(solver.root_constraints.is_empty());
        }
    }

    #[test]
    fn probing_completes_both_sides_and_retains_best_primal() {
        let solver = cycle_solver(11, 0.0);
        let mut node = presolved_node(&solver, FixedVarMap::default());
        assert!(node.fixed_variables.is_empty());
        let result = solver.probe_node(&mut node);
        assert!(result.closed);
        let (solution, value) = result.candidate.unwrap();
        assert!((value + 10.0).abs() < 1e-9);
        assert!((solver.qubo.eval_usize(&solution) - value).abs() < 1e-9);
        let stats = solver.node_probing_statistics();
        assert_eq!(stats.assignments, 2);
        assert_eq!(stats.pruned_nodes, 1);
        assert_eq!(stats.subproblem_calls, 0);
    }

    #[test]
    fn cutoff_probe_fixes_the_only_improving_side() {
        let mut solver = cycle_solver(13, 1.0);
        solver.warm_start(Array1::from_iter((0..13).map(|i| 1 - i % 2)));
        assert!((solver.best_solution_value + 11.0).abs() < 1e-9);
        let mut node = presolved_node(&solver, FixedVarMap::default());
        assert!(!node.fixed_variables.contains_key(&0));
        let result = solver.probe_node(&mut node);
        assert!(!result.closed);
        assert_eq!(node.fixed_variables.get(&0), Some(&0));
        assert!((minimum(&solver.qubo, &node.fixed_variables) + 12.0).abs() < 1e-9);
        assert_eq!(solver.node_probing_statistics().failed_literals, 1);
    }

    #[test]
    fn unclosed_disjunction_uses_minimum_and_keeps_conditional_children() {
        let solver = cycle_solver(13, 1.0);
        let mut node = presolved_node(&solver, FixedVarMap::default());
        let result = solver.probe_node(&mut node);
        assert!(!result.closed);
        let children = result.children.unwrap();
        assert_eq!(children.variable, 0);
        assert!((children.zero.1 + 12.0).abs() < 1e-9);
        assert!((children.one.1 + 11.0).abs() < 1e-9);
        assert!((node.lower_bound + 12.0).abs() < 1e-9);
        assert_eq!(children.zero.0.get(&0), Some(&0));
        assert_eq!(children.one.0.get(&0), Some(&1));
    }

    #[test]
    fn node_processing_keeps_probe_completion_without_a_relaxation_call() {
        let mut solver = cycle_solver(11, 0.0);
        let node = presolved_node(&solver, FixedVarMap::default());
        let state = solver.process_node(&node);
        assert!(state.branches.is_none());
        assert_eq!(solver.node_probing_statistics().subproblem_calls, 0);
        solver.apply_process_result(state);
        assert!((solver.best_solution_value + 10.0).abs() < 1e-9);
    }

    #[test]
    fn node_processing_reuses_matching_conditional_children() {
        use crate::branch_stratagy::BranchStrategy;
        use crate::branch_subproblem::{SubProblemOptions, SubProblemResult, SubProblemSolver};
        use crate::subproblemsolvers::mixingcut_sdp::MixingCutSubProblemResult;

        struct WeakBound;
        impl SubProblemSolver for WeakBound {
            fn solve_lower_bound(
                &self,
                _: &BBSolver,
                node: &QuboBBNode,
                _: Option<SubProblemOptions>,
            ) -> Box<dyn SubProblemResult> {
                Box::new(MixingCutSubProblemResult {
                    lower_bound: -20.0,
                    relaxed_solution: node.solution.clone(),
                    candidate_primal_solution: None,
                    subproblem_state: None,
                    conditional_bounds: Vec::new(),
                })
            }
        }

        let mut solver = cycle_solver(13, 1.0);
        solver.branch_strategy = BranchStrategy::FirstNotFixed;
        solver.subproblem_solver = Box::new(WeakBound);
        let node = presolved_node(&solver, FixedVarMap::default());
        let (zero, one) = solver.process_node(&node).branches.unwrap();
        assert_eq!(zero.fixed_variables.get(&0), Some(&0));
        assert_eq!(one.fixed_variables.get(&0), Some(&1));
        assert!((zero.lower_bound + 12.0).abs() < 1e-9);
        assert!((one.lower_bound + 11.0).abs() < 1e-9);
        let stats = solver.node_probing_statistics();
        assert_eq!(stats.subproblem_calls, 1);
        assert_eq!(stats.children_reused, 2);
    }

    #[test]
    fn zero_budgets_and_time_limits_never_manufacture_a_proof() {
        let mut solver = cycle_solver(13, 1.0);
        for mode in 0..4 {
            solver.options.node_probe_candidates = if mode == 0 { 0 } else { 1 };
            solver.options.node_probe_max_free = if mode == 1 { 5 } else { 64 };
            solver.options.node_probe_max_seconds = if mode == 2 { 0.0 } else { 10.0 };
            if mode == 3 {
                solver.options.max_time = -1.0;
            }
            let mut node = presolved_node(&solver, FixedVarMap::default());
            let fixed = node.fixed_variables.clone();
            let bound = node.lower_bound;
            let result = solver.probe_node(&mut node);
            assert!(!result.closed && result.children.is_none() && result.candidate.is_none());
            assert_eq!(node.fixed_variables, fixed);
            assert_eq!(node.lower_bound, bound);
            assert_eq!(solver.node_probing_statistics().assignments, 0);
        }
    }

    #[test]
    fn generated_probes_preserve_the_best_improving_objective() {
        use smolprng::{JsfLarge, PRNG};
        let mut rng = PRNG {
            generator: JsfLarge::from(817249u64),
        };
        for sample in 0..24 {
            let mut qubo = Qubo::make_random_qubo(12, &mut rng, 0.6);
            qubo.q
                .data_mut()
                .iter_mut()
                .for_each(|v| *v = (*v * 16.0).round());
            qubo.c.mapv_inplace(|v| (v * 8.0).round());
            let mut options = SolverOptions::new();
            options.verbose = 0;
            options.node_probe_candidates = 3;
            options.node_probe_max_seconds = 10.0;
            options.roof_dual_weak_persistencies = sample % 2 == 0;
            let mut solver = BBSolver::new(qubo, options);
            let candidate = Array1::from_iter((0..12).map(|i| (i + sample) % 2));
            solver.warm_start(candidate);
            for fixed in [FixedVarMap::default(), [(0, 1)].into_iter().collect()] {
                let expected = minimum(&solver.qubo, &fixed).min(solver.best_solution_value);
                let mut node = presolved_node(&solver, fixed);
                let result = solver.probe_node(&mut node);
                let candidate = result
                    .candidate
                    .as_ref()
                    .map_or(solver.best_solution_value, |(_, value)| {
                        value.min(solver.best_solution_value)
                    });
                let remaining = if result.closed {
                    f64::INFINITY
                } else {
                    minimum(&solver.qubo, &node.fixed_variables)
                };
                assert!((candidate.min(remaining) - expected).abs() < 1e-8);
                if !result.closed {
                    assert!(node.lower_bound <= remaining + 1e-8);
                }
                if let Some(children) = result.children {
                    for (map, bound) in [children.zero, children.one] {
                        assert!(bound <= minimum(&solver.qubo, &map) + 1e-8);
                    }
                }
            }
        }
    }
}

#[derive(Default)]
pub(super) struct NodeProbingCounters {
    nodes: AtomicUsize,
    assignments: AtomicUsize,
    failed_literals: AtomicUsize,
    fixed_variables: AtomicUsize,
    pruned_nodes: AtomicUsize,
    bound_improvements: AtomicUsize,
    children_reused: AtomicUsize,
    subproblem_calls: AtomicUsize,
    weak_roof_fixings: AtomicUsize,
    root_rounds: AtomicUsize,
    root_assignments: AtomicUsize,
    root_fixed_variables: AtomicUsize,
    root_symmetry_fixings: AtomicUsize,
}

impl NodeProbingCounters {
    pub(super) fn take(&self) -> NodeProbingStatistics {
        NodeProbingStatistics {
            nodes: self.nodes.swap(0, Ordering::Relaxed),
            assignments: self.assignments.swap(0, Ordering::Relaxed),
            failed_literals: self.failed_literals.swap(0, Ordering::Relaxed),
            fixed_variables: self.fixed_variables.swap(0, Ordering::Relaxed),
            pruned_nodes: self.pruned_nodes.swap(0, Ordering::Relaxed),
            bound_improvements: self.bound_improvements.swap(0, Ordering::Relaxed),
            children_reused: self.children_reused.swap(0, Ordering::Relaxed),
            subproblem_calls: self.subproblem_calls.swap(0, Ordering::Relaxed),
            weak_roof_fixings: self.weak_roof_fixings.swap(0, Ordering::Relaxed),
            root_rounds: self.root_rounds.swap(0, Ordering::Relaxed),
            root_assignments: self.root_assignments.swap(0, Ordering::Relaxed),
            root_fixed_variables: self.root_fixed_variables.swap(0, Ordering::Relaxed),
            root_symmetry_fixings: self.root_symmetry_fixings.swap(0, Ordering::Relaxed),
        }
    }

    pub(super) fn merge(&self, work: NodeProbingStatistics) {
        self.record(work);
        self.children_reused
            .fetch_add(work.children_reused, Ordering::Relaxed);
        self.subproblem_calls
            .fetch_add(work.subproblem_calls, Ordering::Relaxed);
        self.weak_roof_fixings
            .fetch_add(work.weak_roof_fixings, Ordering::Relaxed);
        self.root_symmetry_fixings
            .fetch_add(work.root_symmetry_fixings, Ordering::Relaxed);
    }

    pub(super) fn symmetry_fixing(&self) {
        self.root_symmetry_fixings.fetch_add(1, Ordering::Relaxed);
    }

    fn record(&self, work: NodeProbingStatistics) {
        self.nodes.fetch_add(work.nodes, Ordering::Relaxed);
        self.assignments
            .fetch_add(work.assignments, Ordering::Relaxed);
        self.failed_literals
            .fetch_add(work.failed_literals, Ordering::Relaxed);
        self.fixed_variables
            .fetch_add(work.fixed_variables, Ordering::Relaxed);
        self.pruned_nodes
            .fetch_add(work.pruned_nodes, Ordering::Relaxed);
        self.bound_improvements
            .fetch_add(work.bound_improvements, Ordering::Relaxed);
        self.root_rounds
            .fetch_add(work.root_rounds, Ordering::Relaxed);
        self.root_assignments
            .fetch_add(work.root_assignments, Ordering::Relaxed);
        self.root_fixed_variables
            .fetch_add(work.root_fixed_variables, Ordering::Relaxed);
    }

    pub(super) fn relaxation(&self) {
        self.subproblem_calls.fetch_add(1, Ordering::Relaxed);
    }

    pub(super) fn weak_roof_fixings(&self, count: usize) {
        if count != 0 {
            self.weak_roof_fixings.fetch_add(count, Ordering::Relaxed);
        }
    }

    pub(super) fn reuse_children(&self) {
        self.children_reused.fetch_add(2, Ordering::Relaxed);
    }

    pub(super) fn snapshot(&self) -> NodeProbingStatistics {
        NodeProbingStatistics {
            nodes: self.nodes.load(Ordering::Relaxed),
            assignments: self.assignments.load(Ordering::Relaxed),
            failed_literals: self.failed_literals.load(Ordering::Relaxed),
            fixed_variables: self.fixed_variables.load(Ordering::Relaxed),
            pruned_nodes: self.pruned_nodes.load(Ordering::Relaxed),
            bound_improvements: self.bound_improvements.load(Ordering::Relaxed),
            children_reused: self.children_reused.load(Ordering::Relaxed),
            subproblem_calls: self.subproblem_calls.load(Ordering::Relaxed),
            weak_roof_fixings: self.weak_roof_fixings.load(Ordering::Relaxed),
            root_rounds: self.root_rounds.load(Ordering::Relaxed),
            root_assignments: self.root_assignments.load(Ordering::Relaxed),
            root_fixed_variables: self.root_fixed_variables.load(Ordering::Relaxed),
            root_symmetry_fixings: self.root_symmetry_fixings.load(Ordering::Relaxed),
        }
    }
}

type PresolvedSide = (FixedVarMap, f64);

pub(super) struct ProbedChildren {
    pub variable: usize,
    pub zero: PresolvedSide,
    pub one: PresolvedSide,
}

#[derive(Default)]
pub(super) struct NodeProbeResult {
    pub closed: bool,
    pub candidate: Option<(Array1<usize>, f64)>,
    pub children: Option<ProbedChildren>,
}

impl BBSolver {
    pub(super) fn probe_node(&self, node: &mut QuboBBNode) -> NodeProbeResult {
        self.probe_with_budget(
            node,
            self.options.node_probe_candidates,
            self.options.node_probe_max_free,
            self.options.node_probe_max_seconds,
            false,
        )
    }

    pub(super) fn probe_root(&mut self, node: &mut QuboBBNode) -> bool {
        let budget = self.options.root_roof_probe_candidates;
        let seconds = self.options.root_roof_probe_max_seconds;
        if budget == 0
            || !seconds.is_finite()
            || seconds <= 0.0
            || !matches!(
                self.options.node_lower_bound,
                crate::solver_options::NodeLowerBoundSelection::RoofDual
            )
        {
            return false;
        }
        let start = Instant::now();
        loop {
            let remaining = seconds - start.elapsed().as_secs_f64();
            if remaining <= 0.0 {
                return false;
            }
            let before = node.fixed_variables.len();
            // Conditional weak choices are safe within each side. Only the
            // common batch survives the union; no new global relations escape.
            let result = self.probe_with_budget(node, budget, usize::MAX, remaining, true);
            if let Some((solution, value)) = result.candidate {
                self.update_solution_if_better(&solution, value);
            }
            if result.closed {
                return true;
            }
            if node.fixed_variables.len() == before {
                return false;
            }
        }
    }

    fn probe_with_budget(
        &self,
        node: &mut QuboBBNode,
        budget: usize,
        max_free: usize,
        seconds: f64,
        root: bool,
    ) -> NodeProbeResult {
        let mut result = NodeProbeResult::default();
        let free = self.qubo.num_x() - node.fixed_variables.len();
        if budget == 0 || free == 0 || free > max_free || !seconds.is_finite() || seconds <= 0.0 {
            return result;
        }
        let start = Instant::now();
        let expired = || {
            start.elapsed().as_secs_f64() >= seconds
                || self.early_stop
                || crate::branchbound_utils::get_current_time() - self.time_start
                    > self.options.max_time
        };
        let initial_fixed = node.fixed_variables.len();
        let initial_bound = node.lower_bound;
        let mut work = NodeProbingStatistics {
            nodes: 1,
            root_rounds: usize::from(root),
            ..Default::default()
        };
        let candidates = select_probe_candidates(&self.qubo_pp_form, &node.fixed_variables, budget);
        let mut cutoff = self.best_solution_value;
        for variable in candidates {
            if node.fixed_variables.contains_key(&variable) {
                continue;
            }
            if expired() {
                break;
            }
            let mut sides: [Option<PresolvedSide>; 2] = [None, None];
            let mut evaluated = 0;
            for value in 0..2 {
                if expired() {
                    break;
                }
                let mut assumed = node.fixed_variables.clone();
                assumed.insert(variable, value);
                sides[value] = self.presolve_node_with_weak_roof(
                    assumed,
                    root || self.options.roof_dual_weak_persistencies,
                );
                work.assignments += 1;
                evaluated += 1;
                if let Some((fixed, _)) = &sides[value] {
                    if fixed.len() == self.qubo.num_x() {
                        let solution = Array1::from_iter((0..self.qubo.num_x()).map(|i| fixed[&i]));
                        let objective = self.qubo.eval_usize(&solution);
                        cutoff = cutoff.min(objective);
                        result.candidate = Self::better_candidate(
                            result.candidate.take(),
                            Some((solution, objective)),
                        );
                    }
                }
            }
            // An interrupted pair is not a disjunction proof. Any feasible
            // completion found so far is still useful as a primal candidate.
            if evaluated != 2 {
                break;
            }
            for side in &mut sides {
                if side
                    .as_ref()
                    .is_none_or(|(_, bound)| self.prunes_with_incumbent(*bound, cutoff))
                {
                    *side = None;
                    work.failed_literals += 1;
                }
            }
            let [zero, one] = sides;
            match (zero, one) {
                (None, None) => {
                    result.closed = true;
                    break;
                }
                (Some((fixed, bound)), None) | (None, Some((fixed, bound))) => {
                    node.fixed_variables = fixed;
                    node.lower_bound = node.lower_bound.max(bound);
                    result.children = None;
                }
                (Some(zero), Some(one)) => {
                    // min(L0,L1) bounds the union, never max(L0,L1).
                    node.lower_bound = node.lower_bound.max(zero.1.min(one.1));
                    let before = node.fixed_variables.len();
                    for (&i, &value) in &zero.0 {
                        if one.0.get(&i) == Some(&value) {
                            node.fixed_variables.insert(i, value);
                        }
                    }
                    if node.fixed_variables.len() != before {
                        // Component tie choices are local optimum-preserving
                        // reductions, not global implications. Rebase each probe.
                        result.children = None;
                        if let Some((fixed, bound)) = self.presolve_node_with_weak_roof(
                            std::mem::take(&mut node.fixed_variables),
                            root || self.options.roof_dual_weak_persistencies,
                        ) {
                            node.fixed_variables = fixed;
                            node.lower_bound = node.lower_bound.max(bound);
                        } else {
                            result.closed = true;
                            break;
                        }
                    } else if result.children.is_none() {
                        result.children = Some(ProbedChildren {
                            variable,
                            zero,
                            one,
                        });
                    }
                }
            }
            if self.prunes_with_incumbent(node.lower_bound, cutoff) {
                result.closed = true;
                break;
            }
            if node.fixed_variables.len() == self.qubo.num_x() {
                break;
            }
        }
        work.fixed_variables = node.fixed_variables.len().saturating_sub(initial_fixed);
        work.pruned_nodes = usize::from(result.closed);
        work.bound_improvements = usize::from(node.lower_bound > initial_bound + 1e-9);
        if root {
            work.root_assignments = work.assignments;
            work.root_fixed_variables = work.fixed_variables;
        }
        self.probing_counters.record(work);
        result
    }
}
