use super::branch_history::HistoryFeedback;
use super::{BBSolver, QuboBBNode};
use crate::branch_subproblem::ConditionalLowerBound;
use crate::variable_reduction::select_probe_candidates;
use crate::FixedVarMap;
use ndarray::Array1;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

#[derive(Clone, Copy, Debug, Default)]
pub struct NodeProbingStatistics {
    pub cycle_bound_trials: usize,
    pub cycle_bound_improvements: usize,
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
    pub retained_bound_fixings: usize,
    pub retained_bound_prunes: usize,
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
            branch_origin: None,
            probe_bounds: None,
            probe_presolved: None,
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

    #[test]
    fn retained_pairs_share_storage_and_fix_nonbranch_variables_after_cutoff_improves() {
        // f(x)=x0+x1: conditional optima are exactly 0 and 1 for both variables.
        let solver = BBSolver::new(
            Qubo::new_with_c(sprs::CsMat::zero((2, 2)), Array1::ones(2)),
            SolverOptions::new(),
        );
        let mut node = QuboBBNode {
            fixed_variables: FixedVarMap::default(),
            lower_bound: -1.0,
            solution: Array1::from_elem(2, 0.5),
            run_heuristic: false,
            subproblem_state: None,
            branch_origin: None,
            probe_bounds: None,
            probe_presolved: None,
        };
        for variable in 0..2 {
            BBSolver::retain_probe_bound(
                &mut node,
                ConditionalLowerBound {
                    variable,
                    zero: 0.0,
                    one: 1.0,
                },
            );
        }
        assert_eq!(solver.apply_probe_bounds(&mut node, 2.0), (false, false));
        let (mut zero, mut one) = BBSolver::branch(node.clone(), 0, 0.0, node.solution.clone());
        assert!(std::sync::Arc::ptr_eq(
            zero.probe_bounds.as_ref().unwrap(),
            one.probe_bounds.as_ref().unwrap()
        ));
        // The new incumbent fixes x1 even though we branched on x0.
        assert_eq!(solver.apply_probe_bounds(&mut zero, 1.0), (false, true));
        assert_eq!(zero.fixed_variables[&1], 0);
        assert_eq!(zero.solution[1], 0.0);
        assert_eq!(solver.apply_probe_bounds(&mut one, 1.0), (true, false));
        assert!(node.fixed_variables.is_empty());
        BBSolver::retain_probe_bound(
            &mut one,
            ConditionalLowerBound {
                variable: 1,
                zero: 1.0,
                one: 2.0,
            },
        );
        assert!(!std::sync::Arc::ptr_eq(
            zero.probe_bounds.as_ref().unwrap(),
            one.probe_bounds.as_ref().unwrap()
        ));
        assert_eq!(node.probe_bounds.as_ref().unwrap()[1].zero, 0.0);
        assert_eq!(one.probe_bounds.as_ref().unwrap().len(), 2);
        let stats = solver.node_probing_statistics();
        assert_eq!(stats.retained_bound_fixings, 1);
        assert_eq!(stats.retained_bound_prunes, 1);
    }

    #[test]
    fn retained_bound_pruning_records_external_cutoff_certificate() {
        let mut solver = cycle_solver(13, 0.0);
        solver.best_solution_value = 0.0;
        solver.tighten_external_cutoff(-12.5);
        let mut node = presolved_node(&solver, FixedVarMap::default());
        // Both conditional optima of an odd cycle are -12.
        BBSolver::retain_probe_bound(
            &mut node,
            ConditionalLowerBound {
                variable: 0,
                zero: -12.0,
                one: -12.0,
            },
        );
        assert_eq!(solver.apply_probe_bounds(&mut node, 0.0), (true, false));
        assert_eq!(solver.cutoff.discarded_bound(), -12.0);
        assert!(solver.root_constraints.is_empty());
        assert_eq!(solver.best_solution_value, 0.0);
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
        // Keep the root-probe counters in this context rather than reducing it first.
        solver.options.width_region_elimination = false;
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
    fn root_probing_rechecks_after_sdp_incumbent_improves() {
        let mut qubo = Qubo::read_qubo("test_data/be/be100.8.qubo");
        qubo.q = &qubo.q * 2.0;
        for enabled in [false, true] {
            let mut options = SolverOptions::new();
            options.verbose = 0;
            options.root_low_degree_elimination = false;
            options.root_dominant_edge_contraction = false;
            options.width_region_elimination = false;
            options.root_roof_probe_candidates = 1;
            options.root_roof_probe_max_seconds = 10.0;
            options.root_probe_after_relaxation = enabled;
            options.sub_problem_solver =
                crate::branch_subproblem::SubProblemSelection::MixingCutSDP;
            let mut solver = BBSolver::new(qubo.clone(), options);
            solver.time_start = crate::branchbound_utils::get_current_time();
            assert!(solver.initialize_search().is_none());
            assert!(solver.best_solution_value < 0.0);
            assert_eq!(
                solver.node_probing_statistics().root_rounds,
                1 + usize::from(enabled)
            );
            assert!(solver
                .nodes
                .iter()
                .all(|node| node.lower_bound <= -18649.0 + 1e-8));
        }
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
                    options.root_probe_after_relaxation = true;
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
        // This test isolates the first deduction; retry closure is tested separately.
        solver.options.node_probe_retries = false;
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
    fn retries_require_productive_fixings() {
        let mut solver = cycle_solver(13, 1.0);
        solver.options.node_probe_retries = true;
        let mut node = presolved_node(&solver, FixedVarMap::default());
        solver.probe_node(&mut node);
        assert_eq!(solver.node_probing_statistics().assignments, 2);

        let mut solver = cycle_solver(13, 1.0);
        solver.options.node_probe_retries = true;
        solver.warm_start(Array1::from_iter((0..13).map(|i| 1 - i % 2)));
        let mut node = presolved_node(&solver, FixedVarMap::default());
        let expected = minimum(&solver.qubo, &node.fixed_variables);
        let result = solver.probe_node(&mut node);
        assert!(solver.node_probing_statistics().assignments > 2);
        let candidate = result
            .candidate
            .as_ref()
            .map_or(solver.best_solution_value, |(_, value)| {
                solver.best_solution_value.min(*value)
            });
        let remaining = if result.closed {
            f64::INFINITY
        } else {
            minimum(&solver.qubo, &node.fixed_variables)
        };
        assert!((candidate.min(remaining) - expected).abs() < 1e-8);
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
    fn new_backend_candidate_reuses_probe_bounds_before_branching() {
        use crate::branch_subproblem::{SubProblemOptions, SubProblemResult, SubProblemSolver};
        use crate::subproblemsolvers::mixingcut_sdp::MixingCutSubProblemResult;
        struct Candidate;
        impl SubProblemSolver for Candidate {
            fn solve_lower_bound(
                &self,
                _: &BBSolver,
                node: &QuboBBNode,
                _: Option<SubProblemOptions>,
            ) -> Box<dyn SubProblemResult> {
                Box::new(MixingCutSubProblemResult {
                    lower_bound: -20.0,
                    relaxed_solution: node.solution.clone(),
                    candidate_primal_solution: Some(Array1::from_iter((0..13).map(|i| 1 - i % 2))),
                    subproblem_state: None,
                    conditional_bounds: Vec::new(),
                })
            }
        }
        let mut solver = cycle_solver(13, 1.0);
        solver.subproblem_solver = Box::new(Candidate);
        // This test needs the backend candidate, not an exact node closure.
        solver.options.bounded_width_elimination = false;
        let node = presolved_node(&solver, FixedVarMap::default());
        let state = solver.process_node(&node);
        assert_eq!(state.best_update.as_ref().unwrap().1, -11.0);
        assert!(state.branches.is_none());
        let resumed = state.continuation.as_ref().unwrap();
        assert_eq!(resumed.fixed_variables.get(&0), Some(&0));
        assert!(resumed.subproblem_state.is_none());
        assert!(node.fixed_variables.is_empty());
        assert_eq!(solver.node_probing_statistics().subproblem_calls, 1);
        assert_eq!(solver.node_probing_statistics().retained_bound_fixings, 1);
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
        // Preserve the conditional children whose reuse is under test.
        solver.options.bounded_width_elimination = false;
        let node = presolved_node(&solver, FixedVarMap::default());
        let (zero, one) = solver.process_node(&node).branches.unwrap();
        assert_eq!(zero.fixed_variables.get(&0), Some(&0));
        assert_eq!(one.fixed_variables.get(&0), Some(&1));
        assert!((zero.lower_bound + 12.0).abs() < 1e-9);
        assert!((one.lower_bound + 11.0).abs() < 1e-9);
        let stats = solver.node_probing_statistics();
        assert_eq!(stats.subproblem_calls, 1);
        assert_eq!(stats.children_reused, 2);

        // Check actual consumption even when a later probe beats the edge-mass leader.
        let mut different_choices = 0;
        for sample in 0..16 {
            let n = 32;
            let mut q = sprs::TriMat::new((n, n));
            let mut c = Array1::zeros(n);
            for i in 0..n {
                for j in i + 1..n {
                    let w = ((i * 13 + j * 7 + i * j * 3 + sample * 11) % 19) as f64 - 9.0;
                    q.add_triplet(i, j, w);
                    q.add_triplet(j, i, w);
                    c[i] -= 0.5 * w;
                    c[j] -= 0.5 * w;
                }
                c[i] += ((i * 11 + sample) % 9) as f64 - 4.0;
            }
            let mut options = cycle_solver(n, 0.0).options;
            options.branch_strategy = BranchStrategy::ProbeLargestEdges;
            options.node_probe_candidates = 4;
            options.component_decomposition = false;
            let mut solver = BBSolver::new(Qubo::new_with_c(q.to_csr(), c), options);
            solver.subproblem_solver = Box::new(WeakBound);
            let node = presolved_node(&solver, FixedVarMap::default());
            let mut probed = node.clone();
            let result = solver.probe_node(&mut probed);
            if result.closed {
                continue;
            }
            let Some(expected) = result.children else {
                continue;
            };
            different_choices +=
                usize::from(expected.variable != solver.make_branch(&probed).branch_variable);
            let (zero, one) = solver.process_node(&node).branches.unwrap();
            assert_eq!(zero.fixed_variables, expected.zero.0);
            assert_eq!(one.fixed_variables, expected.one.0);
            assert!(zero.lower_bound >= expected.zero.1);
            assert!(one.lower_bound >= expected.one.1);
            assert_eq!(solver.node_probing_statistics().children_reused, 2);
            for mut child in [zero, one] {
                assert!(child.probe_presolved.is_some());
                let cold = solver.presolve_node(child.fixed_variables.clone()).unwrap();
                let reused = solver.presolve_or_reuse_probe(&mut child).unwrap();
                assert_eq!(reused.0, cold.0);
                assert!(reused.1 + 1e-8 >= cold.1);
                assert!(child.probe_presolved.is_none());
            }
            assert_eq!(solver.search_work_statistics().probe_presolve_reuses, 2);
        }
        assert!(different_choices > 0);
    }

    #[test]
    fn zero_budgets_and_time_limits_never_manufacture_a_proof() {
        let mut solver = cycle_solver(13, 1.0);
        solver.branch_strategy = crate::branch_stratagy::BranchStrategy::ProbeLargestEdges;
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
            assert_eq!(solver.make_branch(&node).branch_variable, 0);
        }
    }

    #[test]
    fn probe_ranking_uses_both_bounds_then_component_size_and_fixings() {
        let adjacency = vec![
            vec![(1, 1.0)],
            vec![(0, 1.0), (2, 1.0)],
            vec![(1, 1.0), (3, 1.0)],
            vec![(2, 1.0)],
        ];
        let children = |variable, bounds: [f64; 2]| ProbedChildren {
            variable,
            zero: ([(variable, 0)].into_iter().collect(), bounds[0]),
            one: ([(variable, 1)].into_iter().collect(), bounds[1]),
        };
        let mut ranking = ProbeRanking::new(4, f64::NEG_INFINITY);
        assert!(ranking.consider(&adjacency, &children(0, [-10.0, -5.0])));
        assert!(!ranking.consider(&adjacency, &children(1, [-11.0, 100.0])));
        assert!(ranking.consider(&adjacency, &children(1, [-9.0, -8.0])));
        assert!(!ranking.consider(&adjacency, &children(0, [-9.0, -8.0])));
        assert!(!ranking.consider(&adjacency, &children(2, [-9.0, -8.0])));
        let mut reduced = children(1, [-9.0, -8.0]);
        reduced.zero.0.insert(0, 0);
        reduced.one.0.insert(0, 1);
        assert!(ranking.consider(&adjacency, &reduced));
        ranking.best = None;
        assert!(ranking.consider(&adjacency, &children(0, [-20.0, -20.0])));

        let mut ranking = ProbeRanking::new(4, -1.0);
        assert!(ranking.consider(&adjacency, &children(0, [-2.0, -2.0])));
        // Both candidate bounds are redundant; prefer the smaller residual components.
        assert!(ranking.consider(&adjacency, &children(1, [-10.0, -10.0])));
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
            options.node_cycle_probing = sample % 3 != 0;
            options.node_probe_retries = sample % 2 == 0;
            options.branch_strategy = crate::branch_stratagy::BranchStrategy::ProbeLargestEdges;
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
                    if let Some(bounds) = &node.probe_bounds {
                        for b in bounds.iter() {
                            for (value, bound) in [(0, b.zero), (1, b.one)] {
                                if node
                                    .fixed_variables
                                    .get(&b.variable)
                                    .is_some_and(|&v| v != value)
                                {
                                    continue;
                                }
                                let mut conditioned = node.fixed_variables.clone();
                                conditioned.insert(b.variable, value);
                                assert!(bound <= minimum(&solver.qubo, &conditioned) + 1e-8);
                            }
                        }
                    }
                    let mut revisited = node.clone();
                    let (closed, _) = solver.apply_probe_bounds(&mut revisited, candidate);
                    let left = if closed {
                        f64::INFINITY
                    } else {
                        minimum(&solver.qubo, &revisited.fixed_variables)
                    };
                    assert!((candidate.min(left) - expected).abs() < 1e-8);
                }
                if let Some(children) = result.children {
                    let mut child_optimum = f64::INFINITY;
                    for (map, bound) in [children.zero, children.one] {
                        let optimum = minimum(&solver.qubo, &map);
                        assert!(bound <= optimum + 1e-8);
                        child_optimum = child_optimum.min(optimum);
                    }
                    assert!((candidate.min(child_optimum) - expected).abs() < 1e-8);
                }
            }
        }
    }


    #[test]
    fn conditional_cycles_bound_both_sides_without_sdp_calls() {
        let n = 13;
        let mut terms = sprs::TriMat::new((n, n));
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    terms.add_triplet(i, j, 2.0);
                }
            }
        }
        let q = Qubo::new_with_c(terms.to_csr(), Array1::from_elem(n, -(n as f64 - 1.0)));
        let mut options = SolverOptions::new();
        options.verbose = 0;
        options.node_cycle_probing = true;
        options.node_probe_candidates = 1;
        options.node_probe_max_seconds = 10.0;
        let solver = BBSolver::new(q, options);
        let mut node = presolved_node(&solver, FixedVarMap::default());
        let expected = minimum(&solver.qubo, &node.fixed_variables);
        let result = solver.probe_node(&mut node);
        assert!(!result.closed);
        assert!(node.lower_bound <= expected + 1e-8);
        for b in node.probe_bounds.as_ref().unwrap().iter() {
            for (value, bound) in [(0, b.zero), (1, b.one)] {
                let fixed = [(b.variable, value)].into_iter().collect();
                assert!(bound <= minimum(&solver.qubo, &fixed) + 1e-8);
            }
        }
        assert_eq!(solver.node_probing_statistics().cycle_bound_trials, 2);
        assert_eq!(solver.node_probing_statistics().subproblem_calls, 0);
    }
}

#[derive(Default)]
pub(super) struct NodeProbingCounters {
    cycle_bound_trials: AtomicUsize,
    cycle_bound_improvements: AtomicUsize,
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
    retained_bound_fixings: AtomicUsize,
    retained_bound_prunes: AtomicUsize,
}

impl NodeProbingCounters {
    pub(super) fn take(&self) -> NodeProbingStatistics {
        NodeProbingStatistics {
            cycle_bound_trials: self.cycle_bound_trials.swap(0, Ordering::Relaxed),
            cycle_bound_improvements: self.cycle_bound_improvements.swap(0, Ordering::Relaxed),
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
            retained_bound_fixings: self.retained_bound_fixings.swap(0, Ordering::Relaxed),
            retained_bound_prunes: self.retained_bound_prunes.swap(0, Ordering::Relaxed),
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
        self.retained_bound_fixings
            .fetch_add(work.retained_bound_fixings, Ordering::Relaxed);
        self.retained_bound_prunes
            .fetch_add(work.retained_bound_prunes, Ordering::Relaxed);
    }

    pub(super) fn symmetry_fixing(&self) {
        self.root_symmetry_fixings.fetch_add(1, Ordering::Relaxed);
    }

    fn record(&self, work: NodeProbingStatistics) {
        self.cycle_bound_trials
            .fetch_add(work.cycle_bound_trials, Ordering::Relaxed);
        self.cycle_bound_improvements
            .fetch_add(work.cycle_bound_improvements, Ordering::Relaxed);
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
            cycle_bound_trials: self.cycle_bound_trials.load(Ordering::Relaxed),
            cycle_bound_improvements: self.cycle_bound_improvements.load(Ordering::Relaxed),
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
            retained_bound_fixings: self.retained_bound_fixings.load(Ordering::Relaxed),
            retained_bound_prunes: self.retained_bound_prunes.load(Ordering::Relaxed),
        }
    }
}

type PresolvedSide = (FixedVarMap, f64);

pub(super) struct ProbedChildren {
    pub variable: usize,
    pub zero: PresolvedSide,
    pub one: PresolvedSide,
}

struct ProbeBranchScore {
    weaker_bound: f64,
    stronger_bound: f64,
    largest_component: usize,
    worst_free: usize,
    total_free: usize,
}

impl ProbeBranchScore {
    fn better_than(&self, other: &Self) -> bool {
        self.weaker_bound
            .total_cmp(&other.weaker_bound)
            .then_with(|| self.stronger_bound.total_cmp(&other.stronger_bound))
            .then_with(|| other.largest_component.cmp(&self.largest_component))
            .then_with(|| other.worst_free.cmp(&self.worst_free))
            .then_with(|| other.total_free.cmp(&self.total_free))
            .is_gt()
    }
}

/// Reuse the graph walk storage across candidate pairs; do not materialize components.
struct ProbeRanking {
    visited: Vec<bool>,
    stack: Vec<usize>,
    best: Option<ProbeBranchScore>,
    inherited_bound: f64,
}

impl ProbeRanking {
    fn new(n: usize, inherited_bound: f64) -> Self {
        Self {
            visited: vec![false; n],
            stack: Vec::new(),
            best: None,
            inherited_bound,
        }
    }

    fn largest_component(&mut self, adjacency: &[Vec<(usize, f64)>], fixed: &FixedVarMap) -> usize {
        self.visited.fill(false);
        for &i in fixed.keys() {
            self.visited[i] = true;
        }
        let mut largest = 0;
        for i in 0..adjacency.len() {
            if self.visited[i] {
                continue;
            }
            self.visited[i] = true;
            self.stack.push(i);
            let mut size = 0;
            while let Some(i) = self.stack.pop() {
                size += 1;
                for &(j, _) in &adjacency[i] {
                    if !self.visited[j] {
                        self.visited[j] = true;
                        self.stack.push(j);
                    }
                }
            }
            largest = largest.max(size);
        }
        largest
    }

    fn consider(&mut self, adjacency: &[Vec<(usize, f64)>], children: &ProbedChildren) -> bool {
        let largest = self
            .largest_component(adjacency, &children.zero.0)
            .max(self.largest_component(adjacency, &children.one.0));
        self.consider_with_largest(adjacency.len(), children, largest)
    }

    fn consider_with_largest(
        &mut self,
        n: usize,
        children: &ProbedChildren,
        largest: usize,
    ) -> bool {
        let zero_free = n - children.zero.0.len();
        let one_free = n - children.one.0.len();
        let score = ProbeBranchScore {
            // A cheap bound below the inherited SDP/QP bound offers no new gain.
            weaker_bound: children
                .zero
                .1
                .min(children.one.1)
                .max(self.inherited_bound),
            stronger_bound: children
                .zero
                .1
                .max(children.one.1)
                .max(self.inherited_bound),
            largest_component: largest,
            worst_free: zero_free.max(one_free),
            total_free: zero_free + one_free,
        };
        // Equal scores retain LargestEdges order, including its deterministic tie break.
        if self
            .best
            .as_ref()
            .is_none_or(|best| score.better_than(best))
        {
            self.best = Some(score);
            true
        } else {
            false
        }
    }
}

#[derive(Default)]
pub(super) struct NodeProbeResult {
    pub closed: bool,
    pub candidate: Option<(Array1<usize>, f64)>,
    pub children: Option<ProbedChildren>,
    pub history: HistoryFeedback,
}

impl BBSolver {
    pub(super) fn retain_probe_bound(node: &mut QuboBBNode, bound: ConditionalLowerBound) {
        if !bound.zero.is_finite() || !bound.one.is_finite() {
            return;
        }
        let bounds =
            std::sync::Arc::make_mut(node.probe_bounds.get_or_insert_with(Default::default));
        if let Some(previous) = bounds.iter_mut().find(|b| b.variable == bound.variable) {
            previous.zero = previous.zero.max(bound.zero);
            previous.one = previous.one.max(bound.one);
        } else {
            bounds.push(bound);
        }
    }

    /// Reuse completed probe certificates, including candidates not chosen for branching.
    /// Returns (closed, changed); caller must invalidate pre-fixing solver state/children.
    pub(super) fn apply_probe_bounds(&self, node: &mut QuboBBNode, incumbent: f64) -> (bool, bool) {
        let Some(bounds) = node.probe_bounds.as_ref() else {
            return (false, false);
        };
        let mut changed = false;
        for b in bounds.iter() {
            if b.variable >= self.qubo.num_x() || !b.zero.is_finite() || !b.one.is_finite() {
                continue;
            }
            if let Some(&value) = node.fixed_variables.get(&b.variable) {
                let bound = if value == 0 { b.zero } else { b.one };
                node.lower_bound = node.lower_bound.max(bound);
                if self.prunes_with_incumbent(bound, incumbent) {
                    self.probing_counters
                        .retained_bound_prunes
                        .fetch_add(1, Ordering::Relaxed);
                    return (true, changed);
                }
                continue;
            }
            node.lower_bound = node.lower_bound.max(b.zero.min(b.one));
            let zero = self.prunes_with_incumbent(b.zero, incumbent);
            let one = self.prunes_with_incumbent(b.one, incumbent);
            if zero && one {
                self.probing_counters
                    .retained_bound_prunes
                    .fetch_add(1, Ordering::Relaxed);
                return (true, changed);
            }
            if zero || one {
                let value = usize::from(zero);
                node.fixed_variables.insert(b.variable, value);
                node.solution[b.variable] = value as f64;
                node.lower_bound = node.lower_bound.max(if zero { b.one } else { b.zero });
                self.probing_counters
                    .retained_bound_fixings
                    .fetch_add(1, Ordering::Relaxed);
                changed = true;
            }
        }
        (false, changed)
    }

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
        let adaptive = !root && self.adaptive_probing();
        let outcomes = adaptive && self.options.outcome_probe_selection;
        let candidates = if adaptive {
            let pool_budget = if outcomes {
                budget.saturating_mul(4)
            } else {
                budget
            };
            let pool =
                select_probe_candidates(&self.qubo_pp_form, &node.fixed_variables, pool_budget);
            let available = budget.min(pool.len());
            let (candidates, changed) = if outcomes {
                let (selected, changed, promotions) =
                    self.branch_history.plan_outcomes(pool, budget);
                result.history.pool_promotions += promotions;
                (selected, changed)
            } else {
                self.branch_history.plan(pool, budget)
            };
            result.history.plans = 1;
            result.history.reduced_plans = usize::from(candidates.len() < available);
            result.history.avoided_candidates = available.saturating_sub(candidates.len());
            result.history.history_choices = usize::from(changed);
            candidates
        } else {
            select_probe_candidates(&self.qubo_pp_form, &node.fixed_variables, budget)
        };
        let mut ranking = (!root
            && matches!(
                self.branch_strategy,
                crate::branch_stratagy::BranchStrategy::ProbeLargestEdges
                    | crate::branch_stratagy::BranchStrategy::AdaptiveProbeLargestEdges
            ))
        .then(|| ProbeRanking::new(self.qubo.num_x(), node.lower_bound));
        let mut cutoff = self.best_solution_value;
        let mut parent_component = None;
        let mut candidates = candidates.into_iter();
        let mut round_fixed = node.fixed_variables.len();
        loop {
            if expired() {
                break;
            }
            let variable = match candidates.next() {
                Some(variable) => variable,
                None => {
                    if root
                        || !self.options.node_probe_retries
                        || node.fixed_variables.len() == round_fixed
                        || node.fixed_variables.len() == self.qubo.num_x()
                    {
                        break;
                    }
                    // Only actual fixings justify another round. Keep the same
                    // timer, and select against the newly reduced residual.
                    round_fixed = node.fixed_variables.len();
                    candidates =
                        select_probe_candidates(&self.qubo_pp_form, &node.fixed_variables, budget)
                            .into_iter();
                    let Some(variable) = candidates.next() else {
                        break;
                    };
                    variable
                }
            };
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
                if !root && self.options.node_cycle_probing && !expired() {
                    if let (Some(prepared), Some((fixed, bound))) =
                        (&self.prepared_cycle_bound, &mut sides[value])
                    {
                        let available = (seconds - start.elapsed().as_secs_f64()).min(0.0005);
                        if available > 0.0
                            && fixed.len() < self.qubo.num_x()
                            && !self.prunes_with_incumbent(*bound, cutoff)
                        {
                            let original: Vec<_> = (0..self.qubo.num_x())
                                .filter(|i| !fixed.contains_key(i))
                                .collect();
                            work.cycle_bound_trials += 1;
                            if let Some(cycle) = prepared.conditioned_to(
                                &original,
                                fixed,
                                std::time::Duration::from_secs_f64(available),
                                cutoff.min(self.pruning_upper_bound()),
                            ) {
                                work.cycle_bound_improvements += usize::from(cycle.lower > *bound);
                                *bound = bound.max(cycle.lower);
                            }
                        }
                    }
                }
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
            // Only completed, finite pairs are cached. Infeasible/pruned sides
            // continue through the existing immediate deduction path below.
            if let [Some((_, zero)), Some((_, one))] = &sides {
                Self::retain_probe_bound(
                    node,
                    ConditionalLowerBound {
                        variable,
                        zero: *zero,
                        one: *one,
                    },
                );
            }
            let components = if outcomes {
                let ranking = ranking.as_mut().expect("adaptive probing has a ranking");
                let adjacency = self.prepared_preprocess.adjacency();
                // Probe deductions only extend this map. Reuse the parent walk
                // until new fixings change the residual graph.
                let fixed_count = node.fixed_variables.len();
                let parent = match parent_component {
                    Some((count, largest)) if count == fixed_count => largest,
                    _ => {
                        let largest = ranking.largest_component(adjacency, &node.fixed_variables);
                        parent_component = Some((fixed_count, largest));
                        largest
                    }
                };
                let sizes = std::array::from_fn::<_, 2, _>(|v| {
                    sides[v]
                        .as_ref()
                        .map_or(0, |(fixed, _)| ranking.largest_component(adjacency, fixed))
                });
                Some((parent, sizes))
            } else {
                None
            };
            if adaptive {
                let origin = self.branch_origin(node, variable, node.lower_bound, cutoff);
                for (value, side) in sides.iter().enumerate() {
                    let closed = side
                        .as_ref()
                        .is_none_or(|(_, bound)| self.prunes_with_incumbent(*bound, cutoff));
                    let (fixed, bound) = side.as_ref().map_or(
                        (node.fixed_variables.len() + 1, node.lower_bound),
                        |(fixed, bound)| (fixed.len(), *bound),
                    );
                    result.history.observe(
                        crate::branch_node::BranchOrigin { value, ..origin },
                        bound,
                        fixed,
                        closed,
                        !closed
                            && components.is_some_and(|(parent, sizes)| {
                                // Discount vertex removal itself and pre-existing fragmentation.
                                sizes[value] > 0
                                    && sizes[value]
                                        + fixed.saturating_sub(node.fixed_variables.len())
                                        < parent
                            }),
                        false,
                    );
                }
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
                    if let Some(ranking) = &mut ranking {
                        ranking.best = None;
                        ranking.inherited_bound = node.lower_bound;
                    }
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
                        if let Some(ranking) = &mut ranking {
                            ranking.best = None;
                            ranking.inherited_bound = node.lower_bound;
                        }
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
                    } else {
                        let children = ProbedChildren {
                            variable,
                            zero,
                            one,
                        };
                        let retain = ranking.as_mut().map_or_else(
                            || result.children.is_none(),
                            |ranking| {
                                if let Some((_, sizes)) = components {
                                    ranking.consider_with_largest(
                                        self.qubo.num_x(),
                                        &children,
                                        sizes[0].max(sizes[1]),
                                    )
                                } else {
                                    ranking
                                        .consider(self.prepared_preprocess.adjacency(), &children)
                                }
                            },
                        );
                        if retain {
                            result.children = Some(children);
                        }
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
