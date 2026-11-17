//! Heuristic observations only. Nothing in this module certifies a bound or fixing.
use super::{BBSolver, ProcessNodeState};
use crate::branch_node::BranchOrigin;
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Clone, Copy, Default)]
struct Samples {
    count: usize,
    mean: f64,
}

impl Samples {
    fn observe(&mut self, reward: f64) {
        self.count = self.count.saturating_add(1);
        self.mean += (reward - self.mean) / self.count as f64;
    }
}

#[derive(Clone, Default)]
struct VariableHistory {
    // Keep cheap probe outcomes separate from ordinary solved-child outcomes.
    probes: [Samples; 2],
    children: [Samples; 2],
}

impl VariableHistory {
    fn reliability(&self) -> usize {
        (0..2)
            .map(|v| self.probes[v].count.saturating_add(self.children[v].count))
            .min()
            .unwrap_or(0)
    }

    fn score(&self) -> f64 {
        let reliability = self.reliability();
        if reliability < 4 {
            return 0.0;
        }
        let values = [0, 1].map(|v| match (self.probes[v].count, self.children[v].count) {
            (0, _) => self.children[v].mean,
            (_, 0) => self.probes[v].mean,
            _ => 0.5 * (self.probes[v].mean + self.children[v].mean),
        });
        (values[0].min(values[1]) + 0.25 * values[0].max(values[1])) * reliability as f64
            / (reliability as f64 + 4.0)
    }
}

pub(super) struct Observation {
    variable: usize,
    value: usize,
    child: bool,
    reward: f64,
}

#[derive(Default)]
pub(super) struct BranchHistory {
    variables: Vec<VariableHistory>,
    batches: usize,
}

#[derive(Default)]
pub(super) struct HistoryFeedback {
    pub observations: Vec<Observation>,
    pub plans: usize,
    pub reduced_plans: usize,
    pub avoided_candidates: usize,
    pub history_choices: usize,
    pub pool_promotions: usize,
    pub probe_splits: usize,
    pub bound: f64,
    pub fixed: usize,
    pub split: bool,
}

impl HistoryFeedback {
    pub fn absorb(&mut self, other: Self) {
        if self.observations.is_empty() {
            self.observations = other.observations;
        } else {
            self.observations.extend(other.observations);
        }
        self.plans += other.plans;
        self.reduced_plans += other.reduced_plans;
        self.avoided_candidates += other.avoided_candidates;
        self.history_choices += other.history_choices;
        self.pool_promotions += other.pool_promotions;
        self.probe_splits += other.probe_splits;
    }

    pub fn observe(
        &mut self,
        origin: BranchOrigin,
        bound: f64,
        fixed: usize,
        closed: bool,
        split: bool,
        child: bool,
    ) {
        let gain = if bound.is_finite() && origin.parent_bound.is_finite() {
            ((bound - origin.parent_bound) / origin.gap_scale).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let remaining = origin.size.saturating_sub(fixed);
        // Do not count the branch assignment itself as a presolve inference.
        let inferred = origin.parent_free.saturating_sub(remaining + 1);
        let reduction = inferred as f64 / origin.parent_free.max(1) as f64;
        let reward = reduction + 0.5 * gain + f64::from(closed) + 0.5 * f64::from(split);
        if reward.is_finite() {
            self.probe_splits += usize::from(split && !child);
            self.observations.push(Observation {
                variable: origin.variable,
                value: origin.value,
                child,
                reward,
            });
        }
    }

    pub fn record_child(&mut self, origin: BranchOrigin, state: &ProcessNodeState) {
        self.observe(
            origin,
            self.bound,
            self.fixed,
            state.branches.is_none() && state.continuation.is_none(),
            self.split,
            true,
        );
    }
}

impl BranchHistory {
    /// Spend the full budget, selecting from a wider edge-ranked pool. History
    /// only orders experiments; it never supplies a bound or fixes a variable.
    pub fn plan_outcomes(
        &self,
        mut candidates: Vec<usize>,
        budget: usize,
    ) -> (Vec<usize>, bool, usize) {
        let count = budget.min(candidates.len());
        if count == 0 {
            return (Vec::new(), false, 0);
        }
        let original = candidates[0];
        let original_last = candidates[count - 1];
        // Keep all but one edge-ranked slot: noisy historical rewards should
        // not replace the entire structural shortlist. Stable ties keep edge order.
        let tail = &mut candidates[count - 1..];
        tail.sort_by(|&i, &j| {
            let score = |v| self.variables.get(v).map_or(0.0, VariableHistory::score);
            score(j).total_cmp(&score(i))
        });
        if self.batches > 0 && self.batches % 8 == 0 && candidates.len() > count {
            let explorer = count + (self.batches / 8 - 1) % (candidates.len() - count);
            candidates.swap(count - 1, explorer);
        }
        candidates.truncate(count);
        let changed = candidates[0] != original;
        // Only the final slot can change; candidates contain distinct variables.
        let promotions = usize::from(candidates[count - 1] != original_last);
        (candidates, changed, promotions)
    }

    /// Input order is LargestEdges order. Return a bounded, exploratory shortlist.
    pub fn plan(&self, mut candidates: Vec<usize>, budget: usize) -> (Vec<usize>, bool) {
        if budget == 0 || candidates.is_empty() {
            return (Vec::new(), false);
        }
        let original = candidates[0];
        let score = |i: usize| self.variables.get(i).map_or(0.0, VariableHistory::score);
        let count = |i: usize| {
            self.variables
                .get(i)
                .map_or(0, VariableHistory::reliability)
        };
        candidates.truncate(budget);
        let leader = *candidates
            .iter()
            .min_by(|&&i, &&j| score(j).total_cmp(&score(i)))
            .unwrap();
        let second = candidates
            .iter()
            .copied()
            .filter(|&i| i != leader)
            .min_by(|&i, &j| score(j).total_cmp(&score(i)));
        let refresh = self.batches % 8 == 0;
        let confident = second.is_some_and(|second| {
            count(leader) >= 8
                && candidates.iter().all(|&i| count(i) >= 4)
                && score(leader) > 1.25 * score(second) + 0.05
        });
        if confident && !refresh {
            candidates.clear();
            candidates.push(leader);
            return (candidates, leader != original);
        }
        // Preserve the measured-probe tie break, not just the candidate set.
        (candidates, false)
    }

    pub fn merge(&mut self, n: usize, feedback: HistoryFeedback, counters: &HistoryCounters) {
        // Each reduced/component solver owns its own variable namespace.
        self.variables.resize_with(n, VariableHistory::default);
        counters.plans.fetch_add(feedback.plans, Ordering::Relaxed);
        counters
            .pool_promotions
            .fetch_add(feedback.pool_promotions, Ordering::Relaxed);
        counters
            .probe_splits
            .fetch_add(feedback.probe_splits, Ordering::Relaxed);
        counters
            .reduced_plans
            .fetch_add(feedback.reduced_plans, Ordering::Relaxed);
        counters
            .avoided_candidates
            .fetch_add(feedback.avoided_candidates, Ordering::Relaxed);
        counters
            .history_choices
            .fetch_add(feedback.history_choices, Ordering::Relaxed);
        for observation in feedback.observations {
            if let Some(variable) = self.variables.get_mut(observation.variable) {
                let samples = if observation.child {
                    counters.children.fetch_add(1, Ordering::Relaxed);
                    &mut variable.children
                } else {
                    counters.probes.fetch_add(1, Ordering::Relaxed);
                    &mut variable.probes
                };
                samples[observation.value].observe(observation.reward);
            }
        }
    }

    pub fn finish_batch(&mut self) {
        self.batches = self.batches.wrapping_add(1);
    }
}

#[derive(Default)]
pub(super) struct HistoryCounters {
    pool_promotions: AtomicUsize,
    probe_splits: AtomicUsize,
    probes: AtomicUsize,
    children: AtomicUsize,
    plans: AtomicUsize,
    reduced_plans: AtomicUsize,
    avoided_candidates: AtomicUsize,
    history_choices: AtomicUsize,
}

#[derive(Debug)]
pub struct BranchHistoryStatistics {
    /// Planned slots outside the original edge shortlist, including exploration.
    pub pool_promotions: usize,
    pub probe_split_observations: usize,
    pub probe_observations: usize,
    pub child_observations: usize,
    pub plans: usize,
    pub reduced_budget_nodes: usize,
    /// Planned candidate slots skipped, not a count of measured presolve calls saved.
    pub avoided_probe_candidates: usize,
    pub history_first_choices: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{branch_stratagy::BranchStrategy, qubo::Qubo, solver_options::SolverOptions};
    use ndarray::Array1;

    fn train(history: &mut BranchHistory, variable: usize, reward: f64, count: usize) {
        let mut feedback = HistoryFeedback::default();
        for _ in 0..count {
            for value in 0..2 {
                feedback.observations.push(Observation {
                    variable,
                    value,
                    child: false,
                    reward,
                });
            }
        }
        history.merge(12, feedback, &HistoryCounters::default());
    }

    #[test]
    fn cold_history_preserves_edges_and_respects_budget() {
        let history = BranchHistory::default();
        assert_eq!(history.plan(vec![3, 1, 7, 4], 3), (vec![3, 1, 7], false));
        assert_eq!(history.plan(vec![3, 1], 0), (vec![], false));
        assert_eq!(history.plan(vec![], 3), (vec![], false));
        assert_eq!(history.plan(vec![3], 3), (vec![3], false));
    }

    #[test]
    fn outcome_planning_keeps_budget_promotes_learned_candidates_and_explores() {
        let mut history = BranchHistory::default();
        assert_eq!(
            history.plan_outcomes(vec![0, 1, 2, 3], 2),
            (vec![0, 1], false, 0)
        );
        assert_eq!(history.plan_outcomes(vec![0], 0), (vec![], false, 0));
        assert_eq!(history.plan_outcomes(vec![], 3), (vec![], false, 0));
        train(&mut history, 3, 2.0, 8);
        train(&mut history, 2, 1.0, 8);
        assert_eq!(
            history.plan_outcomes(vec![0, 1, 2, 3], 2),
            (vec![0, 3], false, 1)
        );
        let mut explored = std::collections::HashSet::new();
        for batch in 1..=32 {
            history.finish_batch();
            let (selected, _, _) = history.plan_outcomes(vec![0, 1, 2, 3], 2);
            assert_eq!(selected.len(), 2);
            assert_ne!(selected[0], selected[1]);
            assert_eq!(selected[0], 0);
            if batch % 8 == 0 {
                explored.insert(selected[1]);
            }
        }
        assert_eq!(explored, [1, 2].into_iter().collect());
        assert_eq!(history.plan_outcomes(vec![3], 8).0, vec![3]);
        for _ in 0..16 {
            history.finish_batch();
            let (selected, _, _) = history.plan_outcomes(vec![0, 1, 2, 3], 1);
            assert_eq!(selected.len(), 1);
            assert!(selected[0] < 4);
        }
    }

    #[test]
    fn learned_preferences_cannot_override_disabled_or_expired_probing() {
        use crate::{branch_node::QuboBBNode, FixedVarMap};
        for mode in 0..4 {
            let mut options = SolverOptions::new();
            options.verbose = 0;
            options.branch_strategy = BranchStrategy::AdaptiveProbeLargestEdges;
            options.outcome_probe_selection = true;
            options.node_probe_candidates = if mode == 0 { 0 } else { 3 };
            options.node_probe_max_free = if mode == 1 { 1 } else { 12 };
            options.node_probe_max_seconds = if mode == 2 { 0.0 } else { 10.0 };
            if mode == 3 {
                options.max_time = -1.0;
            }
            let mut solver = BBSolver::new(Qubo::new(sprs::CsMat::eye(12)), options);
            train(&mut solver.branch_history, 0, 2.0, 64);
            let mut node = QuboBBNode {
                lower_bound: -100.0,
                solution: Array1::from_elem(12, 0.5),
                fixed_variables: FixedVarMap::default(),
                run_heuristic: false,
                subproblem_state: None,
                branch_origin: None,
                probe_bounds: None,
                probe_presolved: None,
            };
            let probe = solver.probe_node(&mut node);
            assert!(!probe.closed && probe.children.is_none() && probe.candidate.is_none());
            assert!(probe.history.observations.is_empty());
            assert!(node.fixed_variables.is_empty());
            assert_eq!(node.lower_bound, -100.0);
            assert_eq!(solver.node_probing_statistics().assignments, 0);
        }
    }

    #[test]
    fn reliable_history_reduces_budget_but_refreshes_and_explores() {
        let mut history = BranchHistory::default();
        train(&mut history, 1, 1.0, 8);
        train(&mut history, 0, 0.1, 8);
        train(&mut history, 2, 0.1, 8);
        let pool = vec![0, 1, 2, 3];
        assert_eq!(history.plan(pool.clone(), 3), (vec![0, 1, 2], false));
        for _ in 0..7 {
            history.finish_batch();
            assert_eq!(history.plan(pool.clone(), 3), (vec![1], true));
        }
        history.finish_batch();
        assert_eq!(history.plan(pool.clone(), 3), (vec![0, 1, 2], false));
        history.finish_batch();
        // Do not skip an untested candidate that enters the edge shortlist.
        assert_eq!(history.plan(vec![0, 1, 3], 3), (vec![0, 1, 3], false));
    }

    #[test]
    fn reliability_requires_both_directions_and_can_use_child_outcomes() {
        let mut variable = VariableHistory::default();
        for _ in 0..20 {
            variable.probes[0].observe(2.0);
        }
        assert_eq!(variable.reliability(), 0);
        assert_eq!(variable.score(), 0.0);
        for _ in 0..4 {
            variable.children[1].observe(1.0);
        }
        assert_eq!(variable.reliability(), 4);
        assert!(variable.score() > 0.0);
        let before = variable.score();
        variable.children[1].observe(0.0);
        assert!(variable.score().is_finite());
        assert_ne!(variable.score(), before);
    }

    #[test]
    fn feedback_is_finite_normalized_and_separate_from_bounds() {
        let origin = BranchOrigin {
            variable: 2,
            value: 1,
            parent_bound: -100.0,
            gap_scale: 50.0,
            parent_free: 10,
            size: 12,
        };
        let mut feedback = HistoryFeedback::default();
        feedback.observe(origin, -75.0, 3, false, false, false);
        assert_eq!(feedback.observations[0].reward, 0.25);
        feedback.observe(origin, f64::INFINITY, 5, true, true, true);
        assert_eq!(feedback.observations[1].reward, 1.7);
        feedback.observe(origin, -100.0, 3, false, true, false);
        assert_eq!(feedback.probe_splits, 1);
        assert_eq!(feedback.observations[2].reward, 0.5);
        let counters = HistoryCounters::default();
        let mut history = BranchHistory::default();
        history.merge(12, feedback, &counters);
        assert_eq!(counters.probes.load(Ordering::Relaxed), 2);
        assert_eq!(counters.children.load(Ordering::Relaxed), 1);
        // A different reduced solver has no inherited variable numbering/history.
        let solver = BBSolver::new(Qubo::new(sprs::CsMat::eye(3)), SolverOptions::new());
        assert!(solver.branch_history.variables.is_empty());
        assert_eq!(solver.branch_history_statistics().probe_observations, 0);
    }

    #[test]
    fn child_provenance_is_recorded_and_cleared_on_branch() {
        use crate::branch_node::QuboBBNode;
        let mut options = SolverOptions::new();
        options.verbose = 0;
        options.branch_strategy = BranchStrategy::AdaptiveProbeLargestEdges;
        let solver = BBSolver::new(Qubo::new(sprs::CsMat::eye(3)), options);
        let origin = BranchOrigin {
            variable: 0,
            value: 0,
            parent_bound: -1.0,
            gap_scale: 1.0,
            parent_free: 3,
            size: 3,
        };
        let node = QuboBBNode {
            lower_bound: 0.0,
            solution: Array1::zeros(3),
            fixed_variables: [(0, 0)].into_iter().collect(),
            run_heuristic: false,
            subproblem_state: None,
            branch_origin: Some(origin),
            probe_bounds: None,
            probe_presolved: None,
        };
        let mut feedback = HistoryFeedback::default();
        let state =
            solver.process_node_recorded(std::borrow::Cow::Owned(node.clone()), &mut feedback);
        assert!(state.branches.is_none());
        assert_eq!(feedback.observations.len(), 1);
        assert!(feedback.observations[0].child);
        let (zero, one) = BBSolver::branch(node, 1, -1.0, Array1::zeros(3));
        assert!(zero.branch_origin.is_none() && one.branch_origin.is_none());
        let mut feedback = HistoryFeedback::default();
        solver.process_node_recorded(std::borrow::Cow::Owned(zero), &mut feedback);
        assert!(feedback.observations.iter().all(|o| !o.child));
    }

    #[test]
    fn trained_adaptive_solves_preserve_exhaustive_optima() {
        use smolprng::{JsfLarge, PRNG};
        let mut rng = PRNG {
            generator: JsfLarge::from(186243u64),
        };
        for sample in 0..24 {
            let mut qubo = Qubo::make_random_qubo(12, &mut rng, 0.6);
            qubo.q
                .data_mut()
                .iter_mut()
                .for_each(|v| *v = (*v * 32.0).round());
            qubo.c.mapv_inplace(|v| (v * 8.0).round());
            let (expected, _) = crate::subproblemsolvers::enumerate_qubo::enumerate_solve(&qubo);
            let mut options = SolverOptions::new();
            options.verbose = 0;
            options.threads = 4;
            options.branch_strategy = BranchStrategy::AdaptiveProbeLargestEdges;
            options.node_probe_candidates = 3;
            options.outcome_probe_selection = sample % 2 == 0;
            options.node_probe_max_seconds = 10.0;
            options.sub_problem_solver =
                crate::branch_subproblem::SubProblemSelection::MixingCutSDP;
            let mut solver = BBSolver::new(qubo.clone(), options);
            if solver.initialize_search().is_none() {
                // Deliberately arbitrary preferences: history cannot be trusted as a proof.
                train(&mut solver.branch_history, sample % 12, 2.0, 32);
                train(&mut solver.branch_history, (sample + 1) % 12, 0.1, 8);
                for _ in 0..1000 {
                    if solver.nodes.is_empty() {
                        break;
                    }
                    solver.search_batch(4);
                }
            }
            assert!(solver.nodes.is_empty(), "case {sample}");
            assert!(
                (solver.best_solution_value - expected).abs() < 1e-8,
                "case {sample}"
            );
            assert!((qubo.eval_usize(&solver.best_solution) - expected).abs() < 1e-8);
        }
    }
}

impl BBSolver {
    pub fn branch_history_statistics(&self) -> BranchHistoryStatistics {
        let c = &self.history_counters;
        BranchHistoryStatistics {
            pool_promotions: c.pool_promotions.load(Ordering::Relaxed),
            probe_split_observations: c.probe_splits.load(Ordering::Relaxed),
            probe_observations: c.probes.load(Ordering::Relaxed),
            child_observations: c.children.load(Ordering::Relaxed),
            plans: c.plans.load(Ordering::Relaxed),
            reduced_budget_nodes: c.reduced_plans.load(Ordering::Relaxed),
            avoided_probe_candidates: c.avoided_candidates.load(Ordering::Relaxed),
            history_first_choices: c.history_choices.load(Ordering::Relaxed),
        }
    }
}
