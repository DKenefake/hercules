use crate::constraint::{Constraint, ImplicationGraph};
use crate::early_termination::beck_proof;
use crate::qubo::Qubo;
use crate::FixedVarMap;
use ndarray::Array1;
use rayon::prelude::*;

use crate::branch_node::QuboBBNode;
use crate::branch_stratagy::{BranchResult, BranchStrategy};
use crate::branch_subproblem::{get_sub_problem_solver, SubProblemResult, SubProblemSolver};
use crate::branchbound_utils::{check_integer_feasibility, get_current_time};
use crate::branchboundlogger::SolverOutputLogger;
use crate::lower_bound::li_lower_bound;
use crate::preprocess;
use crate::preprocess::{
    make_sub_problem, prepare_preprocess, preprocess_owned_with_prepared, preprocess_with_prepared,
    PreparedPreprocess,
};
use crate::solver_options::{NodeLowerBoundSelection, SolverOptions};
use crate::subproblemsolvers::roofdual::PreparedRoofDual;
use crate::variable_reduction::{find_pair_dominance, probe_limited_with_prepared};
use std::borrow::Cow;
use std::collections::BinaryHeap;

mod components;
mod cutoff;
mod node_probing;
mod node_reduction;
mod root_reduction;
mod sdp_fixing;
use components::ComponentCounters;
pub use components::{ComponentSearch, ComponentStatistics};
pub use cutoff::CutoffStatistics;
use cutoff::{CutoffCounters, CutoffState};
use node_probing::NodeProbingCounters;
pub use node_probing::NodeProbingStatistics;
use node_reduction::NodeReductionCounters;
pub use node_reduction::NodeReductionStatistics;
use sdp_fixing::SdpFixingCounters;
pub use sdp_fixing::SdpFixingStatistics;

/// Struct for the B&B Solver
pub struct BBSolver {
    input_qubo: Qubo,
    pub qubo: Qubo,
    pub qubo_pp_form: Qubo,
    pub best_solution: Array1<usize>,
    pub best_solution_value: f64,
    pub nodes: BinaryHeap<QuboBBNode>,
    pub nodes_processed: usize,
    pub nodes_solved: usize,
    pub nodes_visited: usize,
    pub time_start: f64,
    pub branch_strategy: BranchStrategy,
    pub subproblem_solver: Box<dyn SubProblemSolver + Sync>,
    prepared_preprocess: PreparedPreprocess,
    prepared_roof_dual: PreparedRoofDual,
    root_symmetry_components: Vec<preprocess::ComplementComponent>,
    pub options: SolverOptions,
    pub early_stop: bool,
    pub solver_logger: SolverOutputLogger,
    pub root_constraints: Vec<Constraint>,
    root_implications: ImplicationGraph,
    probing_counters: NodeProbingCounters,
    pub root_reduction_statistics: preprocess::RootReductionStatistics,
    pub(crate) objective_offset: f64,
    suppress_exit_log: bool,
    component_counters: std::sync::Arc<ComponentCounters>,
    component_depth: usize,
    reduction_counters: std::sync::Arc<NodeReductionCounters>,
    structural_root_reduced: bool,
    cutoff: CutoffState,
    cutoff_counters: std::sync::Arc<CutoffCounters>,
    sdp_fixing_counters: std::sync::Arc<SdpFixingCounters>,
}

pub enum NodeLoggingAction {
    Visited,
    Processed,
    Solved,
}

pub enum PruneAction {
    Prune,
    Dont,
}

pub struct ProcessNodeState {
    pub best_update: Option<(Array1<usize>, f64)>,
    pub branches: Option<(QuboBBNode, QuboBBNode)>,
    pub logging: NodeLoggingAction,
    pub continuation: Option<QuboBBNode>,
    /// Additional (visited, processed, solved) work inside component searches.
    pub component_work: (usize, usize, usize),
}

pub enum SolverResult {
    OptimalSolution(Array1<f64>, f64),
    SubOptimalSolution(Array1<f64>, f64),
}

const ROOT_PROBE_LIMIT: usize = 25;

impl BBSolver {
    fn lower_bound_prunes(&self, lower_bound: f64) -> bool {
        self.prunes_with_incumbent(lower_bound, self.best_solution_value)
    }

    fn bound_closes_gap(lower_bound: f64, incumbent: f64) -> bool {
        let scale = incumbent.abs().max(1.0);
        let tol = 1e-10 * scale;
        lower_bound >= incumbent - tol
    }

    /// Creates a new B&B solver
    pub fn new(qubo: Qubo, options: SolverOptions) -> Self {
        // Check the input polynomial before floating-point convexification.
        // Smaller components are already solved outright by cheap presolve.
        let root_symmetry_components = preprocess::complement_components(&qubo, 11);
        let input_qubo = qubo.clone();
        let qubo = qubo.convex_symmetric_form();
        let num_x = qubo.num_x();
        let subproblem_solver = get_sub_problem_solver(&qubo, &options.sub_problem_solver);
        let branch_strategy = options.branch_strategy;
        let start_time = get_current_time();
        let output_level = options.verbose;
        let pp_form = preprocess::shift_qubo(&qubo);
        let prepared_preprocess = prepare_preprocess(&pp_form, true);
        let prepared_roof_dual = PreparedRoofDual::new(&pp_form);

        Self {
            input_qubo,
            qubo,
            qubo_pp_form: pp_form,
            best_solution: Array1::zeros(num_x),
            best_solution_value: 0.0,
            nodes: BinaryHeap::new(),
            nodes_processed: 0,
            nodes_visited: 0,
            nodes_solved: 0,
            time_start: start_time,
            branch_strategy,
            subproblem_solver,
            prepared_preprocess,
            prepared_roof_dual,
            root_symmetry_components,
            options,
            early_stop: false,
            solver_logger: SolverOutputLogger::new(output_level),
            root_constraints: Vec::new(),
            root_implications: ImplicationGraph::default(),
            probing_counters: NodeProbingCounters::default(),
            root_reduction_statistics: preprocess::RootReductionStatistics::default(),
            objective_offset: 0.0,
            suppress_exit_log: false,
            component_counters: std::sync::Arc::default(),
            component_depth: 0,
            reduction_counters: std::sync::Arc::default(),
            structural_root_reduced: false,
            cutoff: CutoffState::default(),
            cutoff_counters: std::sync::Arc::default(),
            sdp_fixing_counters: std::sync::Arc::default(),
        }
    }

    /// This function is used to warm start the solver with an initial solution if one is not provided
    pub fn warm_start(&mut self, initial_solution: Array1<usize>) {
        let warm_start_value = self.qubo.eval_usize(&initial_solution);
        self.update_solution_if_better(&initial_solution, warm_start_value);
    }

    /// The main solve function of the B&B algorithm
    pub fn solve(&mut self) -> (Array1<usize>, f64) {
        self.time_start = get_current_time();
        self.nodes_visited = 0;
        self.nodes_processed = 0;
        self.nodes_solved = 0;
        self.early_stop = false;
        self.component_counters = std::sync::Arc::default();
        self.reduction_counters = std::sync::Arc::default();
        self.cutoff_counters = std::sync::Arc::default();
        self.sdp_fixing_counters = std::sync::Arc::default();
        if let Some(result) = self.initialize_search() {
            return result;
        }
        self.solver_logger.output_header(self);
        if self.best_solution_value < 0.0 {
            self.solver_logger.output_warm_start_info(self);
        }
        while !self.termination_condition() {
            self.search_batch(self.options.threads.max(1));
            self.solver_logger.generate_output_line(self);
        }
        if !self.suppress_exit_log {
            self.solver_logger.generate_exit_line(self);
        }
        (self.best_solution.clone(), self.best_solution_value)
    }

    /// Initialize once. Component searches subsequently advance existing frontiers.
    fn initialize_search(&mut self) -> Option<(Array1<usize>, f64)> {
        self.nodes.clear();
        self.cutoff.reset_proof();
        if self
            .options
            .fixed_variables
            .iter()
            .any(|(&i, &v)| self.best_solution[i] != v)
        {
            for (&i, &v) in &self.options.fixed_variables {
                self.best_solution[i] = v;
            }
            self.best_solution_value = self.qubo.eval_usize(&self.best_solution);
        }
        self.probing_counters = NodeProbingCounters::default();
        self.root_reduction_statistics = preprocess::RootReductionStatistics::default();
        self.root_constraints.clear();
        self.root_implications = ImplicationGraph::default();
        self.prepared_roof_dual.set_strong_relations(&[]);
        let mut root_fixed = self.options.fixed_variables.clone();
        if self.options.root_complement_symmetry {
            for component in &self.root_symmetry_components {
                if let Some(anchor) = component.anchor_if_unfixed(&root_fixed) {
                    root_fixed.insert(anchor, 0);
                    self.probing_counters.symmetry_fixing();
                }
            }
        }
        let (mut fixed_variables, mut root_bound) = self
            .presolve_node(root_fixed)
            .expect("unconstrained root presolve must retain an optimum");

        let (probe_constraints, probe_fixings) = probe_limited_with_prepared(
            &self.prepared_preprocess,
            &fixed_variables,
            ROOT_PROBE_LIMIT,
        );
        self.root_constraints = probe_constraints.constraints;
        if self.options.root_pair_dominance {
            self.root_constraints.extend(find_pair_dominance(
                &self.prepared_preprocess,
                &fixed_variables,
                std::time::Duration::from_millis(50),
            ));
            self.root_constraints.sort_unstable();
            self.root_constraints.dedup();
        }
        self.root_implications = ImplicationGraph::new(self.qubo.num_x(), &self.root_constraints);
        let use_relation_penalties = self.options.roof_dual_relation_penalties
            && matches!(
                self.options.node_lower_bound,
                NodeLowerBoundSelection::RoofDual
            )
            && !self.root_constraints.is_empty();
        if use_relation_penalties {
            self.prepared_roof_dual
                .set_strong_relations(&self.root_constraints);
        }
        if !probe_fixings.is_empty() || use_relation_penalties || self.options.root_pair_dominance {
            fixed_variables.extend(probe_fixings);
            let (closed, bound) = self
                .presolve_node(fixed_variables)
                .expect("strict root probe deductions must retain an optimum");
            fixed_variables = closed;
            root_bound = root_bound.max(bound);
        }
        if self.options.root_low_degree_elimination
            || self.options.root_dominant_edge_contraction
            || (self.options.root_cut_dominance
                && self.component_depth == 0
                && !self.structural_root_reduced)
            || (self.options.small_block_elimination
                && self.component_depth == 0
                && !self.structural_root_reduced)
        {
            if let Some(result) = self.try_reduced_root(&fixed_variables, root_bound) {
                return Some(result);
            }
        }
        self.root_reduction_statistics.remaining = self.qubo.num_x() - fixed_variables.len();
        // create the root node
        let mut root_node = QuboBBNode {
            lower_bound: root_bound,
            solution: 0.5 * Array1::ones(self.qubo.num_x()), // initial guess is 0.5 for all variables
            fixed_variables,
            run_heuristic: true,
            subproblem_state: None,
        };

        let mut root_closed = self.probe_root(&mut root_node);
        self.options
            .fixed_variables
            .clone_from(&root_node.fixed_variables);
        if !root_closed {
            if let Some(search) = self.make_component_search(&root_node) {
                root_node.subproblem_state =
                    Some(crate::branch_node::SubProblemNodeState::Components(search));
            }
        }
        // A component root is processed immediately by its first bounded slice;
        // its presolve bound suffices until then. Avoid solving it twice.
        if self.component_depth == 0
            && !root_closed
            && !matches!(
                root_node.subproblem_state,
                Some(crate::branch_node::SubProblemNodeState::Components(_))
            )
        {
            let mut root_result = self.solve_node(&root_node);
            let conditional_bounds = root_result.take_conditional_bounds();
            let (
                root_lower_bound,
                root_relaxed_solution,
                root_primal_solution,
                root_subproblem_state,
            ) = root_result.into_parts();
            root_node.lower_bound = root_node.lower_bound.max(root_lower_bound);
            if let Some(solution) = root_relaxed_solution {
                root_node.solution = solution;
            }
            root_node.subproblem_state = root_subproblem_state;
            if let Some(solution) = root_primal_solution {
                let value = self.qubo.eval_usize(&solution);
                self.update_solution_if_better(&solution, value);
            }
            root_closed = self
                .apply_sdp_bounds(
                    &mut root_node,
                    &conditional_bounds,
                    self.best_solution_value,
                )
                .0;
        }

        if root_closed {
            // The disjunction was proved directly; do not manufacture an
            // incumbent-valued lower bound just to prune the root via the queue.
            self.nodes.clear();
            self.apply_logging_action(NodeLoggingAction::Visited);
            self.apply_logging_action(NodeLoggingAction::Processed);
        } else {
            self.nodes.push(root_node);
        }

        None
    }

    fn search_batch(&mut self, limit: usize) {
        let nodes = self.get_next_nodes(limit);
        let results = nodes
            .into_par_iter()
            .map(|node| self.process_node_inner(Cow::Owned(node)))
            .collect::<Vec<_>>();
        for state in results {
            self.apply_process_result(state);
        }
    }

    /// Checks if we can prune the node, based on the lower bound and best solution, returns an action
    pub fn can_prune_action(
        &self,
        node: &QuboBBNode,
    ) -> (PruneAction, Option<(Array1<usize>, f64)>) {
        if self
            .root_constraints
            .iter()
            .any(|constraint| !constraint.check(&node.fixed_variables))
        {
            return (PruneAction::Prune, None);
        }

        // if our parent solution is above our current feasible soltion then prune
        if self.lower_bound_prunes(node.lower_bound) {
            return (PruneAction::Prune, None);
        }

        // if the solution is complete, then we can update the best solution if better
        // we can also prune the node, as there are no more variables to fix
        if node.fixed_variables.len() == self.qubo.num_x() {
            // generate the solution vector
            let mut solution = Array1::zeros(self.qubo.num_x());
            for (&index, &value) in &node.fixed_variables {
                solution[index] = value;
            }

            let value = self.qubo.eval_usize(&solution);
            // evaluate the solution against the best solution we have so far
            // if we have a better solution update it
            return (PruneAction::Prune, Some((solution, value)));
        }

        (PruneAction::Dont, None)
    }

    // apply the logging action to the solver
    pub const fn apply_logging_action(&mut self, action: NodeLoggingAction) {
        match action {
            NodeLoggingAction::Visited => {
                // increment the number of nodes visited
                self.nodes_visited += 1;
            }
            NodeLoggingAction::Processed => {
                // increment the number of nodes processed
                self.nodes_processed += 1;
            }
            NodeLoggingAction::Solved => {
                // increment the number of nodes solved and processed
                self.nodes_processed += 1;
                self.nodes_solved += 1;
            }
        }
    }

    /// main loop of the branch and bound algorithm
    pub fn process_node(&self, node: &QuboBBNode) -> ProcessNodeState {
        self.process_node_inner(Cow::Borrowed(node))
    }

    fn process_node_inner(&self, node: Cow<'_, QuboBBNode>) -> ProcessNodeState {
        let (prune_action, event) = self.can_prune_action(&node);

        // if we are pruning at this stage, then we can early return without cloning the node
        if matches!(prune_action, PruneAction::Prune) {
            return ProcessNodeState {
                best_update: event,
                branches: None,
                logging: NodeLoggingAction::Processed,
                continuation: None,
                component_work: (0, 0, 0),
            };
        }

        // The solve loop transfers ownership; borrowed callers clone only if needed.
        let mut node = node.into_owned();
        if matches!(
            node.subproblem_state,
            Some(crate::branch_node::SubProblemNodeState::Components(_))
        ) {
            return self.advance_component_node(node, None);
        }

        let Some((fixed, node_bound)) =
            self.presolve_node(std::mem::take(&mut node.fixed_variables))
        else {
            return ProcessNodeState {
                best_update: None,
                branches: None,
                logging: NodeLoggingAction::Processed,
                continuation: None,
                component_work: (0, 0, 0),
            };
        };
        node.fixed_variables = fixed;
        node.lower_bound = node.lower_bound.max(node_bound);

        // with this expanded set, can we prune the node?
        let (prune_action, event) = self.can_prune_action(&node);

        // if we are pruning at this stage, then we can early return
        if matches!(prune_action, PruneAction::Prune) {
            return ProcessNodeState {
                best_update: event,
                branches: None,
                logging: NodeLoggingAction::Processed,
                continuation: None,
                component_work: (0, 0, 0),
            };
        }

        let probe = self.probe_node(&mut node);
        let (prune_action, event) = self.can_prune_action(&node);
        let probe_update = Self::better_candidate(probe.candidate, event);
        if probe.closed || matches!(prune_action, PruneAction::Prune) {
            return ProcessNodeState {
                best_update: probe_update,
                branches: None,
                logging: NodeLoggingAction::Processed,
                continuation: None,
                component_work: (0, 0, 0),
            };
        }

        if let Some(state) = self.try_reduced_node(&node, probe_update.as_ref()) {
            return state;
        }
        if let Some(search) = self.make_component_search(&node) {
            node.subproblem_state =
                Some(crate::branch_node::SubProblemNodeState::Components(search));
            return self.advance_component_node(node, probe_update);
        }

        // An inherited integral point is a candidate, not an optimality proof
        // after new node fixings. Fathom only when a valid bound closes its gap.
        let (is_int_feasible, rounded_sol) = check_integer_feasibility(&node);
        let integer_update = is_int_feasible.then(|| {
            let value = self.qubo.eval_usize(&rounded_sol);
            (rounded_sol, value)
        });
        if integer_update
            .as_ref()
            .is_some_and(|(_, value)| self.prunes_with_incumbent(node.lower_bound, *value))
        {
            return ProcessNodeState {
                best_update: Self::better_candidate(probe_update, integer_update),
                branches: None,
                logging: NodeLoggingAction::Solved,
                continuation: None,
                component_work: (0, 0, 0),
            };
        }
        let probe_update = Self::better_candidate(probe_update, integer_update);

        // We now need to solve the node to generate the lower bound and solution
        let mut solve_result = self.solve_node(&node);
        let conditional_bounds = solve_result.take_conditional_bounds();
        let (
            solve_lower_bound,
            solve_relaxed_solution,
            solve_primal_solution,
            solve_subproblem_state,
        ) = solve_result.into_parts();
        let lower_bound = solve_lower_bound.max(node.lower_bound);

        // inject the relaxed solution back into the node when available
        if let Some(solution) = solve_relaxed_solution {
            node.solution = solution;
        }
        node.subproblem_state = solve_subproblem_state;

        let primal_update = Self::better_candidate(
            probe_update,
            solve_primal_solution.map(|solution| {
                let value = self.qubo.eval_usize(&solution);
                (solution, value)
            }),
        );

        let incumbent = primal_update
            .as_ref()
            .map_or(self.best_solution_value, |(_, value)| {
                self.best_solution_value.min(*value)
            });
        // The new bound may already close this node. Keep its primal candidate,
        // but do not pay for branching, child copies or another local search.
        if self.prunes_with_incumbent(lower_bound, incumbent) {
            return ProcessNodeState {
                best_update: primal_update,
                branches: None,
                logging: NodeLoggingAction::Solved,
                continuation: None,
                component_work: (0, 0, 0),
            };
        }

        if !conditional_bounds.is_empty() {
            node.lower_bound = lower_bound;
            let (closed, changed) =
                self.apply_sdp_bounds(&mut node, &conditional_bounds, incumbent);
            if closed || changed {
                // Re-enter cheap presolve/structural reduction before another SDP.
                // Do not reuse pre-fixing probed children or stale backend state.
                node.subproblem_state = None;
                return ProcessNodeState {
                    best_update: primal_update,
                    branches: None,
                    logging: NodeLoggingAction::Solved,
                    continuation: (!closed).then_some(node),
                    component_work: (0, 0, 0),
                };
            }
        }
        let lower_bound = lower_bound.max(node.lower_bound);

        // determine what variable we are branching on
        let branch_result = self.make_branch(&node);

        // we now apply the new fixed variables to the base node before we branch
        for (&index, &value) in &branch_result.found_fixed_vars {
            node.fixed_variables.insert(index, value);
        }

        // if we have now fixed all variables, we can check if we have a solution
        if node.fixed_variables.len() == self.qubo.num_x() {
            // generate the solution vector
            let mut solution = Array1::zeros(self.qubo.num_x());
            for (&index, &value) in &node.fixed_variables {
                solution[index] = value;
            }

            let value = self.qubo.eval_usize(&solution);
            // evaluate the solution against the best solution we have so far
            // if we have a better solution update it
            return ProcessNodeState {
                best_update: Self::better_candidate(primal_update, Some((solution, value))),
                branches: None,
                logging: NodeLoggingAction::Solved,
                continuation: None,
                component_work: (0, 0, 0),
            };
        }

        let best_update = node
            .run_heuristic
            .then(|| self.options.heuristic.make_heuristic(self, &node));

        let best_update = match (primal_update, best_update) {
            (Some(primal), Some(heuristic)) => Some(if primal.1 <= heuristic.1 {
                primal
            } else {
                heuristic
            }),
            (Some(primal), None) => Some(primal),
            (None, heuristic) => heuristic,
        };

        // generate the branches
        let branch_solution = std::mem::take(&mut node.solution);
        let (mut zero_branch, mut one_branch) = Self::branch(
            node,
            branch_result.branch_variable,
            lower_bound,
            branch_solution,
        );
        if branch_result.found_fixed_vars.is_empty() {
            if let Some(children) = probe
                .children
                .filter(|children| children.variable == branch_result.branch_variable)
            {
                zero_branch.fixed_variables = children.zero.0;
                zero_branch.lower_bound = zero_branch.lower_bound.max(children.zero.1);
                one_branch.fixed_variables = children.one.0;
                one_branch.lower_bound = one_branch.lower_bound.max(children.one.1);
                self.probing_counters.reuse_children();
            }
        }

        ProcessNodeState {
            best_update,
            branches: Some((zero_branch, one_branch)),
            logging: NodeLoggingAction::Solved,
            continuation: None,
            component_work: (0, 0, 0),
        }
    }

    pub fn apply_process_result(&mut self, state: ProcessNodeState) {
        self.nodes_visited += state.component_work.0;
        self.nodes_processed += state.component_work.1;
        self.nodes_solved += state.component_work.2;
        if let Some((solution, value)) = state.best_update {
            self.update_solution_if_better(&solution, value);
        }

        if let Some((zero_branch, one_branch)) = state.branches {
            // only add the branches if their lower bound is better than the current best solution
            if !self.lower_bound_prunes(zero_branch.lower_bound) {
                self.nodes.push(zero_branch);
            }
            if !self.lower_bound_prunes(one_branch.lower_bound) {
                self.nodes.push(one_branch);
            }
        }

        if let Some(node) = state.continuation {
            if !self.lower_bound_prunes(node.lower_bound) {
                self.nodes.push(node);
            }
        }

        self.apply_logging_action(state.logging);
    }

    /// update the best solution if better than the current best solution
    pub fn update_solution_if_better(&mut self, solution: &Array1<usize>, solution_value: f64) {
        if solution_value < self.best_solution_value {
            self.best_solution.clone_from(solution);
            self.best_solution_value = solution_value;
            let best_solution_value = self.best_solution_value;
            let tol = if self.cutoff.is_tighter_than(best_solution_value) {
                0.0
            } else {
                1e-10 * best_solution_value.abs().max(1.0)
            };

            // if we have an early stopping condition, then we can check if we have a solution
            // let beck_proof = beck_proof(&self.qubo, &self.best_solution);
            //
            // // if we have a beck proof, then we can stop early
            // if beck_proof {
            //     self.early_stop = true;
            //     self.solver_logger.early_termination();
            // }

            // We can remove all nodes that are worse than the current best solution
            self.nodes.retain(|node| {
                // if the node's lower bound is worse than the best solution, we can prune it
                node.lower_bound < best_solution_value - tol
            });
        }
    }

    /// This function is used to get the next node to process, popping it from the list of nodes
    pub fn get_next_node(&mut self) -> Option<QuboBBNode> {
        while !self.nodes.is_empty() {
            // we pull a node from our node list
            let optional_node = self.nodes.pop();

            // check and unwrap the node if it is safe
            let mut node = optional_node?;

            // we increment the number of nodes we have visited
            self.apply_logging_action(NodeLoggingAction::Visited);
            node.run_heuristic |= self.nodes_visited.is_multiple_of(31);

            // if we can't prune it, then we return it
            let (prune, best_update) = self.can_prune_action(&node);

            // if we have stumbled into a better solution, then we can take it
            if let Some((solution, value)) = best_update {
                self.update_solution_if_better(&solution, value);
            }

            // if we don't prune the node then we can return it
            if matches!(prune, PruneAction::Dont) {
                return Some(node);
            }
        }

        None
    }

    pub fn get_next_nodes(&mut self, n: usize) -> Vec<QuboBBNode> {
        let mut nodes = Vec::with_capacity(n);

        // loop while we haven't filled our vector OR the node list is not empty
        while nodes.len() < n {
            let next_node = self.get_next_node();

            // if there is a node to add, do so, else break out as there aren't any nodes left
            if let Some(node) = next_node {
                nodes.push(node);
            } else {
                break;
            }
        }

        nodes
    }

    /// Checks for termination conditions of the B&B algorithm, such as time limit or no more nodes
    pub fn termination_condition(&self) -> bool {
        // get current time to check if we have exceeded the maximum time
        let current_time = get_current_time();

        // check if we violated the time limit
        if current_time - self.time_start > self.options.max_time {
            return true;
        }

        // check if we have no more nodes to process
        if self.nodes.is_empty() {
            return true;
        }

        // if we have an early stopping condition, then we can check if we have a solution
        if self.early_stop {
            return true;
        }

        false
    }

    /// Branch Selection Strategy - Currently selects the first variable that is not fixed
    pub fn make_branch(&self, node: &QuboBBNode) -> BranchResult {
        self.branch_strategy.make_branch(self, node)
    }

    /// Actually branches the node into two new nodes
    pub fn branch(
        mut node: QuboBBNode,
        branch_id: usize,
        lower_bound: f64,
        solution: Array1<f64>,
    ) -> (QuboBBNode, QuboBBNode) {
        // Public callers may choose to branch an AND node instead of resuming it.
        // Its shared search has different domains and cannot belong to either child.
        if matches!(
            node.subproblem_state,
            Some(crate::branch_node::SubProblemNodeState::Components(_))
        ) {
            node.subproblem_state = None;
        }
        node.solution = solution;
        node.lower_bound = lower_bound;
        node.run_heuristic = false;
        let mut zero_branch = node.clone();
        let mut one_branch = node;

        // add fixed variables
        zero_branch.fixed_variables.insert(branch_id, 0);
        one_branch.fixed_variables.insert(branch_id, 1);
        (zero_branch, one_branch)
    }

    pub fn solve_node(&self, node: &QuboBBNode) -> Box<dyn SubProblemResult> {
        self.probing_counters.relaxation();
        self.subproblem_solver.solve_lower_bound(self, node, None)
    }

    pub fn node_probing_statistics(&self) -> NodeProbingStatistics {
        self.probing_counters.snapshot()
    }

    fn better_candidate(
        left: Option<(Array1<usize>, f64)>,
        right: Option<(Array1<usize>, f64)>,
    ) -> Option<(Array1<usize>, f64)> {
        match (left, right) {
            (Some(a), Some(b)) => Some(if a.1 <= b.1 { a } else { b }),
            (Some(a), None) => Some(a),
            (None, b) => b,
        }
    }

    pub fn preprocess_fixed_variables(&self, fixed_variables: &FixedVarMap) -> FixedVarMap {
        preprocess_with_prepared(&self.prepared_preprocess, fixed_variables)
    }

    fn presolve_node(&self, fixed: FixedVarMap) -> Option<(FixedVarMap, f64)> {
        self.presolve_node_with_weak_roof(fixed, self.options.roof_dual_weak_persistencies)
    }

    fn presolve_node_with_weak_roof(
        &self,
        mut fixed: FixedVarMap,
        weak_roof: bool,
    ) -> Option<(FixedVarMap, f64)> {
        let mut bound = f64::NEG_INFINITY;
        let mut have_bound = false;
        loop {
            let before_cheap = fixed.len();
            // Reach cheap closure before paying for another roof-dual solve.
            loop {
                if !self.root_implications.propagate(&mut fixed) {
                    return None;
                }
                let before_preprocess = fixed.len();
                fixed = preprocess_owned_with_prepared(&self.prepared_preprocess, fixed);
                let count = fixed.len();
                if count == before_preprocess {
                    break;
                }
                if !self.root_implications.propagate(&mut fixed) {
                    return None;
                }
                if fixed.len() == count {
                    break;
                }
            }
            if have_bound && fixed.len() == before_cheap {
                return Some((fixed, bound));
            }
            let count = fixed.len();
            bound = bound.max(self.apply_node_lower_bound(&mut fixed, weak_roof));
            have_bound = true;
            if fixed.len() == count {
                return Some((fixed, bound));
            }
            // Roof-dual fixings can disconnect components or trigger implications.
        }
    }

    fn beck_proof_node_candidate(&self, node: &QuboBBNode) -> Option<(Array1<usize>, f64)> {
        let (sub_qubo, mapping, _constant) = make_sub_problem(&self.qubo, &node.fixed_variables);
        if sub_qubo.num_x() == 0 {
            return None;
        }

        let mut reduced_candidate = Array1::zeros(sub_qubo.num_x());
        let mut full_solution = self.best_solution.clone();
        for (&index, &value) in &node.fixed_variables {
            full_solution[index] = value;
        }

        for (&original_index, &reduced_index) in &mapping {
            reduced_candidate[reduced_index] = full_solution[original_index];
        }

        if !beck_proof(&sub_qubo, &reduced_candidate) {
            return None;
        }

        let value = self.qubo.eval_usize(&full_solution);
        Some((full_solution, value))
    }

    fn apply_node_lower_bound(&self, fixed_variables: &mut FixedVarMap, weak_roof: bool) -> f64 {
        match self.options.node_lower_bound {
            NodeLowerBoundSelection::Li => li_lower_bound(&self.qubo, fixed_variables),
            NodeLowerBoundSelection::RoofDual => {
                let roof_result = if weak_roof {
                    self.prepared_roof_dual
                        .solve_iterative_with_weak_persistencies(fixed_variables, self.qubo.num_x())
                } else {
                    self.prepared_roof_dual
                        .solve_iterative(fixed_variables, self.qubo.num_x())
                };
                for (index, value) in roof_result.fixed_variables {
                    fixed_variables.insert(index, value);
                }
                self.probing_counters
                    .weak_roof_fixings(roof_result.weak_fixed_variables.len());
                // These choices preserve some optimum jointly; they must stay
                // local to this node, not enter the global implication graph.
                fixed_variables.extend(roof_result.weak_fixed_variables);
                roof_result.lower_bound.unwrap_or(f64::NEG_INFINITY)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::branch_stratagy::BranchStrategy;
    use crate::branch_subproblem::SubProblemSelection;
    use crate::preprocess::preprocess_qubo;
    use crate::qubo::Qubo;
    use crate::solver_options::SolverOptions;
    use crate::tests::make_test_prng;
    use crate::FixedVarMap as HashMap;
    use crate::{branchbound, local_search};
    use ndarray::Array1;
    use sprs::CsMat;

    pub fn get_default_solver_options() -> SolverOptions {
        let mut options = SolverOptions::new();
        options.verbose = 1;
        options.max_time = 1000.0;
        options.threads = 20;
        options
    }

    #[test]
    pub fn branch_bound_test() {
        let mut prng = make_test_prng();
        let eye = CsMat::eye(3);
        let c = Array1::from_vec(vec![-1.1, -2.0, -3.0]);
        let p = Qubo::new_with_c(eye, c);

        let guess = local_search::particle_swarm_search(&p, 100, 1000, &mut prng);
        let mut solver = branchbound::BBSolver::new(p, SolverOptions::new());
        solver.warm_start(guess);
        solver.solve();

        assert_eq!(solver.best_solution, Array1::from_vec(vec![1, 1, 1]));
    }

    #[test]
    fn children_own_independent_updated_solutions_and_inherit_fixings() {
        let node = crate::branch_node::QuboBBNode {
            lower_bound: -20.0,
            solution: ndarray::array![0.5, 0.5],
            fixed_variables: [(0, 1)].into_iter().collect(),
            run_heuristic: true,
            subproblem_state: None,
        };
        let point = ndarray::array![1.0, 0.75];
        let allocation = point.as_ptr();
        let (mut zero, one) = branchbound::BBSolver::branch(node, 1, -10.0, point);
        assert_eq!(one.solution.as_ptr(), allocation);
        assert_ne!(zero.solution.as_ptr(), one.solution.as_ptr());
        assert_eq!(zero.solution, one.solution);
        assert_eq!(zero.fixed_variables, [(0, 1), (1, 0)].into_iter().collect());
        assert_eq!(one.fixed_variables, [(0, 1), (1, 1)].into_iter().collect());
        assert_eq!(zero.lower_bound, -10.0);
        assert_eq!(one.lower_bound, -10.0);
        assert!(!zero.run_heuristic && !one.run_heuristic);
        zero.solution[1] = 0.0;
        assert_eq!(one.solution[1], 0.75);
    }

    #[test]
    fn fresh_bound_closes_node_before_strong_branching_and_keeps_primal() {
        use crate::branch_node::QuboBBNode;
        use crate::branch_subproblem::{SubProblemOptions, SubProblemResult, SubProblemSolver};
        use crate::subproblemsolvers::mixingcut_sdp::MixingCutSubProblemResult;
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        };

        struct BoundSolver {
            calls: Arc<AtomicUsize>,
            bound: f64,
            candidate: Option<Array1<usize>>,
        }
        impl SubProblemSolver for BoundSolver {
            fn solve_lower_bound(
                &self,
                _: &branchbound::BBSolver,
                node: &QuboBBNode,
                _: Option<SubProblemOptions>,
            ) -> Box<dyn SubProblemResult> {
                assert_eq!(
                    self.calls.fetch_add(1, Ordering::Relaxed),
                    0,
                    "a closed node should not launch strong-branching solves"
                );
                Box::new(MixingCutSubProblemResult {
                    lower_bound: self.bound,
                    relaxed_solution: node.solution.clone(),
                    candidate_primal_solution: self.candidate.clone(),
                    subproblem_state: None,
                    conditional_bounds: Vec::new(),
                })
            }
        }
        // Odd-cycle max cut: roof bound -11, exact value -10, no small component.
        let mut terms = sprs::TriMat::new((11, 11));
        for i in 0..11 {
            let j = (i + 1) % 11;
            terms.add_triplet(i, j, 2.0);
            terms.add_triplet(j, i, 2.0);
        }
        for with_candidate in [false, true] {
            let qubo = Qubo::new_with_c(terms.to_csr(), Array1::from_elem(11, -2.0));
            let mut options = SolverOptions::new();
            options.verbose = 0;
            options.branch_strategy = BranchStrategy::FullStrongBranching;
            options.node_probe_candidates = 0;
            options.node_structural_reductions = false;
            let mut solver = branchbound::BBSolver::new(qubo, options);
            let candidate = Array1::from_iter((0..11).map(|i| i % 2));
            let optimum = solver.qubo.eval_usize(&candidate);
            assert!((optimum + 10.0).abs() < 1e-9);
            if !with_candidate {
                solver.warm_start(candidate.clone());
            }
            let calls = Arc::new(AtomicUsize::new(0));
            solver.subproblem_solver = Box::new(BoundSolver {
                calls: calls.clone(),
                bound: optimum,
                candidate: with_candidate.then_some(candidate.clone()),
            });
            let node = QuboBBNode {
                lower_bound: f64::NEG_INFINITY,
                // A feasible inherited integer point is not a proof that this
                // node is solved. The fresh bound must still be obtained.
                solution: Array1::zeros(11),
                fixed_variables: HashMap::default(),
                run_heuristic: true,
                subproblem_state: None,
            };
            let state = if with_candidate {
                solver.process_node_inner(std::borrow::Cow::Owned(node))
            } else {
                solver.process_node(&node)
            };
            assert_eq!(calls.load(Ordering::Relaxed), 1);
            assert!(state.branches.is_none());
            assert!(matches!(state.logging, super::NodeLoggingAction::Solved));
            if with_candidate {
                assert_eq!(state.best_update.as_ref().unwrap().0, candidate);
            }
            solver.apply_process_result(state);
            assert_eq!(solver.best_solution_value, optimum);
        }
    }

    #[test]
    fn root_node_preserves_heuristic_flag_on_first_pop() {
        let p = Qubo::new(CsMat::eye(2));
        let mut solver = branchbound::BBSolver::new(p, SolverOptions::new());

        solver.nodes.push(crate::branch_node::QuboBBNode {
            lower_bound: f64::NEG_INFINITY,
            solution: 0.5 * Array1::ones(2),
            fixed_variables: HashMap::default(),
            run_heuristic: true,
            subproblem_state: None,
        });

        let node = solver.get_next_node().expect("expected root node");
        assert!(node.run_heuristic);
    }

    #[test]
    fn root_consumes_common_probe_fixings() {
        // x0+x1=1 is favored by a strong interaction; under either x0
        // assumption x2 must be zero. Leaves keep the component above size 10.
        let mut terms = sprs::TriMat::new((13, 13));
        for (i, j, coefficient) in [(0, 1, 32.0), (0, 2, 1.0), (1, 2, 1.0)] {
            terms.add_triplet(i, j, coefficient);
            terms.add_triplet(j, i, coefficient);
        }
        for j in 3..13 {
            terms.add_triplet(0, j, 1.0);
            terms.add_triplet(j, 0, 1.0);
        }
        let mut linear = Array1::from_elem(13, -0.5);
        linear[0] = -16.0;
        linear[1] = -16.0;
        let qubo = Qubo::new_with_c(terms.to_csr(), linear);
        let mut options = SolverOptions::new();
        options.verbose = 0;
        options.node_lower_bound = crate::solver_options::NodeLowerBoundSelection::Li;
        options.sub_problem_solver = SubProblemSelection::MixingCutSDP;
        let mut solver = branchbound::BBSolver::new(qubo, options);
        let (initial, _) = solver.presolve_node(HashMap::default()).unwrap();
        assert!(!initial.contains_key(&2));
        let (_, objective) = solver.solve();
        assert_eq!(solver.options.fixed_variables.get(&2), Some(&0));
        assert!((objective + 21.0).abs() < 1e-8);
        assert!(solver.nodes.is_empty());
    }

    #[test]
    fn node_presolve_propagates_root_relations_before_solving() {
        use crate::constraint::{Constraint, ConstraintType, ImplicationGraph};
        let qubo = Qubo::new_with_c(CsMat::zero((3, 3)), ndarray::array![1.0, 0.0, 0.0]);
        let mut solver = branchbound::BBSolver::new(qubo, SolverOptions::new());
        solver.root_constraints = vec![
            Constraint::new(0, 1, ConstraintType::Equal),
            Constraint::new(1, 2, ConstraintType::ExactlyOne),
        ];
        solver.root_implications = ImplicationGraph::new(3, &solver.root_constraints);
        let (fixed, _) = solver
            .presolve_node([(0, 0)].into_iter().collect())
            .unwrap();
        assert_eq!(fixed, [(0, 0), (1, 0), (2, 1)].into_iter().collect());
        assert!(solver
            .presolve_node([(0, 0), (2, 0)].into_iter().collect())
            .is_none());
    }

    #[test]
    fn propagated_solver_matches_exhaustive_optima() {
        use smolprng::{JsfLarge, PRNG};
        let mut prng = PRNG {
            generator: JsfLarge::from(920487u64),
        };
        for sample in 0..32 {
            let mut qubo =
                Qubo::make_random_qubo(12, &mut prng, if sample % 2 == 0 { 0.25 } else { 0.8 });
            qubo.q
                .data_mut()
                .iter_mut()
                .for_each(|v| *v = (*v * 32.0).round());
            qubo.c.mapv_inplace(|v| (v * 8.0).round());
            let (expected, _) = crate::subproblemsolvers::enumerate_qubo::enumerate_solve(&qubo);
            for (relations, weak) in [(false, false), (true, false), (true, true)] {
                let mut options = SolverOptions::new();
                options.verbose = 0;
                options.threads = 4;
                options.sub_problem_solver = SubProblemSelection::MixingCutSDP;
                options.roof_dual_relation_penalties = relations;
                options.root_pair_dominance = relations;
                options.roof_dual_weak_persistencies = weak;
                options.root_roof_probe_candidates = if relations { 4 } else { 0 };
                options.root_roof_probe_max_seconds = 10.0;
                let mut solver = branchbound::BBSolver::new(qubo.clone(), options);
                let (solution, value) = solver.solve();
                assert!(solver.nodes.is_empty());
                assert!(
                    (value - expected).abs() < 1e-8,
                    "case {sample}, relations={relations}, weak={weak}: {value} != {expected}"
                );
                assert!((qubo.eval_usize(&solution) - expected).abs() < 1e-8);
            }
        }
    }

    #[test]
    pub fn branch_bound_roof_dual_subproblem_test() {
        let eye = CsMat::eye(3);
        let c = Array1::from_vec(vec![-1.1, -2.0, -3.0]);
        let p = Qubo::new_with_c(eye, c);

        let mut options = SolverOptions::new();
        options.sub_problem_solver = SubProblemSelection::RoofDualQPBO;
        options.verbose = 0;
        options.threads = 1;

        let mut solver = branchbound::BBSolver::new(p.clone(), options);
        let (solution, value) = solver.solve();

        assert_eq!(solution, Array1::from_vec(vec![1, 1, 1]));
        assert!((value - p.eval_usize(&solution)).abs() <= 1e-9);
    }

    #[test]
    pub fn test_gka2b_solve() {
        let file_path = "test_data/gka/gka2b.qubo";
        let p = Qubo::read_qubo(file_path);

        let sol_val = Array1::from_vec(vec![
            0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
            0, 0,
        ]);

        solve_qubo_with_all_permutations(&p, &sol_val);
    }

    #[test]
    pub fn test_gka1b_solve() {
        let file_path = "test_data/gka/gka1b.qubo";
        let p = Qubo::read_qubo(file_path);

        let sol_val = Array1::from_vec(vec![
            0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0,
        ]);
        solve_qubo_with_all_permutations(&p, &sol_val);
    }

    #[test]
    pub fn test_gka6a_solve() {
        let file_path = "test_data/gka/gka6a.qubo";
        let p = Qubo::read_qubo(file_path);

        let sol_val = Array1::from_vec(vec![
            0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1,
            0, 1,
        ]);
        solve_qubo_with_all_permutations(&p, &sol_val);
    }

    #[test]
    pub fn test_gka7a_solve() {
        let file_path = "test_data/gka/gka7a.qubo";
        let p = Qubo::read_qubo(file_path);

        let sol_val = Array1::from_vec(vec![
            0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
            0, 1,
        ]);
        solve_qubo_with_all_permutations(&p, &sol_val);
    }

    pub fn solve_qubo_with_all_permutations(qubo: &Qubo, sol_val: &Array1<usize>) {
        let branch_options = vec![
            BranchStrategy::FirstNotFixed,
            BranchStrategy::MostViolated,
            BranchStrategy::Random,
            BranchStrategy::WorstApproximation,
            BranchStrategy::WorstApproximation2,
            BranchStrategy::MostEdges,
            BranchStrategy::LargestEdges,
            BranchStrategy::MostFixed,
            BranchStrategy::FullStrongBranching,
            BranchStrategy::PartialStrongBranching,
            BranchStrategy::LargestDiag,
            BranchStrategy::MoveingEdges,
            BranchStrategy::RoundRobin,
        ];

        let sub_problem_solvers = vec![
            SubProblemSelection::ClarabelQP,
            SubProblemSelection::ClarabelLP,
            SubProblemSelection::HerculesABQP,
            SubProblemSelection::HerculesCDQP,
        ];

        for branch in &branch_options {
            for sup_problem_solver in &sub_problem_solvers {
                setup_and_solve_problem(*branch, *sup_problem_solver, qubo, sol_val);
            }
        }
    }

    pub fn setup_and_solve_problem(
        branch: BranchStrategy,
        sup_problem_solver: SubProblemSelection,
        qubo: &Qubo,
        true_sol: &Array1<usize>,
    ) {
        let mut prng = make_test_prng();

        let fixed_variables = preprocess_qubo(qubo, &HashMap::default(), false);

        let guess = local_search::particle_swarm_search(qubo, 10, 100, &mut prng);

        let mut options = get_default_solver_options();

        options.branch_strategy = branch;
        options.fixed_variables = fixed_variables;
        options.sub_problem_solver = sup_problem_solver;
        options.verbose = 0;

        let mut solver = branchbound::BBSolver::new(qubo.clone(), options);

        solver.warm_start(guess);

        let (_, sol_value) = solver.solve();

        let actual_obj = solver.qubo.eval_usize(true_sol);

        // the solution should be within 1E-5 of the actual solution
        // we don't check against the solution as there can be multiple optimal solutions
        assert!((sol_value - actual_obj).abs() <= 1E-5);
    }

    fn solve_public_bqp_instance_to_expected_objective_with_options(
        file_path: &str,
        expected_objective: f64,
        branch_strategy: BranchStrategy,
        sub_problem_solver: SubProblemSelection,
        threads: usize,
    ) {
        let mut qubo = Qubo::read_qubo(file_path);
        let mut tri_q = sprs::TriMat::<f64>::new((qubo.num_x(), qubo.num_x()));
        for (&value, (i, j)) in &qubo.q {
            let scale = if i == j { 2.0 } else { 4.0 };
            tri_q.add_triplet(i, j, scale * value);
        }
        qubo.q = tri_q.to_csr();

        let mut options = SolverOptions::new();
        options.branch_strategy = branch_strategy;
        options.sub_problem_solver = sub_problem_solver;
        options.verbose = 0;
        options.threads = threads;
        options.max_time = f64::INFINITY;

        let mut solver = branchbound::BBSolver::new(qubo, options);
        let (_, sol_value) = solver.solve();

        assert!(
            (sol_value - expected_objective).abs() <= 1E-5,
            "instance {file_path} solved to {sol_value}, expected {expected_objective}"
        );
    }

    fn solve_public_bqp_instance_to_expected_objective(file_path: &str, expected_objective: f64) {
        solve_public_bqp_instance_to_expected_objective_with_options(
            file_path,
            expected_objective,
            BranchStrategy::LargestEdges,
            SubProblemSelection::MixingCutSDP,
            64,
        );
    }

    fn solve_public_upper_triangle_instance_to_expected_objective_with_options(
        file_path: &str,
        expected_objective: f64,
        branch_strategy: BranchStrategy,
        sub_problem_solver: SubProblemSelection,
        threads: usize,
    ) {
        let mut qubo = Qubo::read_qubo(file_path);
        qubo.q = &qubo.q * 2.0;

        let mut options = SolverOptions::new();
        options.branch_strategy = branch_strategy;
        options.sub_problem_solver = sub_problem_solver;
        options.verbose = 0;
        options.threads = threads;
        options.max_time = f64::INFINITY;

        let mut solver = branchbound::BBSolver::new(qubo, options);
        let (_, sol_value) = solver.solve();

        assert!(
            (sol_value - expected_objective).abs() <= 1E-5,
            "instance {file_path} solved to {sol_value}, expected {expected_objective}"
        );
    }

    #[test]
    fn test_bqp50_objectives() {
        // Expected objectives taken from:
        // http://bqp.cs.uni-bonn.de/library/html/instances.html
        let cases = [
            ("test_data/bqp/bqp50-1.qubo", -2098.0),
            ("test_data/bqp/bqp50-2.qubo", -3702.0),
            ("test_data/bqp/bqp50-3.qubo", -4626.0),
            ("test_data/bqp/bqp50-4.qubo", -3544.0),
            ("test_data/bqp/bqp50-5.qubo", -4012.0),
            ("test_data/bqp/bqp50-6.qubo", -3693.0),
            ("test_data/bqp/bqp50-7.qubo", -4520.0),
            ("test_data/bqp/bqp50-8.qubo", -4216.0),
            ("test_data/bqp/bqp50-9.qubo", -3780.0),
            ("test_data/bqp/bqp50-10.qubo", -3507.0),
        ];

        for (file_path, expected_objective) in cases {
            solve_public_bqp_instance_to_expected_objective(file_path, expected_objective);
        }
    }

    #[test]
    fn test_bqp100_objectives() {
        // Expected objectives taken from:
        // http://bqp.cs.uni-bonn.de/library/html/instances.html
        let cases = [
            ("test_data/bqp/bqp100-1.qubo", -7970.0),
            ("test_data/bqp/bqp100-2.qubo", -11036.0),
            ("test_data/bqp/bqp100-3.qubo", -12723.0),
            ("test_data/bqp/bqp100-4.qubo", -10368.0),
            ("test_data/bqp/bqp100-5.qubo", -9083.0),
            ("test_data/bqp/bqp100-6.qubo", -10210.0),
            ("test_data/bqp/bqp100-7.qubo", -10125.0),
            ("test_data/bqp/bqp100-8.qubo", -11435.0),
            ("test_data/bqp/bqp100-9.qubo", -11455.0),
            ("test_data/bqp/bqp100-10.qubo", -12565.0),
        ];

        for (file_path, expected_objective) in cases {
            solve_public_bqp_instance_to_expected_objective(file_path, expected_objective);
        }
    }

    #[test]
    #[ignore = "expensive BE regression with MixingCutSDP + LargestEdges"]
    fn test_be100_objectives_mixingcut_largest_edges() {
        // Expected objectives taken from:
        // http://bqp.cs.uni-bonn.de/library/html/instances.html
        let cases = [
            ("test_data/be/be100.1.qubo", -19412.0),
            ("test_data/be/be100.2.qubo", -17290.0),
            ("test_data/be/be100.3.qubo", -17565.0),
            ("test_data/be/be100.4.qubo", -19125.0),
            ("test_data/be/be100.5.qubo", -15868.0),
            ("test_data/be/be100.6.qubo", -17368.0),
            ("test_data/be/be100.7.qubo", -18629.0),
            ("test_data/be/be100.8.qubo", -18649.0),
            ("test_data/be/be100.9.qubo", -13294.0),
            ("test_data/be/be100.10.qubo", -15352.0),
        ];

        for (file_path, expected_objective) in cases {
            solve_public_upper_triangle_instance_to_expected_objective_with_options(
                file_path,
                expected_objective,
                BranchStrategy::LargestEdges,
                SubProblemSelection::MixingCutSDP,
                64,
            );
        }
    }

    #[test]
    fn test_gka_small_objectives() {
        // Expected objectives taken from:
        // http://bqp.cs.uni-bonn.de/library/html/instances.html
        let cases = [
            ("test_data/gka/gka1a.qubo", -3414.0),
            ("test_data/gka/gka1b.qubo", -133.0),
            ("test_data/gka/gka1c.qubo", -5058.0),
            ("test_data/gka/gka2a.qubo", -6063.0),
            ("test_data/gka/gka2b.qubo", -121.0),
            ("test_data/gka/gka2c.qubo", -6213.0),
            ("test_data/gka/gka3a.qubo", -6037.0),
            ("test_data/gka/gka3b.qubo", -118.0),
            ("test_data/gka/gka3c.qubo", -6665.0),
            ("test_data/gka/gka4a.qubo", -8598.0),
            ("test_data/gka/gka4b.qubo", -129.0),
            ("test_data/gka/gka5a.qubo", -5737.0),
            ("test_data/gka/gka5b.qubo", -150.0),
            ("test_data/gka/gka6b.qubo", -146.0),
        ];

        for (file_path, expected_objective) in cases {
            solve_public_upper_triangle_instance_to_expected_objective_with_options(
                file_path,
                expected_objective,
                BranchStrategy::LargestEdges,
                SubProblemSelection::MixingCutSDP,
                64,
            );
        }
    }

    #[test]
    #[ignore = "expensive W regression with MixingCutSDP + LargestEdges"]
    fn test_w_small_objectives() {
        // Expected objectives taken from:
        // http://bqp.cs.uni-bonn.de/library/html/instances.html
        let cases = [
            ("test_data/w/g05_60.0.qubo", -536.0),
            ("test_data/w/pm1s_80.0.qubo", -79.0),
            ("test_data/w/pm1d_80.0.qubo", -227.0),
            ("test_data/w/w01_100.0.qubo", -651.0),
            ("test_data/w/pw01_100.1.qubo", -2060.0),
        ];

        for (file_path, expected_objective) in cases {
            solve_public_upper_triangle_instance_to_expected_objective_with_options(
                file_path,
                expected_objective,
                BranchStrategy::LargestEdges,
                SubProblemSelection::MixingCutSDP,
                64,
            );
        }
    }
}
