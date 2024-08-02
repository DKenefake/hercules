use crate::qubo::Qubo;
use ndarray::Array1;
use rayon::prelude::*;

use crate::branch_node::QuboBBNode;
use crate::branch_stratagy::BranchStrategy;
use crate::branch_subproblem::{
    get_sub_problem_solver, ClarabelSubProblemSolver, SubProblemSolver,
};
use crate::branchbound_utils::{check_integer_feasibility, get_current_time};
use crate::branchboundlogger::SolverOutputLogger;
use crate::early_termination::beck_proof;
use crate::lower_bound::li_lower_bound;
use crate::preprocess;
use crate::preprocess::preprocess_qubo;
use crate::solver_options::SolverOptions;
use std::collections::BinaryHeap;

/// Struct for the B&B Solver
pub struct BBSolver {
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
    pub subproblem_solver: ClarabelSubProblemSolver,
    pub options: SolverOptions,
    pub early_stop: bool,
    pub solver_logger: SolverOutputLogger,
}

pub enum Event {
    UpdateBestSolution(Array1<usize>, f64),
    AddBranches(QuboBBNode, QuboBBNode),
    Nill,
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
    pub prune_action: PruneAction,
    pub events: Vec<Event>,
    pub logging: NodeLoggingAction,
}

pub enum SolverResult {
    OptimalSolution(Array1<f64>, f64),
    SubOptimalSolution(Array1<f64>, f64),
}

impl BBSolver {
    /// Creates a new B&B solver
    pub fn new(qubo: Qubo, options: SolverOptions) -> Self {
        // create auxiliary variables
        let num_x = qubo.num_x();

        let subproblem_solver = get_sub_problem_solver(&qubo, &options.sub_problem_solver);
        let branch_strategy = BranchStrategy::get_branch_strategy(&options.branch_strategy);
        let start_time = get_current_time();
        let output_level = options.verbose;
        let pp_form = preprocess::shift_qubo(&qubo);

        Self {
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
            options,
            early_stop: false,
            solver_logger: SolverOutputLogger { output_level },
        }
    }

    /// This function is used to warm start the solver with an initial solution if one is not provided
    pub fn warm_start(&mut self, initial_solution: Array1<usize>) {
        self.best_solution = initial_solution;
        self.best_solution_value = self.qubo.eval_usize(&self.best_solution);

        // if we have an early stopping condition, then we can check if we have a solution
        let beck_proof = beck_proof(&self.qubo, &self.best_solution);

        // if we have a beck proof, then we can stop early
        if beck_proof {
            self.early_stop = true;
            self.solver_logger.early_termination();
        }
    }

    /// The main solve function of the B&B algorithm
    pub fn solve(&mut self) -> (Array1<usize>, f64) {
        // preprocess the problem
        let fixed_variables =
            preprocess_qubo(&self.qubo_pp_form, &self.options.fixed_variables, true);
        self.options.fixed_variables = fixed_variables.clone();

        // create the root node
        let root_node = QuboBBNode {
            lower_bound: f64::NEG_INFINITY,
            solution: Array1::zeros(self.qubo.num_x()),
            fixed_variables,
        };

        // add the root node to the list of nodes
        self.nodes.push(root_node);

        // Reset start time as it can be different from the time we created the solver instance
        self.time_start = get_current_time();

        // set up the output of the solver
        // display the header
        self.solver_logger.output_header(self);

        // if the best solution is negative, then we output the warm start information
        if self.best_solution_value < 0.0 {
            self.solver_logger.output_warm_start_info(self);
        }

        // until we have hit a termination condition, we will keep iterating
        while !(*self).termination_condition() {
            // get the most recent 25 nodes to process
            let nodes = self.get_next_nodes(self.options.threads);

            let process_results = nodes
                .par_iter()
                .map(|node| self.process_node(node))
                .collect::<Vec<_>>();

            // apply all the events from the parallel loop back to the solver
            for state in process_results {
                self.apply_events(state.events);
                self.apply_logging_action(state.logging);
            }

            // display the line, if verbose
            self.solver_logger.generate_output_line(self);
        }

        // display the exit line
        self.solver_logger.generate_exit_line(self);

        (self.best_solution.clone(), self.best_solution_value)
    }

    /// Checks if we can prune the node, based on the lower bound and best solution, returns an action
    pub fn can_prune_action(&self, node: &QuboBBNode) -> (PruneAction, Event) {
        // if our parent solution is above our current feasible soltion then prune
        if node.lower_bound > self.best_solution_value {
            return (PruneAction::Prune, Event::Nill);
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
            return (
                PruneAction::Prune,
                Event::UpdateBestSolution(solution, value),
            );
        }

        (PruneAction::Dont, Event::Nill)
    }

    // apply the logging action to the solver
    pub fn apply_logging_action(&mut self, action: NodeLoggingAction) {
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
        // create a mutable copy of the node
        let mut node = node.clone();

        // pass to the presolver to see if there are any variables we can fix
        node.fixed_variables = preprocess_qubo(&self.qubo_pp_form, &node.fixed_variables, true);

        // calculate the lower bound via the li lower bound formula
        let li_bound = li_lower_bound(&self.qubo, &node.fixed_variables);
        node.lower_bound = node.lower_bound.max(li_bound);

        // with this expanded set, can we prune the node?
        let (prune_action, event) = self.can_prune_action(&node);

        // if we are pruning at this stage, then we can early return
        if matches!(prune_action, PruneAction::Prune) {
            return ProcessNodeState {
                prune_action,
                events: vec![event],
                logging: NodeLoggingAction::Processed,
            };
        }

        // We now need to solve the node to generate the lower bound and solution
        let (lower_bound, solution) = self.solve_node(&node);

        // inject the solution back into the node
        node.solution = solution.clone();

        // check if integer-feasible solution
        // if not all variables are fixed, we can still check if we are 'near' integer-feasible (within 1E-10) of 0 or 1
        let (is_int_feasible, rounded_sol) = check_integer_feasibility(&node);

        // if we are integer-feasible, then we can prune this branch and return the solution
        if is_int_feasible {
            // compute the objective
            let value = self.qubo.eval_usize(&rounded_sol);

            // if it is better, then we will attempt to update the solution otherwise prune
            if value < self.best_solution_value {
                return ProcessNodeState {
                    prune_action,
                    events: vec![Event::UpdateBestSolution(rounded_sol, value)],
                    logging: NodeLoggingAction::Solved,
                };
            }
            return ProcessNodeState {
                prune_action,
                events: vec![Event::Nill],
                logging: NodeLoggingAction::Solved,
            };
        }

        // if we are going to branch then we can generate a heuristic solution
        let (heur_sol, heur_obj) = self.options.heuristic.make_heuristic(self, &node);

        // determine what variable we are branching on
        let branch_id = self.make_branch(&node);

        // generate the branches
        let (zero_branch, one_branch) = Self::branch(node, branch_id, lower_bound, solution);

        ProcessNodeState {
            prune_action,
            events: vec![
                Event::AddBranches(zero_branch, one_branch),
                Event::UpdateBestSolution(heur_sol, heur_obj),
            ],
            logging: NodeLoggingAction::Solved,
        }
    }

    pub fn apply_events(&mut self, events: Vec<Event>) {
        for action in events {
            match action {
                Event::UpdateBestSolution(solution, value) => {
                    self.update_solution_if_better(&solution, value);
                }
                Event::AddBranches(zero_branch, one_branch) => {
                    self.nodes.push(zero_branch);
                    self.nodes.push(one_branch);
                }
                Event::Nill => {}
            }
        }
    }

    /// update the best solution if better than the current best solution
    pub fn update_solution_if_better(&mut self, solution: &Array1<usize>, solution_value: f64) {
        if solution_value < self.best_solution_value {
            self.best_solution = solution.clone();
            self.best_solution_value = solution_value;

            // if we have an early stopping condition, then we can check if we have a solution
            let beck_proof = beck_proof(&self.qubo, &self.best_solution);

            // if we have a beck proof, then we can stop early
            if beck_proof {
                self.early_stop = true;
                self.solver_logger.early_termination();
            }
        }
    }

    /// This function is used to get the next node to process, popping it from the list of nodes
    pub fn get_next_node(&mut self) -> Option<QuboBBNode> {
        while !self.nodes.is_empty() {
            // we pull a node from our node list
            let optional_node = self.nodes.pop();

            // guard against the case where another thread might have popped the last node between the
            // check and unwrap the node if it is safe
            let node = optional_node?;

            // we increment the number of nodes we have visited
            self.apply_logging_action(NodeLoggingAction::Visited);

            // if we can't prune it, then we return it
            let (prune, event) = self.can_prune_action(&node);

            // if we have stumbled into a better solution, then we can take it
            if let Event::UpdateBestSolution(solution, value) = event {
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
        let mut nodes = Vec::new();

        // loop while we haven't filled our vector OR the node list is not empty
        while nodes.len() <= n {
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
    pub fn make_branch(&self, node: &QuboBBNode) -> usize {
        self.branch_strategy.make_branch(self, node)
    }

    /// Actually branches the node into two new nodes
    pub fn branch(
        node: QuboBBNode,
        branch_id: usize,
        lower_bound: f64,
        solution: Array1<f64>,
    ) -> (QuboBBNode, QuboBBNode) {
        // make two new nodes that are clones of the parent, one with the variable set to 0 and
        // the other set to 1
        let mut zero_branch = node.clone();
        let mut one_branch = node;

        // add fixed variables
        zero_branch.fixed_variables.insert(branch_id, 0);
        one_branch.fixed_variables.insert(branch_id, 1);

        // update the solution and lower bound for the new nodes
        zero_branch.solution = solution.clone();
        one_branch.solution = solution;

        // set the lower bound for the new nodes
        zero_branch.lower_bound = lower_bound;
        one_branch.lower_bound = lower_bound;

        (zero_branch, one_branch)
    }

    pub fn solve_node(&self, node: &QuboBBNode) -> (f64, Array1<f64>) {
        self.subproblem_solver.solve_lower_bound(self, node)
    }
}

#[cfg(test)]
mod tests {
    use crate::branch_stratagy::BranchStrategySelection;
    use crate::preprocess::preprocess_qubo;
    use crate::qubo::Qubo;
    use crate::solver_options::SolverOptions;
    use crate::tests::{make_solver_qubo, make_test_prng};
    use crate::{branchbound, local_search};
    use ndarray::Array1;
    use sprs::CsMat;
    use std::collections::HashMap;

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

        assert_eq!(solver.best_solution_value, -4.6);
        assert_eq!(solver.best_solution, Array1::from_vec(vec![1, 1, 1]));
    }
    #[test]
    pub fn branch_bound_most_violated_branching() {
        setup_and_solve_problem(BranchStrategySelection::MostViolated)
    }

    #[test]
    pub fn branch_bound_random_branching() {
        setup_and_solve_problem(BranchStrategySelection::Random)
    }

    #[test]
    pub fn branch_bound_worst_approximation_branching() {
        setup_and_solve_problem(BranchStrategySelection::WorstApproximation)
    }

    #[test]
    pub fn branch_bound_best_approximation_branching() {
        setup_and_solve_problem(BranchStrategySelection::BestApproximation)
    }

    #[test]
    pub fn branch_bound_finf_branching() {
        setup_and_solve_problem(BranchStrategySelection::FirstNotFixed)
    }

    pub fn setup_and_solve_problem(branch: BranchStrategySelection) {
        let mut prng = make_test_prng();

        let p = make_solver_qubo();

        let p_symm = p.make_symmetric();
        let fixed_variables = preprocess_qubo(&p_symm, &HashMap::new(), false);

        let p_symm_conv = p.convex_symmetric_form();

        let guess = local_search::particle_swarm_search(&p_symm_conv, 10, 100, &mut prng);

        let mut options = get_default_solver_options();

        options.branch_strategy = branch;
        options.fixed_variables = fixed_variables.clone();

        let mut solver = branchbound::BBSolver::new(p_symm_conv, options);
        solver.warm_start(guess);

        let (solution, _) = solver.solve();

        // ensure that the solution is actually feasible with the preprocessor
        for (&index, &val) in fixed_variables.iter() {
            assert_eq!(val, solution[index]);
        }
    }
}
