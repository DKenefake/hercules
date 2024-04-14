use crate::qubo::Qubo;
use ndarray::Array1;
use std::ops::Index;
use std::time;
use rayon::prelude::*;

use crate::branch_node::QuboBBNode;
use crate::branch_stratagy::BranchStrategy;
use crate::branch_subproblem::{
    get_sub_problem_solver, ClarabelSubProblemSolver, SubProblemSolver,
};
use crate::branchbound_utils::check_integer_feasibility;
use crate::branchboundlogger::{
    generate_exit_line, generate_output_line, output_header, output_warm_start_info,
};
use crate::persistence::compute_iterative_persistence;
use crate::solver_options::SolverOptions;

/// Struct for the B&B Solver
pub struct BBSolver {
    pub qubo: Qubo,
    pub best_solution: Array1<f64>,
    pub best_solution_value: f64,
    pub nodes: Vec<QuboBBNode>,
    pub nodes_processed: usize,
    pub nodes_solved: usize,
    pub nodes_visited: usize,
    pub time_start: f64,
    pub branch_strategy: BranchStrategy,
    pub subproblem_solver: ClarabelSubProblemSolver,
    pub options: SolverOptions,
}

pub enum Event {
    UpdateBestSolution(Array1<f64>),
    AddBranches(QuboBBNode, QuboBBNode),
    Nill
}

pub enum SolverLoggingAction{
    NodeVisited,
    NodeProcessed,
    NodeSolved,
}

pub enum PruneAction{
    Prune,
    Dont
}

pub enum IntegerFeasibility{
    IntegerFeasible(Array1<f64>),
    NotIntegerFeasible
}

pub struct ProcessNodeState{
    pub prune_action: PruneAction,
    pub event: Option<Event>
}

impl BBSolver {
    /// Creates a new B&B solver
    pub fn new(qubo: Qubo, options: SolverOptions) -> Self {
        // create auxiliary variables
        let num_x = qubo.num_x();

        let subproblem_solver = get_sub_problem_solver(&qubo, &options.sub_problem_solver);
        let branch_strategy = BranchStrategy::get_branch_strategy(&options.branch_strategy);
        let start_time = time::SystemTime::now()
            .duration_since(time::SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        Self {
            qubo,
            best_solution: Array1::zeros(num_x),
            best_solution_value: 0.0,
            nodes: Vec::new(),
            nodes_processed: 0,
            nodes_visited: 0,
            nodes_solved: 0,
            time_start: start_time,
            branch_strategy,
            subproblem_solver,
            options,
        }
    }

    /// This function is used to warm start the solver with an initial solution if one is not provided
    pub fn warm_start(&mut self, initial_solution: Array1<f64>) {
        self.best_solution = initial_solution;
        self.best_solution_value = self.qubo.eval(&self.best_solution);
    }

    /// The main solve function of the B&B algorithm
    pub fn solve(&mut self) -> (Array1<f64>, f64) {
        // preprocess the problem
        let initial_fixed = self.options.fixed_variables.clone();

        self.options.fixed_variables =
            compute_iterative_persistence(&self.qubo, &initial_fixed, self.qubo.num_x());

        // create the root node
        let root_node = QuboBBNode {
            lower_bound: f64::NEG_INFINITY,
            solution: Array1::zeros(self.qubo.num_x()),
            fixed_variables: self.options.fixed_variables.clone(),
        };

        // add the root node to the list of nodes
        self.nodes.push(root_node);

        // set the start time
        self.time_start = time::SystemTime::now()
            .duration_since(time::SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        // set up the output of the solver
        if self.options.verbose {
            // display the header
            output_header(&self);
            // if the best solution is negative, then we output the warm start information
            if self.best_solution_value < 0.0 {
                output_warm_start_info(self);
            }
        }

        // until we have hit a termination condition, we will keep iterating
        while !(*self).termination_condition() {

            // get the most recent 25 nodes to process
            let nodes = self.get_next_nodes(self.options.threads);

            if nodes.is_empty() {
                break;
            }

            let process_results = nodes.par_iter().map(|node| {
                self.process_node(node.clone())
            }).collect::<Vec<ProcessNodeState>>();

            self.nodes_processed += nodes.len();

            for state in process_results {
                match state.event {
                    Some(Event::UpdateBestSolution(solution)) => {
                        self.update_solution_if_better(&solution);
                    },
                    Some(Event::AddBranches(zero_branch, one_branch)) => {
                        self.nodes.push(zero_branch);
                        self.nodes.push(one_branch);
                        self.nodes_solved += 1;
                    },
                    _ => {}
                }
            }


            if self.options.verbose {
                generate_output_line(&self);
            }

        }

        if self.options.verbose {
            generate_exit_line(&self);
        }

        (self.best_solution.clone(), self.best_solution_value)
    }


    /// Checks if we can prune the node, based on the lower bound and best solution, returns an action
    pub fn can_prune_action(&self, node: &QuboBBNode) -> (PruneAction, Event) {
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

            // evaluate the solution against the best solution we have so far
            // if we have a better solution update it
            return (PruneAction::Prune, Event::UpdateBestSolution(solution));
        }

        (PruneAction::Dont, Event::Nill)
    }

    /// main loop of the branch and bound algorithm
    pub fn process_node(&self, node: QuboBBNode) -> ProcessNodeState {

        let mut x = ProcessNodeState {
            prune_action: PruneAction::Dont,
            event: None
        };

        let mut node = node.clone();

        // see if there are any variables we can fix
        node.fixed_variables =
            compute_iterative_persistence(&self.qubo, &node.fixed_variables, self.qubo.num_x());

        // with this expanded set can we prune the node?
        let (prune_action, event) = self.can_prune_action(&node);

        match prune_action {
            PruneAction::Prune => {
                x.prune_action = PruneAction::Prune;

                match event {
                    Event::UpdateBestSolution(solution) => {
                        x.event = Some(Event::UpdateBestSolution(solution));
                    },
                    _ => {}
                }

                return x;
            },
            PruneAction::Dont => {}
        };

        // We now need to solve the node to generate the lower bound and solution
        let (lower_bound, solution) = self.solve_node(&node);

        // inject the solution back into the node
        node.solution = solution.clone();

        // check if integer feasible solution
        // if not all variables are fixed, we can still check if we are 'near' integer-feasible (within 1E-10) of 0 or 1
        let (is_int_feasible, rounded_sol) = check_integer_feasibility(&node);

        if is_int_feasible {
            x.event = Some(Event::UpdateBestSolution(rounded_sol));
            return x;
        }

        // determine what variable we are branching on
        let branch_id = self.make_branch(&node);

        // generate the branches
        let (zero_branch, one_branch) = Self::branch(node, branch_id, lower_bound, solution);

        // add the branches to the list of nodes
        x.event = Some(Event::AddBranches(zero_branch, one_branch));;

        x

    }

    /// update the best solution if better than the current best solution
    pub fn update_solution_if_better(&mut self, solution: &Array1<f64>) {
        let solution_value = self.qubo.eval(solution);
        if solution_value < self.best_solution_value {
            self.best_solution = solution.clone();
            self.best_solution_value = solution_value;
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
            self.nodes_visited += 1;

            // if we can't prune it, then we return it
            let (prune, event) = self.can_prune_action(&node);

            match event {
                Event::UpdateBestSolution(solution) => {
                    self.update_solution_if_better(&solution);
                },
                _ => {}
            }

            match prune {
                PruneAction::Dont => {
                    return Some(node);
                },
                PruneAction::Prune => {}
            }
        }

        None
    }

    pub fn get_next_nodes(&mut self, n: usize) -> Vec<QuboBBNode> {

        let mut nodes = Vec::new();

        while nodes.len() <= n {

            let next_node = self.get_next_node();

            match next_node {
                Some(node) => {
                    nodes.push(node);
                },
                None => {
                    break;
                }
            }

        }

        nodes
    }

    /// Checks for termination conditions of the B&B algorithm, such as time limit or no more nodes
    pub fn termination_condition(&self) -> bool {
        // get current time to check if we have exceeded the maximum time
        let current_time = time::SystemTime::now()
            .duration_since(time::SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        // check if we violated the time limit
        if current_time - self.time_start > self.options.max_time {
            return true;
        }

        // check if we have no more nodes to process
        if self.nodes.is_empty() {
            return true;
        }

        false
    }

    /// Branch Selection Strategy - Currently selects the first variable that is not fixed
    pub fn make_branch(&self, node: &QuboBBNode) -> usize {
        return self.branch_strategy.make_branch(self, node);
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
        let mut one_branch = node.clone();

        // add fixed variables
        zero_branch.fixed_variables.insert(branch_id, 0.0);
        one_branch.fixed_variables.insert(branch_id, 1.0);

        // update the solution and lower bound for the new nodes
        zero_branch.solution = solution.clone();
        one_branch.solution = solution.clone();

        // set the lower bound for the new nodes
        zero_branch.lower_bound = lower_bound;
        one_branch.lower_bound = lower_bound;

        (zero_branch, one_branch)
    }

    pub fn solve_node(&self, node: &QuboBBNode) -> (f64, Array1<f64>) {
        self.subproblem_solver.solve(self, node)
    }

}

#[cfg(test)]
mod tests {
    use crate::branch_stratagy::BranchStrategySelection;
    use crate::qubo::Qubo;
    use crate::solver_options::SolverOptions;
    use crate::tests::{make_solver_qubo, make_test_prng};
    use crate::{branchbound, local_search};
    use ndarray::Array1;
    use sprs::CsMat;

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
        assert_eq!(solver.best_solution, Array1::from_vec(vec![1.0, 1.0, 1.0]));
    }
    #[test]
    pub fn branch_bound_bench() {
        let mut prng = make_test_prng();

        let p = make_solver_qubo();

        let p_new = p.make_symmetric();
        // get the eigenvalues of the hessian
        let eig = p_new.hess_eigenvalues();
        // get the smallest eigenvalue
        let min_eig = eig.iter().fold(f64::INFINITY, |a, b| a.min(*b));

        let p_fixed = p_new.make_convex(min_eig.abs() * 1.1);

        let guess = local_search::particle_swarm_search(&p_fixed, 10, 100, &mut prng);

        let mut options = SolverOptions::new();
        options.verbose = true;
        options.max_time = 1000.0;
        options.branch_strategy = BranchStrategySelection::MostViolated;
        options.threads = 32;

        let mut solver = branchbound::BBSolver::new(p_fixed, options);
        solver.warm_start(guess);
        solver.solve();
    }
}
