use crate::qubo::Qubo;
use ndarray::Array1;
use std::collections::HashMap;
use std::time;

use crate::persistence::compute_iterative_persistence;
use clarabel::algebra::*;
use clarabel::solver::*;
use sprs::{CsMat, TriMat};

/// Bare bones implimentation of B&B. Currently requires the QUBO to be symmetrical and convex.
/// Currently, the determanistic solver is solved via Clarabel.rs.

/// Struct the describes the branch and bound tree nodes
#[derive(Clone)]
pub struct QuboBBNode {
    pub lower_bound: f64,
    pub solution: Array1<f64>,
    pub fixed_variables: HashMap<usize, f64>,
}

/// Struct for the B&B Solver
pub struct BBSolver {
    pub qubo: Qubo,
    pub best_solution: Array1<f64>,
    pub best_solution_value: f64,
    pub nodes: Vec<QuboBBNode>,
    pub nodes_processed: usize,
    pub nodes_visited: usize,
    pub time_start: f64,
    pub clarabel_wrapper: ClarabelWrapper,
    pub options: SolverOptions,
}

/// Options for the B&B solver for run time
pub struct SolverOptions {
    pub max_time: f64,
}

enum BranchStrategy {
    First,
    MostViolated,
    Random,
}


/// Wrapper to help convert the QUBO to the format required by Clarabel.rs
pub struct ClarabelWrapper {
    pub q: CscMatrix,
    pub c: Array1<f64>,
}

impl ClarabelWrapper {
    pub fn new(qubo: &Qubo) -> ClarabelWrapper {
        let q_new = ClarabelWrapper::make_cb_form(&(qubo.q));
        ClarabelWrapper {
            q: q_new,
            c: qubo.c.clone(),
        }
    }

    pub fn make_cb_form(p0: &CsMat<f64>) -> CscMatrix {
        let (t, y, u) = p0.to_csc().into_raw_storage();
        CscMatrix::new(p0.rows(), p0.cols(), t, y, u)
    }
}

impl BBSolver {
    /// Creates a new B&B solver
    pub fn new(qubo: Qubo, options: SolverOptions) -> BBSolver {
        let num_x = qubo.num_x();
        let wrapper = ClarabelWrapper::new(&qubo);

        BBSolver {
            qubo,
            best_solution: Array1::zeros(num_x),
            best_solution_value: 0.0,
            nodes: Vec::new(),
            nodes_processed: 0,
            nodes_visited: 0,
            time_start: 0.0,
            clarabel_wrapper: wrapper,
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

        // compute an initial set of persistent variables
        let initial_persistent = compute_iterative_persistence(&self.qubo, &HashMap::new(), self.qubo.num_x());

        // create the root node
        let root_node = QuboBBNode {
            lower_bound: f64::NEG_INFINITY,
            solution: Array1::zeros(self.qubo.num_x()),
            fixed_variables: initial_persistent,
        };

        // set the start time
        self.time_start = time::SystemTime::now()
            .duration_since(time::SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        // add the root node to the list of nodes
        self.nodes.push(root_node);

        // until we have hit a termination condition, we will keep iterating
        while !(*self).termination_condition() {
            // get next node, if it exists
            let next_node = self.get_next_node();

            // there are no more nodes to process, so we are done iterating
            if next_node.is_none() {
                break;
            }

            // unwrap the node
            let node = next_node.unwrap();

            // as we are processing the node we increment the number of nodes processed
            self.nodes_processed += 1;

            // We now need to solve the node to generate the lower bound and solution
            let (lower_bound, solution) = self.solve_node(&node);

            // determine what variable we are branching on
            let branch_id = self.make_branch(&node);

            // generate the branches
            let (mut zero_branch, mut one_branch) = self.branch(node, branch_id);

            zero_branch.lower_bound = lower_bound;
            zero_branch.solution = solution.clone();
            one_branch.lower_bound = lower_bound;
            one_branch.solution = solution;

            // add the branches to the list of nodes
            self.nodes.push(zero_branch);
            self.nodes.push(one_branch);
        }

        return (self.best_solution.clone(), self.best_solution_value);
    }

    /// Checks if we can prune the node, based on the lower bound and best solution
    pub fn can_prune(&mut self, node: &QuboBBNode) -> bool {
        // if the lower bound is greater than the best solution, then we can prune
        if node.lower_bound > self.best_solution_value {
            return true;
        }

        // if the solution is complete, then we can update the best solution if better
        // we can also prune the node, as there are no more variables to fix
        if node.fixed_variables.len() == self.qubo.num_x() {

            // generate the solution vector
            let mut solution = Array1::zeros(self.qubo.num_x());
            for (&index, &value) in node.fixed_variables.iter() {
                solution[index] = value;
            }

            let solution_value = self.qubo.eval(&solution);
            if solution_value < self.best_solution_value {
                self.best_solution = node.solution.clone();
                self.best_solution_value = solution_value;
            }
            return true;
        }

        // if we can not remove the node, then we return false as we cannot provably prune it yet
        return false;
    }

    /// This function is used to get the next node to process, popping it from the list of nodes
    pub fn get_next_node(&mut self) -> Option<QuboBBNode> {
        while self.nodes.len() > 0 {
            // we pull a node from our node list
            let node = self.nodes.pop().unwrap();

            // we increment the number of nodes we have visited
            self.nodes_visited += 1;

            // if we can't prune it, then we return it
            if !self.can_prune(&node) {
                return Some(node);
            }
        }
        return None;
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
        if self.nodes.len() == 0 {
            return true;
        }

        return false;
    }

    /// Branch Selection Strategy - Currently selects the first variable that is not fixed
    pub fn make_branch(&self, node: &QuboBBNode) -> usize {
        (0..self.qubo.num_x())
            .filter(|x| !node.fixed_variables.contains_key(x))
            .next()
            .unwrap()
    }

    /// Actually branches the node into two new nodes
    pub fn branch(&self, node: QuboBBNode, branch_id: usize) -> (QuboBBNode, QuboBBNode) {
        let mut zero_branch = node.clone();
        let mut one_branch = node;

        zero_branch.fixed_variables.insert(branch_id, 0.0);
        one_branch.fixed_variables.insert(branch_id, 1.0);

        zero_branch.solution[branch_id] = 0.0;
        one_branch.solution[branch_id] = 1.0;

        return (zero_branch, one_branch);
    }

    /// Solves the subproblem via a QP solver, in this case Clarabel.rs
    pub fn solve_node(&self, node: &QuboBBNode) -> (f64, Array1<f64>) {
        // solve QP associated with the node

        // generate default settings
        let mut settings = DefaultSettings::default();
        settings.verbose = false;

        // generate the constraint matrix
        let A_size = 2 * self.qubo.num_x() + node.fixed_variables.len();
        let mut A = TriMat::new((A_size, self.qubo.num_x()));
        let mut b = Array1::zeros(A_size);

        // add the equality constraints
        for (index, (&key, &value)) in node.fixed_variables.iter().enumerate() {
            A.add_triplet(index, key, 1.0);
            b[index] = value;
        }

        // add the inequality constraints
        for (index, i) in (0..self.qubo.num_x()).enumerate() {
            let offset = node.fixed_variables.len() + index * 2;
            A.add_triplet(offset, i, 1.0);
            A.add_triplet(offset + 1, i, -1.0);
            b[offset] = 1.0;
            b[offset + 1] = 0.0;
        }

        // convert the matrix to CSC format and then Clarabel format
        let A_csc = A.to_csc();
        let A_clara = ClarabelWrapper::make_cb_form(&A_csc);

        // generate the cones for the solver
        let cones = [
            ZeroConeT(node.fixed_variables.len()),
            NonnegativeConeT(2 * self.qubo.num_x()),
        ];

        // set up the solver with the matrices
        let mut solver = DefaultSolver::new(
            &self.clarabel_wrapper.q,
            &self.qubo.c.as_slice().unwrap(),
            &A_clara,
            &b.as_slice().unwrap(),
            &cones,
            settings,
        );

        // actually solve the problem
        solver.solve();

        return (solver.solution.obj_val, Array1::from(solver.solution.x));
    }
}
