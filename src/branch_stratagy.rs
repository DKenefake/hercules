use crate::branch_node::QuboBBNode;
use crate::branchbound::BBSolver;
use crate::lower_bound::pardalos_rodgers_lower_bound;
use crate::preprocess::preprocess_qubo;
use ndarray::Array1;
use smolprng::{JsfLarge, PRNG};
use std::collections::{HashMap, HashSet};

#[derive(Copy, Clone)]
pub enum BranchStrategy {
    FirstNotFixed,
    MostViolated,
    Random,
    WorstApproximation,
    BestApproximation,
    MostEdges,
    LargestEdges,
    MostFixed,
    FullStrongBranching,
    PartialStrongBranching,
    RoundRobin,
    LargestDiag,
}

pub(crate) struct BranchResult {
    pub branch_variable: usize,
    pub found_fixed_vars: HashMap<usize, usize>,
}

impl BranchStrategy {
    pub fn make_branch(self, bb_solver: &BBSolver, node: &QuboBBNode) -> BranchResult {
        let branch_result = match self {
            Self::FirstNotFixed => first_not_fixed(bb_solver, node),
            Self::MostViolated => most_violated(bb_solver, node),
            Self::Random => random(bb_solver, node),
            Self::WorstApproximation => worst_approximation(bb_solver, node),
            Self::BestApproximation => best_approximation(bb_solver, node),
            Self::MostEdges => most_edges(bb_solver, node),
            Self::LargestEdges => largest_edges(bb_solver, node),
            Self::MostFixed => most_fixed(bb_solver, node),
            Self::FullStrongBranching => full_strong_branching(bb_solver, node),
            Self::PartialStrongBranching => partial_strong_branching(bb_solver, node),
            Self::RoundRobin => round_robin(bb_solver, node),
            Self::LargestDiag => largest_diag(bb_solver, node),
        };

        // hard assert that the variable is not fixed
        assert!(
            !node
                .fixed_variables
                .contains_key(&branch_result.branch_variable),
            "Branching on a fixed variable"
        );

        branch_result
    }
}

/// Branches on the variable that has the most edges in the graph equivalent to the QUBO
fn most_edges(solver: &BBSolver, node: &QuboBBNode) -> BranchResult {
    // as a QUBO can be viewed as a graph, we can find the variable with the most (remaining) edges
    let mut edge_count = Array1::<usize>::zeros(solver.qubo.num_x());

    // scan through the Q matrix and count the number of edges
    for (_, (i, j)) in &solver.qubo.q {
        // if we have a fixed variable, then we can skip it
        if node.fixed_variables.contains_key(&i) {
            continue;
        }

        // same again
        if node.fixed_variables.contains_key(&j) {
            continue;
        }

        edge_count[i] += 1;
        edge_count[j] += 1;
    }

    // find the variable with the most edges (fixed variables in the node are not counted)
    let mut max_edges = 0;
    let mut index_max_edges = 0;

    for i in 0..solver.qubo.num_x() {
        if !node.fixed_variables.contains_key(&i) && edge_count[i] > max_edges {
            max_edges = edge_count[i];
            index_max_edges = i;
        }
    }

    BranchResult {
        branch_variable: index_max_edges,
        found_fixed_vars: HashMap::new(),
    }
}

/// Branches on the largest edges in the qubo, ones that are most likely to be the largest
/// determaning factor in the problem
fn largest_edges(solver: &BBSolver, node: &QuboBBNode) -> BranchResult {
    // as a QUBO can be viewed as a graph, we can find the variable with the most (remaining) edges
    let mut edge_count = Array1::<f64>::zeros(solver.qubo.num_x());

    // scan through the Q matrix and count the number of edges
    for (&value, (i, j)) in &solver.qubo.q {
        // if we have a fixed variable, then we can skip it
        if node.fixed_variables.contains_key(&i) {
            continue;
        }

        // same again
        if node.fixed_variables.contains_key(&j) {
            continue;
        }

        edge_count[i] += value.abs();
        edge_count[j] += value.abs();
    }

    // find the variable with the most edges (fixed variables in the node are not counted)
    let mut min_edge_value = 0.0;
    let mut index_max_edges = 0;

    for i in 0..solver.qubo.num_x() {
        if !node.fixed_variables.contains_key(&i) && edge_count[i] > min_edge_value {
            min_edge_value = edge_count[i];
            index_max_edges = i;
        }
    }

    BranchResult {
        branch_variable: index_max_edges,
        found_fixed_vars: HashMap::new(),
    }
}

/// Computes what branch will generate the most fixed variables via the preprocesser
pub fn most_fixed(solver: &BBSolver, node: &QuboBBNode) -> BranchResult {
    let mut most_fixed = 0;
    let mut branch_var = 0;

    for i in 0..solver.qubo.num_x() {
        if !node.fixed_variables.contains_key(&i) {
            let mut list_0 = node.fixed_variables.clone();
            let mut list_1 = node.fixed_variables.clone();

            list_0.insert(i, 0);
            list_1.insert(i, 1);

            let fixed_0 = preprocess_qubo(&solver.qubo_pp_form, &list_0, true).len();
            let fixed_1 = preprocess_qubo(&solver.qubo_pp_form, &list_1, true).len();

            let min_fixed = fixed_0.min(fixed_1);

            if min_fixed > most_fixed {
                most_fixed = min_fixed;
                branch_var = i;
            }
        }
    }

    BranchResult {
        branch_variable: branch_var,
        found_fixed_vars: HashMap::new(),
    }
}

/// #Panics if the node does not have an unfixed variable
pub fn first_not_fixed(solver: &BBSolver, node: &QuboBBNode) -> BranchResult {
    // scan through the variables and find the first one that is not fixed
    for i in 0..solver.qubo.num_x() {
        if !node.fixed_variables.contains_key(&i) {
            return BranchResult {
                branch_variable: i,
                found_fixed_vars: HashMap::new(),
            };
        }
    }
    panic!("No variable to branch on");
}

pub fn largest_diag(solver: &BBSolver, node: &QuboBBNode) -> BranchResult {
    // find the variable with the largest diagonal value in the Q matrix
    let mut max_diag = f64::NEG_INFINITY;
    let mut index_max_diag = 0;

    for i in 0..solver.qubo.num_x() {
        if !node.fixed_variables.contains_key(&i) {
            let diag_value = solver.qubo.q[[i, i]];

            if diag_value > max_diag {
                max_diag = diag_value;
                index_max_diag = i;
            }
        }
    }

    BranchResult {
        branch_variable: index_max_diag,
        found_fixed_vars: HashMap::new(),
    }
}

pub fn most_violated(solver: &BBSolver, node: &QuboBBNode) -> BranchResult {
    let mut most_violated = 1.0;
    let mut index_most_violated = 0;

    for i in 0..solver.qubo.num_x() {
        if !node.fixed_variables.contains_key(&i) {
            let violation = (node.solution[i] - 0.5).abs();

            if violation <= most_violated {
                most_violated = violation;
                index_most_violated = i;
            }
        }
    }

    BranchResult {
        branch_variable: index_most_violated,
        found_fixed_vars: HashMap::new(),
    }
}

pub fn full_strong_branching(solver: &BBSolver, node: &QuboBBNode) -> BranchResult {
    let unfixed_variables = (0..solver.qubo.num_x())
        .filter(|i| !node.fixed_variables.contains_key(i))
        .collect::<Vec<usize>>();
    let mut fixed_variables = node.fixed_variables.clone();

    let mut best_score = f64::NEG_INFINITY;
    let mut best_variable = *unfixed_variables.first().unwrap();

    let mut found_fixes: HashMap<usize, usize> = HashMap::new();

    for i in &unfixed_variables {
        let mut list_0 = fixed_variables.clone();
        let mut list_1 = fixed_variables.clone();

        list_0.insert(*i, 0);
        list_1.insert(*i, 1);

        // make new nodes
        let node_0 = QuboBBNode {
            lower_bound: 0.0,
            fixed_variables: list_0,
            solution: node.solution.clone(),
        };

        let node_1 = QuboBBNode {
            lower_bound: 0.0,
            fixed_variables: list_1,
            solution: node.solution.clone(),
        };

        let bound_0 = solver.subproblem_solver.solve_lower_bound(solver, &node_0);
        let bound_1 = solver.subproblem_solver.solve_lower_bound(solver, &node_1);

        // find the minimum of the two objectives
        let score = bound_0.0.min(bound_1.0);

        if bound_0.0 >= solver.best_solution_value {
            found_fixes.insert(*i, 1);
            fixed_variables.insert(*i, 1);
        } else if bound_1.0 >= solver.best_solution_value {
            found_fixes.insert(*i, 0);
            fixed_variables.insert(*i, 0);
        }

        if score > best_score {
            best_score = score;
            best_variable = *i;
        }
    }

    BranchResult {
        branch_variable: best_variable,
        found_fixed_vars: found_fixes,
    }
}

pub fn partial_strong_branching(solver: &BBSolver, node: &QuboBBNode) -> BranchResult {
    // first compute the approximate objective change for each variable
    let (zero_flip, one_flip) = compute_strong_branch(solver, node);
    let mut score = Array1::zeros(solver.qubo.num_x());
    let unfixed_vars = (0..solver.qubo.num_x())
        .filter(|i| !node.fixed_variables.contains_key(i))
        .collect::<Vec<usize>>();

    // scan through the unfixed variable and compute the scores
    for &i in &unfixed_vars {
        // find the minimum of the two objective changes
        score[i] = zero_flip[i].abs() * (one_flip[i].abs());
    }

    let mut indx = unfixed_vars.clone();
    indx.sort_by(|&i, &j| score[i].total_cmp(&score[j]).reverse());

    // test strong branching on the most likely candidate set of 5 variables

    let end = usize::min(5, unfixed_vars.len());

    let mut found_fixes = HashMap::new();

    let mut best_score = f64::NEG_INFINITY;
    let mut best_variable = *indx.first().unwrap();

    for i in 0..end {
        let mut list_0 = node.fixed_variables.clone();
        let mut list_1 = node.fixed_variables.clone();

        let j = *indx.get(i).unwrap();

        list_0.insert(j, 0);
        list_1.insert(j, 1);

        // make new nodes
        let node_0 = QuboBBNode {
            lower_bound: 0.0,
            fixed_variables: list_0,
            solution: node.solution.clone(),
        };

        let node_1 = QuboBBNode {
            lower_bound: 0.0,
            fixed_variables: list_1,
            solution: node.solution.clone(),
        };

        let bound_0 = solver.subproblem_solver.solve_lower_bound(solver, &node_0);
        let bound_1 = solver.subproblem_solver.solve_lower_bound(solver, &node_1);

        let score_i = bound_0.0.min(bound_1.0);

        if bound_0.0 >= solver.best_solution_value {
            found_fixes.insert(best_variable, 1);
        } else if bound_1.0 >= solver.best_solution_value {
            found_fixes.insert(best_variable, 0);
        }

        if score_i > best_score {
            best_score = score_i;
            best_variable = j;
        }
    }

    BranchResult {
        branch_variable: best_variable,
        found_fixed_vars: found_fixes,
    }
}

pub fn random(solver: &BBSolver, node: &QuboBBNode) -> BranchResult {
    // generate a prng
    let mut prng = PRNG {
        generator: JsfLarge::from(solver.options.seed as u64 + solver.nodes_visited as u64),
    };

    // generate a random index in the list of variables
    // This unwrap is 'safe' in that, the 32-bit system would crash trying to solve a QUBO with 2^32 variables
    let index = usize::try_from(prng.gen_u64() % solver.qubo.num_x() as u64).unwrap();

    // scan thru the variables and find the first one that is not fixed starting at the random point
    for i in index..solver.qubo.num_x() {
        if !node.fixed_variables.contains_key(&i) {
            return BranchResult {
                branch_variable: i,
                found_fixed_vars: HashMap::new(),
            };
        }
    }

    // scan through the variables and find the first one that is not fixed starting at the beginning
    for i in 0..index {
        if !node.fixed_variables.contains_key(&i) {
            return BranchResult {
                branch_variable: i,
                found_fixed_vars: HashMap::new(),
            };
        }
    }

    panic!("No Variable to branch on")
}

/// Branches on the variable that has an estimated worst result, pushing up the lower bound as fast as possible
pub fn worst_approximation(solver: &BBSolver, node: &QuboBBNode) -> BranchResult {
    let (zero_flip, one_flip) = compute_strong_branch(solver, node);

    // tracking variables for the worst approximation
    let mut worst_approximation = f64::NEG_INFINITY;
    let mut index_worst_approximation = 0;

    // scan through the variables and find the worst gain
    for i in 0..solver.qubo.num_x() {
        // if it is a fixed node, then skip it
        if node.fixed_variables.contains_key(&i) {
            continue;
        }

        // take the product of the approximate objective change for the zero and one flips as the metric
        let min_obj_gain = zero_flip[i].abs() * (one_flip[i].abs());

        // if it is the highest growing variable, then update the tracking variables
        if min_obj_gain > worst_approximation {
            worst_approximation = min_obj_gain;
            index_worst_approximation = i;
        }
    }

    BranchResult {
        branch_variable: index_worst_approximation,
        found_fixed_vars: HashMap::new(),
    }
}

/// Branches on the variable that has an estimated best result, keeping the lower bound as low as possible
pub fn best_approximation(solver: &BBSolver, node: &QuboBBNode) -> BranchResult {
    let (zero_flip, one_flip) = compute_strong_branch(solver, node);

    // tracking variables for the worst approximation
    let mut worst_approximation = f64::INFINITY;
    let mut index_best_approximation = 0;

    // scan through the variables and find the worst gain
    for i in 0..solver.qubo.num_x() {
        // if it is a fixed node, then skip it
        if node.fixed_variables.contains_key(&i) {
            continue;
        }

        // find the minimum of the two objective changes
        let max_obj_gain = zero_flip[i].abs().max(one_flip[i].abs());

        // if it is the highest growing variable, then update the tracking variables
        if max_obj_gain <= worst_approximation {
            worst_approximation = max_obj_gain;
            index_best_approximation = i;
        }
    }

    BranchResult {
        branch_variable: index_best_approximation,
        found_fixed_vars: HashMap::new(),
    }
}

pub fn compute_strong_branch(solver: &BBSolver, node: &QuboBBNode) -> (Array1<f64>, Array1<f64>) {
    let mut base_solution = Array1::<f64>::zeros(solver.qubo.num_x());
    let mut delta_zero = Array1::<f64>::zeros(solver.qubo.num_x());
    let mut delta_one = Array1::<f64>::zeros(solver.qubo.num_x());

    for i in 0..solver.qubo.num_x() {
        // fill in the current vector
        match node.fixed_variables.get(&i) {
            Some(val) => base_solution[i] = (*val) as f64,
            None => base_solution[i] = node.solution[i],
        }

        // compute the delta values for the zero and one flips
        delta_zero[i] = -base_solution[i];
        delta_one[i] = 1.0 - base_solution[i];
    }

    // build the intermediate vectors
    let q_jj = solver.qubo.q.diag().to_dense();
    let q_x = &solver.qubo.q * &base_solution;
    let x_q = &solver.qubo.q.transpose_view() * &base_solution;

    // build the result vectors
    let mut zero_result = Array1::zeros(solver.qubo.num_x());
    let mut one_result = Array1::zeros(solver.qubo.num_x());

    // compute the deltas in the objective compared to the current solution
    for i in 0..solver.qubo.num_x() {
        zero_result[i] = 0.5
            * delta_zero[i]
            * (delta_zero[i] * q_jj[i] + x_q[i] + q_x[i] + 2.0 * solver.qubo.c[i]);
        one_result[i] = 0.5
            * delta_one[i]
            * (delta_one[i] * q_jj[i] + x_q[i] + q_x[i] + 2.0 * solver.qubo.c[i]);
    }

    (zero_result, one_result)
}

pub fn round_robin(solver: &BBSolver, node: &QuboBBNode) -> BranchResult {
    // fun branching strat based on pseudo randomly picking a decent (and cheap branching strat)

    // make a random seed that is unique to each node
    let node_seed = node.fixed_variables.keys().sum::<usize>() as u64;
    let solver_seed = solver.options.seed as u64 + solver.nodes_solved as u64;

    let mut prng = PRNG {
        generator: JsfLarge::from(node_seed + solver_seed),
    };

    match prng.gen_u64() % 4 {
        0 => largest_edges(solver, node),
        1 => most_edges(solver, node),
        2 => worst_approximation(solver, node),
        3 => best_approximation(solver, node),
        _ => panic!("Random branch selection failed"),
    }
}
