use crate::branch_node::QuboBBNode;
use crate::branchbound::BBSolver;
use crate::preprocess::preprocess_qubo;
use ndarray::Array1;
use smolprng::{JsfLarge, PRNG};
use std::collections::HashMap;

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
    SmallestDiag,
    MoveingEdges,
    ConnectedComponents,
}

pub struct BranchResult {
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
            Self::SmallestDiag => smallest_diag(bb_solver, node),
            Self::MoveingEdges => moving_edges(bb_solver, node),
            Self::ConnectedComponents => connected_components(bb_solver, node),
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

fn connected_components(solver: &BBSolver, node: &QuboBBNode) -> BranchResult {
    let mut selected_variable = 0;
    let mut max_components = 0;
    // scan through the variables and find the variable that breaks the graph into the most components
    for i in 0..solver.qubo.num_x() {
        if !node.fixed_variables.contains_key(&i) {
            let mut list_0 = node.fixed_variables.clone();
            list_0.insert(i, 0);

            let num_components =
                crate::graph_utils::get_all_disconnected_graphs(&solver.qubo, &list_0);

            if num_components.len() > max_components {
                max_components = num_components.len();
                selected_variable = i;
            }
        }
    }

    BranchResult {
        branch_variable: selected_variable,
        found_fixed_vars: HashMap::new(),
    }
}

fn smallest_diag(solver: &BBSolver, node: &QuboBBNode) -> BranchResult {
    // find the variable with the lowest diagonal value in the Q matrix
    let mut max_diag = f64::INFINITY;
    let mut index_min_diag = 0;

    for i in 0..solver.qubo.num_x() {
        if !node.fixed_variables.contains_key(&i) {
            let diag_value = solver.qubo.q[[i, i]];

            if diag_value < max_diag {
                max_diag = diag_value;
                index_min_diag = i;
            }
        }
    }

    BranchResult {
        branch_variable: index_min_diag,
        found_fixed_vars: HashMap::new(),
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
    let mut branch_variable = 0;

    let mut found_fixed_vars = HashMap::new();

    for i in 0..solver.qubo.num_x() {
        if !node.fixed_variables.contains_key(&i) && !found_fixed_vars.contains_key(&i) {
            let mut list_0 = node.fixed_variables.clone();
            let mut list_1 = node.fixed_variables.clone();

            list_0.insert(i, 0);
            list_1.insert(i, 1);

            // add in any already found fixed variables
            for (&key, &value) in &found_fixed_vars{
                list_0.insert(key, value);
                list_1.insert(key, value);
            }

            let fixed_0 = preprocess_qubo(&solver.qubo_pp_form, &list_0, true);
            let fixed_1 = preprocess_qubo(&solver.qubo_pp_form, &list_1, true);

            for (&key, &value) in &fixed_0 {
                // if this variable is not already fixed then we have the potential to fix it via a check
                if !node.fixed_variables.contains_key(&key)  && fixed_1.contains_key(&key) {
                    // if x_i being fixed to any value forces x_j to be the same value
                    // then we can fix it to that value

                    if fixed_1[&key] == value {
                        found_fixed_vars.insert(key, value);
                    }
                }
            }

            let min_fixed = fixed_0.len().min(fixed_1.len());

            if min_fixed > most_fixed {
                most_fixed = min_fixed;
                branch_variable = i;
            }
        }
    }

    BranchResult {
        branch_variable,
        found_fixed_vars,
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

        for (&key, &value) in &found_fixes{
            list_0.insert(key, value);
            list_1.insert(key, value);
        }

        list_0 = preprocess_qubo(&solver.qubo_pp_form, &list_0, true);
        list_1 = preprocess_qubo(&solver.qubo_pp_form, &list_1, true);

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

        // solve for the
        let bound_0 = solver
            .subproblem_solver
            .solve_lower_bound(solver, &node_0, None);
        let bound_1 = solver
            .subproblem_solver
            .solve_lower_bound(solver, &node_1, None);

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

    let mut fixed_variables = node.fixed_variables.clone();

    // test strong branching on the most likely candidate set of 5 variables

    let end = usize::min(15, unfixed_vars.len());

    let mut found_fixes = HashMap::new();

    let mut best_score = f64::NEG_INFINITY;
    let mut best_variable = *indx.first().unwrap();

    for i in 0..end {
        let mut list_0 = fixed_variables.clone();
        let mut list_1 = fixed_variables.clone();

        let j = *indx.get(i).unwrap();

        list_0.insert(j, 0);
        list_1.insert(j, 1);

        for (&key, &value) in &found_fixes{
            list_0.insert(key, value);
            list_1.insert(key, value);
        }

        list_0 = preprocess_qubo(&solver.qubo_pp_form, &list_0, true);
        list_1 = preprocess_qubo(&solver.qubo_pp_form, &list_1, true);

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

        let bound_0 = solver
            .subproblem_solver
            .solve_lower_bound(solver, &node_0, None);
        let bound_1 = solver
            .subproblem_solver
            .solve_lower_bound(solver, &node_1, None);

        let score_i = bound_0.0.min(bound_1.0);

        if bound_0.0 >= solver.best_solution_value {
            found_fixes.insert(j, 1);
            fixed_variables.insert(j, 1);
        } else if bound_1.0 >= solver.best_solution_value {
            found_fixes.insert(j, 0);
            fixed_variables.insert(j, 0);
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
    // makes the assumption that the node solution is the solution of the relaxed problem above

    // create the result vectors
    let mut zero_result = Array1::zeros(solver.qubo.num_x());
    let mut one_result = Array1::zeros(solver.qubo.num_x());

    // compute the deltas in the objective compared to the current solution
    for i in 0..solver.qubo.num_x() {
        let diag_ii = solver.qubo.q[[i, i]];
        zero_result[i] = diag_ii * node.solution[i] * node.solution[i];
        one_result[i] = diag_ii * (1.0 - node.solution[i]) * (1.0 - node.solution[i]);
    }

    (zero_result, one_result)
}

pub fn moving_edges(solver: &BBSolver, node: &QuboBBNode) -> BranchResult {
    // this is basically largest edges with a bias to movement in the binary violation
    let mut edge_size = Array1::<f64>::zeros(solver.qubo.num_x());

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

        edge_size[i] += value.abs();
        edge_size[j] += value.abs();
    }

    // find the variable with the most edges (fixed variables in the node are not counted)
    let mut min_edge_value = -10.0;
    let mut index_max_edges = 0;

    for i in 0..solver.qubo.num_x() {
        if !node.fixed_variables.contains_key(&i) {
            let movement = node.solution[i].min(1.0 - node.solution[i]);
            if edge_size[i] * movement > min_edge_value {
                min_edge_value = edge_size[i] * movement;
                index_max_edges = i;
            }
        }
    }

    BranchResult {
        branch_variable: index_max_edges,
        found_fixed_vars: HashMap::new(),
    }
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
