use crate::branch_node::QuboBBNode;
use crate::branchbound::BBSolver;
use ndarray::Array1;
use smolprng::{JsfLarge, PRNG};
use crate::preprocess::preprocess_qubo;

pub enum BranchStrategy {
    FirstNotFixed,
    MostViolated,
    Random,
    WorstApproximation,
    BestApproximation,
    MostEdges,
    LargestEdges,
    MostFixed,
}

pub enum BranchStrategySelection {
    FirstNotFixed,
    MostViolated,
    Random,
    WorstApproximation,
    BestApproximation,
    MostEdges,
    LargestEdges,
    MostFixed,
}

impl BranchStrategy {
    pub fn make_branch(&self, bb_solver: &BBSolver, node: &QuboBBNode) -> usize {
        match self {
            Self::FirstNotFixed => first_not_fixed(bb_solver, node),
            Self::MostViolated => most_violated(bb_solver, node),
            Self::Random => random(bb_solver, node),
            Self::WorstApproximation => worst_approximation(bb_solver, node),
            Self::BestApproximation => best_approximation(bb_solver, node),
            Self::MostEdges => most_edges(bb_solver, node),
            Self::LargestEdges => largest_edges(bb_solver, node),
            Self::MostFixed => most_fixed(bb_solver, node),
        }
    }

    pub const fn get_branch_strategy(branch_strategy_selection: &BranchStrategySelection) -> Self {
        match branch_strategy_selection {
            BranchStrategySelection::FirstNotFixed => Self::FirstNotFixed,
            BranchStrategySelection::MostViolated => Self::MostViolated,
            BranchStrategySelection::Random => Self::Random,
            BranchStrategySelection::WorstApproximation => Self::WorstApproximation,
            BranchStrategySelection::BestApproximation => Self::BestApproximation,
            BranchStrategySelection::MostEdges => Self::MostEdges,
            BranchStrategySelection::LargestEdges => Self::LargestEdges,
            BranchStrategySelection::MostFixed => Self::MostFixed,
        }
    }
}

/// Branches on the variable that has the most edges in the graph equivalent to the QUBO
fn most_edges(solver: &BBSolver, node: &QuboBBNode) -> usize {
    // as a QUBO can be viewed as a graph, we can find the variable with the most (remaining) edges
    let mut edge_count = Array1::<usize>::zeros(solver.qubo.num_x());

    // scan through the Q matrix and count the number of edges
    for (_, (i, j)) in &solver.qubo.q {

        // if we have a fixed variable, then we can skip it
        if node.fixed_variables.contains_key(&i){
            continue;
        }

        // same again
        if node.fixed_variables.contains_key(&j){
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

    index_max_edges
}

/// Branches on the largest edges in the qubo, ones that are most likely to be the largest
/// determaning factor in the problem
fn largest_edges(solver: &BBSolver, node: &QuboBBNode) -> usize {
    // as a QUBO can be viewed as a graph, we can find the variable with the most (remaining) edges
    let mut edge_count = Array1::<f64>::zeros(solver.qubo.num_x());

    // scan through the Q matrix and count the number of edges
    for (&value, (i, j)) in &solver.qubo.q {

        // if we have a fixed variable, then we can skip it
        if node.fixed_variables.contains_key(&i){
            continue;
        }

        // same again
        if node.fixed_variables.contains_key(&j){
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

    index_max_edges
}

/// Computes what branch will generate the most fixed variables via the preprocesser
pub fn most_fixed(solver: &BBSolver, node: &QuboBBNode) -> usize {

    let mut most_fixed = 0;
    let mut branch_var = 0;

    for i in 0..solver.qubo.num_x() {
        if !node.fixed_variables.contains_key(&i) {

            let mut list_0 = node.fixed_variables.clone();
            let mut list_1 = node.fixed_variables.clone();

            list_0.insert(i, 0);
            list_1.insert(i, 1);

            let fixed_0 = preprocess_qubo(&solver.qubo, &list_0).len();
            let fixed_1 = preprocess_qubo(&solver.qubo, &list_1).len();

            let min_fixed = fixed_0.min(fixed_1);

            if min_fixed > most_fixed {
                most_fixed = min_fixed;
                branch_var = i;
            }
        }
    }

    branch_var
}

/// #Panics if the node does not have an unfixed variable
pub fn first_not_fixed(solver: &BBSolver, node: &QuboBBNode) -> usize {
    // scan through the variables and find the first one that is not fixed
    for i in 0..solver.qubo.num_x() {
        if !node.fixed_variables.contains_key(&i) {
            return i;
        }
    }
    panic!("No variable to branch on");
}

pub fn most_violated(solver: &BBSolver, node: &QuboBBNode) -> usize {
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

    index_most_violated
}

pub fn random(solver: &BBSolver, node: &QuboBBNode) -> usize {
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
            return i;
        }
    }

    // scan through the variables and find the first one that is not fixed starting at the beginning
    for i in 0..index {
        if !node.fixed_variables.contains_key(&i) {
            return i;
        }
    }

    panic!("No Variable to branch on")
}

/// Branches on the variable that has an estimated worst result, pushing up the lower bound as fast as possible
pub fn worst_approximation(solver: &BBSolver, node: &QuboBBNode) -> usize {
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
        let min_obj_gain = zero_flip[i].abs().max(one_flip[i].abs());

        // if it is the highest growing variable, then update the tracking variables
        if min_obj_gain > worst_approximation {
            worst_approximation = min_obj_gain;
            index_worst_approximation = i;
        }
    }

    index_worst_approximation
}

/// Branches on the variable that has an estimated best result, keeping the lower bound as low as possible
pub fn best_approximation(solver: &BBSolver, node: &QuboBBNode) -> usize {
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

    index_best_approximation
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
