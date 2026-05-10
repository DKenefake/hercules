//! Roof duality routines based on the rooted-noose / max-flow
//! construction of Boros, Hammer, Sun, and Tavares (2008).
//!
//! The implementation here follows the paper's graph-theoretic route rather
//! than a generic QPBO formulation:
//! - project out already fixed variables,
//! - convert the reduced QUBO into a rooted bi-form,
//! - build the literal network on `x_i` and `xÌ„_i`,
//! - compute the maximum flow from `x_0` to `xÌ„_0`,
//! - read strong persistencies from the source side of the residual graph.

use crate::branch_node::QuboBBNode;
use crate::branch_subproblem::{SubProblemOptions, SubProblemResult, SubProblemSolver};
use crate::branchbound::BBSolver;
use crate::preprocess::make_sub_problem;
use crate::qubo::Qubo;
use ndarray::Array1;
use petgraph::algo::maximum_flow::dinics;
use petgraph::graph::{DiGraph, EdgeIndex, NodeIndex};
use petgraph::visit::EdgeRef;
use std::collections::{HashMap, VecDeque};

/// Result of a roof-duality pass.
#[derive(Debug, Clone)]
pub struct RoofDualityResult {
    pub fixed_variables: HashMap<usize, usize>,
    pub lower_bound: Option<f64>,
    pub unlabeled_variables: Vec<usize>,
}

/// A bi-term in the rooted bi-form.
///
/// `Equal` is the XOR-type term `x_i xÌ„_j + xÌ„_i x_j`, which vanishes when
/// the two variables are equal.
///
/// `Different` is the XNOR-type term `x_i x_j + xÌ„_i xÌ„_j`, which vanishes when
/// the two variables are different.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BiTermKind {
    Equal,
    Different,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BiTerm {
    pub i: usize,
    pub j: usize,
    pub weight: f64,
    pub kind: BiTermKind,
}

#[derive(Debug, Clone)]
pub struct ReducedRoofDualProblem {
    /// Number of non-root variables in the reduced problem.
    pub num_variables: usize,
    /// Constant objective shift such that:
    /// `original objective = constant + rooted bi-form(root=1, x)`.
    pub constant: f64,
    /// Rooted bi-form terms over variables `{0, 1, ..., num_variables}`, where
    /// variable `0` is the artificial root fixed to value `1`.
    pub biterms: Vec<BiTerm>,
    /// Maps original variable indices to reduced indices in `0..num_variables`.
    pub original_to_reduced: HashMap<usize, usize>,
}

#[derive(Clone)]
pub struct RoofDualSolver {}

impl RoofDualSolver {
    pub fn new(_: &Qubo) -> Self {
        Self {}
    }
}

impl SubProblemSolver for RoofDualSolver {
    fn solve_lower_bound(
        &self,
        bbsolver: &BBSolver,
        node: &QuboBBNode,
        _: Option<SubProblemOptions>,
    ) -> SubProblemResult {
        let result = roof_duality_presolve(&bbsolver.qubo_pp_form, &node.fixed_variables);

        let mut solution = Array1::from_elem(bbsolver.qubo.num_x(), 0.5);
        for (&index, &value) in &node.fixed_variables {
            solution[index] = value as f64;
        }
        for (&index, &value) in &result.fixed_variables {
            solution[index] = value as f64;
        }

        (result.lower_bound.unwrap_or(f64::NEG_INFINITY), solution)
    }
}

/// Build the rooted bi-form corresponding to the reduced subproblem.
pub fn build_reduced_roof_dual_problem(
    qubo: &Qubo,
    fixed_variables: &HashMap<usize, usize>,
) -> ReducedRoofDualProblem {
    let (sub_problem, original_to_reduced, mut constant) = make_sub_problem(qubo, fixed_variables);
    let num_variables = sub_problem.num_x();

    let mut linear = sub_problem.c.to_vec();
    let mut biterms = Vec::new();
    let mut diagonal = vec![0.0; num_variables];
    let mut pairwise = HashMap::<(usize, usize), f64>::new();

    for (&value, (i, j)) in &sub_problem.q {
        if i == j {
            diagonal[i] += value;
        } else {
            let key = if i < j { (i, j) } else { (j, i) };
            *pairwise.entry(key).or_insert(0.0) += 0.5 * value;
        }
    }

    for i in 0..num_variables {
        linear[i] += 0.5 * diagonal[i];
    }

    for ((i, j), coeff) in pairwise {
        if coeff == 0.0 {
            continue;
        }

        if coeff > 0.0 {
            // x_i x_j = 1/2 * Different(i,j) + 1/2 * x_i + 1/2 * x_j - 1/2
            biterms.push(BiTerm {
                i: i + 1,
                j: j + 1,
                weight: 0.5 * coeff,
                kind: BiTermKind::Different,
            });
            linear[i] += 0.5 * coeff;
            linear[j] += 0.5 * coeff;
            constant -= 0.5 * coeff;
        } else {
            // -x_i x_j = 1/2 * Equal(i,j) - 1/2 * x_i - 1/2 * x_j
            biterms.push(BiTerm {
                i: i + 1,
                j: j + 1,
                weight: -0.5 * coeff,
                kind: BiTermKind::Equal,
            });
            linear[i] += 0.5 * coeff;
            linear[j] += 0.5 * coeff;
        }
    }

    for (i, &coeff) in linear.iter().enumerate() {
        if coeff == 0.0 {
            continue;
        }

        if coeff > 0.0 {
            // x_i = Different(root, i)
            biterms.push(BiTerm {
                i: 0,
                j: i + 1,
                weight: coeff,
                kind: BiTermKind::Different,
            });
        } else {
            // -x_i = Equal(root, i) - 1
            biterms.push(BiTerm {
                i: 0,
                j: i + 1,
                weight: -coeff,
                kind: BiTermKind::Equal,
            });
            constant += coeff;
        }
    }

    ReducedRoofDualProblem {
        num_variables,
        constant,
        biterms,
        original_to_reduced,
    }
}

/// Solve the roof-duality relaxation and extract strong persistencies.
pub fn roof_duality_presolve(
    qubo: &Qubo,
    fixed_variables: &HashMap<usize, usize>,
) -> RoofDualityResult {
    let reduced_problem = build_reduced_roof_dual_problem(qubo, fixed_variables);

    if reduced_problem.num_variables == 0 {
        return RoofDualityResult {
            fixed_variables: HashMap::new(),
            lower_bound: Some(reduced_problem.constant),
            unlabeled_variables: Vec::new(),
        };
    }

    let reduced_result = solve_roof_dual_network(&reduced_problem);
    map_reduced_result(reduced_result, &reduced_problem.original_to_reduced)
}

fn solve_roof_dual_network(problem: &ReducedRoofDualProblem) -> RoofDualityResult {
    let num_biform_variables = problem.num_variables + 1;
    let num_literal_nodes = 2 * num_biform_variables;

    let mut graph = DiGraph::<(), f64>::new();
    let nodes = (0..num_literal_nodes)
        .map(|_| graph.add_node(()))
        .collect::<Vec<_>>();

    for term in &problem.biterms {
        let capacity = 0.5 * term.weight;
        if capacity <= 0.0 {
            continue;
        }

        let (u1, v1, u2, v2) = match term.kind {
            BiTermKind::Equal => (
                literal_node(term.i, false),
                literal_node(term.j, false),
                literal_node(term.i, true),
                literal_node(term.j, true),
            ),
            BiTermKind::Different => (
                literal_node(term.i, false),
                literal_node(term.j, true),
                literal_node(term.i, true),
                literal_node(term.j, false),
            ),
        };

        add_undirected_arc_pair(&mut graph, nodes[u1], nodes[v1], capacity);
        add_undirected_arc_pair(&mut graph, nodes[u2], nodes[v2], capacity);
    }

    let source = nodes[literal_node(0, false)];
    let sink = nodes[literal_node(0, true)];
    let (flow_value, flows) = dinics(&graph, source, sink);
    let residual = ResidualNetwork::from_graph_and_flow(&graph, &flows);
    let source_side = residual.reachable_from(source);

    let mut fixed_variables = HashMap::new();
    let mut unlabeled_variables = Vec::new();

    for reduced_var in 0..problem.num_variables {
        let bi_var = reduced_var + 1;
        let pos_in_source = source_side[literal_node(bi_var, false)];
        let neg_in_source = source_side[literal_node(bi_var, true)];

        match (pos_in_source, neg_in_source) {
            (true, false) => {
                fixed_variables.insert(reduced_var, 1);
            }
            (false, true) => {
                fixed_variables.insert(reduced_var, 0);
            }
            _ => {
                unlabeled_variables.push(reduced_var);
            }
        }
    }

    RoofDualityResult {
        fixed_variables,
        lower_bound: Some(problem.constant + flow_value),
        unlabeled_variables,
    }
}

fn add_undirected_arc_pair(
    graph: &mut DiGraph<(), f64>,
    u: NodeIndex,
    v: NodeIndex,
    capacity: f64,
) {
    graph.add_edge(u, v, capacity);
    graph.add_edge(v, u, capacity);
}

fn literal_node(variable: usize, complemented: bool) -> usize {
    2 * variable + usize::from(complemented)
}

fn invert_index_map(map: &HashMap<usize, usize>) -> HashMap<usize, usize> {
    map.iter().map(|(&original, &reduced)| (reduced, original)).collect()
}

fn map_reduced_result(
    reduced_result: RoofDualityResult,
    original_to_reduced: &HashMap<usize, usize>,
) -> RoofDualityResult {
    let reduced_to_original = invert_index_map(original_to_reduced);
    let fixed_variables = reduced_result
        .fixed_variables
        .into_iter()
        .filter_map(|(reduced_index, value)| {
            reduced_to_original
                .get(&reduced_index)
                .copied()
                .map(|original_index| (original_index, value))
        })
        .collect();

    let unlabeled_variables = reduced_result
        .unlabeled_variables
        .into_iter()
        .filter_map(|reduced_index| reduced_to_original.get(&reduced_index).copied())
        .collect();

    RoofDualityResult {
        fixed_variables,
        lower_bound: reduced_result.lower_bound,
        unlabeled_variables,
    }
}

#[derive(Debug, Clone)]
struct ResidualNetwork {
    outgoing: Vec<Vec<usize>>,
}

impl ResidualNetwork {
    fn from_graph_and_flow(graph: &DiGraph<(), f64>, flows: &[f64]) -> Self {
        let mut outgoing = vec![Vec::new(); graph.node_count()];
        let eps = 1e-12;

        for edge in graph.edge_references() {
            let edge_index: EdgeIndex = edge.id();
            let idx = edge_index.index();
            let flow = flows[idx];
            let capacity = *edge.weight();
            let u = edge.source().index();
            let v = edge.target().index();

            if capacity - flow > eps {
                outgoing[u].push(v);
            }
            if flow > eps {
                outgoing[v].push(u);
            }
        }

        Self { outgoing }
    }

    fn reachable_from(&self, source: NodeIndex) -> Vec<bool> {
        let mut seen = vec![false; self.outgoing.len()];
        let mut queue = VecDeque::new();
        seen[source.index()] = true;
        queue.push_back(source.index());

        while let Some(node) = queue.pop_front() {
            for &next in &self.outgoing[node] {
                if !seen[next] {
                    seen[next] = true;
                    queue.push_back(next);
                }
            }
        }

        seen
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preprocess::shift_qubo;
    use crate::subproblemsolvers::enumerate_qubo::enumerate_solve;
    use crate::tests::make_test_prng;
    use ndarray::Array1;
    use smolprng::PRNG;
    use sprs::TriMat;
    use std::collections::HashMap;

    fn make_small_qubo() -> Qubo {
        let mut q = TriMat::new((2, 2));
        q.add_triplet(0, 0, 2.0);
        q.add_triplet(0, 1, -4.0);
        q.add_triplet(1, 1, 1.0);
        let c = Array1::from_vec(vec![1.0, -2.0]);
        Qubo::new_with_c(q.to_csr(), c)
    }

    fn make_paper_example_1_qubo() -> Qubo {
        let mut q = TriMat::new((5, 5));
        q.add_triplet(0, 1, -20.0);
        q.add_triplet(0, 2, 24.0);
        q.add_triplet(0, 4, -12.0);
        q.add_triplet(1, 2, -28.0);
        q.add_triplet(2, 3, 8.0);
        q.add_triplet(3, 4, -20.0);

        let c = Array1::from_vec(vec![-3.0, 12.0, -1.0, 3.0, 14.0]);
        Qubo::new_with_c(q.to_csr(), c)
    }

    fn evaluate_reduced_problem(problem: &ReducedRoofDualProblem, mask: usize) -> f64 {
        let mut value = problem.constant;

        for term in &problem.biterms {
            let xi = if term.i == 0 {
                1usize
            } else {
                (mask >> (term.i - 1)) & 1
            };
            let xj = if term.j == 0 {
                1usize
            } else {
                (mask >> (term.j - 1)) & 1
            };

            let term_value = match term.kind {
                BiTermKind::Equal => usize::from(xi != xj) as f64,
                BiTermKind::Different => usize::from(xi == xj) as f64,
            };

            value += term.weight * term_value;
        }

        value
    }

    fn evaluate_biform(problem: &ReducedRoofDualProblem, assignment: &[usize]) -> f64 {
        let mut value = 0.0;

        for term in &problem.biterms {
            let xi = assignment[term.i];
            let xj = assignment[term.j];

            let term_value = match term.kind {
                BiTermKind::Equal => usize::from(xi != xj) as f64,
                BiTermKind::Different => usize::from(xi == xj) as f64,
            };

            value += term.weight * term_value;
        }

        value
    }

    fn exact_value_with_fixings(qubo: &Qubo, fixed_variables: &HashMap<usize, usize>) -> f64 {
        let (sub_qubo, mapping, _constant) =
            crate::preprocess::make_sub_problem(qubo, fixed_variables);
        let (sub_value, sub_solution) = enumerate_solve(&sub_qubo);

        let mut full_solution = Array1::<usize>::zeros(qubo.num_x());
        for (&index, &value) in fixed_variables {
            full_solution[index] = value;
        }
        for (&original, &sub_index) in &mapping {
            full_solution[original] = sub_solution[sub_index];
        }

        let _ = sub_value;
        qubo.eval_usize(&full_solution)
    }

    fn assert_roof_dual_fixings_are_persistent(
        qubo: &Qubo,
        fixed_variables: &HashMap<usize, usize>,
        check_lower_bound: bool,
    ) {
        let result = roof_duality_presolve(qubo, fixed_variables);
        let best = exact_value_with_fixings(qubo, fixed_variables);

        if check_lower_bound {
            assert!(
                result.lower_bound.unwrap() <= best + 1e-9,
                "roof dual lower bound must remain valid"
            );
        }

        for (&index, &value) in &result.fixed_variables {
            let mut contradictory_fixings = fixed_variables.clone();
            contradictory_fixings.insert(index, 1 - value);
            let contradictory_value = exact_value_with_fixings(qubo, &contradictory_fixings);

            assert!(
                contradictory_value > best + 1e-9,
                "roof dual fixed x_{index}={value}, but the opposite fixing remains optimal"
            );
        }
    }

    #[test]
    fn test_build_reduced_roof_dual_problem_matches_original_energy() {
        let qubo = make_small_qubo();
        let problem = build_reduced_roof_dual_problem(&qubo, &HashMap::new());

        for mask in 0..(1usize << problem.num_variables) {
            let x = Array1::from_vec(vec![(mask & 1), ((mask >> 1) & 1)]);
            let qubo_value = qubo.eval_usize(&x);
            let roof_value = evaluate_reduced_problem(&problem, mask);
            assert!((qubo_value - roof_value).abs() <= 1e-9);
        }
    }

    #[test]
    fn test_roof_dual_linear_problem_fixes_positive_cost_to_zero() {
        let q = TriMat::<f64>::new((1, 1)).to_csr();
        let qubo = Qubo::new_with_c(q, Array1::from_vec(vec![3.0]));

        let result = roof_duality_presolve(&qubo, &HashMap::new());

        assert_eq!(result.fixed_variables.get(&0), Some(&0));
        assert_eq!(result.lower_bound, Some(0.0));
    }

    #[test]
    fn test_roof_dual_linear_problem_fixes_negative_cost_to_one() {
        let q = TriMat::<f64>::new((1, 1)).to_csr();
        let qubo = Qubo::new_with_c(q, Array1::from_vec(vec![-2.0]));

        let result = roof_duality_presolve(&qubo, &HashMap::new());

        assert_eq!(result.fixed_variables.get(&0), Some(&1));
        assert_eq!(result.lower_bound, Some(-2.0));
    }

    #[test]
    fn test_roof_dual_lower_bound_is_valid_on_small_instance() {
        let qubo = make_small_qubo();
        let result = roof_duality_presolve(&qubo, &HashMap::new());

        let mut best = f64::INFINITY;
        for mask in 0..4usize {
            let x = Array1::from_vec(vec![(mask & 1), ((mask >> 1) & 1)]);
            best = best.min(qubo.eval_usize(&x));
        }

        assert!(result.lower_bound.unwrap() <= best + 1e-9);
    }

    #[test]
    fn test_paper_example_1_biform_terms_match_published_example() {
        // Example 1 of:
        // Boros, Hammer, Sun, and Tavares, "A max-flow approach to improved
        // lower bounds for quadratic unconstrained binary optimization (QUBO)",
        // Discrete Optimization 5 (2008) 501-529.
        let qubo = make_paper_example_1_qubo();
        let problem = build_reduced_roof_dual_problem(&qubo, &HashMap::new());

        assert_eq!(problem.constant, -13.0);
        assert_eq!(problem.biterms.len(), 8);
        assert!(problem.biterms.contains(&BiTerm {
            i: 0,
            j: 1,
            weight: 5.0,
            kind: BiTermKind::Equal,
        }));
        assert!(problem.biterms.contains(&BiTerm {
            i: 0,
            j: 5,
            weight: 6.0,
            kind: BiTermKind::Different,
        }));
        assert!(problem.biterms.contains(&BiTerm {
            i: 1,
            j: 2,
            weight: 5.0,
            kind: BiTermKind::Equal,
        }));
        assert!(problem.biterms.contains(&BiTerm {
            i: 1,
            j: 3,
            weight: 6.0,
            kind: BiTermKind::Different,
        }));
        assert!(problem.biterms.contains(&BiTerm {
            i: 1,
            j: 5,
            weight: 3.0,
            kind: BiTermKind::Equal,
        }));
        assert!(problem.biterms.contains(&BiTerm {
            i: 2,
            j: 3,
            weight: 7.0,
            kind: BiTermKind::Equal,
        }));
        assert!(problem.biterms.contains(&BiTerm {
            i: 3,
            j: 4,
            weight: 2.0,
            kind: BiTermKind::Different,
        }));
        assert!(problem.biterms.contains(&BiTerm {
            i: 4,
            j: 5,
            weight: 5.0,
            kind: BiTermKind::Equal,
        }));
    }

    #[test]
    fn test_paper_example_2_matches_fixing_x1_to_one() {
        let qubo = make_paper_example_1_qubo();
        let problem = build_reduced_roof_dual_problem(&qubo, &HashMap::new());

        for mask in 0..(1usize << 5) {
            let x0 = (mask & 1) as usize;
            let x2 = ((mask >> 1) & 1) as usize;
            let x3 = ((mask >> 2) & 1) as usize;
            let x4 = ((mask >> 3) & 1) as usize;
            let x5 = ((mask >> 4) & 1) as usize;

            let original = evaluate_biform(&problem, &[x0, 1, x2, x3, x4, x5]);
            let reduced = 21.0
                - 11.0 * x0 as f64
                + 2.0 * x2 as f64
                + 11.0 * x3 as f64
                + 3.0 * x4 as f64
                - 4.0 * x5 as f64
                + 12.0 * x0 as f64 * x5 as f64
                - 14.0 * x2 as f64 * x3 as f64
                + 4.0 * x3 as f64 * x4 as f64
                - 10.0 * x4 as f64 * x5 as f64;

            assert!((original - reduced).abs() <= 1e-9);
        }
    }

    #[test]
    fn test_generated_small_qubos_match_exhaustive_persistencies() {
        let mut prng: PRNG<_> = make_test_prng();

        for case_idx in 0..8 {
            let num_x = 10;
            let sparsity = 0.35 + 0.1 * ((case_idx % 3) as f64);
            let qubo = Qubo::make_random_qubo(num_x, &mut prng, sparsity);
            assert_roof_dual_fixings_are_persistent(&qubo, &HashMap::new(), true);
        }
    }

    #[test]
    fn test_generated_small_qubos_with_existing_fixings_match_exhaustive_persistencies() {
        let mut prng: PRNG<_> = make_test_prng();

        for case_idx in 0..6 {
            let num_x = 10;
            let sparsity = 0.4 + 0.1 * ((case_idx % 2) as f64);
            let qubo = Qubo::make_random_qubo(num_x, &mut prng, sparsity);

            let mut fixed_variables = HashMap::new();
            fixed_variables.insert(case_idx % num_x, case_idx % 2);

            assert_roof_dual_fixings_are_persistent(&qubo, &fixed_variables, false);
        }
    }

    #[test]
    fn test_repo_test_qubo_convex_shift_matches_enumeration() {
        let qubo = Qubo::read_qubo("test_data/test.qubo");
        let convex = qubo.convex_symmetric_form();
        let shifted = shift_qubo(&convex);

        let result = roof_duality_presolve(&shifted, &HashMap::new());
        let (exact_value, exact_solution) = enumerate_solve(&shifted);

        assert!(
            (result.lower_bound.unwrap() - exact_value).abs() <= 1e-9,
            "roof-dual lower bound should match the enumerated optimum on test.qubo after convexification and diagonal shifting"
        );

        for (&index, &value) in &result.fixed_variables {
            assert_eq!(
                exact_solution[index], value,
                "roof dual fixed x_{index}={value}, but enumeration found {}",
                exact_solution[index]
            );
        }
    }
}
