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
use crate::branch_subproblem::{
    BasicSubProblemResult, SubProblemOptions, SubProblemResult, SubProblemSolver,
};
use crate::branchbound::BBSolver;
use crate::constraint::Constraint;
use crate::preprocess::make_sub_problem;
use crate::qubo::Qubo;
use crate::FixedVarMap;
use ndarray::Array1;
use std::collections::HashMap;

pub(crate) mod flow;
use flow::FlowNetwork;
mod relations;
use relations::RelationPenalties;

/// Result of a roof-duality pass.
#[derive(Debug, Clone)]
pub struct RoofDualityResult {
    /// Strong persistencies, valid in every optimum of the incoming subproblem.
    pub fixed_variables: FixedVarMap,
    /// Jointly compatible, optimum-preserving reductions. Apply as a batch;
    /// never export these as all-optima implications. Empty in strong-only APIs.
    pub weak_fixed_variables: FixedVarMap,
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

/// Immutable binary coefficients shared by repeated roof-dual node solves.
/// Node projection does not construct a sparse matrix or a pair hash table.
#[derive(Clone)]
pub struct PreparedRoofDual {
    linear: Vec<f64>,
    edges: Vec<(usize, usize, f64)>,
    relations: RelationPenalties,
}

impl PreparedRoofDual {
    pub fn new(qubo: &Qubo) -> Self {
        let mut linear = qubo.c.to_vec();
        let mut pairs = rustc_hash::FxHashMap::<(usize, usize), f64>::default();
        for (&value, (i, j)) in &qubo.q {
            if i == j {
                linear[i] += 0.5 * value;
            } else {
                *pairs.entry((i.min(j), i.max(j))).or_default() += 0.5 * value;
            }
        }
        let mut edges: Vec<_> = pairs
            .into_iter()
            .filter(|(_, weight)| *weight != 0.0)
            .map(|((i, j), weight)| (i, j, weight))
            .collect();
        edges.sort_unstable_by_key(|&(i, j, _)| (i, j));
        Self {
            linear,
            edges,
            relations: RelationPenalties::default(),
        }
    }

    /// Caller must prove these relations for all retained root optima. Bounds
    /// and fixings then concern the penalized problem, not arbitrary original
    /// node optima that violate the relations. Never pass weak or cutoff-based
    /// probe choices here. An empty slice clears the previous solve's relations.
    pub(crate) fn set_strong_relations(&mut self, relations: &[Constraint]) {
        self.relations = RelationPenalties::new(&self.linear, &self.edges, relations);
    }

    fn project(&self, fixed: &FixedVarMap) -> ReducedRoofDualProblem {
        let (coefficients, edges, mut constant) =
            self.relations.coefficients(&self.linear, &self.edges);
        let mut values = vec![None; self.linear.len()];
        for (&i, &value) in fixed {
            values[i] = Some(value);
        }
        let free = self.linear.len() - fixed.len();
        let mut indices = vec![usize::MAX; self.linear.len()];
        let mut original_to_reduced = HashMap::with_capacity(free);
        let mut linear = Vec::with_capacity(free);
        for (i, &coefficient) in coefficients.iter().enumerate() {
            if let Some(value) = values[i] {
                constant += coefficient * value as f64;
            } else {
                indices[i] = linear.len();
                original_to_reduced.insert(i, linear.len());
                linear.push(coefficient);
            }
        }
        let free_edges = edges
            .iter()
            .filter(|&&(i, j, _)| values[i].is_none() && values[j].is_none())
            .count();
        let mut biterms = Vec::with_capacity(free_edges + free);
        for &(i, j, coefficient) in edges {
            match (values[i], values[j]) {
                (Some(a), Some(b)) => constant += coefficient * (a * b) as f64,
                (None, Some(b)) => linear[indices[i]] += coefficient * b as f64,
                (Some(a), None) => linear[indices[j]] += coefficient * a as f64,
                (None, None) => biterms.push(BiTerm {
                    i: indices[i] + 1,
                    j: indices[j] + 1,
                    weight: 0.5 * coefficient.abs(),
                    kind: if coefficient > 0.0 {
                        BiTermKind::Different
                    } else {
                        BiTermKind::Equal
                    },
                }),
            }
        }
        for term in &biterms {
            let shift = match term.kind {
                BiTermKind::Different => {
                    constant -= term.weight;
                    term.weight
                }
                BiTermKind::Equal => -term.weight,
            };
            linear[term.i - 1] += shift;
            linear[term.j - 1] += shift;
        }
        append_linear_biterms(&linear, &mut biterms, &mut constant);
        ReducedRoofDualProblem {
            num_variables: free,
            constant,
            biterms,
            original_to_reduced,
        }
    }

    pub fn solve(&self, fixed: &FixedVarMap) -> RoofDualityResult {
        self.solve_with_mode(fixed, false)
    }

    fn solve_with_mode(&self, fixed: &FixedVarMap, weak: bool) -> RoofDualityResult {
        solve_reduced_roof_dual_problem(&self.project(fixed), weak)
    }

    pub fn solve_iterative(&self, fixed: &FixedVarMap, iter_limit: usize) -> RoofDualityResult {
        iterate_roof_duality(fixed, iter_limit, |current| self.solve(current))
    }

    /// Preserve at least one optimum, including SCC-based weak persistencies.
    /// Later deductions conditional on a weak choice are also classified weak.
    pub fn solve_iterative_with_weak_persistencies(
        &self,
        fixed: &FixedVarMap,
        iter_limit: usize,
    ) -> RoofDualityResult {
        iterate_roof_duality(fixed, iter_limit, |current| {
            self.solve_with_mode(current, true)
        })
    }
}

#[derive(Clone)]
pub struct RoofDualSolver {}

impl RoofDualSolver {
    pub const fn new(_: &Qubo) -> Self {
        Self {}
    }
}

impl SubProblemSolver for RoofDualSolver {
    fn for_reduced_qubo(&self, qubo: &Qubo) -> Option<Box<dyn SubProblemSolver + Sync>> {
        Some(Box::new(Self::new(qubo)))
    }

    fn solve_lower_bound(
        &self,
        bbsolver: &BBSolver,
        node: &QuboBBNode,
        _: Option<SubProblemOptions>,
    ) -> Box<dyn SubProblemResult> {
        let result = roof_duality_presolve(&bbsolver.qubo_pp_form, &node.fixed_variables);

        let mut solution = Array1::from_elem(bbsolver.qubo.num_x(), 0.5);
        for (&index, &value) in &node.fixed_variables {
            solution[index] = value as f64;
        }
        for (&index, &value) in &result.fixed_variables {
            solution[index] = value as f64;
        }

        Box::new(BasicSubProblemResult {
            lower_bound: result.lower_bound.unwrap_or(f64::NEG_INFINITY),
            relaxed_solution: solution,
        })
    }
}

/// Build the rooted bi-form corresponding to the reduced subproblem.
pub fn build_reduced_roof_dual_problem(
    qubo: &Qubo,
    fixed_variables: &FixedVarMap,
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

    append_linear_biterms(&linear, &mut biterms, &mut constant);

    ReducedRoofDualProblem {
        num_variables,
        constant,
        biterms,
        original_to_reduced,
    }
}

fn append_linear_biterms(linear: &[f64], biterms: &mut Vec<BiTerm>, constant: &mut f64) {
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
            *constant += coeff;
        }
    }
}

/// Solve the roof-duality relaxation and extract strong persistencies.
pub fn roof_duality_presolve(qubo: &Qubo, fixed_variables: &FixedVarMap) -> RoofDualityResult {
    let reduced_problem = build_reduced_roof_dual_problem(qubo, fixed_variables);
    solve_reduced_roof_dual_problem(&reduced_problem, false)
}

fn solve_reduced_roof_dual_problem(
    reduced_problem: &ReducedRoofDualProblem,
    weak_persistencies: bool,
) -> RoofDualityResult {
    if reduced_problem.num_variables == 0 {
        return RoofDualityResult {
            fixed_variables: FixedVarMap::default(),
            weak_fixed_variables: FixedVarMap::default(),
            lower_bound: Some(reduced_problem.constant),
            unlabeled_variables: Vec::new(),
        };
    }

    let reduced_result = solve_roof_dual_network(reduced_problem, weak_persistencies);
    map_reduced_result(reduced_result, &reduced_problem.original_to_reduced)
}

/// Repeatedly apply roof duality until no additional persistencies are found.
///
/// The returned `fixed_variables` contains only newly discovered fixings and
/// excludes the incoming fixed variables.
pub fn iterative_roof_duality_presolve(
    qubo: &Qubo,
    fixed_variables: &FixedVarMap,
    iter_limit: usize,
) -> RoofDualityResult {
    iterate_roof_duality(fixed_variables, iter_limit, |fixed| {
        roof_duality_presolve(qubo, fixed)
    })
}

fn iterate_roof_duality(
    fixed_variables: &FixedVarMap,
    iter_limit: usize,
    mut solve: impl FnMut(&FixedVarMap) -> RoofDualityResult,
) -> RoofDualityResult {
    let mut all_fixed = fixed_variables.clone();
    let mut last_lower_bound = None;
    let mut last_unlabeled = Vec::new();
    let mut strong_fixed = FixedVarMap::default();
    let mut weak_fixed = FixedVarMap::default();
    let max_iters = iter_limit.max(1);

    for _ in 0..max_iters {
        let result = solve(&all_fixed);
        last_lower_bound = result.lower_bound;
        last_unlabeled = result.unlabeled_variables;

        let previous_len = all_fixed.len();
        let target = if weak_fixed.is_empty() {
            &mut strong_fixed
        } else {
            &mut weak_fixed
        };
        for (&index, &value) in &result.fixed_variables {
            target.insert(index, value);
            all_fixed.insert(index, value);
        }
        for (index, value) in result.weak_fixed_variables {
            weak_fixed.insert(index, value);
            all_fixed.insert(index, value);
        }

        if all_fixed.len() == previous_len {
            break;
        }
    }

    RoofDualityResult {
        fixed_variables: strong_fixed,
        weak_fixed_variables: weak_fixed,
        lower_bound: last_lower_bound,
        unlabeled_variables: last_unlabeled,
    }
}

fn solve_roof_dual_network(
    problem: &ReducedRoofDualProblem,
    weak_persistencies: bool,
) -> RoofDualityResult {
    let num_biform_variables = problem.num_variables + 1;
    let num_literal_nodes = 2 * num_biform_variables;

    let mut graph = FlowNetwork::new(num_literal_nodes, 2 * problem.biterms.len());

    for term in &problem.biterms {
        let capacity = 0.5 * term.weight;
        if capacity <= 0.0 {
            continue;
        }

        let (u, v) = match term.kind {
            BiTermKind::Equal => (literal_node(term.i, false), literal_node(term.j, false)),
            BiTermKind::Different => (literal_node(term.i, false), literal_node(term.j, true)),
        };

        graph.add_literal_edge(u, v, capacity);
    }

    let source = literal_node(0, false);
    let sink = literal_node(0, true);
    let flow_value = graph.max_flow(source, sink);
    let source_side = graph.reachable_from(source);

    let mut fixed_variables = FixedVarMap::default();
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

    let mut weak_fixed_variables = FixedVarMap::default();
    if weak_persistencies && !unlabeled_variables.is_empty() {
        let labels = graph.weak_labels(source);
        // Strong fixings must be respected by the joint weak labeling.
        if fixed_variables
            .iter()
            .all(|(&i, &value)| labels[i + 1] == Some(value))
        {
            unlabeled_variables.retain(|&i| {
                if let Some(value) = labels[i + 1] {
                    weak_fixed_variables.insert(i, value);
                    false
                } else {
                    true
                }
            });
        }
    }

    RoofDualityResult {
        fixed_variables,
        weak_fixed_variables,
        lower_bound: Some(problem.constant + flow_value),
        unlabeled_variables,
    }
}

fn literal_node(variable: usize, complemented: bool) -> usize {
    2 * variable + usize::from(complemented)
}

fn map_reduced_result(
    reduced_result: RoofDualityResult,
    original_to_reduced: &HashMap<usize, usize>,
) -> RoofDualityResult {
    let mut reduced_to_original = vec![0; original_to_reduced.len()];
    for (&original, &reduced) in original_to_reduced {
        reduced_to_original[reduced] = original;
    }
    let fixed_variables = reduced_result
        .fixed_variables
        .into_iter()
        .map(|(reduced_index, value)| (reduced_to_original[reduced_index], value))
        .collect();

    let unlabeled_variables = reduced_result
        .unlabeled_variables
        .into_iter()
        .map(|reduced_index| reduced_to_original[reduced_index])
        .collect();
    let weak_fixed_variables = reduced_result
        .weak_fixed_variables
        .into_iter()
        .map(|(reduced_index, value)| (reduced_to_original[reduced_index], value))
        .collect();

    RoofDualityResult {
        fixed_variables,
        weak_fixed_variables,
        lower_bound: reduced_result.lower_bound,
        unlabeled_variables,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::preprocess::shift_qubo;
    use crate::subproblemsolvers::enumerate_qubo::enumerate_solve;
    use crate::tests::make_test_prng;
    use crate::FixedVarMap as HashMap;
    use ndarray::Array1;
    use smolprng::PRNG;
    use sprs::TriMat;

    fn make_small_qubo() -> Qubo {
        let mut q = TriMat::new((2, 2));
        q.add_triplet(0, 0, 2.0);
        q.add_triplet(0, 1, -4.0);
        q.add_triplet(1, 1, 1.0);
        let c = Array1::from_vec(vec![1.0, -2.0]);
        Qubo::new_with_c(q.to_csr(), c)
    }

    fn assert_weak_reduction_preserves_optimum(qubo: &Qubo, fixed: &FixedVarMap) {
        let result = PreparedRoofDual::new(qubo)
            .solve_iterative_with_weak_persistencies(fixed, qubo.num_x());
        let mut reduced = fixed.clone();
        for (&i, &value) in result
            .fixed_variables
            .iter()
            .chain(&result.weak_fixed_variables)
        {
            assert!(reduced.insert(i, value).is_none());
        }
        assert_eq!(
            reduced.len() + result.unlabeled_variables.len(),
            qubo.num_x()
        );
        let best = exact_value_with_fixings(qubo, fixed);
        assert!(result.lower_bound.unwrap() <= best + 1e-8);
        assert!((exact_value_with_fixings(qubo, &reduced) - best).abs() < 1e-8);
        // Check the stronger autarky property, including non-optimal inputs:
        // overwriting with the complete returned batch cannot increase energy.
        for mask in 0..1usize << qubo.num_x() {
            if fixed.iter().any(|(&i, &v)| (mask >> i) & 1 != v) {
                continue;
            }
            let original = Array1::from_iter((0..qubo.num_x()).map(|i| (mask >> i) & 1));
            let mut mapped = original.clone();
            for (&i, &v) in &reduced {
                mapped[i] = v;
            }
            assert!(qubo.eval_usize(&mapped) <= qubo.eval_usize(&original) + 1e-8);
        }
        for (&i, &value) in &result.fixed_variables {
            let mut opposite = fixed.clone();
            opposite.insert(i, 1 - value);
            assert!(exact_value_with_fixings(qubo, &opposite) > best + 1e-9);
        }
    }

    #[test]
    fn scc_tie_choices_are_compatible_and_not_reported_as_strong() {
        // The two optima are 00 and 11; choosing each variable independently
        // could manufacture the non-optimal assignment 01.
        let mut q = TriMat::new((2, 2));
        q.add_triplet(0, 1, -4.0);
        let qubo = Qubo::new_with_c(q.to_csr(), ndarray::array![1.0, 1.0]);
        let prepared = PreparedRoofDual::new(&qubo);
        let strong = prepared.solve(&FixedVarMap::default());
        assert!(strong.fixed_variables.is_empty());
        assert!(strong.weak_fixed_variables.is_empty());
        let weak = prepared.solve_iterative_with_weak_persistencies(&FixedVarMap::default(), 2);
        assert!(weak.fixed_variables.is_empty());
        assert_eq!(weak.weak_fixed_variables.len(), 2);
        assert_eq!(weak.weak_fixed_variables[&0], weak.weak_fixed_variables[&1]);
        assert_weak_reduction_preserves_optimum(&qubo, &FixedVarMap::default());
    }

    #[test]
    fn scc_leaves_frustrated_core_unfixed_and_respects_its_dependencies() {
        let mut q = TriMat::new((5, 5));
        for (i, j, weight) in [
            (0, 1, 4.0),
            (0, 2, 4.0),
            (1, 2, 4.0),
            (0, 3, -2.0),
            (3, 4, -4.0),
        ] {
            q.add_triplet(i, j, weight);
        }
        let qubo = Qubo::new_with_c(q.to_csr(), ndarray::array![-2.0, -2.0, -2.0, 2.0, 1.0]);
        let result = PreparedRoofDual::new(&qubo)
            .solve_iterative_with_weak_persistencies(&FixedVarMap::default(), 5);
        assert!(result.fixed_variables.is_empty());
        assert_eq!(
            result.weak_fixed_variables,
            [(3, 0), (4, 0)].into_iter().collect()
        );
        assert_eq!(result.unlabeled_variables.len(), 3);
        assert_weak_reduction_preserves_optimum(&qubo, &FixedVarMap::default());
    }

    #[test]
    fn scc_exhaustive_three_variable_coefficients_and_all_conditionings() {
        for coefficients in 0..3usize.pow(6) {
            let mut code = coefficients;
            let mut c = Array1::zeros(3);
            for value in &mut c {
                *value = (code % 3) as f64 - 1.0;
                code /= 3;
            }
            let mut q = TriMat::new((3, 3));
            for (i, j) in [(0, 1), (0, 2), (1, 2)] {
                q.add_triplet(i, j, 2.0 * ((code % 3) as f64 - 1.0));
                code /= 3;
            }
            let qubo = Qubo::new_with_c(q.to_csr(), c);
            for pattern in 0..27 {
                let mut code = pattern;
                let mut fixed = FixedVarMap::default();
                for i in 0..3 {
                    if code % 3 != 0 {
                        fixed.insert(i, code % 3 - 1);
                    }
                    code /= 3;
                }
                assert_weak_reduction_preserves_optimum(&qubo, &fixed);
            }
        }
    }

    #[test]
    fn scc_generated_fractional_asymmetric_and_tied_qubos() {
        let mut rng = make_test_prng();
        for sample in 0..256 {
            let mut qubo = Qubo::make_random_qubo(8, &mut rng, 0.5);
            if sample % 2 == 0 {
                qubo.q
                    .data_mut()
                    .iter_mut()
                    .for_each(|v| *v = (*v * 4.0).round());
                qubo.c.mapv_inplace(|v| (v * 2.0).round());
            }
            if sample % 3 == 0 {
                qubo.q = qubo.q.to_csc();
            }
            for fixed in [
                FixedVarMap::default(),
                [(0, 1), (5, 0)].into_iter().collect(),
            ] {
                assert_weak_reduction_preserves_optimum(&qubo, &fixed);
            }
        }
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

    fn exact_value_with_fixings(qubo: &Qubo, fixed_variables: &HashMap) -> f64 {
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
        fixed_variables: &HashMap,
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
        let problem = build_reduced_roof_dual_problem(&qubo, &HashMap::default());

        for mask in 0..(1usize << problem.num_variables) {
            let x = Array1::from_vec(vec![(mask & 1), ((mask >> 1) & 1)]);
            let qubo_value = qubo.eval_usize(&x);
            let roof_value = evaluate_reduced_problem(&problem, mask);
            assert!((qubo_value - roof_value).abs() <= 1e-9);
        }
    }

    #[test]
    fn prepared_projection_preserves_every_conditional_assignment() {
        let mut terms = TriMat::new((5, 5));
        for (i, j, value) in [
            (0, 0, 3.0),
            (0, 1, -7.0),
            (1, 0, 2.0),
            (2, 1, 4.0),
            (2, 2, -3.0),
            (3, 4, 5.0),
            (4, 3, -1.0),
        ] {
            terms.add_triplet(i, j, value);
        }
        for matrix in [terms.to_csr(), terms.to_csc()] {
            let qubo = Qubo::new_with_c(matrix, ndarray::array![2.0, -5.0, 7.0, -1.0, 3.0]);
            let prepared = super::PreparedRoofDual::new(&qubo);
            for pattern in 0..3usize.pow(5) {
                let mut code = pattern;
                let mut fixed = HashMap::default();
                for i in 0..5 {
                    if code % 3 != 0 {
                        fixed.insert(i, code % 3 - 1);
                    }
                    code /= 3;
                }
                let reduced = prepared.project(&fixed);
                for mask in 0..(1 << reduced.num_variables) {
                    let full = Array1::from_iter((0..5).map(|i| {
                        fixed
                            .get(&i)
                            .copied()
                            .unwrap_or_else(|| (mask >> reduced.original_to_reduced[&i]) & 1)
                    }));
                    assert!(
                        (evaluate_reduced_problem(&reduced, mask) - qubo.eval_usize(&full)).abs()
                            < 1e-10
                    );
                }
            }
        }
    }

    #[test]
    fn prepared_roof_bounds_and_fixings_match_cold_and_exhaustive_solves() {
        let mut prng = crate::tests::make_test_prng();
        for _ in 0..64 {
            let mut qubo = Qubo::make_random_qubo(7, &mut prng, 0.5);
            // Binary fractions keep ties exact, independent of network ordering.
            qubo.q
                .data_mut()
                .iter_mut()
                .for_each(|v| *v = (*v * 16.0).round());
            qubo.c.mapv_inplace(|v| (v * 8.0).round());
            let prepared = super::PreparedRoofDual::new(&qubo);
            for fixed in [
                HashMap::default(),
                [(0, 1), (3, 0)].into_iter().collect(),
                (0..7).map(|i| (i, i % 2)).collect(),
            ] {
                let warm = prepared.solve_iterative(&fixed, 7);
                let cold = super::iterative_roof_duality_presolve(&qubo, &fixed, 7);
                let best = exact_value_with_fixings(&qubo, &fixed);
                assert_eq!(warm.lower_bound, cold.lower_bound);
                assert_eq!(warm.fixed_variables, cold.fixed_variables);
                assert!(warm.lower_bound.unwrap() <= best + 1e-10);
                for (&i, &v) in &warm.fixed_variables {
                    let mut opposite = fixed.clone();
                    opposite.insert(i, 1 - v);
                    assert!(exact_value_with_fixings(&qubo, &opposite) > best + 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_roof_dual_linear_problem_fixes_positive_cost_to_zero() {
        let q = TriMat::<f64>::new((1, 1)).to_csr();
        let qubo = Qubo::new_with_c(q, Array1::from_vec(vec![3.0]));

        let result = roof_duality_presolve(&qubo, &HashMap::default());

        assert_eq!(result.fixed_variables.get(&0), Some(&0));
        assert_eq!(result.lower_bound, Some(0.0));
    }

    #[test]
    fn test_roof_dual_linear_problem_fixes_negative_cost_to_one() {
        let q = TriMat::<f64>::new((1, 1)).to_csr();
        let qubo = Qubo::new_with_c(q, Array1::from_vec(vec![-2.0]));

        let result = roof_duality_presolve(&qubo, &HashMap::default());

        assert_eq!(result.fixed_variables.get(&0), Some(&1));
        assert_eq!(result.lower_bound, Some(-2.0));
    }

    #[test]
    fn test_roof_dual_lower_bound_is_valid_on_small_instance() {
        let qubo = make_small_qubo();
        let result = roof_duality_presolve(&qubo, &HashMap::default());

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
        let problem = build_reduced_roof_dual_problem(&qubo, &HashMap::default());

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
        let problem = build_reduced_roof_dual_problem(&qubo, &HashMap::default());

        for mask in 0..(1usize << 5) {
            let x0 = (mask & 1) as usize;
            let x2 = ((mask >> 1) & 1) as usize;
            let x3 = ((mask >> 2) & 1) as usize;
            let x4 = ((mask >> 3) & 1) as usize;
            let x5 = ((mask >> 4) & 1) as usize;

            let original = evaluate_biform(&problem, &[x0, 1, x2, x3, x4, x5]);
            let reduced =
                21.0 - 11.0 * x0 as f64 + 2.0 * x2 as f64 + 11.0 * x3 as f64 + 3.0 * x4 as f64
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
            assert_roof_dual_fixings_are_persistent(&qubo, &HashMap::default(), true);
        }
    }

    #[test]
    fn test_generated_small_qubos_with_existing_fixings_match_exhaustive_persistencies() {
        let mut prng: PRNG<_> = make_test_prng();

        for case_idx in 0..6 {
            let num_x = 10;
            let sparsity = 0.4 + 0.1 * ((case_idx % 2) as f64);
            let qubo = Qubo::make_random_qubo(num_x, &mut prng, sparsity);

            let mut fixed_variables = HashMap::default();
            fixed_variables.insert(case_idx % num_x, case_idx % 2);

            assert_roof_dual_fixings_are_persistent(&qubo, &fixed_variables, false);
        }
    }

    #[test]
    fn test_repo_test_qubo_convex_shift_matches_enumeration() {
        let qubo = Qubo::read_qubo("test_data/test.qubo");
        let convex = qubo.convex_symmetric_form();
        let shifted = shift_qubo(&convex);

        let result = roof_duality_presolve(&shifted, &HashMap::default());
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
