use crate::persistence::{
    build_gradient_adjacency, compute_iterative_persistence_with_adjacency, GradientAdjacency,
    GradientProbe,
};
use crate::qubo::Qubo;
use crate::FixedVarMap;
use crate::VarMap;
use ndarray::Array1;
use sprs::{CsMat, TriMat};
use std::collections::HashMap;

mod symmetry;
pub(crate) use symmetry::{complement_components, ComplementComponent};
pub(crate) mod low_degree;
pub use low_degree::{CutDominanceStatistics, RootReductionStatistics};

/// This file is the main module that defines the preprocessing functions.
pub(crate) struct PreparedPreprocess {
    no_effect_vars: FixedVarMap,
    qubo_pp: Qubo,
    adjacency: GradientAdjacency,
}

pub(crate) struct ProbePreprocess<'a> {
    prepared: &'a PreparedPreprocess,
    workspace: GradientProbe,
}

impl<'a> ProbePreprocess<'a> {
    pub(crate) fn new(prepared: &'a PreparedPreprocess, base_fixed: &FixedVarMap) -> Self {
        let workspace = GradientProbe::new(&prepared.qubo_pp, base_fixed, &prepared.adjacency);
        Self {
            prepared,
            workspace,
        }
    }

    pub(crate) fn probe(&mut self, variable: usize, value: usize) -> FixedVarMap {
        self.workspace.probe(
            &self.prepared.qubo_pp,
            &self.prepared.adjacency,
            variable,
            value,
        )
    }
}

impl PreparedPreprocess {
    pub(crate) fn qubo(&self) -> &Qubo {
        &self.qubo_pp
    }

    pub(crate) fn adjacency(&self) -> &GradientAdjacency {
        &self.adjacency
    }
}

pub(crate) fn prepare_preprocess(qubo: &Qubo, in_standard_form: bool) -> PreparedPreprocess {
    let no_effect_vars = fix_no_effect_variables(qubo);
    let qubo_pp = if in_standard_form {
        qubo.clone()
    } else {
        shift_qubo(qubo)
    };
    let adjacency = build_gradient_adjacency(&qubo_pp);

    PreparedPreprocess {
        no_effect_vars,
        qubo_pp,
        adjacency,
    }
}

pub(crate) fn preprocess_with_prepared(
    prepared: &PreparedPreprocess,
    fixed_variables: &FixedVarMap,
) -> FixedVarMap {
    preprocess_owned_with_prepared(prepared, fixed_variables.clone())
}

pub(crate) fn preprocess_owned_with_prepared(
    prepared: &PreparedPreprocess,
    fixed_variables: FixedVarMap,
) -> FixedVarMap {
    let mut persistent = compute_iterative_persistence_with_adjacency(
        &prepared.qubo_pp,
        fixed_variables,
        &prepared.no_effect_vars,
        prepared.qubo_pp.num_x(),
        &prepared.adjacency,
    );

    merge_no_effect_variables(prepared, &mut persistent);
    persistent
}

fn merge_no_effect_variables(prepared: &PreparedPreprocess, persistent: &mut FixedVarMap) {
    persistent.reserve(prepared.no_effect_vars.len());
    for (&index, &value) in &prepared.no_effect_vars {
        persistent.entry(index).or_insert(value);
    }
}

/// This is the main entry point for preprocessing
pub fn preprocess_qubo(
    qubo: &Qubo,
    fixed_variables: &FixedVarMap,
    in_standard_form: bool,
) -> FixedVarMap {
    let prepared = prepare_preprocess(qubo, in_standard_form);
    preprocess_with_prepared(&prepared, fixed_variables)
}

/// This is the heavy entry point for preprocessing + variable probing
pub fn preprocess_qubo_heavy(
    qubo: &Qubo,
    fixed_variables: &FixedVarMap,
    in_standard_form: bool,
) -> FixedVarMap {
    // copy the fixed variables
    let mut new_persistent = fixed_variables.clone();
    let prepared = prepare_preprocess(qubo, in_standard_form);

    // the number of required iterations is always below the number of variables
    let iters = qubo.num_x();

    // loop over the number of iters
    for _ in 0..iters {
        let previous_len = new_persistent.len();
        let mut incoming_persistent = preprocess_with_prepared(&prepared, &new_persistent);

        let (_, probe_fixes) =
            crate::variable_reduction::probe(qubo, &incoming_persistent, in_standard_form);

        // add the probe fixes to the incoming persistent
        for (key, value) in probe_fixes {
            incoming_persistent.insert(key, value);
        }

        if incoming_persistent.len() == previous_len {
            return new_persistent;
        }
        new_persistent = incoming_persistent;
    }

    new_persistent
}

/// Find variables that have no effect in the QUBO, where the linear term is zero and the quadratic
/// terms are zero. This is useful for reducing the size of the QUBO.
pub fn find_no_effect_variables(qubo: &Qubo) -> Vec<usize> {
    let mut is_no_effect_var = Array1::from_elem(qubo.num_x(), true);

    // check the quadratic terms
    for (&_value, (i, j)) in &qubo.q {
        is_no_effect_var[i] = false;
        is_no_effect_var[j] = false;
    }

    // check the linear terms
    for i in 0..qubo.num_x() {
        if qubo.c[i] != 0.0 {
            is_no_effect_var[i] = false;
        }
    }

    is_no_effect_var
        .indexed_iter()
        .filter(|(_, &value)| value)
        .map(|(i, _)| i)
        .collect()
}

/// Fixes variables that have no effect in the QUBO, where the linear term is zero and the quadratic
/// terms are zero. This is useful for reducing the size of the QUBO.
pub fn fix_no_effect_variables(qubo: &Qubo) -> FixedVarMap {
    let no_effect_vars = find_no_effect_variables(qubo);

    no_effect_vars.iter().map(|&i| (i, 0)).collect()
}

/// solve small unconnected components optimally via enumerition if the number of variables is small
/// enough
pub fn solve_small_components(
    qubo: &Qubo,
    fixed_vars: &FixedVarMap,
    max_size: usize,
) -> FixedVarMap {
    let mut new_fixed_variables = fixed_vars.clone();
    solve_small_components_in_place_with_extra(qubo, &mut new_fixed_variables, None, max_size);
    new_fixed_variables
}

pub(crate) fn solve_small_components_in_place(
    qubo: &Qubo,
    fixed_vars: &mut FixedVarMap,
    max_size: usize,
) {
    solve_small_components_in_place_with_extra(qubo, fixed_vars, None, max_size);
}

pub(crate) fn solve_small_components_in_place_with_extra(
    qubo: &Qubo,
    fixed_vars: &mut FixedVarMap,
    extra_fixed_vars: Option<&FixedVarMap>,
    max_size: usize,
) {
    let adjacency = build_gradient_adjacency(qubo);
    solve_small_components_with_adjacency(qubo, fixed_vars, extra_fixed_vars, max_size, &adjacency);
}

pub(crate) fn solve_small_components_with_adjacency(
    qubo: &Qubo,
    fixed_vars: &mut FixedVarMap,
    extra_fixed_vars: Option<&FixedVarMap>,
    max_size: usize,
    adjacency: &GradientAdjacency,
) {
    let components = crate::graph_utils::small_components_with_adjacency(
        adjacency,
        fixed_vars,
        extra_fixed_vars,
        max_size,
    );
    if components.is_empty() {
        return;
    }
    let indexed_terms =
        (max_size > 10 && components.len() > 1).then(|| index_component_terms(qubo, &components));

    for (index, component) in components.iter().enumerate() {
        if component.len() <= 10 {
            let (_, mask) =
                enumerate_component(qubo, component, fixed_vars, extra_fixed_vars, adjacency);
            for (bit, &variable) in component.iter().enumerate() {
                fixed_vars.insert(variable, (mask >> bit) & 1);
            }
            continue;
        }
        // remove the fixed variables from the component
        let (sub_qubo, mapping) = if let Some(terms) = &indexed_terms {
            make_component_qubo_with_terms(
                qubo,
                component,
                fixed_vars,
                extra_fixed_vars,
                terms[index].iter().copied(),
            )
        } else {
            make_component_qubo_with_extra(qubo, component, fixed_vars, extra_fixed_vars)
        };

        // solve the subproblem via enumeration
        let (_, solution) = crate::subproblemsolvers::enumerate_qubo::enumerate_solve(&sub_qubo);

        // map the solution back to the original variables
        for (key, &value) in &mapping {
            fixed_vars.insert(*key, solution[value]);
        }
    }
}

fn enumerate_component(
    qubo: &Qubo,
    component: &[usize],
    fixed: &FixedVarMap,
    extra: Option<&FixedVarMap>,
    adjacency: &GradientAdjacency,
) -> (f64, usize) {
    assert!(component.len() <= 10);
    let mut linear = [0.0; 10];
    for (i, &original) in component.iter().enumerate() {
        linear[i] = qubo.c[original];
        // Incident terms retain the original sparse iteration order, including
        // both orientations of an asymmetric Q. No component matrix is needed.
        for &(neighbor, coefficient) in &adjacency[original] {
            if let Some(&value) = fixed
                .get(&neighbor)
                .or_else(|| extra.and_then(|map| map.get(&neighbor)))
            {
                linear[i] += coefficient * value as f64;
            }
        }
    }
    let terms = linear[..component.len()]
        .iter()
        .enumerate()
        .map(|(i, &value)| (1 << i, value))
        .chain(component.iter().enumerate().flat_map(|(j, &original_j)| {
            component
                .iter()
                .enumerate()
                .filter_map(move |(i, &original_i)| {
                    qubo.q
                        .get(original_i, original_j)
                        .map(|&value| ((1 << i) | (1 << j), 0.5 * value))
                })
        }));
    // Match the previous component CSC's term order and smallest-mask tie break.
    crate::subproblemsolvers::enumerate_qubo::enumerate_small_terms(component.len(), terms)
}

type ComponentTerm = (f64, (usize, usize));

fn index_component_terms(qubo: &Qubo, components: &[Vec<usize>]) -> Vec<Vec<ComponentTerm>> {
    let mut owner = vec![usize::MAX; qubo.num_x()];
    for (index, component) in components.iter().enumerate() {
        for &variable in component {
            owner[variable] = index;
        }
    }
    let mut counts = vec![0; components.len()];
    for (_, (i, j)) in &qubo.q {
        if owner[i] != usize::MAX {
            counts[owner[i]] += 1;
        }
        if owner[j] != usize::MAX && owner[j] != owner[i] {
            counts[owner[j]] += 1;
        }
    }
    let mut terms = counts
        .into_iter()
        .map(Vec::with_capacity)
        .collect::<Vec<_>>();
    // Retain original indices and edge order. Fixings are still substituted
    // when each component is solved, not prematurely while indexing terms.
    for (&value, (i, j)) in &qubo.q {
        if owner[i] != usize::MAX {
            terms[owner[i]].push((value, (i, j)));
        }
        if owner[j] != usize::MAX && owner[j] != owner[i] {
            terms[owner[j]].push((value, (i, j)));
        }
    }
    terms
}

pub fn make_component_qubo(
    qubo: &Qubo,
    component: &[usize],
    fixed_vars: &FixedVarMap,
) -> (Qubo, VarMap) {
    make_component_qubo_with_extra(qubo, component, fixed_vars, None)
}

pub fn make_component_qubo_with_extra(
    qubo: &Qubo,
    component: &[usize],
    fixed_vars: &FixedVarMap,
    extra_fixed_vars: Option<&FixedVarMap>,
) -> (Qubo, VarMap) {
    make_component_qubo_with_terms(
        qubo,
        component,
        fixed_vars,
        extra_fixed_vars,
        qubo.q.iter().map(|(&value, indices)| (value, indices)),
    )
}

fn make_component_qubo_with_terms(
    qubo: &Qubo,
    component: &[usize],
    fixed_vars: &FixedVarMap,
    extra_fixed_vars: Option<&FixedVarMap>,
    terms: impl IntoIterator<Item = ComponentTerm>,
) -> (Qubo, VarMap) {
    let mut Q_tri = TriMat::new((component.len(), component.len()));
    let mut c_new = Array1::<f64>::zeros(component.len());

    let mut index_map = VarMap::default();
    index_map.reserve(component.len());

    for (new_index, &old_index) in component.iter().enumerate() {
        index_map.insert(old_index, new_index);
        c_new[new_index] = qubo.c[old_index];
    }

    for (value, (i, j)) in terms {
        let fixed_i = fixed_vars
            .get(&i)
            .copied()
            .or_else(|| extra_fixed_vars.and_then(|extra| extra.get(&i).copied()));
        let fixed_j = fixed_vars
            .get(&j)
            .copied()
            .or_else(|| extra_fixed_vars.and_then(|extra| extra.get(&j).copied()));

        match (index_map.get(&i), index_map.get(&j), fixed_i, fixed_j) {
            (Some(&i_new), Some(&j_new), _, _) => {
                Q_tri.add_triplet(i_new, j_new, value);
            }
            (Some(&i_new), None, _, Some(fixed_j)) => {
                c_new[i_new] += 0.5 * value * fixed_j as f64;
            }
            (None, Some(&j_new), Some(fixed_i), _) => {
                c_new[j_new] += 0.5 * value * fixed_i as f64;
            }
            _ => {}
        }
    }

    (Qubo::new_with_c(Q_tri.to_csc(), c_new), index_map)
}

/// Given a QUBO and a set of fixed variables, create a new QUBO where the fixed variables are
/// removed and the linear term is adjusted accordingly. Also return a mapping between the new
/// variable indices and the old variable indices, as well as the constant term that was added to
/// the objective function due to the fixed variables.
///
/// # Panics
/// This function will not panic if there are no free variables left removing unfixed varaibles.
pub fn make_sub_problem(
    qubo: &Qubo,
    fixed_vars: &FixedVarMap,
) -> (Qubo, HashMap<usize, usize>, f64) {
    let num_unfixed = qubo.num_x() - fixed_vars.len();
    let mut reduced_index = vec![usize::MAX; qubo.num_x()];
    let mut fixed_values = vec![0.0; qubo.num_x()];
    let mut c_new = Array1::<f64>::zeros(num_unfixed);
    let mut constant = 0.0;
    let mut unfixed_map = HashMap::with_capacity(num_unfixed);

    for (i, index) in reduced_index.iter_mut().enumerate() {
        if let Some(&value) = fixed_vars.get(&i) {
            fixed_values[i] = value as f64;
        } else {
            *index = unfixed_map.len();
            unfixed_map.insert(i, *index);
        }
    }

    // Count columns while projecting fixed terms, avoiding hashes in the edge loop.
    let mut indptr = vec![0; num_unfixed + 1];
    for (&q_ij, (i, j)) in &qubo.q {
        let i_new = reduced_index[i];
        let j_new = reduced_index[j];
        match (i_new != usize::MAX, j_new != usize::MAX) {
            (false, false) => {
                constant += 0.5 * q_ij * fixed_values[i] * fixed_values[j];
            }
            (false, true) => {
                c_new[j_new] += 0.5 * q_ij * fixed_values[i];
            }
            (true, false) => {
                c_new[i_new] += 0.5 * q_ij * fixed_values[j];
            }
            (true, true) => {
                indptr[j_new + 1] += 1;
            }
        }
    }

    for (i, &c_i) in qubo.c.iter().enumerate() {
        if reduced_index[i] == usize::MAX {
            constant += c_i * fixed_values[i];
        } else {
            c_new[reduced_index[i]] += c_i;
        }
    }

    for column in 0..num_unfixed {
        indptr[column + 1] += indptr[column];
    }
    let mut next = indptr[..num_unfixed].to_vec();
    let mut indices = vec![0; indptr[num_unfixed]];
    let mut values = vec![0.0; indptr[num_unfixed]];
    // CSR and CSC iteration both visit each column's surviving rows in order.
    for (&value, (i, j)) in &qubo.q {
        let i_new = reduced_index[i];
        let j_new = reduced_index[j];
        if i_new != usize::MAX && j_new != usize::MAX {
            let position = next[j_new];
            indices[position] = i_new;
            values[position] = value;
            next[j_new] += 1;
        }
    }
    (
        Qubo::new_with_c(
            CsMat::new_csc((num_unfixed, num_unfixed), indptr, indices, values),
            c_new,
        ),
        unfixed_map,
        constant,
    )
}

/// Creates a new QUBO where the diagonal elements are zeroed out and the linear term is adjusted
/// accordingly
pub fn shift_qubo(qubo: &Qubo) -> Qubo {
    let mut new_q = TriMat::new((qubo.num_x(), qubo.num_x()));
    let mut new_c = qubo.c.clone();

    for (&value, (i, j)) in &qubo.q {
        if i == j {
            new_c[i] += 0.5 * value;
        } else {
            new_q.add_triplet(i, j, value);
        }
    }

    Qubo::new_with_c(new_q.to_csr(), new_c)
}

#[cfg(test)]
mod tests {
    use crate::preprocess::preprocess_qubo;
    use crate::qubo::Qubo;
    use crate::FixedVarMap as HashMap;
    use ndarray::Array1;
    use sprs::{CsMat, TriMat};

    fn check_component_presolve(qubo: &Qubo, fixed: &HashMap, extra: &HashMap) {
        let mut expected = fixed.clone();
        let adjacency = crate::persistence::build_gradient_adjacency(qubo);
        let components =
            crate::graph_utils::small_components_with_adjacency(&adjacency, fixed, Some(extra), 10);
        for component in components.iter().filter(|component| component.len() <= 10) {
            let (sub, mapping) =
                super::make_component_qubo_with_extra(qubo, component, &expected, Some(extra));
            let (_, solution) = crate::subproblemsolvers::enumerate_qubo::enumerate_solve(&sub);
            for (&i, &j) in &mapping {
                expected.insert(i, solution[j]);
            }
        }
        let mut actual = fixed.clone();
        super::solve_small_components_in_place_with_extra(qubo, &mut actual, Some(extra), 10);
        assert_eq!(actual, expected);
    }

    #[test]
    fn direct_component_enumeration_preserves_objectives_and_ties() {
        use smolprng::{JsfLarge, PRNG};
        let mut prng = PRNG {
            generator: JsfLarge::from(817632u64),
        };
        let check = |qubo: &Qubo, component: &[usize], fixed: &HashMap, extra: &HashMap| {
            let adjacency = crate::persistence::build_gradient_adjacency(qubo);
            let (sub, _) =
                super::make_component_qubo_with_extra(qubo, component, fixed, Some(extra));
            let (value, solution) = crate::subproblemsolvers::enumerate_qubo::enumerate_solve(&sub);
            let mask = solution
                .iter()
                .enumerate()
                .fold(0, |mask, (i, &v)| mask | (v << i));
            let direct =
                super::enumerate_component(qubo, component, fixed, Some(extra), &adjacency);
            assert_eq!(direct.0.to_bits(), value.to_bits());
            assert_eq!(direct.1, mask);
        };
        for size in 1..=10 {
            let component: Vec<_> = (0..size).rev().collect();
            let fixed = [(size, 1), (size + 1, 0)].into_iter().collect();
            let extra = [(size + 2, 1)].into_iter().collect();
            for _ in 0..8 {
                let qubo = Qubo::make_random_qubo(size + 3, &mut prng, 0.5);
                for matrix in [qubo.q.to_csr(), qubo.q.to_csc()] {
                    check(
                        &Qubo::new_with_c(matrix, qubo.c.clone()),
                        &component,
                        &fixed,
                        &extra,
                    );
                }
            }
            check(
                &Qubo::new(CsMat::zero((size + 3, size + 3))),
                &component,
                &fixed,
                &extra,
            );
        }
        let mut terms = TriMat::new((6, 6));
        for (i, j, value) in [
            (0, 1, 1e16),
            (1, 0, -1e16),
            (1, 2, 0.1),
            (1, 4, 1e-16),
            (2, 3, -0.2),
            (3, 3, 2.0),
            (4, 5, -1e-16),
        ] {
            terms.add_triplet(i, j, value);
        }
        for matrix in [terms.to_csr(), terms.to_csc()] {
            check(
                &Qubo::new_with_c(matrix, ndarray::array![0.0, -0.1, 0.0, 0.2, 0.0, 0.0]),
                &[4, 1, 3],
                &[(0, 1), (2, 1)].into_iter().collect(),
                &[(5, 1)].into_iter().collect(),
            );
        }
    }

    #[test]
    fn indexed_components_match_full_matrix_scans() {
        use smolprng::{JsfLarge, PRNG};
        let mut prng = PRNG {
            generator: JsfLarge::from(182341u64),
        };
        for _ in 0..40 {
            let qubo = Qubo::make_random_qubo(24, &mut prng, 0.07);
            for matrix in [qubo.q.to_csr(), qubo.q.to_csc()] {
                let qubo = Qubo::new_with_c(matrix, qubo.c.clone());
                check_component_presolve(&qubo, &HashMap::default(), &HashMap::default());
                check_component_presolve(
                    &qubo,
                    &[(0, 1), (4, 0)].into_iter().collect(),
                    &[(7, 1)].into_iter().collect(),
                );
            }
        }
        // Fixing the center splits the star into many singleton components.
        let mut terms = TriMat::new((20, 20));
        for i in 0..20 {
            terms.add_triplet(0, i, i as f64 - 9.5);
            if i > 0 {
                terms.add_triplet(i, 0, i as f64 - 9.5);
            }
        }
        let qubo = Qubo::new(terms.to_csr());
        check_component_presolve(&qubo, &[(0, 1)].into_iter().collect(), &HashMap::default());
        check_component_presolve(&qubo, &HashMap::default(), &[(0, 1)].into_iter().collect());
    }

    fn check_cached_probes(qubo: &Qubo, fixed: &HashMap, candidates: &[usize]) {
        let prepared = super::prepare_preprocess(qubo, false);
        let mut cached = super::ProbePreprocess::new(&prepared, fixed);
        for &i in candidates {
            if fixed.contains_key(&i) {
                continue;
            }
            for value in [0, 1, 0] {
                let mut assumption = fixed.clone();
                assumption.insert(i, value);
                let mut expected = crate::persistence::cold_gradient_probe(
                    &prepared.qubo_pp,
                    assumption,
                    &prepared.adjacency,
                );
                expected.retain(|i, _| !fixed.contains_key(i));
                assert_eq!(cached.probe(i, value), expected, "probe x_{i}={value}");
            }
        }
    }

    #[test]
    fn rollback_probes_match_cold_strict_propagation_on_generated_qubos() {
        use smolprng::{JsfLarge, PRNG};
        let mut prng = PRNG {
            generator: JsfLarge::from(192731u64),
        };
        for sample in 0..40 {
            let qubo = Qubo::make_random_qubo(12, &mut prng, 0.3).make_symmetric();
            let qubo = if sample % 2 == 0 {
                qubo
            } else {
                Qubo::new_with_c(qubo.q.to_csc(), qubo.c)
            };
            for fixed in [HashMap::default(), [(0, 1), (3, 0)].into_iter().collect()] {
                check_cached_probes(&qubo, &fixed, &(0..12).collect::<Vec<_>>());
            }
        }
        // Exercise no-effect assumptions, ties, cancellation and nonsymmetric Q.
        let mut terms = TriMat::new((6, 6));
        for (i, j, value) in [
            (0, 1, 1e16),
            (0, 2, -1e16),
            (0, 3, 1.0),
            (1, 2, -1.0),
            (2, 3, 1e-16),
            (3, 3, 1.0),
            (4, 4, 0.0),
        ] {
            terms.add_triplet(i, j, value);
        }
        for matrix in [terms.to_csr(), terms.to_csc()] {
            let qubo = Qubo::new_with_c(
                matrix,
                Array1::from_vec(vec![-0.5, 1.0, 0.0, -0.5, 0.0, 0.0]),
            );
            check_cached_probes(&qubo, &HashMap::default(), &[0, 1, 2, 3, 4, 5]);
        }
    }

    #[test]
    fn rollback_probes_match_cold_strict_propagation_on_benchmark_instances() {
        for path in [
            "test_data/mk487a.qubo",
            "test_data/mk487b.qubo",
            "test_data/bqp/bqp100-1.qubo",
            "test_data/sg3dl101000.qubo",
        ] {
            let qubo = Qubo::read_qubo(path).make_symmetric();
            let convex = qubo.convex_symmetric_form();
            let candidates = (0..qubo.num_x())
                .step_by((qubo.num_x() / 12).max(1))
                .collect::<Vec<_>>();
            for fixed in [
                HashMap::default(),
                (0..qubo.num_x()).step_by(5).map(|i| (i, i % 2)).collect(),
            ] {
                check_cached_probes(&qubo, &fixed, &candidates);
                check_cached_probes(&convex, &fixed, &candidates);
            }
        }
    }

    #[test]
    fn reduced_qubo_preserves_every_assignment_and_storage_order() {
        let mut terms = TriMat::new((5, 5));
        for (i, j, value) in [
            (0, 0, 3.0),
            (0, 1, -7.0),
            (1, 0, 2.0),
            (0, 4, 0.0),
            (2, 1, 4.0),
            (2, 2, -3.0),
            (3, 4, 5.0),
            (4, 3, -1.0),
        ] {
            terms.add_triplet(i, j, value);
        }
        for matrix in [terms.to_csr(), terms.to_csc()] {
            let qubo = Qubo::new_with_c(matrix, Array1::from_vec(vec![2.0, -5.0, 7.0, -1.0, 3.0]));
            for pattern in 0..3usize.pow(5) {
                let mut code = pattern;
                let mut fixed = HashMap::default();
                for i in 0..5 {
                    if code % 3 != 0 {
                        fixed.insert(i, code % 3 - 1);
                    }
                    code /= 3;
                }
                let (reduced, mapping, constant) = super::make_sub_problem(&qubo, &fixed);
                assert!(reduced.q.is_csc());
                assert_eq!(reduced.num_x(), 5 - fixed.len());
                for mask in 0..(1 << reduced.num_x()) {
                    let x = Array1::from_iter((0..reduced.num_x()).map(|i| (mask >> i) & 1));
                    let full = Array1::from_iter(
                        (0..5).map(|i| fixed.get(&i).copied().unwrap_or_else(|| x[mapping[&i]])),
                    );
                    assert_eq!(qubo.eval_usize(&full), reduced.eval_usize(&x) + constant);
                }
            }
        }
        let empty = Qubo::new(CsMat::zero((0, 0)));
        let (reduced, mapping, constant) = super::make_sub_problem(&empty, &HashMap::default());
        assert_eq!((reduced.num_x(), mapping.len(), constant), (0, 0, 0.0));
    }

    #[test]
    fn presolve_preserves_conditional_optima_on_generated_qubos() {
        use smolprng::{JsfLarge, PRNG};
        let mut prng = PRNG {
            generator: JsfLarge::from(91237u64),
        };
        for _ in 0..32 {
            let qubo = Qubo::make_random_qubo(7, &mut prng, 0.35).make_symmetric();
            for fixed in [HashMap::default(), [(0, 1), (3, 0)].into_iter().collect()] {
                let result = preprocess_qubo(&qubo, &fixed, false);
                assert!(fixed.iter().all(|(i, v)| result.get(i) == Some(v)));
                let minimum = |fixings: &HashMap| {
                    (0..128)
                        .filter(|mask| fixings.iter().all(|(&i, &v)| (mask >> i) & 1 == v))
                        .map(|mask| {
                            qubo.eval_usize(&Array1::from_iter((0..7).map(|i| (mask >> i) & 1)))
                        })
                        .fold(f64::INFINITY, f64::min)
                };
                assert!((minimum(&fixed) - minimum(&result)).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn component_fixings_leave_large_component_at_gradient_closure() {
        let mut terms = TriMat::new((14, 14));
        for i in 0..12 {
            let j = (i + 1) % 12;
            terms.add_triplet(i, j, 2.0);
            terms.add_triplet(j, i, 2.0);
        }
        terms.add_triplet(12, 13, -2.0);
        terms.add_triplet(13, 12, -2.0);
        let mut linear = Array1::from_elem(14, -2.0);
        linear[12] = 1.0;
        linear[13] = 1.0;
        let qubo = Qubo::new_with_c(terms.to_csr(), linear);
        let empty = HashMap::default();
        let run = |limit| crate::persistence::compute_iterative_persistence(&qubo, &empty, limit);
        assert!(run(0).is_empty());
        let once = run(1);
        assert_eq!(once, [(12, 0), (13, 0)].into_iter().collect());
        assert_eq!(run(14), once);
        let adjacency = crate::persistence::build_gradient_adjacency(&qubo);
        assert_eq!(
            crate::persistence::cold_gradient_probe(&qubo, once.clone(), &adjacency),
            once
        );
    }

    #[test]
    fn small_components_are_solved_without_new_gradient_fixings() {
        let mut terms = TriMat::new((4, 4));
        for (i, j) in [(0, 1), (2, 3)] {
            terms.add_triplet(i, j, -2.0);
            terms.add_triplet(j, i, -2.0);
        }
        let qubo = Qubo::new_with_c(terms.to_csr(), Array1::ones(4));
        assert!(crate::persistence::compute_persistent(&qubo, &HashMap::default()).is_empty());
        let result = preprocess_qubo(&qubo, &HashMap::default(), false);
        assert_eq!(result.len(), 4);
        let completion = Array1::from_iter((0..4).map(|i| result[&i]));
        assert_eq!(qubo.eval_usize(&completion), 0.0);
    }

    #[test]
    fn component_presolve_respects_edges_in_either_triangle_and_storage() {
        for (i, j) in [(0, 1), (1, 0)] {
            let mut terms = TriMat::new((2, 2));
            terms.add_triplet(i, j, -6.0);
            for matrix in [terms.to_csr(), terms.to_csc()] {
                let qubo = Qubo::new_with_c(matrix, Array1::ones(2));
                let result = preprocess_qubo(&qubo, &HashMap::default(), false);
                assert_eq!(result, [(0, 1), (1, 1)].into_iter().collect());
                assert_eq!(qubo.eval_usize(&Array1::ones(2)), -1.0);
            }
        }
    }

    #[test]
    fn test_preprocess_qubo_1() {
        let eye = CsMat::eye(3);
        let c = Array1::from_vec(vec![1.1, 2.0, 3.0]);
        let p = Qubo::new_with_c(eye, c);
        let fixed_variables = HashMap::default();
        let fixed_variables = preprocess_qubo(&p, &fixed_variables, false);
        assert_eq!(fixed_variables.len(), 3);
    }

    #[test]
    fn test_preprocess_qubo_materializes_no_effect_variables() {
        let p = Qubo::new_with_c(CsMat::zero((3, 3)), Array1::zeros(3));
        let fixed_variables = preprocess_qubo(&p, &HashMap::default(), false);

        assert_eq!(fixed_variables.len(), 3);
        assert_eq!(fixed_variables.get(&0), Some(&0));
        assert_eq!(fixed_variables.get(&1), Some(&0));
        assert_eq!(fixed_variables.get(&2), Some(&0));
    }

    #[test]
    fn test_generate_sub_problem_1() {
        // the idea of this test is, given a QUBO & some fixed variables, generate an equivalent problem

        // f(x) = 0.5<x,x>
        let q = CsMat::<f64>::eye(3);
        let c = Array1::<f64>::zeros(3);
        let p = Qubo::new_with_c(q, c);

        // we have variable x_0 fixed to 1
        let mut fixed_variables = HashMap::default();
        fixed_variables.insert(0, 1);

        // this should generate a subproblem with the following matrix
        // [1 0]  [0]
        // [0 1]  [0]

        let (sub_p, _, constant) = super::make_sub_problem(&p, &fixed_variables);

        // fix the expected matrix
        let q_target = CsMat::<f64>::eye(2);
        let c_target = Array1::<f64>::zeros(2);

        // check the linear term
        for i in 0..2 {
            assert_eq!(c_target[i], sub_p.c[i]);
        }

        // check the quadratic term
        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(
                    q_target.get(i, j).unwrap_or(&0.0),
                    sub_p.q.get(i, j).unwrap_or(&0.0)
                );
            }
        }

        // check the constant term
        assert_eq!(0.5, constant);
    }

    #[test]
    fn test_generate_sub_problem_2() {
        // the idea of this test is, given a QUBO & some fixed variables, generate an equivalent problem

        // f(x) = 0.5<x,x>
        let mut q = TriMat::<f64>::new((3, 3));

        q.add_triplet(0, 0, 1.0);
        q.add_triplet(0, 1, 2.0);
        q.add_triplet(0, 2, 3.0);

        q.add_triplet(1, 0, 5.0);
        q.add_triplet(1, 1, 0.0);
        q.add_triplet(1, 2, 1.0);

        q.add_triplet(2, 0, 1.0);
        q.add_triplet(2, 1, 5.0);
        q.add_triplet(2, 2, 6.0);

        let c = Array1::<f64>::from_vec(vec![0.0, 1.0, 3.0]);
        let p = Qubo::new_with_c(q.to_csr(), c);

        // we have variable x_0 fixed to 1
        let mut fixed_variables = HashMap::default();
        fixed_variables.insert(0, 1);

        // this should generate a subproblem with the following matrix
        // [0 1]  [4.5]
        // [5 6]  [5]

        let (sub_p, _, constant) = super::make_sub_problem(&p, &fixed_variables);

        // fix the expected matrix
        let mut q_target_tri = TriMat::<f64>::new((2, 2));

        q_target_tri.add_triplet(0, 0, 0.0);
        q_target_tri.add_triplet(0, 1, 1.0);
        q_target_tri.add_triplet(1, 0, 5.0);
        q_target_tri.add_triplet(1, 1, 6.0);

        let q_target: CsMat<f64> = q_target_tri.to_csr();

        let c_target = Array1::<f64>::from_vec(vec![4.5, 5.0]);

        // check the linear term
        for i in 0..2 {
            assert_eq!(c_target[i], sub_p.c[i]);
        }

        // check the quadratic term
        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(
                    q_target.get(i, j).unwrap_or(&0.0),
                    sub_p.q.get(i, j).unwrap_or(&0.0)
                );
            }
        }

        // check the constant term
        assert_eq!(0.5, constant);
    }

    #[test]
    fn test_generate_sub_problem_includes_fixed_linear_term_in_constant() {
        let q = CsMat::<f64>::zero((2, 2));
        let c = Array1::<f64>::from_vec(vec![3.5, -2.0]);
        let p = Qubo::new_with_c(q, c);

        let mut fixed_variables = HashMap::default();
        fixed_variables.insert(0, 1);

        let (sub_p, _, constant) = super::make_sub_problem(&p, &fixed_variables);

        assert_eq!(sub_p.c[0], -2.0);
        assert_eq!(constant, 3.5);
    }
}
