use crate::preprocess::solve_small_components_with_adjacency;
use crate::qubo::Qubo;
use crate::FixedVarMap;
use std::collections::VecDeque;

pub(crate) type GradientAdjacency = Vec<Vec<(usize, f64)>>;

#[derive(Clone)]
pub(crate) struct GradientBounds {
    fixed_values: Vec<Option<u8>>,
    lower: Vec<f64>,
    upper: Vec<f64>,
}

impl GradientBounds {
    pub(crate) fn new(
        qubo: &Qubo,
        persistent: &FixedVarMap,
        extra_fixed: &FixedVarMap,
        adjacency: &GradientAdjacency,
    ) -> Self {
        let mut bounds = Self {
            fixed_values: vec![None; qubo.num_x()],
            lower: vec![0.0; qubo.num_x()],
            upper: vec![0.0; qubo.num_x()],
        };
        bounds.reset(qubo, persistent, extra_fixed, adjacency);
        bounds
    }

    fn reset(
        &mut self,
        qubo: &Qubo,
        persistent: &FixedVarMap,
        extra_fixed: &FixedVarMap,
        adjacency: &GradientAdjacency,
    ) {
        self.fixed_values.fill(None);
        for (&index, &value) in persistent.iter().chain(extra_fixed) {
            self.fixed_values[index] = Some(value as u8);
        }
        for i in 0..qubo.num_x() {
            if self.fixed_values[i].is_none() {
                self.recompute_row(qubo, adjacency, i);
            }
        }
    }

    fn recompute_row(&mut self, qubo: &Qubo, adjacency: &GradientAdjacency, i: usize) {
        let mut lower = qubo.c[i];
        let mut upper = qubo.c[i];
        for &(neighbor, coeff) in &adjacency[i] {
            if let Some(value) = self.fixed_values[neighbor] {
                apply_fixed_term(&mut lower, &mut upper, coeff, value);
            } else {
                apply_free_term(&mut lower, &mut upper, coeff);
            }
        }
        self.lower[i] = lower;
        self.upper[i] = upper;
    }
}

pub(crate) struct PersistenceWorkspace {
    bounds: GradientBounds,
    queue: VecDeque<(usize, u8)>,
}

impl PersistenceWorkspace {
    pub(crate) fn new(bounds: GradientBounds) -> Self {
        Self {
            bounds,
            queue: VecDeque::new(),
        }
    }
}

/// Conditional strict deductions only. Component tie-breaking is deliberately
/// excluded: a selected minimizer is not a fact about every conditional optimum.
pub(crate) struct GradientProbe {
    base: GradientBounds,
    bounds: GradientBounds,
    base_candidates: Vec<usize>,
    dirty: Vec<usize>,
    is_dirty: Vec<bool>,
    scan: Vec<usize>,
    queue: VecDeque<(usize, u8)>,
}

impl GradientProbe {
    pub(crate) fn new(qubo: &Qubo, fixed: &FixedVarMap, adjacency: &GradientAdjacency) -> Self {
        let base = GradientBounds::new(qubo, fixed, &FixedVarMap::default(), adjacency);
        let base_candidates = (0..qubo.num_x())
            .filter(|&i| {
                base.fixed_values[i].is_none() && (base.lower[i] > 0.0 || base.upper[i] < 0.0)
            })
            .collect();
        Self {
            bounds: base.clone(),
            base,
            base_candidates,
            dirty: Vec::new(),
            is_dirty: vec![false; qubo.num_x()],
            scan: Vec::new(),
            queue: VecDeque::new(),
        }
    }

    fn remember(&mut self, i: usize) {
        if !self.is_dirty[i] {
            self.is_dirty[i] = true;
            self.dirty.push(i);
        }
    }

    fn fix_if_strict(&mut self, i: usize, found: &mut FixedVarMap) {
        if self.bounds.fixed_values[i].is_some() {
            return;
        }
        let value = if self.bounds.lower[i] > 0.0 {
            0
        } else if self.bounds.upper[i] < 0.0 {
            1
        } else {
            return;
        };
        self.remember(i);
        self.bounds.fixed_values[i] = Some(value);
        found.insert(i, usize::from(value));
        self.queue.push_back((i, value));
    }

    /// Returns the assumption and new deductions, not a copy of the input map.
    pub(crate) fn probe(
        &mut self,
        qubo: &Qubo,
        adjacency: &GradientAdjacency,
        variable: usize,
        value: usize,
    ) -> FixedVarMap {
        for i in self.dirty.drain(..) {
            self.bounds.fixed_values[i] = self.base.fixed_values[i];
            self.bounds.lower[i] = self.base.lower[i];
            self.bounds.upper[i] = self.base.upper[i];
            self.is_dirty[i] = false;
        }
        self.queue.clear();
        self.scan.clear();
        debug_assert!(self.base.fixed_values[variable].is_none());
        self.remember(variable);
        self.bounds.fixed_values[variable] = Some(value as u8);
        self.scan.extend_from_slice(&self.base_candidates);
        for &(neighbor, _) in &adjacency[variable] {
            if self.bounds.fixed_values[neighbor].is_some() {
                continue;
            }
            if !self.is_dirty[neighbor] {
                self.remember(neighbor);
                // Preserve the cold probe's summation order near a zero bound.
                self.bounds.recompute_row(qubo, adjacency, neighbor);
            }
            self.scan.push(neighbor);
        }
        self.scan.sort_unstable();
        self.scan.dedup();
        let mut found = FixedVarMap::default();
        found.insert(variable, value);
        for position in 0..self.scan.len() {
            self.fix_if_strict(self.scan[position], &mut found);
        }
        while let Some((i, value)) = self.queue.pop_front() {
            for &(target, coeff) in &adjacency[i] {
                if self.bounds.fixed_values[target].is_some() {
                    continue;
                }
                self.remember(target);
                remove_free_term(
                    &mut self.bounds.lower[target],
                    &mut self.bounds.upper[target],
                    coeff,
                );
                apply_fixed_term(
                    &mut self.bounds.lower[target],
                    &mut self.bounds.upper[target],
                    coeff,
                    value,
                );
                self.fix_if_strict(target, &mut found);
            }
        }
        found
    }
}

/// This function takes a QUBO and a set of persistent variables and returns a new set of persistent variables by repeatedly
/// recomputing the persistent variables until.
pub fn compute_iterative_persistence(
    qubo: &Qubo,
    persistent: &FixedVarMap,
    iter_lim: usize,
) -> FixedVarMap {
    let adjacency = build_gradient_adjacency(qubo);
    compute_iterative_persistence_with_adjacency(
        qubo,
        persistent.clone(),
        &FixedVarMap::default(),
        iter_lim,
        &adjacency,
    )
}

pub(crate) fn compute_iterative_persistence_with_adjacency(
    qubo: &Qubo,
    mut persistent: FixedVarMap,
    extra_fixed: &FixedVarMap,
    iter_lim: usize,
    adjacency: &GradientAdjacency,
) -> FixedVarMap {
    if iter_lim == 0 || qubo.num_x() == 0 {
        return persistent;
    }
    let bounds = GradientBounds::new(qubo, &persistent, extra_fixed, adjacency);
    let mut workspace = PersistenceWorkspace::new(bounds);
    propagate_persistent_in_place(&mut persistent, adjacency, &mut workspace);
    // The queue reaches gradient closure. Solving entire disconnected components
    // cannot change any remaining free variable's bounds or connectivity.
    solve_small_components_with_adjacency(qubo, &mut persistent, Some(extra_fixed), 10, adjacency);
    persistent
}

fn propagate_persistent_in_place(
    persistent: &mut FixedVarMap,
    adjacency: &[Vec<(usize, f64)>],
    workspace: &mut PersistenceWorkspace,
) {
    let PersistenceWorkspace { bounds, queue } = workspace;
    let GradientBounds {
        fixed_values,
        lower,
        upper,
    } = bounds;
    let num_x = fixed_values.len();
    queue.clear();

    for i in 0..num_x {
        if fixed_values[i].is_some() {
            continue;
        }

        if lower[i] > 0.0 {
            fixed_values[i] = Some(0);
            persistent.insert(i, 0);
            queue.push_back((i, 0u8));
        } else if upper[i] < 0.0 {
            fixed_values[i] = Some(1);
            persistent.insert(i, 1);
            queue.push_back((i, 1u8));
        }
    }

    while let Some((fixed_variable, value)) = queue.pop_front() {
        for &(target, coeff) in &adjacency[fixed_variable] {
            if fixed_values[target].is_some() {
                continue;
            }

            remove_free_term(&mut lower[target], &mut upper[target], coeff);
            apply_fixed_term(&mut lower[target], &mut upper[target], coeff, value);

            if lower[target] > 0.0 {
                fixed_values[target] = Some(0);
                persistent.insert(target, 0);
                queue.push_back((target, 0u8));
            } else if upper[target] < 0.0 {
                fixed_values[target] = Some(1);
                persistent.insert(target, 1);
                queue.push_back((target, 1u8));
            }
        }
    }
}

#[cfg(test)]
pub(crate) fn cold_gradient_probe(
    qubo: &Qubo,
    fixed: FixedVarMap,
    adjacency: &GradientAdjacency,
) -> FixedVarMap {
    let bounds = GradientBounds::new(qubo, &fixed, &FixedVarMap::default(), adjacency);
    let mut workspace = PersistenceWorkspace::new(bounds);
    let mut result = fixed;
    propagate_persistent_in_place(&mut result, adjacency, &mut workspace);
    result
}

fn apply_free_term(lower: &mut f64, upper: &mut f64, coeff: f64) {
    if coeff <= 0.0 {
        *lower += coeff;
    }
    if coeff >= 0.0 {
        *upper += coeff;
    }
}

fn remove_free_term(lower: &mut f64, upper: &mut f64, coeff: f64) {
    if coeff <= 0.0 {
        *lower -= coeff;
    }
    if coeff >= 0.0 {
        *upper -= coeff;
    }
}

fn apply_fixed_term(lower: &mut f64, upper: &mut f64, coeff: f64, value: u8) {
    let contribution = coeff * f64::from(value);
    *lower += contribution;
    *upper += contribution;
}

pub(crate) fn build_gradient_adjacency(qubo: &Qubo) -> GradientAdjacency {
    let num_x = qubo.num_x();
    let mut counts = vec![0usize; num_x];

    for (&_value, (i, j)) in &qubo.q {
        counts[i] += 1;
        counts[j] += 1;
    }

    let mut adjacency = counts
        .into_iter()
        .map(Vec::with_capacity)
        .collect::<Vec<_>>();

    for (&value, (i, j)) in &qubo.q {
        let coeff = 0.5 * value;

        adjacency[i].push((j, coeff));
        adjacency[j].push((i, coeff));
    }

    adjacency
}

/// This function takes a QUBO and a set of persistent variables and returns a new set of persistent variables by computing the
/// persistent variables once.
pub fn compute_persistent(qubo: &Qubo, persistent: &FixedVarMap) -> FixedVarMap {
    // create a new hashmap to store the new persistent variables
    let mut new_persistent = persistent.clone();

    // iterate over all the variables in the QUBO
    for i in 0..qubo.num_x() {
        if persistent.contains_key(&i) {
            continue;
        }

        // find the bounds of the gradient in each direction
        let (lower, upper) = grad_bounds(qubo, i, persistent);

        // if the lower bound it positive, then we can set the variable to 0
        if lower > 0.0 {
            new_persistent.insert(i, 0);
        }

        // if the upper bound is below 0, then we can set the variable to 1
        if upper < 0.0 {
            new_persistent.insert(i, 1);
        }
    }

    new_persistent
}

/// Finds bounds of the i-th index of the gradients of the QUBO function
///
/// # Panics
/// This function should not panic as the unwraps are bounded on the size of the QUBO matrix
pub fn grad_bounds(qubo: &Qubo, i: usize, persistent: &FixedVarMap) -> (f64, f64) {
    // set up tracking variables for each bound
    let mut lower = 0.0;
    let mut upper = 0.0;

    // get the i-th row of the Q matrix, this is safe as we are bounded by the size of the Q matrix
    let x = qubo.q.outer_view(i).unwrap();

    // get the i-th column of the Q matrix, this is safe for the same reason
    let binding = qubo.q.transpose_view();
    let y = binding.outer_view(i).unwrap();

    accumulate_grad_terms(x.iter(), persistent, &mut lower, &mut upper);
    accumulate_grad_terms(y.iter(), persistent, &mut lower, &mut upper);

    // add the contribution from the constant term
    lower += qubo.c[i];
    upper += qubo.c[i];

    (lower, upper)
}

fn accumulate_grad_terms<'a, I>(
    terms: I,
    persistent: &FixedVarMap,
    lower: &mut f64,
    upper: &mut f64,
) where
    I: Iterator<Item = (usize, &'a f64)>,
{
    for (index, x_j) in terms {
        let value = 0.5 * *x_j;

        // if it is a fixed variable, we have effectively removed this variable from the QUBO
        if let Some(&fixed) = persistent.get(&index) {
            let fixed = fixed as f64;
            *lower += value * fixed;
            *upper += value * fixed;
        } else {
            // if it is not in the persistent set, then we can choose the best value

            // if the value is negative, we would set it to 1 to minimize the gradient else 0
            if value <= 0.0 {
                *lower += value;
            }

            // if the value is positive, we would set it to 1 to maximize the gradient else 0
            if value >= 0.0 {
                *upper += value;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qubo::Qubo;
    use crate::FixedVarMap as HashMap;
    use ndarray::Array1;
    use sprs::CsMat;

    #[test]
    fn test_persistence() {
        //build the problem
        let eye = CsMat::eye(3);
        let c = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let p = Qubo::new_with_c(eye, c);
        let persist = compute_iterative_persistence(&p, &HashMap::default(), 3);

        assert!(persist.contains_key(&0));
        assert!(persist.contains_key(&1));
        assert!(persist.contains_key(&2));

        assert!(persist[&0].eq(&0));
        assert!(persist[&1].eq(&0));
        assert!(persist[&2].eq(&0));
    }
    #[test]
    fn test_grad_bounds_1() {
        let eye = CsMat::eye(3);
        let c = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let p = Qubo::new_with_c(eye, c);
        assert_eq!(grad_bounds(&p, 0, &HashMap::default()), (1.0, 2.0));
        assert_eq!(grad_bounds(&p, 1, &HashMap::default()), (2.0, 3.0));
        assert_eq!(grad_bounds(&p, 2, &HashMap::default()), (3.0, 4.0));
    }

    #[test]
    fn test_grad_bounds_2() {
        let eye = CsMat::eye(3);
        let c = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let p = Qubo::new_with_c(eye, c);

        let mut fixed_vars = HashMap::default();
        fixed_vars.insert(0, 1);
        fixed_vars.insert(2, 1);

        assert_eq!(grad_bounds(&p, 0, &fixed_vars), (2.0, 2.0));
        assert_eq!(grad_bounds(&p, 1, &fixed_vars), (2.0, 3.0));
        assert_eq!(grad_bounds(&p, 2, &fixed_vars), (4.0, 4.0));
    }

    #[test]
    fn test_grad_bounds_3() {
        let zero = CsMat::zero((3, 3));
        let c = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let p = Qubo::new_with_c(zero, c);

        assert_eq!(grad_bounds(&p, 0, &HashMap::default()), (1.0, 1.0));
        assert_eq!(grad_bounds(&p, 1, &HashMap::default()), (2.0, 2.0));
        assert_eq!(grad_bounds(&p, 2, &HashMap::default()), (3.0, 3.0));
    }

    // old test, not really relevant anymore as the presolve will solve this entirely
    // #[test]
    // fn test_alternating_persistence() {
    //     let p = make_solver_qubo();
    //     let p_symm = p.make_symmetric();
    //     let persist = compute_iterative_persistence(&p_symm, &HashMap::new(), p.num_x());
    //
    //     assert_eq!(persist.len(), 50);
    // }
}
