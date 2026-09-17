//! Exact binary elimination with a reverse reconstruction stack. This operates
//! on the input polynomial, before numerical convexification changes its rows.
use super::preprocess_qubo;
use crate::qubo::Qubo;
use crate::solver_options::{NodeLowerBoundSelection, SolverOptions};
use crate::subproblemsolvers::roofdual::PreparedRoofDual;
use crate::FixedVarMap;
use ndarray::Array1;
use sprs::TriMat;
use std::collections::{BTreeMap, VecDeque};
use std::time::{Duration, Instant};

mod cut_dominance;
mod degree_three;
mod dominant_edge;
mod small_block;
pub use cut_dominance::CutDominanceStatistics;

#[derive(Clone, Copy, Debug, Default)]
pub struct RootReductionStatistics {
    /// All low-degree eliminations, excluding edge contractions.
    pub eliminated: usize,
    /// Subset of `eliminated` with three neighbors.
    pub degree_three_eliminated: usize,
    /// Degree-three attempts skipped for a cubic term or inexact arithmetic.
    pub degree_three_skipped: usize,
    pub contracted: usize,
    /// Contractions with a tie in at least one conditional flip-gain bound.
    pub weak_contractions: usize,
    pub additional_fixed: usize,
    pub remaining: usize,
    pub passes: usize,
    pub blocks_eliminated: usize,
    pub block_variables: usize,
    pub cut_dominance: CutDominanceStatistics,
}

#[derive(Clone)]
struct Elimination {
    variable: usize,
    neighbors: Vec<usize>,
    choices: [usize; 8],
}

#[derive(Clone)]
pub(crate) struct RootReduction {
    pub qubo: Qubo,
    pub constant: f64,
    pub remaining: Vec<usize>,
    pub fixed: FixedVarMap,
    pub statistics: RootReductionStatistics,
    original_size: usize,
    steps: Vec<Elimination>,
}

impl RootReduction {
    pub fn reconstruct(&self, reduced: &Array1<usize>) -> Array1<usize> {
        assert_eq!(reduced.len(), self.remaining.len());
        let mut full = Array1::zeros(self.original_size);
        for (&i, &value) in self.remaining.iter().zip(reduced) {
            full[i] = value;
        }
        for (&i, &value) in &self.fixed {
            full[i] = value;
        }
        for step in self.steps.iter().rev() {
            let mask = step
                .neighbors
                .iter()
                .enumerate()
                .fold(0, |mask, (bit, &i)| mask | (full[i] << bit));
            full[step.variable] = step.choices[mask];
        }
        full
    }

    pub fn project(&self, full: &Array1<usize>) -> Array1<usize> {
        Array1::from_iter(self.remaining.iter().map(|&i| full[i]))
    }
}

// Refuse a transformation rather than silently discard a small coefficient.
// TwoSum detects inexact additions; ordinary integer benchmark data is exact.
pub(crate) fn add(a: f64, b: f64) -> Option<f64> {
    let sum = a + b;
    let z = sum - a;
    let error = (a - (sum - z)) + (b - z);
    (sum.is_finite() && error == 0.0).then_some(sum)
}

struct Graph {
    linear: Vec<f64>,
    edges: Vec<BTreeMap<usize, f64>>,
    active: Vec<bool>,
    constant: f64,
}

impl Graph {
    fn new(qubo: &Qubo) -> Option<Self> {
        let n = qubo.num_x();
        let mut graph = Self {
            linear: qubo.c.to_vec(),
            edges: vec![BTreeMap::new(); n],
            active: vec![true; n],
            constant: 0.0,
        };
        if graph.linear.iter().any(|x| !x.is_finite()) {
            return None;
        }
        for (&value, (i, j)) in &qubo.q {
            let half = value * 0.5;
            if !half.is_finite() || half * 2.0 != value {
                return None;
            }
            if i == j {
                graph.linear[i] = add(graph.linear[i], half)?;
            } else {
                graph.add_edge(i, j, half)?;
            }
        }
        Some(graph)
    }

    fn add_edge(&mut self, i: usize, j: usize, delta: f64) -> Option<()> {
        let value = add(self.edges[i].get(&j).copied().unwrap_or(0.0), delta)?;
        if value == 0.0 {
            self.edges[i].remove(&j);
            self.edges[j].remove(&i);
        } else {
            self.edges[i].insert(j, value);
            self.edges[j].insert(i, value);
        }
        Some(())
    }

    fn fix(&mut self, i: usize, value: usize) -> Option<()> {
        if value > 1 || i >= self.active.len() || !self.active[i] {
            return None;
        }
        if value == 1 {
            self.constant = add(self.constant, self.linear[i])?;
        }
        for (j, weight) in std::mem::take(&mut self.edges[i]) {
            self.edges[j].remove(&i);
            if value == 1 {
                self.linear[j] = add(self.linear[j], weight)?;
            }
        }
        self.active[i] = false;
        Some(())
    }

    fn eliminate(&mut self, i: usize) -> Option<Elimination> {
        let neighbors: Vec<_> = self.edges[i].keys().copied().collect();
        let mut table = [0.0; 4];
        let mut choices = [0; 8];
        for mask in 0..1 << neighbors.len() {
            let mut delta = self.linear[i];
            for (bit, &j) in neighbors.iter().enumerate() {
                if mask & (1 << bit) != 0 {
                    delta = add(delta, self.edges[i][&j])?;
                }
            }
            choices[mask] = usize::from(delta < 0.0);
            table[mask] = delta.min(0.0);
        }
        // min_xi xi*(a + b*xj + c*xk) is a four-entry binary table,
        // hence exactly a constant + two linear terms + one quadratic term.
        self.constant = add(self.constant, table[0])?;
        for (bit, &j) in neighbors.iter().enumerate() {
            self.linear[j] = add(self.linear[j], add(table[1 << bit], -table[0])?)?;
        }
        if neighbors.len() == 2 {
            let weight = add(add(add(table[3], -table[1])?, -table[2])?, table[0])?;
            self.add_edge(neighbors[0], neighbors[1], weight)?;
        }
        for &j in &neighbors {
            self.edges[j].remove(&i);
        }
        self.edges[i].clear();
        self.active[i] = false;
        Some(Elimination {
            variable: i,
            neighbors,
            choices,
        })
    }

    fn compact(&self) -> (Qubo, Vec<usize>) {
        let remaining: Vec<_> = (0..self.active.len()).filter(|&i| self.active[i]).collect();
        let mut index = vec![usize::MAX; self.active.len()];
        for (j, &i) in remaining.iter().enumerate() {
            index[i] = j;
        }
        let mut q = TriMat::new((remaining.len(), remaining.len()));
        for &i in &remaining {
            for (&j, &weight) in &self.edges[i] {
                q.add_triplet(index[i], index[j], weight);
            }
        }
        let c = Array1::from_iter(remaining.iter().map(|&i| self.linear[i]));
        (Qubo::new_with_c(q.to_csr(), c), remaining)
    }
}

pub(crate) fn reduce_root(
    qubo: &Qubo,
    fixed: &FixedVarMap,
    options: &SolverOptions,
    budget: Duration,
) -> Option<RootReduction> {
    reduce(qubo, fixed, options, budget, true)
}

fn reduce(
    qubo: &Qubo,
    fixed: &FixedVarMap,
    options: &SolverOptions,
    budget: Duration,
    presolve: bool,
) -> Option<RootReduction> {
    reduce_tracked(
        qubo,
        fixed,
        options,
        budget,
        presolve,
        &mut CutDominanceStatistics::default(),
    )
}

pub(crate) fn reduce_root_with_statistics(
    qubo: &Qubo,
    fixed: &FixedVarMap,
    options: &SolverOptions,
    budget: Duration,
    cut_stats: &mut CutDominanceStatistics,
) -> Option<RootReduction> {
    reduce_tracked(qubo, fixed, options, budget, true, cut_stats)
}

fn reduce_tracked(
    qubo: &Qubo,
    fixed: &FixedVarMap,
    options: &SolverOptions,
    budget: Duration,
    presolve: bool,
    cut_stats: &mut CutDominanceStatistics,
) -> Option<RootReduction> {
    let start = Instant::now();
    let cut_deadline = start + budget.min(Duration::from_millis(50));
    let eligible_degree = if options.root_degree_three_elimination {
        3
    } else {
        2
    };
    let mut graph = Graph::new(qubo)?;
    let mut ordered_fixed: Vec<_> = fixed.iter().map(|(&i, &v)| (i, v)).collect();
    ordered_fixed.sort_unstable();
    for (i, value) in ordered_fixed {
        graph.fix(i, value)?;
    }
    let mut result = RootReduction {
        qubo: Qubo::new(sprs::CsMat::zero((0, 0))),
        constant: 0.0,
        remaining: Vec::new(),
        fixed: fixed.clone(),
        statistics: RootReductionStatistics::default(),
        original_size: qubo.num_x(),
        steps: Vec::new(),
    };
    loop {
        let mut queued: Vec<_> = (0..qubo.num_x())
            .map(|i| {
                graph.active[i]
                    && (options.root_dominant_edge_contraction
                        || (options.root_low_degree_elimination
                            && graph.edges[i].len() <= eligible_degree))
            })
            .collect();
        let mut queue: VecDeque<_> = (0..qubo.num_x()).filter(|&i| queued[i]).collect();
        while let Some(i) = queue.pop_front() {
            queued[i] = false;
            if start.elapsed() >= budget {
                break;
            }
            if !graph.active[i] {
                continue;
            }
            let reduction = if options.root_low_degree_elimination && graph.edges[i].len() <= 2 {
                let step = graph.eliminate(i)?;
                let affected = step.neighbors.clone();
                result.statistics.eliminated += 1;
                Some((step, affected))
            } else if options.root_dominant_edge_contraction {
                if let Some((j, weak)) = graph.dominant_neighbor(i) {
                    let reduction = graph.contract(i, j)?;
                    result.statistics.contracted += 1;
                    result.statistics.weak_contractions += usize::from(weak);
                    Some(reduction)
                } else {
                    None
                }
            } else {
                None
            };
            let (step, affected) = if let Some(reduction) = reduction {
                reduction
            } else if options.root_low_degree_elimination
                && options.root_degree_three_elimination
                && graph.edges[i].len() == 3
            {
                let Some(step) = graph.try_eliminate_three(i) else {
                    result.statistics.degree_three_skipped += 1;
                    continue;
                };
                let affected = step.neighbors.clone();
                result.statistics.eliminated += 1;
                result.statistics.degree_three_eliminated += 1;
                (step, affected)
            } else {
                continue;
            };
            // Only rows changed by this reduction need another certificate test.
            // Never batch weak relations certified on an earlier graph.
            for j in affected {
                if graph.active[j]
                    && !queued[j]
                    && (options.root_dominant_edge_contraction
                        || (options.root_low_degree_elimination
                            && graph.edges[j].len() <= eligible_degree))
                {
                    queued[j] = true;
                    queue.push_back(j);
                }
            }
            result.steps.push(step);
        }
        if options.root_cut_dominance && start.elapsed() < budget {
            if let Some((i, j, weak)) = graph.dominant_cut_edge(cut_deadline, cut_stats) {
                if j == graph.active.len() {
                    let value = graph.cut_anchor_value(i)?;
                    graph.fix(i, value)?;
                    result.fixed.insert(i, value);
                    result.statistics.additional_fixed += 1;
                    cut_stats.fixed += 1;
                } else {
                    let (step, _) = graph.contract(i, j)?;
                    result.steps.push(step);
                    result.statistics.contracted += 1;
                    result.statistics.weak_contractions += usize::from(weak);
                    cut_stats.contracted += 1;
                }
                result.statistics.passes += 1;
                continue;
            }
        }
        if options.small_block_elimination && start.elapsed() < budget {
            if let Some(steps) = graph.eliminate_small_block(start, budget) {
                result.statistics.blocks_eliminated += 1;
                result.statistics.block_variables += steps.len();
                result.steps.extend(steps);
                // The replacement edge/fields can expose new low-degree or
                // dominance reductions. Recheck them before another block.
                result.statistics.passes += 1;
                continue;
            }
        }
        if result.steps.is_empty() && result.statistics.additional_fixed == 0 {
            return None;
        }
        result.statistics.passes += 1;
        let (reduced, mapping) = graph.compact();
        if !presolve || mapping.is_empty() || start.elapsed() >= budget {
            break;
        }
        let mut new_fixed = preprocess_qubo(&reduced, &FixedVarMap::default(), true);
        if matches!(options.node_lower_bound, NodeLowerBoundSelection::RoofDual) {
            let prepared = PreparedRoofDual::new(&reduced);
            let roof = if options.roof_dual_weak_persistencies {
                prepared.solve_iterative_with_weak_persistencies(&new_fixed, mapping.len())
            } else {
                prepared.solve_iterative(&new_fixed, mapping.len())
            };
            new_fixed.extend(roof.fixed_variables);
            new_fixed.extend(roof.weak_fixed_variables);
        }
        if new_fixed.is_empty() {
            break;
        }
        let mut ordered: Vec<_> = new_fixed.into_iter().collect();
        ordered.sort_unstable();
        for (index, value) in ordered {
            let original = mapping[index];
            graph.fix(original, value)?;
            result.fixed.insert(original, value);
            result.statistics.additional_fixed += 1;
        }
    }
    (result.qubo, result.remaining) = graph.compact();
    result.constant = graph.constant;
    result.statistics.remaining = result.remaining.len();
    result.statistics.cut_dominance = *cut_stats;
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(n: usize, mask: usize) -> Array1<usize> {
        Array1::from_iter((0..n).map(|i| (mask >> i) & 1))
    }

    fn check_projection(qubo: &Qubo, fixed: &FixedVarMap) {
        let mut options = SolverOptions::new();
        options.root_dominant_edge_contraction = false;
        let result = reduce(qubo, fixed, &options, Duration::from_secs(5), false).unwrap();
        let mut minima = vec![f64::INFINITY; 1 << result.remaining.len()];
        for mask in 0..1 << qubo.num_x() {
            let x = point(qubo.num_x(), mask);
            if fixed.iter().any(|(&i, &v)| x[i] != v) {
                continue;
            }
            let reduced_mask = result
                .remaining
                .iter()
                .enumerate()
                .fold(0, |m, (bit, &i)| m | (x[i] << bit));
            minima[reduced_mask] = minima[reduced_mask].min(qubo.eval_usize(&x));
        }
        for (mask, expected) in minima.into_iter().enumerate() {
            let small = point(result.remaining.len(), mask);
            let full = result.reconstruct(&small);
            assert_eq!(result.qubo.eval_usize(&small) + result.constant, expected);
            assert_eq!(qubo.eval_usize(&full), expected);
            assert!(fixed.iter().all(|(&i, &v)| full[i] == v));
            assert!(full.iter().all(|&v| v <= 1));
        }
    }

    #[test]
    fn low_degree_tables_preserve_every_conditional_minimum() {
        // All signs, zeros, ties, asymmetric storage and nonzero diagonals.
        for a in -2..=2 {
            for b in -2..=2 {
                for c in -2..=2 {
                    let mut q = TriMat::new((3, 3));
                    q.add_triplet(0, 0, 2.0);
                    q.add_triplet(0, 1, 2.0 * f64::from(b));
                    q.add_triplet(2, 0, 2.0 * f64::from(c));
                    let qubo = Qubo::new_with_c(
                        q.to_csc(),
                        ndarray::array![f64::from(a) - 1.0, 1.0, -1.0],
                    );
                    check_projection(&qubo, &FixedVarMap::default());
                    check_projection(&qubo, &[(1, 1)].into_iter().collect());
                }
            }
        }
    }

    #[test]
    fn low_degree_preserves_dense_core_and_reverse_reconstruction() {
        // Two chains incident to a dense core; elimination creates and cancels edges.
        for seed in 0..32 {
            let mut q = TriMat::new((9, 9));
            for i in 0..5 {
                for j in i + 1..5 {
                    q.add_triplet(i, j, 2.0 * ((i * 3 + j + seed) % 7) as f64 - 6.0);
                }
            }
            for (i, j) in [(0, 5), (5, 6), (6, 1), (2, 7), (7, 8), (8, 3)] {
                q.add_triplet(i, j, if (i + seed) % 2 == 0 { 4.0 } else { -6.0 });
            }
            let qubo = Qubo::new_with_c(
                q.to_csr(),
                Array1::from_iter((0..9).map(|i| ((i + seed) % 5) as f64 - 2.0)),
            );
            check_projection(&qubo, &FixedVarMap::default());
            check_projection(&qubo, &[(5, 1)].into_iter().collect());
        }
    }

    #[test]
    fn low_degree_rejects_rounding_and_overflow_and_honors_budget() {
        assert_eq!(add(1e20, 1.0), None);
        assert_eq!(add(f64::MAX, f64::MAX), None);
        let q = Qubo::new_with_c(sprs::CsMat::zero((2, 2)), ndarray::array![-1.0, 2.0]);
        assert!(reduce_root(
            &q,
            &FixedVarMap::default(),
            &SolverOptions::new(),
            Duration::ZERO
        )
        .is_none());
        let mut bad = q.clone();
        bad.c[0] = f64::NAN;
        assert!(reduce_root(
            &bad,
            &FixedVarMap::default(),
            &SolverOptions::new(),
            Duration::from_secs(1)
        )
        .is_none());
    }

    #[test]
    fn low_degree_repeats_after_component_presolve() {
        let mut q = TriMat::new((5, 5));
        for i in 0..4 {
            for j in i + 1..4 {
                q.add_triplet(i, j, 4.0);
            }
        }
        q.add_triplet(0, 4, -2.0);
        let qubo = Qubo::new_with_c(q.to_csr(), ndarray::array![-3.0, -3.0, -3.0, -3.0, 0.0]);
        let mut options = SolverOptions::new();
        options.root_dominant_edge_contraction = false;
        options.root_degree_three_elimination = false;
        let r = reduce_root(
            &qubo,
            &FixedVarMap::default(),
            &options,
            Duration::from_secs(5),
        )
        .unwrap();
        assert_eq!(r.statistics.eliminated, 1);
        assert_eq!(r.statistics.additional_fixed, 4);
        assert_eq!(r.statistics.passes, 2);
        assert!(r.remaining.is_empty());
        let expected = (0..32)
            .map(|mask| qubo.eval_usize(&point(5, mask)))
            .fold(f64::INFINITY, f64::min);
        assert_eq!(qubo.eval_usize(&r.reconstruct(&Array1::zeros(0))), expected);
    }

    #[test]
    fn low_degree_presolve_cascade_preserves_optimum() {
        for seed in 0..20 {
            let n = 12;
            let mut q = TriMat::new((n, n));
            for i in 0..n {
                for j in i + 1..n {
                    if j == i + 1 || (j < 8 && (i + j + seed) % 3 == 0) {
                        q.add_triplet(i, j, 2.0 * ((i + 2 * j + seed) % 9) as f64 - 8.0);
                    }
                }
            }
            let qubo = Qubo::new_with_c(
                q.to_csr(),
                Array1::from_iter((0..n).map(|i| ((i + seed) % 7) as f64 - 3.0)),
            );
            let fixed = [(0, seed % 2)].into_iter().collect();
            for weak in [false, true] {
                let mut opts = SolverOptions::new();
                opts.roof_dual_weak_persistencies = weak;
                let r = reduce_root(&qubo, &fixed, &opts, Duration::from_secs(5)).unwrap();
                let expected = (0..1 << n)
                    .filter(|mask| mask & 1 == seed % 2)
                    .map(|mask| qubo.eval_usize(&point(n, mask)))
                    .fold(f64::INFINITY, f64::min);
                let actual = (0..1 << r.remaining.len())
                    .map(|mask| {
                        let x = point(r.remaining.len(), mask);
                        let full = r.reconstruct(&x);
                        assert_eq!(qubo.eval_usize(&full), r.qubo.eval_usize(&x) + r.constant);
                        qubo.eval_usize(&full)
                    })
                    .fold(f64::INFINITY, f64::min);
                assert_eq!(actual, expected);
            }
        }
    }
}
