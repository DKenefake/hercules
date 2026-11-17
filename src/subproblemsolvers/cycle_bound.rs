//! Capacity-accounted frustrated cycles supplement, rather than alter, the SDP.
//! Every packed cycle pays twice its weight; residual edge capacities prevent
//! double counting. Native coefficients and fixed contributions remain exact.
use crate::{qubo::Qubo, FixedVarMap};
use std::collections::{BTreeMap, VecDeque};
use std::time::{Duration, Instant};
mod certificate;
mod root;
pub use root::RootCycleStatistics;
/// Offline small-region bound screening; not used by the solve path.
pub mod regions;
/// Offline cutoff-scoped parity probing; not used by the solve path.
pub mod relation_probe;
pub mod frustration_probe;
pub use certificate::{Certificate, Evaluation};

#[derive(Default)]
struct PackingTrace {
    cycles: Vec<(Vec<usize>, i128)>,
    residual: Vec<i128>,
}

#[derive(Debug)]
pub struct Bound {
    pub lower: f64,
    pub baseline: f64,
    pub cycles: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PackingStrategy {
    /// Original 512-cycle reference, retained for controlled comparisons.
    Greedy,
    ShortCycles,
    WeightedCycles,
    Portfolio,
    ShortPortfolio,
    /// Continue the original packing within the same time budget.
    ExtendedGreedy,
    /// Compare whole-capacity packing with rotating half-capacity packing.
    #[default]
    FractionalPortfolio,
}

// Exact eighth-integral inputs; spin energy is stored in units of 1/64.
// Decline unsupported inputs instead of silently rounding their coefficients.
fn coefficient(x: f64) -> Option<i128> {
    let v = x * 8.0;
    (v.is_finite() && v.abs() <= (1u64 << 40) as f64 && v.fract() == 0.0).then_some(v as i128)
}

// Guarantees projection sums and half-products stay exactly representable.
pub fn exact_projection_supported(q: &Qubo) -> bool {
    let mut total = 0i128;
    for &v in q.q.data().iter().chain(q.c.iter()) {
        let Some(v) = coefficient(v) else {
            return false;
        };
        total += v.abs();
        if total > (1i128 << 44) {
            return false;
        }
    }
    true
}

impl PreparedCycleBound {
    pub fn new(q: &Qubo) -> Option<Self> {
        if !exact_projection_supported(q) {
            return None;
        }
        let n = q.num_x();
        let mut fields = vec![0i128; n];
        let mut constant = 0i128;
        let mut couplings = BTreeMap::new();
        for (i, &c) in q.c.iter().enumerate() {
            let c = 4 * coefficient(c)?;
            fields[i] += c;
            constant += c;
        }
        for (&v, (i, j)) in &q.q {
            let v = coefficient(v)?;
            if i == j {
                fields[i] += 2 * v;
                constant += 2 * v;
            } else {
                *couplings.entry((i.min(j), i.max(j))).or_insert(0i128) += v;
                fields[i] += v;
                fields[j] += v;
                constant += v;
            }
        }
        Some(Self {
            fields,
            constant,
            edges: couplings.into_iter().filter(|(_, j)| *j != 0).collect(),
            certificate: std::sync::OnceLock::new(),
        })
    }

    /// Uses the SDP's free-variable ordering; includes fixed objective contributions exactly.
    pub fn conditioned(
        &self,
        original: &[usize],
        fixed: &FixedVarMap,
        budget: Duration,
    ) -> Option<Bound> {
        self.conditioned_to(original, fixed, budget, f64::INFINITY)
    }

    /// Stop packing once the current node cutoff has been certified.
    pub fn conditioned_to(
        &self,
        original: &[usize],
        fixed: &FixedVarMap,
        budget: Duration,
        cutoff: f64,
    ) -> Option<Bound> {
        self.conditioned_with_strategy(original, fixed, budget, cutoff, PackingStrategy::default())
    }

    pub fn conditioned_with_strategy(
        &self,
        original: &[usize],
        fixed: &FixedVarMap,
        budget: Duration,
        cutoff: f64,
        strategy: PackingStrategy,
    ) -> Option<Bound> {
        let start = Instant::now();
        if original.len() + fixed.len() != self.fields.len() {
            return None;
        }
        let n = original.len();
        let mut index = vec![usize::MAX; self.fields.len()];
        for (k, &i) in original.iter().enumerate() {
            if i >= index.len() || index[i] != usize::MAX || fixed.contains_key(&i) {
                return None;
            }
            index[i] = k;
        }
        let mut spins = vec![0i128; self.fields.len()];
        for (&i, &x) in fixed {
            if i >= spins.len() || x > 1 {
                return None;
            }
            spins[i] = 2 * x as i128 - 1;
        }
        let mut fields = vec![0i128; n];
        let mut constant = self.constant;
        for (i, &h) in self.fields.iter().enumerate() {
            if index[i] == usize::MAX {
                constant += h * spins[i];
            } else {
                fields[index[i]] = h;
            }
        }
        let mut edges = Vec::with_capacity(self.edges.len() + n);
        for &((i, j), weight) in &self.edges {
            let (u, v) = (index[i], index[j]);
            match (u != usize::MAX, v != usize::MAX) {
                (true, true) => edges.push(((u, v), weight)),
                (true, false) => fields[u] += weight * spins[j],
                (false, true) => fields[v] += weight * spins[i],
                (false, false) => constant += weight * spins[i] * spins[j],
            }
        }
        for (i, h) in fields.into_iter().enumerate() {
            if h != 0 {
                edges.push(((i, n), h));
            }
        }
        // Native pairs retain their order; anchor edges appended in index order
        // give the same per-vertex neighbor ordering without sorting each node.
        let fractional = strategy == PackingStrategy::FractionalPortfolio;
        let mut best = pack(
            n,
            constant,
            &edges,
            start,
            if fractional { budget / 2 } else { budget },
            cutoff,
            if fractional {
                PackingStrategy::ExtendedGreedy
            } else {
                strategy
            },
            0,
            None,
        )?;
        if fractional && start.elapsed() < budget && best.lower < cutoff {
            let trial = pack(
                n,
                constant,
                &edges,
                start,
                budget,
                cutoff,
                PackingStrategy::FractionalPortfolio,
                0,
                None,
            )?;
            if trial.lower > best.lower {
                best = trial;
            }
        }
        if strategy == PackingStrategy::ShortPortfolio
            && start.elapsed() < budget
            && best.lower < cutoff
        {
            let trial = pack(
                n,
                constant,
                &edges,
                start,
                budget,
                cutoff,
                PackingStrategy::ShortCycles,
                0,
                None,
            )?;
            if trial.lower > best.lower {
                best = trial;
            }
        }
        if strategy == PackingStrategy::Portfolio {
            // Independent capacity assignments compete; they are never summed.
            for root in [n, n / 3, 2 * n / 3] {
                if start.elapsed() >= budget || best.lower >= cutoff {
                    break;
                }
                let trial = pack(
                    n,
                    constant,
                    &edges,
                    start,
                    budget,
                    cutoff,
                    PackingStrategy::Greedy,
                    root,
                    None,
                )?;
                if trial.lower > best.lower {
                    best = trial;
                }
            }
        }
        Some(best)
    }
}

/// Exact immutable native spin polynomial, prepared once per solver context.
pub struct PreparedCycleBound {
    fields: Vec<i128>,
    edges: Vec<((usize, usize), i128)>,
    constant: i128,
    certificate: std::sync::OnceLock<Option<Certificate>>,
}

impl PreparedCycleBound {
    /// One immutable decomposition per coordinate/objective context.
    pub fn retained_certificate(&self) -> Option<&Certificate> {
        self.certificate
            .get_or_init(|| Certificate::new(self))
            .as_ref()
    }
}

pub fn bound(q: &Qubo, budget: Duration) -> Option<Bound> {
    PreparedCycleBound::new(q)?.conditioned(
        &(0..q.num_x()).collect::<Vec<_>>(),
        &FixedVarMap::default(),
        budget,
    )
}

#[allow(clippy::too_many_arguments)]
fn pack(
    n: usize,
    constant: i128,
    edges: &[((usize, usize), i128)],
    start: Instant,
    budget: Duration,
    cutoff: f64,
    strategy: PackingStrategy,
    first_root: usize,
    mut trace: Option<&mut PackingTrace>,
) -> Option<Bound> {
    let fractional = strategy == PackingStrategy::FractionalPortfolio;
    let scale = if fractional { 2 } else { 1 };
    let denominator = 64.0 * scale as f64;
    let mut capacity: Vec<_> = edges.iter().map(|(_, j)| scale * j.abs()).collect();
    let baseline = scale * constant - capacity.iter().sum::<i128>();
    // Flat adjacency avoids one small allocation per vertex at every node.
    let mut offsets = vec![0; n + 2];
    for &((u, v), _) in edges {
        offsets[u + 1] += 1;
        offsets[v + 1] += 1;
    }
    for i in 0..=n {
        offsets[i + 1] += offsets[i];
    }
    let mut next = offsets[..n + 1].to_vec();
    let mut adjacency = vec![(0, 0); 2 * edges.len()];
    for (e, &((u, v), _)) in edges.iter().enumerate() {
        adjacency[next[u]] = (v, e);
        next[u] += 1;
        adjacency[next[v]] = (u, e);
        next[v] += 1;
    }
    let mut gain = 0;
    let mut cycles = 0;
    let mut parent = vec![usize::MAX; n + 1];
    let mut parent_edge = parent.clone();
    let mut depth = vec![0; n + 1];
    let mut parity = vec![false; n + 1];
    let mut queue = VecDeque::with_capacity(n + 1);
    let mut cycle = Vec::with_capacity(n + 1);
    let mut trial_cycle = Vec::with_capacity(n + 1);
    let limit = if fractional {
        edges.len().saturating_mul(4).min(8192)
    } else if strategy == PackingStrategy::ExtendedGreedy {
        edges.len().min(4096)
    } else {
        512
    };
    while cycles < limit && start.elapsed() < budget {
        if (baseline + gain) as f64 / denominator >= cutoff {
            break;
        }
        parent.fill(usize::MAX);
        depth.fill(0);
        parity.fill(false);
        queue.clear();
        let mut found = None;
        let mut candidates = 0;
        let mut best_weight = 0;
        cycle.clear();
        'forest: for root_index in 0..=n {
            let root = (root_index + first_root + if fractional { cycles } else { 0 }) % (n + 1);
            if parent[root] != usize::MAX {
                continue;
            }
            parent[root] = root;
            queue.push_back(root);
            while let Some(u) = queue.pop_front() {
                let neighbors = &adjacency[offsets[u]..offsets[u + 1]];
                for k in 0..neighbors.len() {
                    let index = if fractional {
                        (k + cycles) % neighbors.len()
                    } else {
                        k
                    };
                    let (v, e) = neighbors[index];
                    if capacity[e] == 0 {
                        continue;
                    }
                    let opposite = edges[e].1 > 0;
                    if parent[v] == usize::MAX {
                        parent[v] = u;
                        parent_edge[v] = e;
                        depth[v] = depth[u] + 1;
                        parity[v] = parity[u] ^ opposite;
                        queue.push_back(v);
                    } else if parity[u] ^ parity[v] != opposite {
                        found = Some((u, v, e));
                        if matches!(
                            strategy,
                            PackingStrategy::Greedy
                                | PackingStrategy::Portfolio
                                | PackingStrategy::ShortPortfolio
                                | PackingStrategy::ExtendedGreedy
                                | PackingStrategy::FractionalPortfolio
                        ) {
                            break 'forest;
                        }
                        // Score a bounded set of simple fundamental cycles.
                        extract_cycle(u, v, e, &parent, &parent_edge, &depth, &mut trial_cycle);
                        let weight = trial_cycle.iter().map(|&e| capacity[e]).min().unwrap();
                        let better = cycle.is_empty()
                            || match strategy {
                                PackingStrategy::ShortCycles => {
                                    trial_cycle.len() < cycle.len()
                                        || (trial_cycle.len() == cycle.len()
                                            && weight > best_weight)
                                }
                                PackingStrategy::WeightedCycles => {
                                    weight * cycle.len() as i128
                                        > best_weight * trial_cycle.len() as i128
                                }
                                _ => unreachable!(),
                            };
                        if better {
                            cycle.clone_from(&trial_cycle);
                            best_weight = weight;
                        }
                        candidates += 1;
                        if candidates >= 24 || start.elapsed() >= budget {
                            break 'forest;
                        }
                    }
                }
            }
        }
        let Some((u, v, e)) = found else {
            break;
        };
        if cycle.is_empty() {
            extract_cycle(u, v, e, &parent, &parent_edge, &depth, &mut cycle);
        }
        let minimum = cycle.iter().map(|&e| capacity[e]).min().unwrap();
        // Exact half-unit capacities allow overlapping cycles to share edges.
        // Always debit the full chosen weight from every participating edge.
        let weight = if fractional {
            // Half of the original capacity, not half of an ever-shrinking
            // remainder: avoid many tiny updates on uniform-weight graphs.
            minimum.min(cycle.iter().map(|&e| edges[e].1.abs()).min().unwrap())
        } else {
            minimum
        };
        if let Some(trace) = trace.as_deref_mut() {
            trace.cycles.push((cycle.clone(), weight));
        }
        // Every frustrated simple cycle pays at least twice its minimum |J|.
        // Debit every participating edge so overlapping cycles cannot overcharge.
        for &e in &cycle {
            capacity[e] -= weight;
        }
        gain += 2 * weight;
        cycles += 1;
    }
    // This cap makes the final integer-to-float conversion exact as well.
    if baseline.abs().max((baseline + gain).abs()) > (1i128 << 52) {
        return None;
    }
    if let Some(trace) = trace {
        trace.residual = capacity;
    }
    Some(Bound {
        lower: (baseline + gain) as f64 / denominator,
        baseline: baseline as f64 / denominator,
        cycles,
    })
}

fn extract_cycle(
    mut u: usize,
    mut v: usize,
    e: usize,
    parent: &[usize],
    parent_edge: &[usize],
    depth: &[usize],
    cycle: &mut Vec<usize>,
) {
    cycle.clear();
    cycle.push(e);
    while u != v {
        if depth[u] < depth[v] {
            std::mem::swap(&mut u, &mut v);
        }
        cycle.push(parent_edge[u]);
        u = parent[u];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    #[test]
    fn overlapping_cycles_can_share_capacity() {
        assert_eq!(
            PackingStrategy::default(),
            PackingStrategy::FractionalPortfolio
        );
        let mut matrix = sprs::TriMat::new((4, 4));
        for i in 0..4 {
            for j in 0..4 {
                if i != j {
                    matrix.add_triplet(i, j, 2.0);
                }
            }
        }
        let q = Qubo::new_with_c(matrix.to_csr(), Array1::from_elem(4, -3.0));
        let prepared = PreparedCycleBound::new(&q).unwrap();
        let free = vec![0, 1, 2, 3];
        let fixed = FixedVarMap::default();
        let ordinary = prepared
            .conditioned_with_strategy(
                &free,
                &fixed,
                Duration::from_secs(1),
                f64::INFINITY,
                PackingStrategy::ExtendedGreedy,
            )
            .unwrap();
        let shared = prepared
            .conditioned_with_strategy(
                &free,
                &fixed,
                Duration::from_secs(1),
                f64::INFINITY,
                PackingStrategy::FractionalPortfolio,
            )
            .unwrap();
        assert!(
            shared.lower > ordinary.lower,
            "{shared:?} versus {ordinary:?}"
        );
        for mask in 0..16 {
            let x = Array1::from_iter((0..4).map(|i| (mask >> i) & 1));
            assert!(shared.lower <= q.eval_usize(&x));
        }
    }
    #[test]
    fn extended_budget_finishes_more_than_512_independent_certificates() {
        let n = 1800;
        let mut matrix = sprs::TriMat::new((n, n));
        for first in (0..n).step_by(3) {
            for (i, j) in [
                (first, first + 1),
                (first, first + 2),
                (first + 1, first + 2),
            ] {
                matrix.add_triplet(i, j, 8.0);
            }
        }
        let q = Qubo::new_with_c(matrix.to_csr(), Array1::from_elem(n, -4.0));
        let prepared = PreparedCycleBound::new(&q).unwrap();
        let free: Vec<_> = (0..n).collect();
        let fixed = FixedVarMap::default();
        let baseline = prepared
            .conditioned_with_strategy(
                &free,
                &fixed,
                Duration::from_secs(10),
                f64::INFINITY,
                PackingStrategy::Greedy,
            )
            .unwrap();
        let extended = prepared
            .conditioned_with_strategy(
                &free,
                &fixed,
                Duration::from_secs(10),
                f64::INFINITY,
                PackingStrategy::ExtendedGreedy,
            )
            .unwrap();
        assert_eq!(baseline.cycles, 512);
        assert_eq!(extended.cycles, 600);
        assert_eq!(extended.lower, -2400.0);
        assert!(extended.lower > baseline.lower);
        let x = Array1::from_iter((0..n).map(|i| usize::from(i % 3 == 0)));
        assert_eq!(q.eval_usize(&x), extended.lower);
    }
    #[test]
    fn conditioned_certificate_covers_every_fixing_and_free_order() {
        let n = 5;
        for seed in 0..12 {
            let mut matrix = sprs::TriMat::new((n, n));
            for i in 0..n {
                for j in 0..n {
                    matrix.add_triplet(
                        i,
                        j,
                        (((seed * 7 + i * 13 + j * 3) % 11) as f64 - 5.0) / 8.0,
                    );
                }
            }
            let q = Qubo::new_with_c(
                matrix.to_csc(),
                Array1::from_iter((0..n).map(|i| (i as f64 - 2.0) / 8.0)),
            );
            let prepared = PreparedCycleBound::new(&q).unwrap();
            for ternary in 0..3usize.pow(n as u32) {
                let mut code = ternary;
                let mut fixed = FixedVarMap::default();
                let mut free = Vec::new();
                for i in 0..n {
                    match code % 3 {
                        0 => free.push(i),
                        v => {
                            fixed.insert(i, v - 1);
                        }
                    }
                    code /= 3;
                }
                free.reverse();
                let b = prepared
                    .conditioned_with_strategy(
                        &free,
                        &fixed,
                        Duration::from_secs(1),
                        f64::INFINITY,
                        match seed % 2 {
                            0 => PackingStrategy::FractionalPortfolio,
                            _ => PackingStrategy::ExtendedGreedy,
                        },
                    )
                    .unwrap();
                for mask in 0..1usize << free.len() {
                    let mut x = Array1::zeros(n);
                    for (&i, &v) in &fixed {
                        x[i] = v;
                    }
                    for (k, &i) in free.iter().enumerate() {
                        x[i] = (mask >> k) & 1;
                    }
                    assert!(b.lower <= q.eval_usize(&x));
                    if free.is_empty() {
                        assert_eq!(b.lower, q.eval_usize(&x));
                    }
                }
            }
        }
    }
    #[test]
    fn exhaustive_signed_graphs_with_fields_and_diagonals() {
        for seed in 0..100usize {
            let n = 7;
            let mut m = sprs::TriMat::new((n, n));
            for i in 0..n {
                for j in i..n {
                    let w = ((seed * 37 + i * 13 + j * 19 + i * j * 7) % 9) as f64 - 4.0;
                    m.add_triplet(i, j, w);
                }
            }
            let q = Qubo::new_with_c(
                m.to_csr(),
                Array1::from_iter((0..n).map(|i| ((seed + i * 3) % 7) as f64 - 3.0)),
            );
            let prepared = PreparedCycleBound::new(&q).unwrap();
            let b = prepared
                .conditioned_with_strategy(
                    &(0..n).collect::<Vec<_>>(),
                    &FixedVarMap::default(),
                    Duration::from_secs(1),
                    f64::INFINITY,
                    PackingStrategy::Greedy,
                )
                .unwrap();
            assert!(b.lower >= b.baseline);
            for strategy in [
                PackingStrategy::Greedy,
                PackingStrategy::ShortCycles,
                PackingStrategy::WeightedCycles,
                PackingStrategy::Portfolio,
                PackingStrategy::ShortPortfolio,
                PackingStrategy::ExtendedGreedy,
                PackingStrategy::FractionalPortfolio,
            ] {
                let trial = prepared
                    .conditioned_with_strategy(
                        &(0..n).collect::<Vec<_>>(),
                        &FixedVarMap::default(),
                        Duration::from_secs(1),
                        f64::INFINITY,
                        strategy,
                    )
                    .unwrap();
                if matches!(
                    strategy,
                    PackingStrategy::Portfolio
                        | PackingStrategy::ShortPortfolio
                        | PackingStrategy::ExtendedGreedy
                ) {
                    assert!(trial.lower >= b.lower);
                }
                for mask in 0..1usize << n {
                    let x = Array1::from_iter((0..n).map(|i| (mask >> i) & 1));
                    assert!(trial.lower <= q.eval_usize(&x));
                }
            }
        }
    }
    #[test]
    fn frustrated_triangle_is_exact_and_unsupported_values_decline() {
        let q = Qubo::from_vec(vec![0, 0, 1], vec![1, 2, 2], vec![8.0; 3], vec![-4.0; 3], 3);
        let b = bound(&q, Duration::from_secs(1)).unwrap();
        assert_eq!(b.lower, -4.0);
        assert_eq!(b.cycles, 1);
        let mut unsupported = q.clone();
        unsupported.c[0] = 0.1;
        assert!(bound(&unsupported, Duration::from_secs(1)).is_none());
        let zero = bound(&q, Duration::ZERO).unwrap();
        assert_eq!(zero.lower, zero.baseline);
    }
}
