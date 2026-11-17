//! Discover narrow pockets during a bounded fill traversal, not separator enumeration.
use super::*;
use crate::preprocess::width_elimination::{solve, MAX_FREE};
use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, Default)]
pub struct RegionPersistencyStatistics {
    pub attempts: usize,
    pub completed: usize,
    pub inner_solves: usize,
    pub fixed: usize,
    pub seconds: f64,
}

impl Graph {
    fn commit_block(
        &mut self,
        interior: &[usize],
        boundary: &[usize],
        table: &[f64; 4],
        minimizers: &[u128; 4],
        start: Instant,
        budget: Duration,
    ) -> Option<Vec<Elimination>> {
        // Compute every update before mutating the graph: a failed exactness
        // or deadline check must leave the current problem unchanged.
        let constant = add(self.constant, table[0])?;
        let mut linear = Vec::new();
        for (bit, &i) in boundary.iter().enumerate() {
            linear.push(add(self.linear[i], add(table[1 << bit], -table[0])?)?);
        }
        let weight = if boundary.len() == 2 {
            let delta = add(add(add(table[3], -table[1])?, -table[2])?, table[0])?;
            Some(add(
                self.edges[boundary[0]]
                    .get(&boundary[1])
                    .copied()
                    .unwrap_or(0.0),
                delta,
            )?)
        } else {
            None
        };
        if start.elapsed() >= budget {
            return None;
        }
        self.constant = constant;
        for (&i, value) in boundary.iter().zip(linear) {
            self.linear[i] = value;
        }
        if let Some(w) = weight {
            let (a, b) = (boundary[0], boundary[1]);
            if w == 0.0 {
                self.edges[a].remove(&b);
                self.edges[b].remove(&a);
            } else {
                self.edges[a].insert(b, w);
                self.edges[b].insert(a, w);
            }
        }
        let mut steps = Vec::new();
        for (bit, &i) in interior.iter().enumerate() {
            for (j, _) in std::mem::take(&mut self.edges[i]) {
                self.edges[j].remove(&i);
            }
            self.active[i] = false;
            let mut choices = [0; 8];
            for b in 0..1 << boundary.len() {
                choices[b] = ((minimizers[b] >> bit) & 1) as usize;
            }
            steps.push(Elimination {
                variable: i,
                neighbors: boundary.to_vec(),
                choices,
            });
        }
        Some(steps)
    }

    pub(super) fn eliminate_width_region(
        &mut self,
        start: Instant,
        budget: Duration,
    ) -> Option<Vec<Elimination>> {
        self.visit_width_regions(2, start, budget, |graph, interior, boundary| {
            graph.eliminate_width_block(interior, boundary, start, budget)
        })
    }

    pub(super) fn region_persistencies(
        &mut self,
        start: Instant,
        budget: Duration,
        stats: &mut RegionPersistencyStatistics,
    ) -> Option<Vec<(usize, usize)>> {
        self.visit_width_regions(4, start, budget, |graph, interior, boundary| {
            if boundary.len() < 3 || interior.len() > 64 || stats.attempts >= 32 {
                return None;
            }
            stats.attempts += 1;
            graph
                .common_width_fixings(interior, boundary, start, budget, stats)
                .filter(|fixes| !fixes.is_empty())
        })
    }

    fn visit_width_regions<T>(
        &mut self,
        max_boundary: usize,
        start: Instant,
        budget: Duration,
        mut evaluate: impl FnMut(&mut Self, &[usize], &[usize]) -> Option<T>,
    ) -> Option<T> {
        // Bound discovery as well as the exact tables. Wide cores are never solved here.
        if self.active.len() > 4096 || start.elapsed() >= budget {
            return None;
        }
        let mut edges: Vec<BTreeSet<usize>> = self
            .edges
            .iter()
            .map(|row| row.keys().copied().collect())
            .collect();
        let mut active = self.active.clone();
        let mut owner: Vec<_> = (0..active.len()).collect();
        let mut regions: Vec<Vec<usize>> = (0..active.len()).map(|_| Vec::new()).collect();
        for _ in 0..512.min(active.len()) {
            if start.elapsed() >= budget {
                return None;
            }
            let v = (0..active.len())
                .filter(|&i| active[i])
                .min_by_key(|&i| (edges[i].len(), i))?;
            if edges[v].len() > 8 {
                return None;
            }
            // Merge connected eliminated prefixes using original edges only.
            // Filled edges describe elimination cost, not actual region boundaries.
            let mut roots = BTreeSet::new();
            for &j in self.edges[v].keys() {
                if !active[j] {
                    roots.insert(owner[j]);
                }
            }
            regions[v].push(v);
            for root in roots {
                for i in std::mem::take(&mut regions[root]) {
                    owner[i] = v;
                    regions[v].push(i);
                }
            }
            active[v] = false;
            let neighbors: Vec<_> = edges[v].iter().copied().collect();
            for &i in &neighbors {
                edges[i].remove(&v);
                for &j in &neighbors {
                    if i != j {
                        edges[i].insert(j);
                    }
                }
            }
            if regions[v].len() < 3 || regions[v].len() > MAX_FREE {
                continue;
            }
            let mut boundary = BTreeSet::new();
            for &i in &regions[v] {
                for &j in self.edges[i].keys() {
                    if active[j] {
                        boundary.insert(j);
                    }
                }
                if boundary.len() > max_boundary {
                    break;
                }
            }
            if boundary.len() <= max_boundary {
                let boundary: Vec<_> = boundary.into_iter().collect();
                if let Some(result) = evaluate(self, &regions[v], &boundary) {
                    return Some(result);
                }
            }
        }
        None
    }

    fn width_block_qubo(
        &self,
        interior: &[usize],
        boundary: &[usize],
    ) -> Option<(Qubo, Vec<usize>)> {
        let n = interior.len();
        if n == 0 || n > MAX_FREE || boundary.len() > 4 {
            return None;
        }
        let mut index = vec![usize::MAX; self.active.len()];
        for (i, &v) in interior.iter().chain(boundary).enumerate() {
            if v >= index.len() || !self.active[v] || index[v] != usize::MAX {
                return None;
            }
            index[v] = i;
        }
        let mut q = TriMat::new((n + boundary.len(), n + boundary.len()));
        let mut c = Array1::zeros(n + boundary.len());
        for (i, &v) in interior.iter().enumerate() {
            c[i] = self.linear[v];
            for (&w, &weight) in &self.edges[v] {
                let j = index[w];
                if j == usize::MAX {
                    return None;
                }
                if j > i {
                    q.add_triplet(i, j, weight);
                    q.add_triplet(j, i, weight);
                }
            }
        }
        Some((Qubo::new_with_c(q.to_csr(), c), index))
    }

    fn common_width_fixings(
        &self,
        interior: &[usize],
        boundary: &[usize],
        start: Instant,
        budget: Duration,
        stats: &mut RegionPersistencyStatistics,
    ) -> Option<Vec<(usize, usize)>> {
        let (qubo, _) = self.width_block_qubo(interior, boundary)?;
        let n = interior.len();
        let mut common: Option<Array1<usize>> = None;
        let mut agrees = vec![true; n];
        for b in 0..1 << boundary.len() {
            let fixed = (0..boundary.len()).map(|i| (n + i, (b >> i) & 1)).collect();
            let remaining = budget.checked_sub(start.elapsed())?;
            stats.inner_solves += 1;
            let x = solve(&qubo, &fixed, remaining).ok()?;
            if let Some(first) = &common {
                for i in 0..n {
                    agrees[i] &= first[i] == x[i];
                }
            } else {
                common = Some(x);
            }
            // No later assignment can restore agreement that was already lost.
            if !agrees.iter().any(|&v| v) {
                return Some(Vec::new());
            }
        }
        if start.elapsed() >= budget {
            return None;
        }
        stats.completed += 1;
        let first = common?;
        // A jointly chosen conditional minimizer exists for every boundary.
        // These are compatible weak fixings, not facts about every optimum.
        Some(
            interior
                .iter()
                .enumerate()
                .filter(|(i, _)| agrees[*i])
                .map(|(i, &v)| (v, first[i]))
                .collect(),
        )
    }

    fn eliminate_width_block(
        &mut self,
        interior: &[usize],
        boundary: &[usize],
        start: Instant,
        budget: Duration,
    ) -> Option<Vec<Elimination>> {
        if boundary.len() > 2 {
            return None;
        }
        let n = interior.len();
        let (qubo, index) = self.width_block_qubo(interior, boundary)?;
        let mut table = [0.0; 4];
        let mut minimizers = [0u128; 4];
        for b in 0..1 << boundary.len() {
            let fixed = (0..boundary.len()).map(|i| (n + i, (b >> i) & 1)).collect();
            let remaining = budget.checked_sub(start.elapsed())?;
            let x = solve(&qubo, &fixed, remaining).ok()?;
            // Exact evaluation of the block polynomial, excluding boundary-only terms.
            let mut value = 0.0;
            for (i, &v) in interior.iter().enumerate() {
                if x[i] == 0 {
                    continue;
                }
                minimizers[b] |= 1u128 << i;
                value = add(value, self.linear[v])?;
                for (&w, &weight) in &self.edges[v] {
                    let j = index[w];
                    if j > i && x[j] == 1 {
                        value = add(value, weight)?;
                    }
                }
            }
            table[b] = value;
        }
        self.commit_block(interior, boundary, &table, &minimizers, start, budget)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boundary_fixings_preserve_each_conditional_optimum_including_ties() {
        let mut certified = 0;
        for boundary_n in 3..=4 {
            for seed in 0..32 {
                let n = 5 + boundary_n;
                let mut q = TriMat::new((n, n));
                for i in 0..5 {
                    for j in i + 1..n {
                        q.add_triplet(i, j, 2.0 * (((i * 13 + j * 7 + seed) % 9) as f64 - 4.0));
                    }
                }
                let mut c = Array1::from_iter((0..n).map(|i| ((i * 5 + seed) % 11) as f64 - 5.0));
                if seed % 4 == 0 {
                    c[0] = -100.0;
                }
                let qubo = Qubo::new_with_c(q.to_csr(), c);
                let graph = Graph::new(&qubo).unwrap();
                let boundary: Vec<_> = (5..n).collect();
                let fixes = graph
                    .common_width_fixings(
                        &[0, 1, 2, 3, 4],
                        &boundary,
                        Instant::now(),
                        Duration::from_secs(2),
                        &mut Default::default(),
                    )
                    .unwrap();
                certified += fixes.len();
                for b in 0..1usize << boundary_n {
                    let mut all = f64::INFINITY;
                    let mut retained = f64::INFINITY;
                    for inside in 0..32 {
                        let x = Array1::from_iter((0..n).map(|i| {
                            if i < 5 {
                                (inside >> i) & 1
                            } else {
                                (b >> (i - 5)) & 1
                            }
                        }));
                        let value = qubo.eval_usize(&x);
                        all = all.min(value);
                        if fixes.iter().all(|&(i, v)| x[i] == v) {
                            retained = retained.min(value);
                        }
                    }
                    assert_eq!(all, retained, "seed={seed}, boundary={b}, fixes={fixes:?}");
                }
            }
        }
        assert!(certified > 0);
        let graph = Graph::new(&Qubo::new(sprs::CsMat::zero((8, 8)))).unwrap();
        let fixes = graph
            .common_width_fixings(
                &[0, 1, 2, 3, 4],
                &[5, 6, 7],
                Instant::now(),
                Duration::from_secs(1),
                &mut Default::default(),
            )
            .unwrap();
        assert_eq!(fixes.len(), 5); // Compatible weak choices, despite every point being optimal.
        assert!(graph
            .common_width_fixings(
                &[0, 1, 2, 3, 4],
                &[5, 6, 7],
                Instant::now(),
                Duration::ZERO,
                &mut Default::default()
            )
            .is_none());
    }

    #[test]
    fn wider_boundary_discovery_and_reconstruction_preserve_optimum() {
        let mut q = TriMat::new((15, 15));
        for group in [0..5, 5..15] {
            for i in group.clone() {
                for j in i + 1..group.end {
                    q.add_triplet(i, j, 2.0);
                }
            }
        }
        for i in 0..5 {
            for j in 5..8 {
                q.add_triplet(i, j, 2.0);
            }
        }
        let mut c = Array1::from_elem(15, -2.0);
        c[0] = -100.0;
        let qubo = Qubo::new_with_c(q.to_csr(), c);
        let mut options = SolverOptions::new();
        options.root_low_degree_elimination = false;
        options.root_dominant_edge_contraction = false;
        options.width_region_elimination = false;
        options.root_region_persistency = true;
        let reduction = reduce(
            &qubo,
            &FixedVarMap::default(),
            &options,
            Duration::from_secs(1),
            false,
        )
        .unwrap();
        assert!(reduction.statistics.region_persistency.fixed > 0);
        assert_eq!(reduction.fixed.get(&0), Some(&1));
        let expected = (0..1usize << 15)
            .map(|mask| qubo.eval_usize(&Array1::from_iter((0..15).map(|i| (mask >> i) & 1))))
            .fold(f64::INFINITY, f64::min);
        let mut found = f64::INFINITY;
        for mask in 0..1usize << reduction.remaining.len() {
            let x = Array1::from_iter((0..reduction.remaining.len()).map(|i| (mask >> i) & 1));
            let value = reduction.qubo.eval_usize(&x) + reduction.constant;
            assert_eq!(value, qubo.eval_usize(&reduction.reconstruct(&x)));
            found = found.min(value);
        }
        assert_eq!(expected, found);
    }

    fn fixture(boundaries: usize, seed: usize) -> Qubo {
        let mut q = TriMat::new((15, 15));
        for group in [0..5, 5..15] {
            for i in group.clone() {
                for j in i + 1..group.end {
                    q.add_triplet(i, j, 2.0 * (((i * 7 + j * 3 + seed) % 9) as f64 - 4.5));
                }
            }
        }
        for b in 0..boundaries {
            q.add_triplet(b, 5 + b, 6.0);
        }
        Qubo::new_with_c(
            q.to_csr(),
            Array1::from_iter((0..15).map(|i| ((i + seed) % 7) as f64 - 3.0)),
        )
    }

    #[test]
    fn wide_core_pockets_match_exhaustive_conditioned_optima() {
        for boundaries in 0..=2 {
            for seed in 0..16 {
                let q = fixture(boundaries, seed);
                let mut graph = Graph::new(&q).unwrap();
                let steps = graph
                    .eliminate_width_region(Instant::now(), Duration::from_secs(2))
                    .unwrap();
                assert!((3..=5).contains(&steps.len()));
                let eliminated: Vec<_> = steps.iter().map(|s| s.variable).collect();
                let (small, remaining) = graph.compact();
                let reduction = RootReduction {
                    qubo: small,
                    constant: graph.constant,
                    remaining,
                    fixed: FixedVarMap::default(),
                    statistics: RootReductionStatistics::default(),
                    original_size: 15,
                    steps,
                };
                for mask in 0..1usize << reduction.remaining.len() {
                    let x =
                        Array1::from_iter((0..reduction.remaining.len()).map(|i| (mask >> i) & 1));
                    let full = reduction.reconstruct(&x);
                    let expected = (0..1usize << eliminated.len())
                        .map(|bits| {
                            let mut trial = full.clone();
                            for (i, &v) in eliminated.iter().enumerate() {
                                trial[v] = (bits >> i) & 1;
                            }
                            q.eval_usize(&trial)
                        })
                        .fold(f64::INFINITY, f64::min);
                    assert_eq!(q.eval_usize(&full), expected);
                    assert_eq!(reduction.qubo.eval_usize(&x) + reduction.constant, expected);
                }
            }
        }
    }

    #[test]
    fn eliminates_more_than_eight_variables_and_preserves_each_boundary_case() {
        let mut q = TriMat::new((22, 22));
        for i in 0..12 {
            for offset in 1..=2 {
                q.add_triplet(i, (i + offset) % 12, 6.0);
            }
            q.add_triplet(i, 12, -2.0);
            q.add_triplet(i, 13, 4.0);
        }
        for i in 12..22 {
            for j in i + 1..22 {
                q.add_triplet(i, j, 2.0);
            }
        }
        let q = Qubo::new_with_c(q.to_csr(), Array1::from_elem(22, -4.0));
        let mut graph = Graph::new(&q).unwrap();
        let steps = graph
            .eliminate_width_region(Instant::now(), Duration::from_secs(2))
            .unwrap();
        assert_eq!(steps.len(), 12);
        let (small, remaining) = graph.compact();
        let r = RootReduction {
            qubo: small,
            constant: graph.constant,
            remaining,
            fixed: FixedVarMap::default(),
            statistics: RootReductionStatistics::default(),
            original_size: 22,
            steps,
        };
        for b in 0..4 {
            let mut x = Array1::zeros(10);
            x[0] = b & 1;
            x[1] = (b >> 1) & 1;
            let full = r.reconstruct(&x);
            let expected = (0..4096)
                .map(|mask| {
                    let mut trial = full.clone();
                    for i in 0..12 {
                        trial[i] = (mask >> i) & 1;
                    }
                    q.eval_usize(&trial)
                })
                .fold(f64::INFINITY, f64::min);
            assert_eq!(q.eval_usize(&full), expected);
            assert_eq!(r.qubo.eval_usize(&x) + r.constant, expected);
        }
    }

    #[test]
    fn deadline_and_inexact_update_are_atomic() {
        let q = fixture(2, 0);
        let mut graph = Graph::new(&q).unwrap();
        assert!(graph
            .eliminate_width_region(Instant::now(), Duration::ZERO)
            .is_none());
        graph.constant = 1e20;
        let linear = graph.linear.clone();
        let edges = graph.edges.clone();
        let active = graph.active.clone();
        assert!(graph
            .eliminate_width_block(
                &[0, 1, 2, 3, 4],
                &[5, 6],
                Instant::now(),
                Duration::from_secs(2)
            )
            .is_none());
        assert_eq!(graph.constant, 1e20);
        assert_eq!(graph.linear, linear);
        assert_eq!(graph.edges, edges);
        assert_eq!(graph.active, active);
    }
}
