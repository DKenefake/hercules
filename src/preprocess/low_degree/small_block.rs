//! Exact elimination through small vertex separators. Removing one vertex and
//! finding articulations exposes two-vertex separators without testing all pairs.
use super::*;

const MAX_INTERIOR: usize = 8;

impl Graph {
    // Iterative Tarjan traversal: large sparse instances must not exhaust the
    // call stack. An optional removed vertex supplies the first separator.
    fn articulations(
        &self,
        removed: Option<usize>,
        start: Instant,
        budget: Duration,
    ) -> Option<Vec<usize>> {
        let n = self.active.len();
        let mut order = vec![usize::MAX; n];
        let mut low = vec![0; n];
        let mut parent = vec![usize::MAX; n];
        let mut children = vec![0; n];
        let mut cut = vec![false; n];
        let mut clock = 0;
        for root in 0..n {
            if !self.active[root] || removed == Some(root) || order[root] != usize::MAX {
                continue;
            }
            order[root] = clock;
            low[root] = clock;
            clock += 1;
            let mut stack = vec![(root, self.edges[root].keys())];
            while let Some((i, neighbors)) = stack.last_mut() {
                if start.elapsed() >= budget {
                    return None;
                }
                let v = *i;
                if let Some(&w) = neighbors.next() {
                    if removed == Some(w) {
                        continue;
                    }
                    if order[w] == usize::MAX {
                        parent[w] = v;
                        children[v] += 1;
                        order[w] = clock;
                        low[w] = clock;
                        clock += 1;
                        stack.push((w, self.edges[w].keys()));
                    } else if w != parent[v] {
                        low[v] = low[v].min(order[w]);
                    }
                } else {
                    stack.pop();
                    let p = parent[v];
                    if p == usize::MAX {
                        cut[v] = children[v] > 1;
                    } else {
                        low[p] = low[p].min(low[v]);
                        if parent[p] != usize::MAX && low[v] >= order[p] {
                            cut[p] = true;
                        }
                    }
                }
            }
        }
        Some((0..n).filter(|&i| cut[i]).collect())
    }

    pub(super) fn eliminate_small_block(
        &mut self,
        start: Instant,
        budget: Duration,
    ) -> Option<Vec<Elimination>> {
        let active: Vec<_> = (0..self.active.len()).filter(|&i| self.active[i]).collect();
        // First try single articulations; only then spend work on pairs.
        for removed in std::iter::once(None).chain(active.iter().copied().map(Some)) {
            for cut in self.articulations(removed, start, budget)? {
                let mut seen = vec![false; self.active.len()];
                seen[cut] = true;
                if let Some(v) = removed {
                    seen[v] = true;
                }
                for &seed in &active {
                    if seen[seed] {
                        continue;
                    }
                    let mut block = vec![seed];
                    seen[seed] = true;
                    let mut head = 0;
                    while head < block.len() {
                        if start.elapsed() >= budget {
                            return None;
                        }
                        for &w in self.edges[block[head]].keys() {
                            if !seen[w] {
                                seen[w] = true;
                                block.push(w);
                            }
                        }
                        head += 1;
                    }
                    if (2..=MAX_INTERIOR).contains(&block.len()) {
                        if let Some(steps) = self.eliminate_block(&block, start, budget) {
                            return Some(steps);
                        }
                    }
                }
            }
        }
        None
    }

    fn eliminate_block(
        &mut self,
        interior: &[usize],
        start: Instant,
        budget: Duration,
    ) -> Option<Vec<Elimination>> {
        if interior.is_empty() || interior.len() > MAX_INTERIOR {
            return None;
        }
        let mut index = vec![usize::MAX; self.active.len()];
        for (bit, &i) in interior.iter().enumerate() {
            if !self.active[i] || index[i] != usize::MAX {
                return None;
            }
            index[i] = bit;
        }
        let mut boundary = Vec::new();
        for &i in interior {
            for &j in self.edges[i].keys() {
                if index[j] == usize::MAX && !boundary.contains(&j) {
                    boundary.push(j);
                    if boundary.len() > 2 {
                        return None;
                    }
                }
            }
        }
        boundary.sort_unstable();
        let mut table = [f64::INFINITY; 4];
        let mut minimizers = [0; 4];
        for b in 0..1 << boundary.len() {
            for mask in 0..1 << interior.len() {
                if start.elapsed() >= budget {
                    return None;
                }
                let mut cost = 0.0;
                for (bit, &i) in interior.iter().enumerate() {
                    if mask & (1 << bit) == 0 {
                        continue;
                    }
                    cost = add(cost, self.linear[i])?;
                    for (&j, &weight) in &self.edges[i] {
                        let jbit = index[j];
                        let include = if jbit != usize::MAX {
                            jbit > bit && mask & (1 << jbit) != 0
                        } else {
                            b & (1 << boundary.iter().position(|&v| v == j)?) != 0
                        };
                        if include {
                            cost = add(cost, weight)?;
                        }
                    }
                }
                if cost < table[b] {
                    table[b] = cost;
                    minimizers[b] = mask;
                }
            }
        }
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
                choices[b] = (minimizers[b] >> bit) & 1;
            }
            steps.push(Elimination {
                variable: i,
                neighbors: boundary.clone(),
                choices,
            });
        }
        Some(steps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(boundaries: usize, seed: usize) -> Qubo {
        // A dense K5 pocket, attached through one or two vertices to a K6 core.
        // No low-degree vertex exists in the original graph.
        let mut q = TriMat::new((11, 11));
        for group in [0..5, 5..11] {
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
            q.to_csc(),
            Array1::from_iter((0..11).map(|i| ((i + seed) % 7) as f64 - 3.0)),
        )
    }

    #[test]
    fn every_boundary_case_matches_exhaustive_minimum_and_reconstructs() {
        for boundary_count in 0..=2 {
            for seed in 0..16 {
                let qubo = fixture(boundary_count, seed);
                let mut graph = Graph::new(&qubo).unwrap();
                let steps = graph
                    .eliminate_block(&[0, 1, 2, 3, 4], Instant::now(), Duration::from_secs(2))
                    .unwrap();
                let (small, remaining) = graph.compact();
                let r = RootReduction {
                    qubo: small,
                    constant: graph.constant,
                    remaining,
                    fixed: FixedVarMap::default(),
                    statistics: RootReductionStatistics::default(),
                    original_size: 11,
                    steps,
                };
                for mask in 0..64 {
                    let x = Array1::from_iter((0..6).map(|i| (mask >> i) & 1));
                    let full = r.reconstruct(&x);
                    let expected = (0..32)
                        .map(|inside| {
                            let mut trial = full.clone();
                            for i in 0..5 {
                                trial[i] = (inside >> i) & 1;
                            }
                            qubo.eval_usize(&trial)
                        })
                        .fold(f64::INFINITY, f64::min);
                    assert_eq!(r.qubo.eval_usize(&x) + r.constant, expected);
                    assert_eq!(qubo.eval_usize(&full), expected);
                }
            }
        }
    }

    #[test]
    fn separator_finds_dense_pockets_and_handles_vertex_order() {
        for boundaries in 1..=2 {
            let q = fixture(boundaries, 3);
            let mut graph = Graph::new(&q).unwrap();
            assert!(graph.edges.iter().all(|row| row.len() >= 4));
            let steps = graph
                .eliminate_small_block(Instant::now(), Duration::from_secs(2))
                .unwrap();
            assert!((2..=MAX_INTERIOR).contains(&steps.len()));
            assert!(steps.iter().all(|s| s.neighbors.len() <= 2));
        }
    }

    #[test]
    fn failed_block_attempts_are_atomic() {
        let q = fixture(2, 0);
        let mut graph = Graph::new(&q).unwrap();
        let edges = graph.edges.clone();
        assert!(graph
            .eliminate_block(&[0, 1, 2, 3, 4], Instant::now(), Duration::ZERO)
            .is_none());
        assert_eq!(graph.edges, edges);
        assert!(graph
            .eliminate_block(&[0, 1], Instant::now(), Duration::from_secs(1))
            .is_none());
        graph.linear[0] = 1e20;
        graph.linear[1] = 1.0;
        let linear = graph.linear.clone();
        assert!(graph
            .eliminate_block(&[0, 1, 2, 3, 4], Instant::now(), Duration::from_secs(1))
            .is_none());
        assert_eq!(graph.edges, edges);
        assert_eq!(graph.linear, linear);
        assert!(graph.active.iter().all(|&a| a));
        assert_eq!(graph.constant, 0.0);
    }
}
