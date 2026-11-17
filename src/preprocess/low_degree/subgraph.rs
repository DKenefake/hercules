//! Small induced-subgraph certificates: Lange et al., Max-Cut Theorem 3.
use super::*;

impl Graph {
    fn subgraph_relation(&self, vertices: &[usize]) -> Option<(usize, usize, bool)> {
        let k = vertices.len();
        if !(3..=6).contains(&k) {
            return None;
        }
        let mut weights = [[0.0; 6]; 6];
        let mut outside = [0.0; 6];
        for (a, &i) in vertices.iter().enumerate() {
            // 2*(f-constant) = -sum b_ij*(x_i XOR x_j), including anchor edges.
            let twice = 2.0 * self.linear[i];
            if !twice.is_finite() || twice * 0.5 != self.linear[i] {
                return None;
            }
            let mut anchor = -twice;
            for (&j, &w) in &self.edges[i] {
                anchor = add(anchor, -w)?;
                if let Some(b) = vertices.iter().position(|&v| v == j) {
                    weights[a][b] = -w;
                } else {
                    outside[a] = add(outside[a], w.abs())?;
                }
            }
            outside[a] = add(outside[a], anchor.abs())?;
        }
        // The internal optimum only proposes a switching; the test below proves it.
        let cost = |mask: usize| -> Option<f64> {
            let mut value = 0.0;
            for a in 0..k {
                for b in a + 1..k {
                    if ((mask >> a) ^ (mask >> b)) & 1 != 0 {
                        value = add(value, weights[a][b])?;
                    }
                }
            }
            Some(value)
        };
        let mut gauge = 0;
        let mut best = 0.0;
        for mask in (0..1usize << k).step_by(2) {
            let value = cost(mask)?;
            if value < best {
                best = value;
                gauge = mask;
            }
        }
        let mut valid = [[true; 6]; 6];
        for mask in (2..1usize << k).step_by(2) {
            let mut internal = 0.0;
            let mut external = [0.0; 2];
            for a in 0..k {
                let side = (mask >> a) & 1;
                external[side] = add(external[side], outside[a])?;
                for b in a + 1..k {
                    if side != (mask >> b) & 1 {
                        let w = weights[a][b];
                        let switched = if ((gauge >> a) ^ (gauge >> b)) & 1 != 0 {
                            -w
                        } else {
                            w
                        };
                        internal = add(internal, switched)?;
                    }
                }
            }
            if internal < external[0].min(external[1]) {
                for a in 0..k {
                    for b in a + 1..k {
                        if ((mask >> a) ^ (mask >> b)) & 1 != 0 {
                            valid[a][b] = false;
                        }
                    }
                }
            }
        }
        for a in 0..k {
            for b in a + 1..k {
                if valid[a][b] && weights[a][b] != 0.0 {
                    return Some((
                        vertices[a],
                        vertices[b],
                        ((gauge >> a) ^ (gauge >> b)) & 1 != 0,
                    ));
                }
            }
        }
        None
    }

    pub(super) fn small_subgraph_relation(
        &self,
        deadline: Instant,
    ) -> Option<(usize, usize, bool)> {
        let mut seeds: Vec<_> = self
            .edges
            .iter()
            .enumerate()
            .filter(|(i, _)| self.active[*i])
            .flat_map(|(i, row)| {
                row.iter()
                    .filter(move |(j, _)| i < **j)
                    .map(move |(&j, &w)| (i, j, w.abs()))
            })
            .collect();
        let compare = |a: &(usize, usize, f64), b: &(usize, usize, f64)| {
            b.2.total_cmp(&a.2).then((a.0, a.1).cmp(&(b.0, b.1)))
        };
        if seeds.len() > 16 {
            seeds.select_nth_unstable_by(16, compare);
            seeds.truncate(16);
        }
        seeds.sort_unstable_by(compare);
        let mut scores = vec![0.0; self.active.len()];
        let mut touched = Vec::with_capacity(self.active.len());
        let mut seen = vec![false; self.active.len()];
        for (i, j, _) in seeds {
            scores.fill(0.0);
            seen.fill(false);
            touched.clear();
            let mut vertices = [0; 6];
            vertices[0] = i;
            vertices[1] = j;
            let mut size = 2;
            let mut scored = 0;
            while size < 6 {
                if Instant::now() >= deadline {
                    return None;
                }
                // Add each row once as the region grows, in the same summation
                // order as the previous map-based implementation.
                for &v in &vertices[scored..size] {
                    for (&u, &w) in &self.edges[v] {
                        if !vertices[..size].contains(&u) {
                            if !seen[u] {
                                seen[u] = true;
                                touched.push(u);
                            }
                            scores[u] += w.abs();
                        }
                    }
                }
                scored = size;
                let Some(next) = touched
                    .iter()
                    .copied()
                    .filter(|u| !vertices[..size].contains(u))
                    .max_by(|&a, &b| scores[a].total_cmp(&scores[b]).then(b.cmp(&a)))
                else {
                    break;
                };
                vertices[size] = next;
                size += 1;
                if let Some(relation) = self.subgraph_relation(&vertices[..size]) {
                    return (Instant::now() < deadline).then_some(relation);
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn reference_discovery(graph: &Graph) -> Option<(usize, usize, bool)> {
        let mut seeds = Vec::new();
        for (i, row) in graph.edges.iter().enumerate() {
            if graph.active[i] {
                seeds.extend(
                    row.iter()
                        .filter(|(j, _)| i < **j)
                        .map(|(&j, &w)| (i, j, w.abs())),
                );
            }
        }
        seeds.sort_unstable_by(|a, b| b.2.total_cmp(&a.2).then((a.0, a.1).cmp(&(b.0, b.1))));
        for (i, j, _) in seeds.into_iter().take(16) {
            let mut vertices = vec![i, j];
            while vertices.len() < 6 {
                let mut scores = BTreeMap::<usize, f64>::new();
                for &v in &vertices {
                    for (&u, &w) in &graph.edges[v] {
                        if !vertices.contains(&u) {
                            *scores.entry(u).or_default() += w.abs();
                        }
                    }
                }
                let Some((next, _)) = scores
                    .into_iter()
                    .max_by(|a, b| a.1.total_cmp(&b.1).then(b.0.cmp(&a.0)))
                else {
                    break;
                };
                vertices.push(next);
                if let Some(r) = graph.subgraph_relation(&vertices) {
                    return Some(r);
                }
            }
        }
        None
    }
    fn minimum(q: &Qubo) -> f64 {
        (0..1usize << q.num_x())
            .map(|m| q.eval_usize(&Array1::from_iter((0..q.num_x()).map(|i| (m >> i) & 1))))
            .fold(f64::INFINITY, f64::min)
    }
    #[test]
    fn collective_certificate_beats_single_edge_dominance() {
        let mut q = TriMat::new((4, 4));
        for i in 0..4 {
            for j in i + 1..4 {
                q.add_triplet(i, j, -2.0);
            }
        }
        let qubo = Qubo::new_with_c(q.to_csr(), Array1::from_elem(4, 0.5));
        let graph = Graph::new(&qubo).unwrap();
        assert!((0..4).all(|i| graph.dominant_neighbor(i).is_none()));
        assert!(graph.subgraph_relation(&[0, 1, 2, 3]).is_some());
        let mut options = SolverOptions::new();
        options.root_low_degree_elimination = false;
        options.root_dominant_edge_contraction = false;
        options.width_region_elimination = false;
        options.subgraph_persistency = true;
        let reduced = reduce(
            &qubo,
            &FixedVarMap::default(),
            &options,
            Duration::from_secs(1),
            false,
        )
        .unwrap();
        assert!(reduced.statistics.subgraph_contracted > 0);
        assert_eq!(minimum(&reduced.qubo) + reduced.constant, minimum(&qubo));
    }

    #[test]
    fn certificates_preserve_conditional_optima_and_reconstruction() {
        let mut accepted = 0;
        for seed in 0..160usize {
            let n = 7;
            let mut q = TriMat::new((n, n));
            let mut c = Array1::zeros(n);
            for i in 0..n {
                c[i] = ((seed * 17 + i * 11) % 9) as f64 - 4.0;
                for j in i + 1..n {
                    let w = ((seed * 13 + i * 7 + j * 19 + i * j * 3) % 13) as f64 - 6.0;
                    q.add_triplet(i, j, 2.0 * w);
                }
            }
            let qubo = Qubo::new_with_c(q.to_csr(), c);
            for fixed in [None, Some(0), Some(1)] {
                let mut graph = Graph::new(&qubo).unwrap();
                if let Some(v) = fixed {
                    graph.fix(6, v).unwrap();
                }
                let original = graph.compact().0;
                let optimum = minimum(&original) + graph.constant;
                for vertices in [&[0, 1, 2][..], &[0, 1, 2, 3][..], &[0, 1, 2, 3, 4, 5][..]] {
                    if let Some((i, j, complement)) = graph.subgraph_relation(vertices) {
                        accepted += 1;
                        let mut reduced = Graph::new(&qubo).unwrap();
                        if let Some(v) = fixed {
                            reduced.fix(6, v).unwrap();
                        }
                        let (step, _) = reduced.contract_parity(i, j, complement).unwrap();
                        let (small, map) = reduced.compact();
                        assert_eq!(minimum(&small) + reduced.constant, optimum);
                        for m in 0..1usize << map.len() {
                            let x = Array1::from_iter((0..map.len()).map(|b| (m >> b) & 1));
                            let mut full = Array1::zeros(n);
                            if let Some(v) = fixed {
                                full[6] = v;
                            }
                            for (b, &v) in map.iter().enumerate() {
                                full[v] = x[b];
                            }
                            full[i] = step.choices[full[j]];
                            assert_eq!(
                                qubo.eval_usize(&full),
                                small.eval_usize(&x) + reduced.constant
                            );
                        }
                    }
                }
                assert!(graph.small_subgraph_relation(Instant::now()).is_none());
                assert_eq!(
                    graph.small_subgraph_relation(Instant::now() + Duration::from_secs(1)),
                    reference_discovery(&graph)
                );
            }
        }
        assert!(accepted > 0);
    }
}
