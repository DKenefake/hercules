//! Experimental necessary conditions for a node-local optimum, not global cuts.
use super::*;

#[derive(Clone)]
struct Parity {
    parent: Vec<usize>,
    rank: Vec<u8>,
    xor: Vec<bool>,
}
impl Parity {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
            xor: vec![false; n],
        }
    }
    fn find(&self, mut v: usize) -> (usize, bool) {
        let mut p = false;
        while self.parent[v] != v {
            p ^= self.xor[v];
            v = self.parent[v];
        }
        (v, p)
    }
    fn relation(&self, a: usize, b: usize) -> Option<bool> {
        let (u, p) = self.find(a);
        let (v, q) = self.find(b);
        (u == v).then_some(p ^ q)
    }
    fn join(&mut self, a: usize, b: usize, opposite: bool) -> bool {
        let (mut u, p) = self.find(a);
        let (mut v, q) = self.find(b);
        if u == v {
            return p ^ q == opposite;
        }
        if self.rank[u] < self.rank[v] {
            std::mem::swap(&mut u, &mut v);
        }
        self.parent[v] = u;
        self.xor[v] = p ^ q ^ opposite;
        if self.rank[u] == self.rank[v] {
            self.rank[u] += 1;
        }
        true
    }
}

#[derive(Debug, Default)]
pub struct Report {
    /// Each tuple requires x_i XOR x_j = opposite; index n is the fixed-one anchor.
    pub relations: Vec<(usize, usize, bool)>,
    pub probes: usize,
    pub rejected: usize,
    pub contradiction: bool,
    pub timed_out: bool,
}

// None means interrupted, never a proof of contradiction.
fn propagate(
    state: &mut Parity,
    edges: &[((usize, usize), i128)],
    adjacency: &[Vec<usize>],
    fixed: &FixedVarMap,
    deadline: Instant,
) -> Option<bool> {
    loop {
        let mut changed = false;
        for (v, incident) in adjacency.iter().enumerate() {
            if Instant::now() >= deadline {
                return None;
            }
            if fixed.contains_key(&v) {
                continue;
            }
            let mut total = 0;
            let mut used = 0;
            for &e in incident {
                let ((i, j), w) = edges[e];
                total += w.abs();
                if state.relation(i, j).is_some_and(|p| p != (w > 0)) {
                    used += w.abs();
                }
            }
            if 2 * used > total {
                return Some(false);
            }
            for &e in incident {
                let ((i, j), w) = edges[e];
                if state.relation(i, j).is_none() && 2 * (used + w.abs()) > total {
                    if !state.join(i, j, w > 0) {
                        return Some(false);
                    }
                    changed = true;
                }
            }
        }
        if !changed {
            return Some(true);
        }
    }
}

/// Root or node diagnostic. Returned relations must stay local to `fixed`.
/// No incumbent, floating-point tolerances, or heuristic assumptions are used.
pub fn probe(q: &Qubo, fixed: &FixedVarMap, limit: usize, budget: Duration) -> Option<Report> {
    let deadline = Instant::now() + budget;
    let prepared = PreparedCycleBound::new(q)?;
    let n = q.num_x();
    let mut edges = prepared.edges;
    for (i, h) in prepared.fields.into_iter().enumerate() {
        if h != 0 {
            edges.push(((i, n), h));
        }
    }
    let mut adjacency = vec![Vec::new(); n];
    for (e, &((i, j), _)) in edges.iter().enumerate() {
        adjacency[i].push(e);
        if j < n {
            adjacency[j].push(e);
        }
    }
    let mut state = Parity::new(n + 1);
    for (&i, &x) in fixed {
        if i >= n || x > 1 {
            return None;
        }
        state.join(i, n, x == 0);
    }
    let initial = state.clone();
    let mut report = Report::default();
    match propagate(&mut state, &edges, &adjacency, fixed, deadline) {
        Some(false) => report.contradiction = true,
        None => report.timed_out = true,
        _ => {}
    }
    for &((i, j), _) in &edges {
        if report.contradiction || report.timed_out || report.probes >= limit {
            break;
        }
        if state.relation(i, j).is_some() {
            continue;
        }
        for opposite in [false, true] {
            if report.probes >= limit {
                break;
            }
            let mut trial = state.clone();
            trial.join(i, j, opposite);
            report.probes += 1;
            match propagate(&mut trial, &edges, &adjacency, fixed, deadline) {
                Some(false) => {
                    report.rejected += 1;
                    if !state.join(i, j, !opposite) {
                        report.contradiction = true;
                        break;
                    }
                    match propagate(&mut state, &edges, &adjacency, fixed, deadline) {
                        Some(false) => report.contradiction = true,
                        None => report.timed_out = true,
                        _ => {}
                    }
                    break;
                }
                None => {
                    report.timed_out = true;
                    break;
                }
                _ => {}
            }
        }
    }
    for ((i, j), _) in edges {
        if initial.relation(i, j).is_none() {
            if let Some(p) = state.relation(i, j) {
                report.relations.push((i, j, p));
            }
        }
    }
    Some(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    #[test]
    fn preserves_every_node_local_optimum() {
        for seed in 0..24 {
            let n = 5;
            let mut m = sprs::TriMat::new((n, n));
            for i in 0..n {
                for j in 0..n {
                    m.add_triplet(
                        i,
                        j,
                        ((seed * 17 + i * 7 + j * 13 + i * j) % 11) as f64 / 8.0 - 0.625,
                    );
                }
            }
            let q = Qubo::new_with_c(
                m.to_csr(),
                Array1::from_iter((0..n).map(|i| ((seed + i) % 5) as f64 / 8.0 - 0.25)),
            );
            for mut code in 0..3usize.pow(n as u32) {
                let mut fixed = FixedVarMap::default();
                for i in 0..n {
                    if code % 3 > 0 {
                        fixed.insert(i, code % 3 - 1);
                    }
                    code /= 3;
                }
                let result = probe(&q, &fixed, 100, Duration::from_secs(1)).unwrap();
                for mask in 0..1 << n {
                    let mut x = Array1::from_iter((0..n).map(|i| (mask >> i) & 1));
                    if fixed.iter().any(|(&i, &v)| x[i] != v) {
                        continue;
                    }
                    let value = q.eval_usize(&x);
                    let local = (0..n).filter(|i| !fixed.contains_key(i)).all(|i| {
                        x[i] ^= 1;
                        let other = q.eval_usize(&x);
                        x[i] ^= 1;
                        other >= value
                    });
                    if local {
                        assert!(!result.contradiction);
                        for &(i, j, p) in &result.relations {
                            assert_eq!(x[i] != if j == n { 1 } else { x[j] }, p);
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn interrupted_probe_is_not_a_contradiction() {
        let q = Qubo::from_vec(vec![0], vec![1], vec![8.0], vec![-2.0; 2], 2);
        let r = probe(&q, &FixedVarMap::default(), 10, Duration::ZERO).unwrap();
        assert!(r.timed_out);
        assert!(!r.contradiction);
    }
}
