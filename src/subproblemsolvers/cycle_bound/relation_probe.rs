//! Probe exact spin contractions without rebuilding a QUBO or SDP backend.
use super::*;

type Edges = Vec<((usize, usize), i128)>;

fn project(p: &PreparedCycleBound, fixed: &FixedVarMap) -> Option<(usize, i128, Edges)> {
    let n = p.fields.len();
    if fixed.iter().any(|(&i, &v)| i >= n || v > 1) {
        return None;
    }
    let free: Vec<_> = (0..n).filter(|i| !fixed.contains_key(i)).collect();
    let mut index = vec![usize::MAX; n];
    for (k, &i) in free.iter().enumerate() {
        index[i] = k;
    }
    let mut fields = vec![0; free.len()];
    let mut constant = p.constant;
    let spin = |i: usize| 2 * fixed[&i] as i128 - 1;
    for (i, &h) in p.fields.iter().enumerate() {
        if fixed.contains_key(&i) {
            constant += h * spin(i);
        } else {
            fields[index[i]] = h;
        }
    }
    let mut edges = Vec::new();
    for &((i, j), w) in &p.edges {
        match (fixed.contains_key(&i), fixed.contains_key(&j)) {
            (true, true) => constant += w * spin(i) * spin(j),
            (true, false) => fields[index[j]] += w * spin(i),
            (false, true) => fields[index[i]] += w * spin(j),
            (false, false) => edges.push(((index[i], index[j]), w)),
        }
    }
    for (i, h) in fields.into_iter().enumerate() {
        if h != 0 {
            edges.push(((i, free.len()), h));
        }
    }
    Some((free.len(), constant, edges))
}

fn contract(constant: i128, edges: &Edges, u: usize, v: usize, opposite: bool) -> (i128, Edges) {
    assert!(u < v);
    let sign = if opposite { -1 } else { 1 };
    let mut constant = constant;
    let mut sums = BTreeMap::new();
    for &((i, j), w) in edges {
        let w = w * if i == v { sign } else { 1 } * if j == v { sign } else { 1 };
        let remap = |i| {
            if i == v {
                u
            } else if i > v {
                i - 1
            } else {
                i
            }
        };
        let (a, b) = (remap(i), remap(j));
        if a == b {
            constant += w;
        } else {
            *sums.entry((a.min(b), a.max(b))).or_insert(0) += w;
        }
    }
    (
        constant,
        sums.into_iter().filter(|(_, w)| *w != 0).collect(),
    )
}

fn bound(n: usize, c: i128, edges: &Edges, budget: Duration, cutoff: f64) -> Option<f64> {
    let start = Instant::now();
    let a = pack(
        n,
        c,
        edges,
        start,
        budget / 2,
        cutoff,
        PackingStrategy::ExtendedGreedy,
        0,
        None,
    )?
    .lower;
    if a >= cutoff {
        return Some(a);
    }
    let b = pack(
        n,
        c,
        edges,
        start,
        budget,
        cutoff,
        PackingStrategy::FractionalPortfolio,
        0,
        None,
    )?
    .lower;
    Some(a.max(b))
}

#[derive(Default, Debug)]
pub struct Report {
    /// Original variable indices, retained parity, rejected bound, retained bound.
    pub certified: Vec<(usize, usize, bool, f64, f64)>,
    pub candidates: usize,
    pub pairs: usize,
    pub relations: usize,
    pub independent_relations: usize,
    pub closed: bool,
    pub baseline_closed: bool,
    pub baseline: f64,
    pub union_bound: f64,
    pub best_side_gain: f64,
    pub nearest_cutoff_shortfall: f64,
    pub micros: u128,
}

pub fn screen(
    p: &PreparedCycleBound,
    fixed: &FixedVarMap,
    incumbent: f64,
    inherited: f64,
    budget: Duration,
    side_budget: Duration,
) -> Option<Report> {
    screen_limited(p, fixed, incumbent, inherited, budget, side_budget, 32, false)
}

pub fn screen_limited(
    p: &PreparedCycleBound, fixed: &FixedVarMap, incumbent: f64, inherited: f64,
    budget: Duration, side_budget: Duration, max_candidates: usize, stop_after_certificate: bool,
) -> Option<Report> {
    let start = Instant::now();
    let (n, c, edges) = project(p, fixed)?;
    let mut trace = PackingTrace::default();
    let traced = pack(
        n,
        c,
        &edges,
        start,
        budget.min(Duration::from_millis(5)),
        f64::INFINITY,
        PackingStrategy::ExtendedGreedy,
        0,
        Some(&mut trace),
    )?
    .lower;
    let baseline = bound(n, c, &edges, side_budget, incumbent)?
        .max(inherited)
        .max(traced);
    let mut r = Report {
        baseline,
        union_bound: baseline,
        baseline_closed: baseline >= incumbent,
        nearest_cutoff_shortfall: f64::INFINITY,
        ..Default::default()
    };
    let mut score = vec![0i128; edges.len()];
    for (cycle, w) in trace.cycles {
        for e in cycle {
            score[e] += w;
        }
    }
    let mut candidates: Vec<_> = (0..edges.len())
        .filter(|&e| score[e] > 0 && edges[e].0 .1 < n)
        .collect();
    candidates.sort_by_key(|&e| {
        (
            std::cmp::Reverse(score[e]),
            std::cmp::Reverse(edges[e].1.abs()),
            e,
        )
    });
    candidates.truncate(max_candidates);
    let original: Vec<_> = (0..p.fields.len()).filter(|i| !fixed.contains_key(i)).collect();
    r.candidates = candidates.len();
    let mut parent: Vec<_> = (0..n).collect();
    let mut xor = vec![false; n];
    let find = |mut i: usize, parent: &[usize], xor: &[bool]| {
        let mut parity = false;
        while parent[i] != i {
            parity ^= xor[i];
            i = parent[i];
        }
        (i, parity)
    };
    if !r.baseline_closed {
        for e in candidates {
            let (u, v) = edges[e].0;
            let mut sides = [0.0; 2];
            let mut complete = true;
            for parity in 0..2 {
                if start.elapsed() >= budget {
                    complete = false;
                    break;
                }
                let (cc, ee) = contract(c, &edges, u, v, parity == 1);
                sides[parity] = bound(
                    n - 1,
                    cc,
                    &ee,
                    side_budget.min(budget.saturating_sub(start.elapsed())),
                    incumbent,
                )?
                .max(baseline);
            }
            if !complete {
                break;
            }
            r.pairs += 1;
            r.union_bound = r.union_bound.max(sides[0].min(sides[1]));
            r.best_side_gain = r.best_side_gain.max(sides[0].max(sides[1]) - baseline);
            r.nearest_cutoff_shortfall = r
                .nearest_cutoff_shortfall
                .min(incumbent - sides[0].max(sides[1]));
            if sides.iter().all(|&b| b >= incumbent) {
                r.closed = true;
                break;
            }
            if sides.iter().any(|&b| b >= incumbent) {
                r.relations += 1;
                // Count independent contractions, not duplicate edges in one class.
                let ((a, pa), (b, pb)) = (find(u, &parent, &xor), find(v, &parent, &xor));
                let retained_opposite = sides[0] >= incumbent;
                r.certified.push((original[u], original[v], retained_opposite,
                    sides[usize::from(!retained_opposite)], sides[usize::from(retained_opposite)]));
                if a != b {
                    parent[b] = a;
                    xor[b] = pa ^ pb ^ retained_opposite;
                    r.independent_relations += 1;
                } else if pa ^ pb != retained_opposite {
                    r.closed = true;
                    break;
                }
                if stop_after_certificate { break; }
            }
        }
    }
    r.micros = start.elapsed().as_micros();
    Some(r)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parity_contractions_and_bounds_match_enumeration() {
        use smolprng::{JsfLarge, PRNG};
        let mut rng = PRNG {
            generator: JsfLarge::from(81573u64),
        };
        for _ in 0..20 {
            let mut q = Qubo::make_random_qubo(7, &mut rng, 0.6);
            q.q.data_mut()
                .iter_mut()
                .for_each(|v| *v = (*v * 8.0).round());
            q.c.mapv_inplace(|v| (v * 8.0).round());
            let p = PreparedCycleBound::new(&q).unwrap();
            let fixed: FixedVarMap = [(1, 0), (5, 1)].into_iter().collect();
            let free: Vec<_> = (0..7).filter(|i| !fixed.contains_key(i)).collect();
            let (n, c, edges) = project(&p, &fixed).unwrap();
            for u in 0..n {
                for v in u + 1..n {
                    for opposite in [false, true] {
                        let (cc, ee) = contract(c, &edges, u, v, opposite);
                        let mut best = f64::INFINITY;
                        for mask in 0..1usize << n {
                            if (((mask >> u) ^ (mask >> v)) & 1 == 1) != opposite {
                                continue;
                            }
                            let mut x = ndarray::Array1::zeros(7);
                            for (&i, &value) in &fixed {
                                x[i] = value;
                            }
                            for (k, &i) in free.iter().enumerate() {
                                x[i] = (mask >> k) & 1;
                            }
                            let mut spins: Vec<_> = (0..n)
                                .filter(|&i| i != v)
                                .map(|i| 2 * ((mask >> i) & 1) as i128 - 1)
                                .collect();
                            spins.push(1);
                            let value = (cc
                                + ee.iter()
                                    .map(|&((i, j), w)| w * spins[i] * spins[j])
                                    .sum::<i128>()) as f64
                                / 64.0;
                            assert_eq!(value, q.eval_usize(&x));
                            best = best.min(value);
                        }
                        assert!(
                            bound(n - 1, cc, &ee, Duration::from_millis(5), f64::INFINITY).unwrap()
                                <= best
                        );
                    }
                }
            }
        }
    }
}
