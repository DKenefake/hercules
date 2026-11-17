//! Certify both parities of an edge without committing to a branch.
use super::*;
use rustc_hash::FxHashSet;

const MAX_PAIRS: usize = 2;
const MAX_PATHS_PER_SIDE: usize = 128;
const EXTRA_PRECISION: i128 = 1 << 20;
const FINE_SCALE: i128 = SCALE * EXTRA_PRECISION;
const FINE_UNIT: f64 = 64.0 * FINE_SCALE as f64;

pub(super) fn probe(
    problem: &PreparedCycleBound,
    graph: &Edges,
    pool: &[Vec<usize>],
    caps: &[i128],
    prices: &[f64],
    baseline: i128,
    budget: Duration,
    max_pairs: usize,
    crosses_lattice: impl Fn(f64) -> bool,
) -> (Option<f64>, usize) {
    if max_pairs == 0 || budget.is_zero() || prices.len() != caps.len() {
        return (None, 0);
    }
    let deadline = Instant::now() + budget;
    let mut usage = vec![0usize; caps.len()];
    for cut in pool {
        for &edge in cut {
            usage[edge] += 1;
        }
    }
    let mut ranked: Vec<_> = prices
        .iter()
        .enumerate()
        .filter_map(|(edge, &p)| (p > 0.05 && p < 0.95 && usage[edge] > 0).then_some(edge))
        .collect();
    ranked.sort_unstable_by(|&a, &b| {
        (prices[a] - 0.5)
            .abs()
            .total_cmp(&(prices[b] - 0.5).abs())
            .then_with(|| usage[b].cmp(&usage[a]))
            .then_with(|| a.cmp(&b))
    });
    let mut best = None;
    let mut lp_results = 0;
    for edge in ranked.into_iter().take(max_pairs.min(MAX_PAIRS)) {
        let now = Instant::now();
        if deadline.saturating_duration_since(now) < Duration::from_millis(75) {
            break;
        }
        let ((u, v), w) = problem.edges[edge];
        let [preferred, violated] = parity_paths(
            graph,
            u,
            v,
            edge,
            w > 0,
            deadline.min(now + Duration::from_millis(20)),
        );
        let ((a, solved_a), (b, solved_b)) = rayon::join(
            || branch(pool, caps, baseline, edge, false, preferred, deadline),
            || branch(pool, caps, baseline, edge, true, violated, deadline),
        );
        // Both parities cover the parent; the weaker side is the certificate.
        let pair = a.min(b);
        best = Some(best.map_or(pair, |old: i128| old.max(pair)));
        lp_results += usize::from(solved_a) + usize::from(solved_b);
        if crosses_lattice(outward_lower(pair)) {
            break;
        }
    }
    (best.map(outward_lower), lp_results)
}

fn outward_lower(value: i128) -> f64 {
    (value as f64 / FINE_UNIT).next_down()
}

fn verify_precise(rows: &[Vec<usize>], caps: &[i128], proposal: &[f64]) -> i128 {
    let mut remaining: Vec<_> = caps.iter().map(|&c| c * EXTRA_PRECISION).collect();
    let fine = pack_into(rows, &mut remaining, Some(proposal), FINE_SCALE);
    // Add only weight that fits every edge of its row in the exact residual.
    fine + pack_into(rows, &mut remaining, None, FINE_SCALE)
}

fn branch(
    pool: &[Vec<usize>],
    caps: &[i128],
    baseline: i128,
    edge: usize,
    unsatisfied: bool,
    paths: Vec<Vec<usize>>,
    deadline: Instant,
) -> (i128, bool) {
    let mut child_caps = caps.to_vec();
    child_caps[edge] = 0;
    let constant = baseline + if unsatisfied { 2 * caps[edge] } else { 0 };
    let mut rows = Vec::with_capacity(pool.len() + paths.len());
    let mut seen = FxHashSet::default();
    for row in pool {
        let row = match row.binary_search(&edge) {
            // An unsatisfied split edge already fulfills any row containing it.
            Ok(_) if unsatisfied => continue,
            Ok(i) => {
                let mut path = row.clone();
                path.remove(i);
                path
            }
            Err(_) => row.clone(),
        };
        if seen.insert(row.clone()) {
            rows.push(row);
        }
    }
    rows.extend(paths.into_iter().filter(|row| seen.insert(row.clone())));
    let greedy = verified_gain(&rows, &child_caps, None);
    if !rows.is_empty() {
        if let Some(gain) = solve_pool_with(&rows, &child_caps, deadline, |proposal, _| {
            // Retain the coarse repair too: different flooring can change which
            // rows receive the remaining capacity. These checks share one LP.
            let coarse = verified_gain(&rows, &child_caps, Some(proposal)).max(greedy);
            verify_precise(&rows, &child_caps, proposal).max(coarse * EXTRA_PRECISION)
        }) {
            return (constant * EXTRA_PRECISION + gain, true);
        }
    }
    ((constant + greedy) * EXTRA_PRECISION, false)
}

fn parity_paths(
    graph: &Edges,
    source: usize,
    target: usize,
    edge: usize,
    preferred: bool,
    deadline: Instant,
) -> [Vec<Vec<usize>>; 2] {
    struct Paths<'a> {
        graph: &'a Edges,
        target: usize,
        edge: usize,
        preferred: bool,
        deadline: Instant,
        visits: usize,
        seen: Vec<bool>,
        path: Vec<usize>,
        out: [Vec<Vec<usize>>; 2],
    }

    impl Paths<'_> {
        fn visit(&mut self, u: usize, parity: bool) -> bool {
            for &(v, e, different) in &self.graph[u] {
                if self.visits >= 60_000
                    || (self.visits % 256 == 0 && Instant::now() >= self.deadline)
                {
                    return false;
                }
                self.visits += 1;
                if e == self.edge || self.seen[v] {
                    continue;
                }
                self.path.push(e);
                let odd = parity ^ different;
                let mut keep = true;
                if v == self.target {
                    // This path needs an unsatisfied edge exactly when its
                    // preferred parity disagrees with the chosen endpoint parity.
                    let side = usize::from(!(odd ^ self.preferred));
                    if self.out[side].len() < MAX_PATHS_PER_SIDE {
                        let mut path = self.path.clone();
                        path.sort_unstable();
                        self.out[side].push(path);
                    }
                    keep = !self.out.iter().all(|p| p.len() >= MAX_PATHS_PER_SIDE);
                } else if self.path.len() < 10 {
                    self.seen[v] = true;
                    keep = self.visit(v, odd);
                    self.seen[v] = false;
                }
                self.path.pop();
                if !keep {
                    return false;
                }
            }
            true
        }
    }

    let mut search = Paths {
        graph,
        target,
        edge,
        preferred,
        deadline,
        visits: 0,
        seen: vec![false; graph.len()],
        path: Vec::with_capacity(10),
        out: [Vec::new(), Vec::new()],
    };
    search.seen[source] = true;
    search.visit(source, false);
    search.out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_crossing_stops_before_another_pair() {
        let mut q = sprs::TriMat::new((6, 6));
        for (i, j) in [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)] {
            q.add_triplet(i, j, 4.0);
        }
        let q = Qubo::new_with_c(q.to_csr(), ndarray::Array1::from_elem(6, -2.0));
        let p = PreparedCycleBound::new(&q).unwrap();
        let mut graph = vec![Vec::new(); 6];
        for (e, &((u, v), w)) in p.edges.iter().enumerate() {
            graph[u].push((v, e, w > 0));
            graph[v].push((u, e, w > 0));
        }
        let caps: Vec<_> = p.edges.iter().map(|(_, w)| w.abs() * SCALE).collect();
        let baseline = (p.constant - p.fields.iter().map(|h| h.abs()).sum::<i128>()) * SCALE
            - caps.iter().sum::<i128>();
        let pool = vec![vec![0, 1, 2], vec![3, 4, 5]];
        let check = |max_pairs, crosses: fn(f64) -> bool| {
            probe(
                &p,
                &graph,
                &pool,
                &caps,
                &[0.5; 6],
                baseline,
                Duration::from_secs(2),
                max_pairs,
                crosses,
            )
        };
        assert_eq!(check(0, |_| false), (None, 0));
        let (bound, lps) = check(2, |bound| bound > -5.0);
        assert!(bound.unwrap() > -5.0);
        assert_eq!(lps, 2);
        assert_eq!(check(2, |_| false).1, 4);
    }

    #[test]
    fn finer_certificate_recovers_fractional_weight_without_overcharging() {
        let rows = vec![vec![0, 1], vec![0, 2]];
        let caps = [SCALE; 3];
        let proposal = [1.0 / 3.0, 2.0 / 3.0];
        let coarse = verified_gain(&rows, &caps, Some(&proposal));
        let tight = verify_precise(&rows, &caps, &proposal);
        assert_eq!(coarse, 2 * (SCALE - 1));
        assert!(tight > coarse * EXTRA_PRECISION);
        assert_eq!(tight, 2 * FINE_SCALE);
    }

    #[test]
    fn both_edge_split_certificates_hold_on_every_matching_assignment() {
        let mut q = sprs::TriMat::new((4, 4));
        for (i, j, w) in [
            (0, 1, 8.0),
            (0, 2, 4.0),
            (1, 2, 4.0),
            (0, 3, 4.0),
            (1, 3, -4.0),
        ] {
            q.add_triplet(i, j, w);
        }
        let q = Qubo::new_with_c(q.to_csr(), ndarray::array![0.5, -2.0, 1.0, 0.0]);
        let p = PreparedCycleBound::new(&q).unwrap();
        let mut graph = vec![Vec::new(); 4];
        for (k, &((u, v), w)) in p.edges.iter().enumerate() {
            graph[u].push((v, k, w > 0));
            graph[v].push((u, k, w > 0));
        }
        let caps: Vec<_> = p.edges.iter().map(|(_, w)| w.abs() * SCALE).collect();
        let baseline = (p.constant - p.fields.iter().map(|h| h.abs()).sum::<i128>()) * SCALE
            - caps.iter().sum::<i128>();
        let deadline = Instant::now() + Duration::from_secs(1);
        let pool = vec![vec![0, 1, 3]];
        let paths = parity_paths(&graph, 0, 1, 0, true, deadline);
        assert_eq!(paths, [vec![vec![1, 3]], vec![vec![2, 4]]]);
        let bounds = paths
            .into_iter()
            .enumerate()
            .map(|(side, paths)| {
                let (bound, solved) = branch(&pool, &caps, baseline, 0, side != 0, paths, deadline);
                assert!(solved);
                outward_lower(bound)
            })
            .collect::<Vec<_>>();
        for mask in 0..16 {
            let x = ndarray::Array1::from_iter((0..4).map(|i| (mask >> i) & 1));
            assert!(bounds[usize::from(x[0] == x[1])] <= q.eval_usize(&x));
        }
    }
}
