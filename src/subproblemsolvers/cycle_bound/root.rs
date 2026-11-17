//! A bounded root packing over overlapping short frustrated cycles.
use super::*;
use clarabel::algebra::CscMatrix;
use clarabel::solver::{DefaultSettings, DefaultSolver, IPSolver, NonnegativeConeT};
mod edge_split;
mod pricing;

const MAX_CYCLES: usize = 2800;
const MAX_EDGE_VISITS: usize = 2_000_000;
const SHORT_LENGTH: usize = 8;
const LONG_LENGTH: usize = 10;
const SCALE: i128 = 256;
const INITIAL_PRICING_ROUNDS: usize = 4;
const MAX_PRICING_ROUNDS: usize = 24;
type Edges = Vec<Vec<(usize, usize, bool)>>;

#[derive(Clone, Copy, Debug)]
pub struct RootCycleStatistics {
    pub packing_bound: f64,
    pub split_bound: Option<f64>,
    pub pricing_rounds: usize,
    pub split_lp_results: usize,
    pub packing_seconds: f64,
    pub split_seconds: f64,
}

impl RootCycleStatistics {
    pub(crate) fn lower_bound(self) -> f64 {
        self.split_bound
            .map_or(self.packing_bound, |b| b.max(self.packing_bound))
    }
}

fn continue_pricing(rounds: usize, recent_gain: [i128; 2], coupling: i128) -> bool {
    // Require half an unsatisfied edge's cost over the past two rounds.
    rounds < INITIAL_PRICING_ROUNDS || recent_gain.into_iter().sum::<i128>() >= coupling * SCALE
}

fn lattice_split_limit(distance: f64, edge_cost: f64) -> usize {
    if distance <= 0.25 * edge_cost {
        2
    } else if distance <= 0.75 * edge_cost {
        1
    } else {
        0
    }
}

struct Discovery<'a> {
    graph: &'a Edges,
    deadline: Instant,
    visits: usize,
    max_visits: usize,
    min_length: usize,
    max_length: usize,
    max_cycles: usize,
    seen: Vec<bool>,
    path: Vec<usize>,
    cycles: Vec<Vec<usize>>,
}

impl Discovery<'_> {
    fn visit(&mut self, root: usize, first: usize, u: usize, odd: bool) -> bool {
        for &(v, edge, sign) in &self.graph[u] {
            if self.visits == self.max_visits
                || (self.visits % 256 == 0 && Instant::now() >= self.deadline)
            {
                return false;
            }
            self.visits += 1;
            if v == root && self.path.len() + 1 >= self.min_length && first < u && odd != sign {
                let mut cycle = self.path.clone();
                cycle.push(edge);
                cycle.sort_unstable();
                self.cycles.push(cycle);
                if self.cycles.len() == self.max_cycles {
                    return false;
                }
            } else if v > root && !self.seen[v] && self.path.len() + 1 < self.max_length {
                let first = if self.path.is_empty() { v } else { first };
                self.seen[v] = true;
                self.path.push(edge);
                let keep_going = self.visit(root, first, v, odd != sign);
                self.path.pop();
                self.seen[v] = false;
                if !keep_going {
                    return false;
                }
            }
        }
        true
    }
}

fn verified_gain(cycles: &[Vec<usize>], caps: &[i128], proposal: Option<&[f64]>) -> i128 {
    let mut residual = caps.to_vec();
    pack_into(cycles, &mut residual, proposal, SCALE)
}

fn pack_into(
    cycles: &[Vec<usize>],
    residual: &mut [i128],
    proposal: Option<&[f64]>,
    scale: i128,
) -> i128 {
    let mut gain = 0;
    for (i, cycle) in cycles.iter().enumerate() {
        let capacity = cycle.iter().map(|&e| residual[e]).min().unwrap();
        let weight = proposal.map_or(capacity, |x| {
            let w = x[i];
            if w.is_finite() {
                ((w * scale as f64).floor() as i128).clamp(0, capacity)
            } else {
                0
            }
        });
        for &e in cycle {
            residual[e] -= weight;
        }
        gain += 2 * weight;
    }
    gain
}

impl PreparedCycleBound {
    pub(crate) fn short_cycle_root_bound(
        &self,
        budget: Duration,
        split_budget: Duration,
        grid_point: impl Fn(f64) -> Option<f64>,
    ) -> Option<RootCycleStatistics> {
        let start = Instant::now();
        let initial_budget = budget.min(Duration::from_millis(500));
        let n = self.fields.len();
        let m = self.edges.len();
        if n > 2000 || m < 32 || m > 4 * n || budget < Duration::from_millis(10) {
            return None;
        }
        // Uniform couplings are the regime where short-cycle packing paid
        // for itself; weighted instances keep their ordinary certified bounds.
        let magnitude = self.edges[0].1.abs();
        if self.edges.iter().any(|&(_, w)| w.abs() != magnitude) {
            return None;
        }
        let mut graph = vec![Vec::new(); n];
        for (k, &((u, v), w)) in self.edges.iter().enumerate() {
            graph[u].push((v, k, w > 0));
            graph[v].push((u, k, w > 0));
        }
        // Anchor couplings contribute to the baseline, but not the cycle pool.
        let mut discovery = Discovery {
            graph: &graph,
            deadline: start + initial_budget / 3,
            visits: 0,
            max_visits: MAX_EDGE_VISITS,
            min_length: 3,
            max_length: SHORT_LENGTH,
            max_cycles: MAX_CYCLES,
            seen: vec![false; n],
            path: Vec::with_capacity(SHORT_LENGTH),
            cycles: Vec::new(),
        };
        for root in 0..n {
            if !discovery.visit(root, root, root, false) {
                break;
            }
        }
        if Instant::now() < discovery.deadline {
            // Preserve the short pool before spending work on longer cycles.
            discovery.max_visits += 5_000_000;
            discovery.max_cycles += 2400;
            discovery.min_length = SHORT_LENGTH + 1;
            discovery.max_length = LONG_LENGTH;
            for root in 0..n {
                if !discovery.visit(root, root, root, false) {
                    break;
                }
            }
        }
        let mut cycles = discovery.cycles;
        if cycles.is_empty() {
            return None;
        }
        cycles.sort_by_key(Vec::len);
        let caps: Vec<_> = self.edges.iter().map(|&(_, w)| w.abs() * SCALE).collect();
        let fields: i128 = self.fields.iter().map(|h| h.abs()).sum();
        let baseline = (self.constant - fields) * SCALE - caps.iter().sum::<i128>();
        let mut gain = verified_gain(&cycles, &caps, None);
        let mut pricing_rounds = 0;
        let mut final_prices = Vec::new();
        if let Some((packed, mut prices)) = solve_pool(&cycles, &caps, start + initial_budget) {
            gain = gain.max(packed);
            let deadline = start + budget;
            let mut recent_gain = [0; 2];
            for round in 0..MAX_PRICING_ROUNDS {
                let now = Instant::now();
                if deadline.saturating_duration_since(now) < Duration::from_millis(100)
                    || prices.iter().any(|p| !p.is_finite())
                {
                    break;
                }
                if !continue_pricing(round, recent_gain, magnitude) {
                    break;
                }
                if pricing::add_cycles(
                    &graph,
                    &prices,
                    &mut cycles,
                    round,
                    deadline.min(now + Duration::from_millis(45)),
                ) == 0
                {
                    break;
                }
                let Some((packed, updated)) = solve_pool(&cycles, &caps, deadline) else {
                    break;
                };
                let best = gain.max(packed);
                recent_gain[round % 2] = best - gain;
                gain = best;
                pricing_rounds += 1;
                prices = updated;
            }
            final_prices = prices;
        }
        let lower = baseline + gain;
        if lower.abs() > 1i128 << 52 {
            return None;
        }
        let packing_seconds = start.elapsed().as_secs_f64();
        let split_start = Instant::now();
        let packing_bound = lower as f64 / (64.0 * SCALE as f64);
        let (split_bound, split_lp_results) = grid_point(packing_bound).map_or((None, 0), |grid| {
            let distance = (grid - packing_bound).max(0.0);
            let max_pairs = lattice_split_limit(distance, 2.0 * magnitude as f64 / 64.0);
            edge_split::probe(
                self,
                &graph,
                &cycles,
                &caps,
                &final_prices,
                baseline,
                split_budget,
                max_pairs,
                |bound| grid_point(bound).is_some_and(|point| point > grid),
            )
        });
        Some(RootCycleStatistics {
            packing_bound,
            split_bound,
            pricing_rounds,
            split_lp_results,
            packing_seconds,
            split_seconds: split_start.elapsed().as_secs_f64(),
        })
    }
}

fn solve_pool(cycles: &[Vec<usize>], caps: &[i128], deadline: Instant) -> Option<(i128, Vec<f64>)> {
    solve_pool_with(cycles, caps, deadline, |weights, prices| {
        (verified_gain(cycles, caps, Some(weights)), prices.to_vec())
    })
}

fn solve_pool_with<T>(
    cycles: &[Vec<usize>],
    caps: &[i128],
    deadline: Instant,
    certificate: impl FnOnce(&[f64], &[f64]) -> T,
) -> Option<T> {
    let m = caps.len();
    let k = cycles.len();
    let mut colptr = vec![0];
    let mut rowval = Vec::with_capacity(k + cycles.iter().map(Vec::len).sum::<usize>());
    let mut nzval = Vec::with_capacity(rowval.capacity());
    for (i, cycle) in cycles.iter().enumerate() {
        rowval.extend(cycle.iter().copied());
        nzval.extend(std::iter::repeat_n(1.0, cycle.len()));
        rowval.push(m + i);
        nzval.push(-1.0);
        colptr.push(rowval.len());
    }
    let constraints = CscMatrix::new(m + k, k, colptr, rowval, nzval);
    let mut rhs: Vec<_> = caps
        .iter()
        .map(|&capacity| capacity as f64 / SCALE as f64)
        .collect();
    rhs.resize(m + k, 0.0);
    let remaining = deadline.saturating_duration_since(Instant::now());
    if remaining >= Duration::from_millis(5) {
        let settings = DefaultSettings {
            verbose: false,
            max_iter: 45,
            time_limit: remaining.as_secs_f64(),
            max_threads: 1,
            ..DefaultSettings::default()
        };
        let mut solver = DefaultSolver::new(
            &CscMatrix::zeros((k, k)),
            &vec![-1.0; k],
            &constraints,
            &rhs,
            &[NonnegativeConeT(m + k)],
            settings,
        );
        // Symbolic setup is bounded by the pool size, but not by the
        // LP's own timer. Debit it before allowing the first iteration.
        solver.settings.time_limit = deadline
            .saturating_duration_since(Instant::now())
            .as_secs_f64();
        if solver.settings.time_limit > 0.0 {
            solver.solve();
            if solver.solution.x.len() == k && solver.solution.z.len() >= m {
                return Some(certificate(&solver.solution.x, &solver.solution.z[..m]));
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn lattice_split_work_scales_with_distance_to_cross() {
        assert_eq!(lattice_split_limit(0.03186, 1.0), 2);
        assert_eq!(lattice_split_limit(0.25, 1.0), 2);
        assert_eq!(lattice_split_limit(0.50, 1.0), 1);
        assert_eq!(lattice_split_limit(0.75, 1.0), 1);
        assert_eq!(lattice_split_limit(0.75001, 1.0), 0);
        assert_eq!(lattice_split_limit(5.0, 10.0), 1);
    }

    #[test]
    fn adaptive_pricing_uses_a_window_of_verified_improvements() {
        let coupling = 32;
        let threshold = coupling * SCALE;
        assert!(continue_pricing(3, [0, 0], coupling));
        assert!(!continue_pricing(4, [threshold - 1, 0], coupling));
        assert!(continue_pricing(4, [threshold, 0], coupling));
        assert!(continue_pricing(5, [threshold - 1, 1], coupling));
        assert!(!continue_pricing(6, [1, 0], coupling));
    }

    #[test]
    fn long_frustrated_cycles_are_discovered_once() {
        let mut graph = vec![Vec::new(); LONG_LENGTH];
        for u in 0..LONG_LENGTH {
            let v = (u + 1) % LONG_LENGTH;
            graph[u].push((v, u, u == 0));
            graph[v].push((u, u, u == 0));
        }
        let mut discovery = Discovery {
            graph: &graph,
            deadline: Instant::now() + Duration::from_secs(1),
            visits: 0,
            max_visits: 1000,
            min_length: SHORT_LENGTH + 1,
            max_length: LONG_LENGTH,
            max_cycles: 100,
            seen: vec![false; LONG_LENGTH],
            path: Vec::new(),
            cycles: Vec::new(),
        };
        for root in 0..LONG_LENGTH {
            assert!(discovery.visit(root, root, root, false));
        }
        assert_eq!(discovery.cycles, vec![(0..LONG_LENGTH).collect::<Vec<_>>()]);
    }

    #[test]
    fn repaired_weights_never_overcharge_an_edge() {
        let cycles = vec![vec![0, 1, 2], vec![0, 3, 4]];
        let cap = vec![9 * SCALE; 5];
        assert_eq!(verified_gain(&cycles, &cap, Some(&[7.0, 7.0])), 18 * SCALE);
        assert_eq!(verified_gain(&cycles, &cap, Some(&[f64::NAN, -1.0])), 0);
    }

    #[test]
    fn root_bound_is_valid_against_exhaustive_minima() {
        let n = 10;
        for seed in 0..8 {
            let mut q = sprs::TriMat::new((n, n));
            for i in 0..n {
                for j in i + 1..n {
                    if (i + j + seed) % 7 != 0 {
                        q.add_triplet(
                            i,
                            j,
                            if (i * 3 + j * 7 + seed) % 3 == 0 {
                                -8.0
                            } else {
                                8.0
                            },
                        );
                    }
                }
            }
            let q = Qubo::new_with_c(q.to_csr(), Array1::from_elem(n, seed as f64 - 4.0));
            let prepared = PreparedCycleBound::new(&q).unwrap();
            let lower = prepared
                .short_cycle_root_bound(Duration::from_millis(250), Duration::ZERO, |_| None)
                .unwrap()
                .lower_bound();
            for mask in 0..1 << n {
                assert!(lower <= q.eval_usize(&Array1::from_iter((0..n).map(|i| (mask >> i) & 1))));
            }
        }
    }
}
