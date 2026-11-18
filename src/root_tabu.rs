//! A bounded primal-only warm start for sparse QUBOs with small exact gains.
use crate::qubo::Qubo;
use ndarray::Array1;
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

mod gain_buckets;
use gain_buckets::{flip_delta, GainBuckets};

const MAX_WORKERS: usize = 16;
const ELITE_SIZE: usize = 8;
const MAX_RADIUS: i32 = 31;

struct Rng(u64);

impl Rng {
    fn below(&mut self, limit: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 % limit as u64) as usize
    }
}

fn worker_seed(seed: usize, worker: usize) -> u64 {
    // Disjoint worker streams, with seed 2 matching the original comparison.
    let ordinal = (seed as u64)
        .wrapping_sub(2)
        .wrapping_mul(MAX_WORKERS as u64)
        .wrapping_add(3 + worker as u64);
    ordinal.wrapping_mul(0x9e3779b97f4a7c15).max(1)
}

#[derive(Clone)]
struct Point {
    x: Vec<u8>,
    value: f64,
}

struct Model {
    linear: Vec<f64>,
    edges: Vec<(usize, usize, f64)>,
    adj: Vec<Vec<(usize, f64)>>,
    radius: i32,
}

impl Model {
    fn new(q: &Qubo) -> Option<Self> {
        let n = q.num_x();
        if !(128..=4096).contains(&n) || q.q.nnz() > 8 * n || q.q.nnz() < n / 2 {
            return None;
        }

        let mut linear = q.c.to_vec();
        let mut pairs = BTreeMap::new();
        for (&value, (i, j)) in &q.q {
            // The supported gain lattice is 2 Z. Verify terms before aggregating.
            if !value.is_finite() || value % 4.0 != 0.0 || value.abs() > 4.0 * f64::from(MAX_RADIUS)
            {
                return None;
            }
            if i == j {
                linear[i] += 0.5 * value;
            } else {
                *pairs.entry((i.min(j), i.max(j))).or_insert(0.0) += 0.5 * value;
            }
        }
        if linear.iter().any(|&w| w % 2.0 != 0.0) {
            return None;
        }

        let mut adj = vec![Vec::new(); n];
        let mut edges = Vec::with_capacity(pairs.len());
        for ((i, j), w) in pairs {
            if w != 0.0 {
                edges.push((i, j, w));
                adj[i].push((j, w));
                adj[j].push((i, w));
            }
        }
        let radius = linear
            .iter()
            .zip(&adj)
            .map(|(&c, row)| {
                let (lo, hi) = row.iter().fold((c, c), |(lo, hi), &(_, w)| {
                    (lo + w.min(0.0), hi + w.max(0.0))
                });
                lo.abs().max(hi.abs())
            })
            .fold(0.0, f64::max)
            * 0.5;
        if radius > f64::from(MAX_RADIUS) {
            return None;
        }
        Some(Self {
            linear,
            edges,
            adj,
            radius: radius as i32,
        })
    }

    fn point(&self, x: Vec<u8>) -> Point {
        let value = self
            .linear
            .iter()
            .zip(&x)
            .map(|(&c, &b)| c * f64::from(b))
            .sum::<f64>()
            + self
                .edges
                .iter()
                .map(|&(i, j, w)| w * f64::from(x[i] * x[j]))
                .sum::<f64>();
        Point { x, value }
    }

    fn tabu(&self, start: Point, rng: &mut Rng, deadline: Instant) -> Point {
        let mut best = start.clone();
        let Point { mut x, mut value } = start;
        let n = x.len();
        let mut gradient = self.linear.clone();
        for &(i, j, w) in &self.edges {
            gradient[i] += w * f64::from(x[j]);
            gradient[j] += w * f64::from(x[i]);
        }
        let mut tabu_until = vec![0; n];
        let mut buckets = GainBuckets::new(&x, &gradient, self.radius);
        let mut last_best = 0;
        for step in 0..n * 5 {
            if (step % 64 == 0 && Instant::now() >= deadline) || step - last_best >= n {
                break;
            }
            buckets.release(step, &tabu_until, &x, &gradient);
            let Some((i, delta)) = buckets.select(rng.below(n), value, best.value) else {
                break;
            };
            x[i] ^= 1;
            value += delta;
            let sign = if x[i] == 1 { 1.0 } else { -1.0 };
            let expiry = step + 9 + rng.below(23);
            buckets.forbid(i, delta, step >= tabu_until[i], expiry);
            tabu_until[i] = expiry;
            for &(j, w) in &self.adj[i] {
                let old = flip_delta(x[j], gradient[j]);
                gradient[j] += w * sign;
                buckets.update(j, old, flip_delta(x[j], gradient[j]), step >= tabu_until[j]);
            }
            if value < best.value {
                last_best = step;
                best.value = value;
                best.x.copy_from_slice(&x);
            }
        }
        best
    }
}

fn elite_insert(elite: &mut Vec<Point>, child: Point) {
    if let Some(i) = elite.iter().position(|p| {
        p.x.iter()
            .zip(&child.x)
            .filter(|(a, b)| a != b)
            .take(8)
            .count()
            < 8
    }) {
        if child.value >= elite[i].value {
            return;
        }
        elite.remove(i);
    }
    elite.push(child);
    elite.sort_by(|a, b| a.value.total_cmp(&b.value));
    elite.truncate(ELITE_SIZE);
}

fn worker(model: &Model, start: &Point, seed: u64, deadline: Instant) -> Point {
    let mut rng = Rng(seed);
    let n = model.linear.len();
    let mut elite = vec![start.clone()];
    let mut trials = 0;
    while Instant::now() < deadline {
        let x = if elite.len() >= 2 && trials % 4 != 0 {
            // Preserve the tested restart sequence without merging parent solutions.
            rng.below(elite.len() - 1);
            let mut x = elite[0].x.clone();
            for _ in 0..(3 + rng.below(20)) {
                x[rng.below(n)] ^= 1;
            }
            x
        } else {
            (0..n).map(|_| rng.below(2) as u8).collect()
        };
        elite_insert(&mut elite, model.tabu(model.point(x), &mut rng, deadline));
        trials += 1;
    }
    elite.swap_remove(0)
}

pub(crate) fn warm_start(
    qubo: &Qubo,
    incumbent: &Array1<usize>,
    seed: usize,
    max_workers: usize,
    budget: Duration,
) -> Option<Array1<usize>> {
    let deadline = Instant::now() + budget;
    let model = Model::new(qubo)?;
    if incumbent.len() != qubo.num_x()
        || incumbent.iter().any(|&v| v > 1)
        || Instant::now() >= deadline
    {
        return None;
    }
    let start = model.point(incumbent.iter().map(|&v| v as u8).collect());
    let workers = max_workers
        .clamp(1, MAX_WORKERS)
        .min(rayon::current_num_threads());
    let best = (0..workers)
        .into_par_iter()
        .map(|i| (i, worker(&model, &start, worker_seed(seed, i), deadline)))
        .min_by(|a, b| a.1.value.total_cmp(&b.1.value).then_with(|| a.0.cmp(&b.0)))?
        .1;
    if best.value >= start.value {
        return None;
    }
    let x = Array1::from_iter(best.x.into_iter().map(usize::from));
    (qubo.eval_usize(&x) == best.value && best.value < qubo.eval_usize(incumbent)).then_some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chain(weight: f64) -> Qubo {
        Qubo::from_vec(
            (0..127).collect(),
            (1..128).collect(),
            vec![weight; 127],
            vec![-2.0; 128],
            128,
        )
    }

    #[test]
    fn optimizer_only_returns_verified_improvements() {
        let q = chain(-4.0);
        let zeros = Array1::zeros(q.num_x());
        let candidate = warm_start(&q, &zeros, 2, 1, Duration::from_millis(100)).unwrap();
        assert!(candidate.iter().all(|&v| v <= 1));
        assert!(q.eval_usize(&candidate) < q.eval_usize(&zeros));
        assert!(warm_start(&q, &Array1::ones(q.num_x()), 2, 1, Duration::from_millis(5)).is_none());
    }

    #[test]
    fn unsupported_coefficients_and_exhausted_budget_are_no_ops() {
        let zeros = Array1::zeros(128);
        assert!(warm_start(&chain(1.5), &zeros, 2, 1, Duration::from_millis(5)).is_none());
        assert!(warm_start(&chain(-4.0), &zeros, 2, 1, Duration::ZERO).is_none());
    }
}
