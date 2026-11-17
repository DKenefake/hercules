//! Exact binary bucket elimination for small-width residuals. Failure is a no-op.
use crate::preprocess::low_degree::add;
use crate::{qubo::Qubo, FixedVarMap};
use ndarray::Array1;
use std::time::{Duration, Instant};

pub(crate) const MAX_FREE: usize = 128;
const MAX_WIDTH: usize = 8;
const MAX_ENTRIES: usize = 32_768;

#[derive(Debug, PartialEq)]
pub(crate) enum Declined {
    Width,
    Budget,
    Arithmetic,
}

struct Factor {
    variables: Vec<usize>,
    values: Vec<f64>,
}

fn exact_add(a: f64, b: f64) -> Result<f64, Declined> {
    add(a, b).ok_or(Declined::Arithmetic)
}

fn check(start: Instant, budget: Duration) -> Result<(), Declined> {
    if start.elapsed() >= budget {
        Err(Declined::Budget)
    } else {
        Ok(())
    }
}

pub(crate) fn solve(
    qubo: &Qubo,
    fixed: &FixedVarMap,
    budget: Duration,
) -> Result<Array1<usize>, Declined> {
    let start = Instant::now();
    check(start, budget)?;
    let free: Vec<_> = (0..qubo.num_x())
        .filter(|i| !fixed.contains_key(i))
        .collect();
    let n = free.len();
    if n > MAX_FREE {
        return Err(Declined::Width);
    }
    let mut index = vec![usize::MAX; qubo.num_x()];
    let mut linear = Vec::with_capacity(n);
    for (i, &original) in free.iter().enumerate() {
        index[original] = i;
        linear.push(exact_add(0.0, qubo.c[original])?);
    }
    let mut quadratic = vec![vec![0.0; n]; n];
    for (entry, (&value, (a, b))) in qubo.q.iter().enumerate() {
        if entry % 256 == 0 {
            check(start, budget)?;
        }
        let weight = value * 0.5;
        if !weight.is_finite() || weight * 2.0 != value {
            return Err(Declined::Arithmetic);
        }
        let (i, j) = (index[a], index[b]);
        match (i == usize::MAX, j == usize::MAX) {
            (true, true) => {}
            (true, false) if fixed[&a] == 1 => linear[j] = exact_add(linear[j], weight)?,
            (false, true) if fixed[&b] == 1 => linear[i] = exact_add(linear[i], weight)?,
            (false, false) if i == j => linear[i] = exact_add(linear[i], weight)?,
            (false, false) => {
                let (lo, hi) = (i.min(j), i.max(j));
                quadratic[lo][hi] = exact_add(quadratic[lo][hi], weight)?;
            }
            _ => {}
        }
    }
    let mut graph = vec![0u128; n];
    for i in 0..n {
        for j in i + 1..n {
            if quadratic[i][j] != 0.0 {
                graph[i] |= 1u128 << j;
                graph[j] |= 1u128 << i;
            }
        }
    }
    let mut active = vec![true; n];
    let mut order = Vec::with_capacity(n);
    let mut entries = 0;
    for _ in 0..n {
        check(start, budget)?;
        let v = (0..n)
            .filter(|&i| active[i])
            .min_by_key(|&i| (graph[i].count_ones(), i))
            .unwrap();
        if graph[v].count_ones() as usize > MAX_WIDTH {
            return Err(Declined::Width);
        }
        let neighbors: Vec<_> = (0..n).filter(|&i| graph[v] & (1u128 << i) != 0).collect();
        entries += 1usize << neighbors.len();
        if entries > MAX_ENTRIES {
            return Err(Declined::Budget);
        }
        for &w in &neighbors {
            graph[w] = (graph[w] | graph[v]) & !(1u128 << w) & !(1u128 << v);
        }
        active[v] = false;
        order.push((v, neighbors));
    }
    let mut rank = vec![0; n];
    for (position, (v, _)) in order.iter().enumerate() {
        rank[*v] = position;
    }
    let mut buckets: Vec<Vec<Factor>> = (0..n).map(|_| Vec::new()).collect();
    for i in 0..n {
        buckets[rank[i]].push(Factor {
            variables: vec![i],
            values: vec![0.0, linear[i]],
        });
        for j in i + 1..n {
            if quadratic[i][j] != 0.0 {
                buckets[rank[i].min(rank[j])].push(Factor {
                    variables: vec![i, j],
                    values: vec![0.0, 0.0, 0.0, quadratic[i][j]],
                });
            }
        }
    }
    let mut choices = Vec::with_capacity(n);
    let mut optimum = 0.0;
    for (position, (v, neighbors)) in order.iter().enumerate() {
        check(start, budget)?;
        let bucket = std::mem::take(&mut buckets[position]);
        let projections: Vec<Vec<usize>> = bucket
            .iter()
            .map(|factor| {
                factor
                    .variables
                    .iter()
                    .map(|i| {
                        if i == v {
                            neighbors.len()
                        } else {
                            neighbors.iter().position(|j| j == i).unwrap()
                        }
                    })
                    .collect()
            })
            .collect();
        let size = 1usize << neighbors.len();
        let mut table = Vec::with_capacity(size);
        let mut selected = Vec::with_capacity(size);
        for mask in 0..size {
            if mask % 16 == 0 {
                check(start, budget)?;
            }
            let mut costs = [0.0; 2];
            for (value, cost) in costs.iter_mut().enumerate() {
                let assignment = mask | (value << neighbors.len());
                for (factor, projection) in bucket.iter().zip(&projections) {
                    let offset = projection
                        .iter()
                        .enumerate()
                        .fold(0, |bits, (b, p)| bits | (((assignment >> p) & 1) << b));
                    *cost = exact_add(*cost, factor.values[offset])?;
                }
            }
            let value = usize::from(costs[1] < costs[0]);
            selected.push(value);
            table.push(costs[value]);
        }
        choices.push(selected);
        if neighbors.is_empty() {
            optimum = exact_add(optimum, table[0])?;
        } else {
            let next = neighbors.iter().map(|i| rank[*i]).min().unwrap();
            buckets[next].push(Factor {
                variables: neighbors.clone(),
                values: table,
            });
        }
    }
    let mut x = Array1::zeros(qubo.num_x());
    for (&i, &value) in fixed {
        x[i] = value;
    }
    for ((v, neighbors), selected) in order.iter().zip(&choices).rev() {
        let mask = neighbors
            .iter()
            .enumerate()
            .fold(0, |bits, (b, i)| bits | (x[free[*i]] << b));
        x[free[*v]] = selected[mask];
    }
    // Check reconstruction in the same exact residual polynomial, excluding
    // the fixed-only constant, which cannot affect its minimizer.
    let mut value = 0.0;
    for i in 0..n {
        if x[free[i]] == 0 {
            continue;
        }
        value = exact_add(value, linear[i])?;
        for j in i + 1..n {
            if x[free[j]] == 1 {
                value = exact_add(value, quadratic[i][j])?;
            }
        }
    }
    check(start, budget)?;
    if value != optimum {
        return Err(Declined::Arithmetic);
    }
    Ok(x)
}
