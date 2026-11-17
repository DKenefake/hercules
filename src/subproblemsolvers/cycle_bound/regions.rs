//! Offline exact patch bounds with disjoint edge-cost allocation.
use super::{pack, PackingStrategy, PreparedCycleBound};
use crate::FixedVarMap;
use std::collections::BTreeSet;
use std::time::{Duration, Instant};

#[derive(Debug, Default)]
pub struct Screen {
    pub lower: f64,
    pub candidates: usize,
    pub exact_completed: usize,
    pub accepted: usize,
    pub local_gain: f64,
    pub exact_micros: u128,
    pub elapsed_micros: u128,
}

pub fn screen(
    prepared: &PreparedCycleBound,
    fixed: &FixedVarMap,
    size: usize,
    budget: Duration,
) -> Option<Screen> {
    screen_with_share(prepared, fixed, size, budget, false)
}

pub fn screen_with_share(
    prepared: &PreparedCycleBound,
    fixed: &FixedVarMap,
    size: usize,
    budget: Duration,
    half: bool,
) -> Option<Screen> {
    if size != 0 && !(3..=16).contains(&size) {
        return None;
    }
    let start = Instant::now();
    let n = prepared.fields.len();
    if fixed.iter().any(|(&i, &v)| i >= n || v > 1) {
        return None;
    }
    let spin = |i: usize| 2 * fixed[&i] as i128 - 1;
    let mut constant = prepared.constant;
    let mut fields = prepared.fields.clone();
    let mut edges = Vec::new();
    for &((u, v), weight) in &prepared.edges {
        match (fixed.contains_key(&u), fixed.contains_key(&v)) {
            (false, false) => edges.push(((u, v), weight)),
            (true, false) => fields[v] += weight * spin(u),
            (false, true) => fields[u] += weight * spin(v),
            (true, true) => constant += weight * spin(u) * spin(v),
        }
    }
    for (i, h) in fields.into_iter().enumerate() {
        if fixed.contains_key(&i) {
            constant += h * spin(i);
        } else if h != 0 {
            edges.push(((i, n), h));
        }
    }
    // Match the production bound's compact free-variable ordering exactly.
    let mut index = vec![usize::MAX; n + 1];
    let mut free_count = 0;
    for (i, entry) in index.iter_mut().take(n).enumerate() {
        if !fixed.contains_key(&i) {
            *entry = free_count;
            free_count += 1;
        }
    }
    index[n] = free_count;
    for ((u, v), _) in &mut edges {
        *u = index[*u];
        *v = index[*v];
    }
    let n = free_count;
    let mut adjacency = vec![Vec::new(); n + 1];
    for (e, &((u, v), _)) in edges.iter().enumerate() {
        adjacency[u].push((v, e));
        adjacency[v].push((u, e));
    }
    let mut seeds: Vec<_> = (0..=n).filter(|&i| adjacency[i].len() >= 2).collect();
    seeds.sort_by_key(|&i| (std::cmp::Reverse(adjacency[i].len()), i));
    let count = if size == 0 { 0 } else { seeds.len().min(64) };
    let mut seen = BTreeSet::new();
    let mut result = Screen::default();
    let patch_budget = budget.saturating_sub((budget / 4).min(Duration::from_millis(5)));
    for trial in 0..count {
        if start.elapsed() >= patch_budget {
            break;
        }
        let seed = seeds[trial * seeds.len() / count];
        let mut vertices = vec![seed];
        let mut selected = vec![false; n + 1];
        selected[seed] = true;
        let mut links = vec![0usize; n + 1];
        let mut strength = vec![0i128; n + 1];
        while vertices.len() < size {
            if start.elapsed() >= patch_budget {
                break;
            }
            let last = *vertices.last().unwrap();
            for &(v, e) in &adjacency[last] {
                if edges[e].1 != 0 {
                    links[v] += 1;
                    strength[v] += edges[e].1.abs();
                }
            }
            let next = (0..=n)
                .filter(|&i| !selected[i] && links[i] > 0)
                .max_by_key(|&i| (links[i], strength[i], std::cmp::Reverse(i)));
            let Some(next) = next else {
                break;
            };
            selected[next] = true;
            vertices.push(next);
        }
        vertices.sort_unstable();
        if vertices.len() < 3 || !seen.insert(vertices.clone()) {
            continue;
        }
        let mut local_index = vec![usize::MAX; n + 1];
        for (i, &v) in vertices.iter().enumerate() {
            local_index[v] = i;
        }
        let ids: Vec<_> = edges
            .iter()
            .enumerate()
            .filter(|(_, ((u, v), w))| *w != 0 && selected[*u] && selected[*v])
            .map(|(e, _)| e)
            .collect();
        // Trees and unicyclic patches cannot improve on an exact cycle bound.
        if ids.len() <= vertices.len() {
            continue;
        }
        let local: Vec<_> = ids
            .iter()
            .map(|&e| {
                let ((u, v), w) = edges[e];
                (
                    (local_index[u], local_index[v]),
                    if half { w / 2 } else { w },
                )
            })
            .collect();
        result.candidates += 1;
        let exact_start = Instant::now();
        let minimum = exact_minimum(vertices.len(), &local, start, patch_budget);
        result.exact_micros += exact_start.elapsed().as_micros();
        let Some(minimum) = minimum else {
            break;
        };
        result.exact_completed += 1;
        let local_start = Instant::now();
        let local_budget = patch_budget
            .saturating_sub(start.elapsed())
            .min(Duration::from_millis(1));
        let greedy = pack(
            vertices.len() - 1,
            0,
            &local,
            local_start,
            local_budget / 2,
            f64::INFINITY,
            PackingStrategy::ExtendedGreedy,
            0,
            None,
        )?;
        let fractional = pack(
            vertices.len() - 1,
            0,
            &local,
            local_start,
            local_budget,
            f64::INFINITY,
            PackingStrategy::FractionalPortfolio,
            0,
            None,
        )?;
        let cycle_lower = greedy.lower.max(fractional.lower);
        if minimum as f64 / 64.0 <= cycle_lower {
            continue;
        }
        result.local_gain += minimum as f64 / 64.0 - cycle_lower;
        result.accepted += 1;
        constant += minimum;
        for (e, (_, allocated)) in ids.into_iter().zip(local) {
            edges[e].1 -= allocated;
        }
    }
    edges.retain(|(_, w)| *w != 0);
    let remaining = budget.saturating_sub(start.elapsed());
    let residual_start = Instant::now();
    let greedy = pack(
        n,
        constant,
        &edges,
        residual_start,
        remaining / 2,
        f64::INFINITY,
        PackingStrategy::ExtendedGreedy,
        0,
        None,
    )?;
    let fractional = pack(
        n,
        constant,
        &edges,
        residual_start,
        remaining,
        f64::INFINITY,
        PackingStrategy::FractionalPortfolio,
        0,
        None,
    )?;
    result.lower = greedy.lower.max(fractional.lower);
    result.elapsed_micros = start.elapsed().as_micros();
    Some(result)
}

fn exact_minimum(
    n: usize,
    edges: &[((usize, usize), i128)],
    start: Instant,
    budget: Duration,
) -> Option<i128> {
    // Every term is a spin product, including anchor edges. Fixing one spin
    // removes global complement symmetry without changing the minimum.
    let mut spins = vec![-1i128; n];
    spins[0] = 1;
    let mut fields = vec![0i128; n];
    let mut adjacency = vec![Vec::new(); n];
    let mut energy = 0;
    for &((u, v), w) in edges {
        energy += w * spins[u] * spins[v];
        fields[u] += w * spins[v];
        fields[v] += w * spins[u];
        adjacency[u].push((v, w));
        adjacency[v].push((u, w));
    }
    let mut best = energy;
    for step in 1..1usize << (n - 1) {
        if step % 256 == 1 && start.elapsed() >= budget {
            return None;
        }
        let i = step.trailing_zeros() as usize + 1;
        let change = -2 * spins[i];
        energy += change * fields[i];
        spins[i] = -spins[i];
        for &(j, w) in &adjacency[i] {
            fields[j] += change * w;
        }
        best = best.min(energy);
    }
    Some(best)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qubo::Qubo;
    use ndarray::Array1;

    #[test]
    fn complete_frustrated_patch_strictly_strengthens_cycles() {
        let mut matrix = sprs::TriMat::new((5, 5));
        for i in 0..5 {
            for j in 0..5 {
                if i != j {
                    matrix.add_triplet(i, j, 2.0);
                }
            }
        }
        let q = Qubo::new_with_c(matrix.to_csr(), Array1::from_elem(5, -4.0));
        let prepared = PreparedCycleBound::new(&q).unwrap();
        let fixed = FixedVarMap::default();
        let ordinary = prepared
            .conditioned(&(0..5).collect::<Vec<_>>(), &fixed, Duration::from_secs(1))
            .unwrap();
        let region = screen(&prepared, &fixed, 8, Duration::from_secs(1)).unwrap();
        assert_eq!(region.lower, -6.0);
        assert!(region.lower > ordinary.lower);
        assert_eq!(region.accepted, 1);
    }

    #[test]
    fn conditioned_regions_never_exceed_exhaustive_optima() {
        for seed in 0..16 {
            let n = 7;
            let mut matrix = sprs::TriMat::new((n, n));
            for i in 0..n {
                for j in 0..n {
                    matrix.add_triplet(i, j, ((seed * 7 + i * 13 + j * 3) % 11) as f64 - 5.0);
                }
            }
            let q = Qubo::new_with_c(
                matrix.to_csr(),
                Array1::from_iter((0..n).map(|i| i as f64 - 3.0)),
            );
            let prepared = PreparedCycleBound::new(&q).unwrap();
            for fixed in [
                FixedVarMap::default(),
                [(0, 0), (4, 1)].into_iter().collect(),
                (0..n).map(|i| (i, i % 2)).collect(),
            ] {
                let free: Vec<_> = (0..n).filter(|i| !fixed.contains_key(i)).collect();
                let baseline = prepared
                    .conditioned(&free, &fixed, Duration::from_secs(1))
                    .unwrap();
                let control = screen(&prepared, &fixed, 0, Duration::from_secs(1)).unwrap();
                assert_eq!(control.lower, baseline.lower);
                for size in [3, 5, 8] {
                    for half in [false, true] {
                        let result = screen_with_share(
                            &prepared,
                            &fixed,
                            size,
                            Duration::from_secs(1),
                            half,
                        )
                        .unwrap();
                        let optimum = (0..1usize << n)
                            .filter(|mask| fixed.iter().all(|(&i, &v)| (mask >> i) & 1 == v))
                            .map(|mask| {
                                q.eval_usize(&Array1::from_iter((0..n).map(|i| (mask >> i) & 1)))
                            })
                            .fold(f64::INFINITY, f64::min);
                        assert!(
                            result.lower <= optimum,
                            "{seed} {size}: {result:?} > {optimum}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn interrupted_enumeration_is_not_a_lower_bound() {
        assert!(exact_minimum(5, &[((0, 1), 1)], Instant::now(), Duration::ZERO).is_none());
    }
}
