//! Bounded exact certificate for zero-field, equal-magnitude layered strips.
use super::PreparedCycleBound;
use std::collections::VecDeque;
use std::time::{Duration, Instant};

const MAX_WIDTH: usize = 18;
const MAX_ENTRIES: usize = 185_000_000;
type Graph = Vec<Vec<(usize, bool)>>;

struct Step {
    remove: usize,
    neighbor: [usize; 2],
    different: [usize; 2],
    bits_after: usize,
}

pub(crate) struct LayeredCertificate {
    pub(crate) lower: f64,
    pub(crate) width: usize,
    pub(crate) entries: usize,
}

impl PreparedCycleBound {
    pub(crate) fn layered_root_certificate(&self, budget: Duration) -> Option<LayeredCertificate> {
        let deadline = Instant::now() + budget;
        let n = self.fields.len();
        if !(6..=2000).contains(&n) || self.fields.iter().any(|&h| h != 0) {
            return None;
        }
        let magnitude = self.edges.first()?.1.abs();
        if self.edges.iter().any(|&(_, w)| w.abs() != magnitude) {
            return None;
        }
        let mut graph = vec![Vec::new(); n];
        for &((u, v), w) in &self.edges {
            graph[u].push((v, w > 0));
            graph[v].push((u, w > 0));
        }
        let (width, steps, entries) = plan(&graph)?;
        let violations = minimize(width, &steps, deadline)?;
        let scaled =
            self.constant + magnitude * (2 * i128::from(violations) - self.edges.len() as i128);
        (scaled.abs() <= 1i128 << 52).then_some(LayeredCertificate {
            lower: scaled as f64 / 64.0,
            width,
            entries,
        })
    }
}

fn layers(graph: &Graph) -> Option<(Vec<Vec<usize>>, Vec<usize>)> {
    let n = graph.len();
    if graph.iter().any(|row| !matches!(row.len(), 2 | 4)) {
        return None;
    }
    let start = graph.iter().position(|row| row.len() == 2)?;
    let mut dist = vec![usize::MAX; n];
    let mut first = VecDeque::from([start]);
    dist[start] = 0;
    let mut work = VecDeque::new();
    // At one end of a strip, boundary vertices share neighbors in the next layer.
    while let Some(u) = first.pop_front() {
        work.push_back(u);
        if work.len() > MAX_WIDTH {
            return None;
        }
        for &(v, _) in &graph[u] {
            for &(w, _) in &graph[v] {
                if graph[w].len() == 2 && dist[w] == usize::MAX {
                    dist[w] = 0;
                    first.push_back(w);
                }
            }
        }
    }
    let width = work.len();
    if width < 3 {
        return None;
    }
    while let Some(u) = work.pop_front() {
        for &(v, _) in &graph[u] {
            if dist[v] == usize::MAX {
                dist[v] = dist[u] + 1;
                work.push_back(v);
            }
        }
    }
    let last = *dist.iter().max()?;
    if last == usize::MAX || last < 2 || (last + 1) * width != n {
        return None;
    }
    let mut rows = vec![Vec::with_capacity(width); last + 1];
    for (u, &d) in dist.iter().enumerate() {
        let (mut inward, mut outward) = (0, 0);
        for &(v, _) in &graph[u] {
            if dist[v] + 1 == d {
                inward += 1;
            } else if dist[v] == d + 1 {
                outward += 1;
            } else {
                return None;
            }
        }
        if inward != 2 * usize::from(d > 0) || outward != 2 * usize::from(d < last) {
            return None;
        }
        rows[d].push(u);
    }
    rows.iter()
        .all(|row| row.len() == width)
        .then_some((rows, dist))
}

fn plan(graph: &Graph) -> Option<(usize, Vec<Step>, usize)> {
    let (rows, dist) = layers(graph)?;
    let width = rows[0].len();
    let mut steps = Vec::with_capacity(graph.len());
    let mut frontier = rows[0].clone();
    let mut seen = vec![false; graph.len()];
    let mut entries = 0;
    for (level, row) in rows.iter().take(rows.len() - 1).enumerate() {
        // Each adjacent layer pair is a union of even cycles. Follow a cycle
        // before starting the next, so at most one extra bit stays in the frontier.
        for &start in row {
            if seen[start] {
                continue;
            }
            let mut u = start;
            let mut previous = None;
            loop {
                if seen[u] {
                    return None;
                }
                seen[u] = true;
                let mut neighbors = graph[u]
                    .iter()
                    .copied()
                    .filter(|&(v, _)| dist[v] == level + 1);
                let adj = [neighbors.next()?, neighbors.next()?];
                let remove = frontier.iter().position(|&v| v == u)?;
                frontier.remove(remove);
                let mut neighbor = [0; 2];
                for (j, &(v, _)) in adj.iter().enumerate() {
                    neighbor[j] = frontier.iter().position(|&w| w == v).unwrap_or_else(|| {
                        frontier.push(v);
                        frontier.len() - 1
                    });
                }
                let bits_after = frontier.len();
                if bits_after > MAX_WIDTH + 1 {
                    return None;
                }
                entries += 1 << (bits_after - 1);
                if entries > MAX_ENTRIES {
                    return None;
                }
                steps.push(Step {
                    remove,
                    neighbor,
                    different: adj.map(|(_, preferred)| usize::from(preferred)),
                    bits_after,
                });
                let &(v, _) = adj.iter().find(|&&(v, _)| Some(v) != previous)?;
                let next = graph[v]
                    .iter()
                    .find(|&&(w, _)| dist[w] == level && w != u)?
                    .0;
                if next == start {
                    break;
                }
                previous = Some(v);
                u = next;
            }
        }
        if frontier.len() != width || frontier.iter().any(|&v| dist[v] != level + 1) {
            return None;
        }
    }
    Some((width, steps, entries))
}

fn minimize(width: usize, steps: &[Step], deadline: Instant) -> Option<u16> {
    let capacity = 1 << MAX_WIDTH;
    let mut old = Vec::with_capacity(capacity);
    let mut next = Vec::with_capacity(capacity);
    // Zero fields give complement symmetry, so store just the states whose
    // highest frontier bit is zero. The other half is its bitwise complement.
    old.resize(1 << (width - 1), 0_u16);
    for step in steps {
        if Instant::now() >= deadline {
            return None;
        }
        let half = 1 << step.remove;
        let remaining_mask = old.len() - 1;
        let old_mask = 2 * old.len() - 1;
        let canonical = |mask: usize| {
            if mask > remaining_mask {
                old_mask ^ mask
            } else {
                mask
            }
        };
        next.resize(1 << (step.bits_after - 1), 0);
        let [p, q] = step.neighbor;
        let [a, b] = step.different;
        for (mask, result) in next.iter_mut().enumerate() {
            let shared = mask & remaining_mask;
            let source = (shared & (half - 1)) | ((shared & !(half - 1)) << 1);
            let cost_zero = ((((mask >> p) & 1) ^ a) + (((mask >> q) & 1) ^ b)) as u16;
            *result = (old[canonical(source)] + cost_zero)
                .min(old[canonical(source | half)] + 2 - cost_zero);
        }
        std::mem::swap(&mut old, &mut next);
    }
    if Instant::now() >= deadline {
        return None;
    }
    old.into_iter().min()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qubo::Qubo;
    use ndarray::Array1;

    #[test]
    fn strip_certificate_matches_exhaustive_signed_minimum() {
        let (width, height) = (3, 5);
        let n = width * height;
        let mut q = sprs::TriMat::new((n, n));
        let mut c = Array1::zeros(n);
        for r in 0..height - 1 {
            for j in 0..width {
                for k in [j, (j + 1) % width] {
                    let (u, v) = (r * width + j, (r + 1) * width + k);
                    let w = if (r * 5 + j * 3 + k) % 7 < 4 {
                        4.0
                    } else {
                        -4.0
                    };
                    q.add_triplet(u, v, w);
                    c[u] -= w / 4.0;
                    c[v] -= w / 4.0;
                }
            }
        }
        let qubo = Qubo::new_with_c(q.to_csr(), c);
        let model = PreparedCycleBound::new(&qubo).unwrap();
        let certificate = model
            .layered_root_certificate(Duration::from_secs(1))
            .unwrap();
        let expected = (0..1 << n)
            .map(|mask| qubo.eval_usize(&Array1::from_iter((0..n).map(|i| (mask >> i) & 1))))
            .fold(f64::INFINITY, f64::min);
        assert_eq!(certificate.width, width);
        assert!((certificate.lower - expected).abs() < 1e-9);
        assert!(model.layered_root_certificate(Duration::ZERO).is_none());
        let mut with_field = qubo;
        with_field.c[0] += 2.0;
        assert!(PreparedCycleBound::new(&with_field)
            .unwrap()
            .layered_root_certificate(Duration::from_secs(1))
            .is_none());
    }
}
