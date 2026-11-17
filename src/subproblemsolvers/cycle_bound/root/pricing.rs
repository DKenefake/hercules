use super::Edges;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};
use std::time::Instant;

const ONE: u32 = 1 << 20;
const LIMIT: u32 = ONE - 32;

pub(super) fn add_cycles(
    graph: &Edges,
    prices: &[f64],
    cycles: &mut Vec<Vec<usize>>,
    round: usize,
    deadline: Instant,
) -> usize {
    let n = graph.len();
    let count = n.min(384);
    let price: Vec<_> = prices
        .iter()
        .map(|x| (x * ONE as f64).ceil().clamp(1.0, ONE as f64) as u32)
        .collect();
    let mut known: HashSet<_> = cycles.iter().cloned().collect();
    let mut queue = BinaryHeap::new();
    let mut distance = vec![u32::MAX; 2 * n];
    let mut predecessor = vec![(0, 0); 2 * n];
    let mut path = Vec::with_capacity(2 * n);
    let mut position = vec![usize::MAX; n];
    let mut found = Vec::new();
    let mut polls = 0;
    'sources: for j in 0..count {
        if Instant::now() >= deadline {
            break;
        }
        let source = (j * n / count + round) % n;
        let target = source + n;
        distance.fill(u32::MAX);
        queue.clear();
        distance[source] = 0;
        queue.push(Reverse((0, source)));
        while let Some(Reverse((d, state))) = queue.pop() {
            polls += 1;
            if polls % 256 == 0 && Instant::now() >= deadline {
                break 'sources;
            }
            if d != distance[state] {
                continue;
            }
            if state == target {
                // A shortest lifted path never repeats a lifted vertex. Its
                // first repeated original vertex therefore encloses a simple
                // cycle in opposite parity layers, which is frustrated.
                path.clear();
                position.fill(usize::MAX);
                let mut v = target;
                loop {
                    let old = position[v % n];
                    if old != usize::MAX {
                        let walk = &path[old..];
                        if (3..=40).contains(&walk.len()) {
                            let mut cycle: Vec<_> =
                                walk.iter().map(|&p: &usize| predecessor[p].1).collect();
                            let cost = cycle.iter().map(|&e| price[e]).sum::<u32>();
                            cycle.sort_unstable();
                            if known.insert(cycle.clone()) {
                                found.push((cost, cycle));
                            }
                        }
                        break;
                    }
                    position[v % n] = path.len();
                    path.push(v);
                    v = predecessor[v].0;
                }
                break;
            }
            for &(neighbor, edge, odd) in &graph[state % n] {
                let next = neighbor + n * ((state >= n) ^ odd) as usize;
                let next_d = d + price[edge];
                if next_d < LIMIT && next_d < distance[next] {
                    distance[next] = next_d;
                    predecessor[next] = (state, edge);
                    queue.push(Reverse((next_d, next)));
                }
            }
        }
    }
    found.sort_unstable();
    let added = found.len().min(200);
    cycles.extend(found.into_iter().take(added).map(|(_, cycle)| cycle));
    added
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn pricing_adds_only_new_frustrated_low_price_cycles() {
        let mut graph = vec![Vec::new(); 5];
        for (e, (u, v, odd)) in [
            (0, 1, true),
            (1, 2, false),
            (2, 0, false),
            (2, 3, false),
            (3, 4, false),
            (4, 2, false),
        ]
        .into_iter()
        .enumerate()
        {
            graph[u].push((v, e, odd));
            graph[v].push((u, e, odd));
        }
        let mut pool = Vec::new();
        let deadline = Instant::now() + Duration::from_secs(1);
        assert_eq!(add_cycles(&graph, &[0.1; 6], &mut pool, 0, deadline), 1);
        assert_eq!(pool, vec![vec![0, 1, 2]]);
        assert_eq!(add_cycles(&graph, &[0.1; 6], &mut pool, 0, deadline), 0);
        assert_eq!(
            add_cycles(&graph, &[0.4; 6], &mut Vec::new(), 0, deadline),
            0
        );
    }
}
