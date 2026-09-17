//! Dinic flow for the roof-dual network's symmetric, nonnegative capacities.
//! Search buffers are allocated once per network, not once per augmenting path.

const NONE: usize = usize::MAX;

struct Arc {
    target: usize,
    next: usize,
    residual: f64,
}

pub(crate) struct FlowNetwork {
    head: Vec<usize>,
    arcs: Vec<Arc>,
}

impl FlowNetwork {
    pub(crate) fn new(nodes: usize, undirected_edges: usize) -> Self {
        Self {
            head: vec![NONE; nodes],
            arcs: Vec::with_capacity(2 * undirected_edges),
        }
    }

    pub(crate) fn add_edge(&mut self, u: usize, v: usize, capacity: f64) {
        // Adjacent arcs are each other's reverse. Both directions start with
        // capacity because every roof-dual edge is undirected.
        self.arcs.push(Arc {
            target: v,
            next: self.head[u],
            residual: capacity,
        });
        self.head[u] = self.arcs.len() - 1;
        self.arcs.push(Arc {
            target: u,
            next: self.head[v],
            residual: capacity,
        });
        self.head[v] = self.arcs.len() - 1;
    }

    /// Add a roof-dual edge and its complemented copy as one quartet of arcs.
    pub(super) fn add_literal_edge(&mut self, u: usize, v: usize, capacity: f64) {
        self.add_edge(u, v, capacity);
        self.add_edge(u ^ 1, v ^ 1, capacity);
    }

    /// Compatible weak labeling of a roof-dual residual network.
    ///
    /// Boros, Hammer, Tavares (RRR 10-2006), Section 3.3.2. Dinic need not
    /// produce a symmetric flow: average each residual arc with its reversed,
    /// complemented mate before computing SCCs. All input edges must have been
    /// added with add_literal_edge. Self-complementary SCCs remain unlabeled.
    pub(super) fn weak_labels(&mut self, source: usize) -> Vec<Option<usize>> {
        for quartet in self.arcs.chunks_exact_mut(4) {
            let forward = 0.5 * quartet[0].residual + 0.5 * quartet[3].residual;
            let backward = 0.5 * quartet[1].residual + 0.5 * quartet[2].residual;
            quartet[0].residual = forward;
            quartet[3].residual = forward;
            quartet[1].residual = backward;
            quartet[2].residual = backward;
        }

        // The artificial root must be true, even if it is isolated. This
        // ordering-only arc cannot create a cycle after an exact maximum flow.
        self.add_edge(source ^ 1, source, 0.0);
        let last = self.arcs.len() - 2;
        self.arcs[last].residual = 1.0;
        let components = self.residual_components();
        if components[source] == components[source ^ 1] {
            // Floating-point residuals may still join source and sink. Do not
            // derive weak persistencies from an inconsistent root orientation.
            return vec![None; self.head.len() / 2];
        }
        // C -> D also implies complement(D) -> complement(C). Choosing the
        // later component of each pair therefore makes the true set closed
        // under outgoing arcs, including arcs next to a self-complementary SCC.
        // All residual terms touching fixed variables then vanish. Independent
        // per-variable tie choices would not have this autarky property.
        (0..self.head.len())
            .step_by(2)
            .map(|positive| {
                let (a, b) = (components[positive], components[positive ^ 1]);
                (a != b).then_some(usize::from(a > b))
            })
            .collect()
    }

    /// Kosaraju with explicit stacks. Component IDs are source-to-sink ordered.
    /// Reverse adjacency is already available through the paired residual arcs.
    fn residual_components(&self) -> Vec<usize> {
        let n = self.head.len();
        let mut seen = vec![false; n];
        let mut order = Vec::with_capacity(n);
        let mut path = Vec::with_capacity(n);
        for root in 0..n {
            if seen[root] {
                continue;
            }
            seen[root] = true;
            path.push((root, self.head[root]));
            while let Some((node, edge)) = path.last_mut() {
                if *edge == NONE {
                    order.push(*node);
                    path.pop();
                    continue;
                }
                let arc = &self.arcs[*edge];
                *edge = arc.next;
                // Do not drop tiny positive arcs: doing so can invent fixings.
                if arc.residual > 0.0 && !seen[arc.target] {
                    seen[arc.target] = true;
                    path.push((arc.target, self.head[arc.target]));
                }
            }
        }
        let mut components = vec![NONE; n];
        let mut stack = Vec::with_capacity(n);
        let mut component = 0;
        for root in order.into_iter().rev() {
            if components[root] != NONE {
                continue;
            }
            components[root] = component;
            stack.push(root);
            while let Some(node) = stack.pop() {
                let mut edge = self.head[node];
                while edge != NONE {
                    let arc = &self.arcs[edge];
                    if self.arcs[edge ^ 1].residual > 0.0 && components[arc.target] == NONE {
                        components[arc.target] = component;
                        stack.push(arc.target);
                    }
                    edge = arc.next;
                }
            }
            component += 1;
        }
        components
    }

    fn build_levels(&self, source: usize, levels: &mut [usize], queue: &mut Vec<usize>) {
        levels.fill(NONE);
        queue.clear();
        queue.push(source);
        levels[source] = 0;
        let mut cursor = 0;
        while cursor < queue.len() {
            let node = queue[cursor];
            cursor += 1;
            let mut edge = self.head[node];
            while edge != NONE {
                let arc = &self.arcs[edge];
                if arc.residual > 0.0 && levels[arc.target] == NONE {
                    levels[arc.target] = levels[node] + 1;
                    queue.push(arc.target);
                }
                edge = arc.next;
            }
        }
    }

    pub(super) fn max_flow(&mut self, source: usize, sink: usize) -> f64 {
        self.flow_until(source, sink, |_| false).unwrap()
    }

    /// A candidate partition only. The caller must verify its capacity against
    /// the original edges before using it as a numerical reduction certificate.
    pub(crate) fn bounded_cut(
        &mut self,
        source: usize,
        sink: usize,
        limit: f64,
        deadline: std::time::Instant,
    ) -> Option<Vec<bool>> {
        self.flow_until(source, sink, |total| {
            total > limit || std::time::Instant::now() >= deadline
        })?;
        let side = self.reachable_from(source);
        (!side[sink]).then_some(side)
    }

    fn flow_until(
        &mut self,
        source: usize,
        sink: usize,
        mut stop: impl FnMut(f64) -> bool,
    ) -> Option<f64> {
        assert_ne!(source, sink);
        let n = self.head.len();
        let mut levels = vec![NONE; n];
        let mut current = vec![NONE; n];
        let mut queue = Vec::with_capacity(n);
        let mut path: Vec<usize> = Vec::with_capacity(n);
        let mut total = 0.0;

        loop {
            if stop(total) {
                return None;
            }
            self.build_levels(source, &mut levels, &mut queue);
            if levels[sink] == NONE {
                return Some(total);
            }
            current.copy_from_slice(&self.head);
            path.clear();
            let mut node = source;

            loop {
                if stop(total) {
                    return None;
                }
                if node == sink {
                    let amount = path
                        .iter()
                        .map(|&edge| self.arcs[edge].residual)
                        .fold(f64::INFINITY, f64::min);
                    for &edge in &path {
                        self.arcs[edge].residual -= amount;
                        self.arcs[edge ^ 1].residual += amount;
                    }
                    total += amount;
                    path.clear();
                    node = source;
                    continue;
                }

                while current[node] != NONE {
                    let arc = &self.arcs[current[node]];
                    if arc.residual > 0.0 && levels[arc.target] == levels[node] + 1 {
                        break;
                    }
                    current[node] = arc.next;
                }
                let edge = current[node];
                if edge != NONE {
                    path.push(edge);
                    node = self.arcs[edge].target;
                } else {
                    // A dead end cannot acquire a forward residual edge in this
                    // level graph. Keep current-arc positions across paths.
                    levels[node] = NONE;
                    if let Some(edge) = path.pop() {
                        node = self.arcs[edge ^ 1].target;
                        current[node] = self.arcs[edge].next;
                    } else {
                        break;
                    }
                }
            }
        }
    }

    pub(super) fn reachable_from(&self, source: usize) -> Vec<bool> {
        let mut seen = vec![false; self.head.len()];
        let mut queue = Vec::with_capacity(self.head.len());
        seen[source] = true;
        queue.push(source);
        let mut cursor = 0;
        while cursor < queue.len() {
            let node = queue[cursor];
            cursor += 1;
            let mut edge = self.head[node];
            while edge != NONE {
                let arc = &self.arcs[edge];
                if arc.residual > 1e-12 && !seen[arc.target] {
                    seen[arc.target] = true;
                    queue.push(arc.target);
                }
                edge = arc.next;
            }
        }
        seen
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph::algo::maximum_flow::dinics;
    use petgraph::graph::DiGraph;
    use petgraph::visit::EdgeRef;
    use smolprng::{JsfLarge, PRNG};

    #[test]
    fn bounded_cut_declines_deadlines_and_flow_limits() {
        let mut flow = FlowNetwork::new(3, 2);
        flow.add_edge(0, 1, 10.0);
        flow.add_edge(1, 2, 10.0);
        assert!(flow
            .bounded_cut(0, 2, 20.0, std::time::Instant::now())
            .is_none());
        assert!(flow
            .bounded_cut(
                0,
                2,
                5.0,
                std::time::Instant::now() + std::time::Duration::from_secs(1)
            )
            .is_none());
        // Restart from the original capacities, not a cutoff-aborted network.
        let mut flow = FlowNetwork::new(3, 2);
        flow.add_edge(0, 1, 10.0);
        flow.add_edge(1, 2, 10.0);
        let side = flow
            .bounded_cut(
                0,
                2,
                10.0,
                std::time::Instant::now() + std::time::Duration::from_secs(1),
            )
            .unwrap();
        assert!(side[0] && !side[2]);
    }

    #[test]
    fn weak_labeling_is_closed_and_residual_is_complement_symmetric() {
        let mut rng = PRNG {
            generator: JsfLarge::from(94121u64),
        };
        for _ in 0..200 {
            let mut graph = FlowNetwork::new(32, 120);
            for i in 0..16 {
                for j in i + 1..16 {
                    if rng.gen_f64() < 0.4 {
                        let parity = usize::from(rng.gen_f64() < 0.5);
                        graph.add_literal_edge(2 * i, 2 * j + parity, rng.gen_f64());
                    }
                }
            }
            graph.max_flow(0, 1);
            let labels = graph.weak_labels(0);
            if labels[0].is_none() {
                continue; // conservative fallback on floating-point residual paths
            }
            assert_eq!(labels[0], Some(1));
            for quartet in graph.arcs[..graph.arcs.len() - 2].chunks_exact(4) {
                assert_eq!(quartet[0].residual, quartet[3].residual);
                assert_eq!(quartet[1].residual, quartet[2].residual);
            }
            let is_true = |node: usize| labels[node / 2] == Some(1 - node % 2);
            for u in 0..32 {
                let mut edge = graph.head[u];
                while edge != NONE {
                    let arc = &graph.arcs[edge];
                    if is_true(u) && arc.residual > 0.0 {
                        assert!(is_true(arc.target), "selected literals must be closed");
                    }
                    edge = arc.next;
                }
            }
        }
    }

    #[test]
    fn weak_labels_on_long_components_do_not_recurse() {
        let n = 20_000;
        let mut graph = FlowNetwork::new(2 * n, 2 * n);
        for i in 2..n {
            graph.add_literal_edge(2 * (i - 1), 2 * i, 1.0);
        }
        graph.max_flow(0, 1);
        let labels = graph.weak_labels(0);
        assert_eq!(labels[0], Some(1));
        assert!(labels[1].is_some());
        assert!(labels[1..].iter().all(|&label| label == labels[1]));
    }

    #[test]
    fn agrees_with_petgraph_on_fractional_and_disconnected_networks() {
        let mut random = PRNG {
            generator: JsfLarge::from(193477u64),
        };
        for n in [2, 3, 7, 19, 40] {
            for case in 0..80 {
                let mut reference = DiGraph::<(), f64>::new();
                let nodes: Vec<_> = (0..n).map(|_| reference.add_node(())).collect();
                let mut compact = FlowNetwork::new(n, n * n);
                for i in 0..n {
                    for j in i + 1..n {
                        if random.gen_f64() < 0.2 {
                            let value = random.gen_f64() * 100.0;
                            let capacity = if case < 40 {
                                value.floor() / 8.0
                            } else {
                                value
                            };
                            compact.add_edge(i, j, capacity);
                            reference.add_edge(nodes[i], nodes[j], capacity);
                            reference.add_edge(nodes[j], nodes[i], capacity);
                        }
                    }
                }
                let (expected, flows) = dinics(&reference, nodes[0], nodes[n - 1]);
                assert!((compact.max_flow(0, n - 1) - expected).abs() < 1e-9);
                let actual = compact.reachable_from(0);
                let mut reachable = vec![false; n];
                reachable[0] = true;
                loop {
                    let mut changed = false;
                    for edge in reference.edge_references() {
                        let (u, v) = (edge.source().index(), edge.target().index());
                        let flow = flows[edge.id().index()];
                        if reachable[u] && !reachable[v] && *edge.weight() - flow > 1e-12 {
                            reachable[v] = true;
                            changed = true;
                        }
                        if reachable[v] && !reachable[u] && flow > 1e-12 {
                            reachable[u] = true;
                            changed = true;
                        }
                    }
                    if !changed {
                        break;
                    }
                }
                assert_eq!(actual, reachable);
            }
        }
    }

    #[test]
    fn long_paths_do_not_need_recursion() {
        let n = 20_000;
        let mut graph = FlowNetwork::new(n, n - 1);
        for i in 1..n {
            graph.add_edge(i - 1, i, 0.125);
        }
        assert_eq!(graph.max_flow(0, n - 1), 0.125);
        let seen = graph.reachable_from(0);
        assert!(seen[0]);
        assert!(seen[1..].iter().all(|&reachable| !reachable));
    }
}
