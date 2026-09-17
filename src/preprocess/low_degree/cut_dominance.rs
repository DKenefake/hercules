//! Whole-cut dominance (Charfreitag et al., SEA 2024, Proposition 3.3).
//! A dominant cut edge has its preferred parity in some optimum. Substitute
//! one certificate at a time, then rebuild the signed graph before retesting.
use super::*;
use crate::subproblemsolvers::roofdual::flow::FlowNetwork;

const MAX_FLOW_TRIALS: usize = 64;

#[derive(Clone, Copy, Debug, Default)]
pub struct CutDominanceStatistics {
    pub flow_trials: usize,
    pub contracted: usize,
    pub fixed: usize,
    pub budget_exhausted: bool,
}

impl Graph {
    fn anchor_weight(&self, i: usize) -> Option<f64> {
        let twice = 2.0 * self.linear[i];
        if !twice.is_finite() || twice * 0.5 != self.linear[i] {
            return None;
        }
        self.edges[i]
            .values()
            .try_fold(-twice, |weight, &b| add(weight, -b))
    }

    pub(super) fn cut_anchor_value(&self, i: usize) -> Option<usize> {
        Some(usize::from(self.anchor_weight(i)? > 0.0))
    }

    fn signed_cut_edges(&self) -> Option<Vec<(usize, usize, f64)>> {
        // With anchor x_a=0: 2*(f-constant) = -sum w_ij*(x_i XOR x_j).
        // Scaling by two avoids halving native coefficients. Fixed-variable
        // fields MUST enter via anchor edges, even in a formerly field-free QUBO.
        let anchor = self.active.len();
        let mut edges = Vec::new();
        for i in 0..anchor {
            if !self.active[i] {
                continue;
            }
            for (&j, &weight) in &self.edges[i] {
                if i < j {
                    edges.push((i, j, weight));
                }
            }
            let field = self.anchor_weight(i)?;
            if field != 0.0 {
                edges.push((i, anchor, field));
            }
        }
        Some(edges)
    }

    pub(super) fn dominant_cut_edge(
        &self,
        deadline: Instant,
        stats: &mut CutDominanceStatistics,
    ) -> Option<(usize, usize, bool)> {
        if stats.flow_trials >= MAX_FLOW_TRIALS || Instant::now() >= deadline {
            stats.budget_exhausted = true;
            return None;
        }
        let edges = self.signed_cut_edges()?;
        let mut candidates: Vec<_> = (0..edges.len()).collect();
        // Heavy edges are likelier to dominate a separator. Sorting is only a
        // search heuristic; the actual cut certificate below decides validity.
        candidates.sort_unstable_by(|&a, &b| {
            edges[b]
                .2
                .abs()
                .total_cmp(&edges[a].2.abs())
                .then(a.cmp(&b))
        });
        for index in candidates {
            if stats.flow_trials >= MAX_FLOW_TRIALS || Instant::now() >= deadline {
                stats.budget_exhausted = true;
                break;
            }
            let (u, v, weight) = edges[index];
            let threshold = add(weight.abs(), weight.abs())?;
            let mut flow = FlowNetwork::new(self.active.len() + 1, edges.len());
            for &(i, j, w) in &edges {
                flow.add_edge(i, j, w.abs());
            }
            stats.flow_trials += 1;
            let Some(side) = flow.bounded_cut(u, v, threshold, deadline) else {
                continue;
            };
            if side[u] == side[v] {
                continue;
            }
            // Certify from original capacities, not approximate max-flow output.
            // We only need SOME separating cut dominated by this edge; minimum
            // cut accuracy affects detection, never validity of an accepted rule.
            let mut other = Some(0.0);
            for (k, &(i, j, w)) in edges.iter().enumerate() {
                if k != index && side[i] != side[j] {
                    other = other.and_then(|sum| add(sum, w.abs()));
                }
            }
            if let Some(other) = other {
                if other <= weight.abs() {
                    return Some((u, v, other == weight.abs()));
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qubo(n: usize, edges: &[(usize, usize, f64)], fields: &[f64]) -> Qubo {
        let mut q = TriMat::new((n, n));
        let mut c = Array1::from_vec(fields.to_vec());
        for &(i, j, b) in edges {
            q.add_triplet(i, j, 2.0 * b);
            c[i] -= b * 0.5;
            c[j] -= b * 0.5;
        }
        Qubo::new_with_c(q.to_csc(), c)
    }

    fn exact(q: &Qubo, fixed: &FixedVarMap) -> f64 {
        (0..1 << q.num_x())
            .filter(|mask| fixed.iter().all(|(&i, &v)| (mask >> i) & 1 == v))
            .map(|mask| q.eval_usize(&Array1::from_iter((0..q.num_x()).map(|i| (mask >> i) & 1))))
            .fold(f64::INFINITY, f64::min)
    }

    #[test]
    fn group_cut_finds_a_bridge_that_endpoint_dominance_misses() {
        let mut edges = Vec::new();
        for group in [0..4, 4..8] {
            for i in group.clone() {
                for j in i + 1..group.end {
                    edges.push((i, j, 4.0));
                }
            }
        }
        edges.push((0, 4, 2.0));
        let q = qubo(8, &edges, &[0.0; 8]);
        let mut graph = Graph::new(&q).unwrap();
        assert!((0..8).all(|i| graph.dominant_neighbor(i).is_none()));
        let mut stats = CutDominanceStatistics::default();
        let (i, j, _) = graph
            .dominant_cut_edge(Instant::now() + Duration::from_secs(1), &mut stats)
            .unwrap();
        assert_eq!((i, j), (0, 4));
        let (step, _) = graph.contract(i, j).unwrap();
        let (small, remaining) = graph.compact();
        let reduction = RootReduction {
            qubo: small,
            constant: graph.constant,
            remaining,
            fixed: FixedVarMap::default(),
            statistics: RootReductionStatistics::default(),
            original_size: 8,
            steps: vec![step],
        };
        let expected = exact(&q, &FixedVarMap::default());
        assert_eq!(
            exact(&reduction.qubo, &FixedVarMap::default()) + reduction.constant,
            expected
        );
        for mask in 0..1 << reduction.remaining.len() {
            let x = Array1::from_iter((0..reduction.remaining.len()).map(|i| (mask >> i) & 1));
            assert_eq!(
                q.eval_usize(&reduction.reconstruct(&x)),
                reduction.qubo.eval_usize(&x) + reduction.constant
            );
        }
    }

    #[test]
    fn cut_reductions_preserve_generated_optima_with_fields_fixings_and_ties() {
        use smolprng::{JsfLarge, PRNG};
        let mut rng = PRNG {
            generator: JsfLarge::from(73301u64),
        };
        let mut total = 0;
        for sample in 0..120 {
            let n = 7;
            let mut edges = Vec::new();
            for i in 0..n {
                for j in i + 1..n {
                    if rng.gen_f64() < 0.4 {
                        edges.push((i, j, 2.0 * ((rng.gen_f64() * 7.0).floor() - 3.0)));
                    }
                }
            }
            let fields: Vec<_> = (0..n)
                .map(|_| (rng.gen_f64() * 5.0).floor() - 2.0)
                .collect();
            let q = qubo(n, &edges, &fields);
            let fixed = if sample % 2 == 0 {
                [(0, sample % 3 % 2)].into_iter().collect()
            } else {
                FixedVarMap::default()
            };
            let mut options = SolverOptions::new();
            options.root_low_degree_elimination = false;
            options.root_dominant_edge_contraction = false;
            options.root_cut_dominance = true;
            if let Some(r) =
                super::super::reduce(&q, &fixed, &options, Duration::from_secs(1), false)
            {
                total += r.statistics.cut_dominance.contracted + r.statistics.cut_dominance.fixed;
                assert_eq!(
                    exact(&r.qubo, &FixedVarMap::default()) + r.constant,
                    exact(&q, &fixed)
                );
                for mask in 0..1 << r.remaining.len() {
                    let x = Array1::from_iter((0..r.remaining.len()).map(|i| (mask >> i) & 1));
                    let full = r.reconstruct(&x);
                    assert!(fixed.iter().all(|(&i, &v)| full[i] == v));
                    assert_eq!(q.eval_usize(&full), r.qubo.eval_usize(&x) + r.constant);
                }
            }
        }
        assert!(total > 100);
    }

    #[test]
    fn fields_are_included_and_expired_budgets_produce_no_certificate() {
        let q = qubo(2, &[(0, 1, 2.0)], &[10.0, 10.0]);
        let graph = Graph::new(&q).unwrap();
        let edges = graph.signed_cut_edges().unwrap();
        assert_eq!(edges, vec![(0, 1, 2.0), (0, 2, -20.0), (1, 2, -20.0)]);
        let mut stats = CutDominanceStatistics::default();
        assert!(graph
            .dominant_cut_edge(Instant::now(), &mut stats)
            .is_none());
        assert_eq!(stats.flow_trials, 0);
        assert!(stats.budget_exhausted);
        let mut stats = CutDominanceStatistics {
            flow_trials: MAX_FLOW_TRIALS,
            ..Default::default()
        };
        assert!(graph
            .dominant_cut_edge(Instant::now() + Duration::from_secs(1), &mut stats)
            .is_none());
    }
}
