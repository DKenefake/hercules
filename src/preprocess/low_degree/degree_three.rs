//! Exact three-neighbor elimination; never project away a genuine cubic term.
use super::{add, Elimination, Graph};

impl Graph {
    pub(super) fn try_eliminate_three(&mut self, i: usize) -> Option<Elimination> {
        let neighbors: Vec<_> = self.edges[i].keys().copied().collect();
        if neighbors.len() != 3 {
            return None;
        }
        let mut coefficients = [0.0; 8];
        let mut choices = [0; 8];
        for mask in 0..8 {
            let mut delta = self.linear[i];
            for (bit, &j) in neighbors.iter().enumerate() {
                if mask & (1 << bit) != 0 {
                    delta = add(delta, self.edges[i][&j])?;
                }
            }
            coefficients[mask] = delta.min(0.0);
            choices[mask] = usize::from(delta < 0.0);
        }
        // Boolean Mobius transform: coefficient[mask] multiplies the product
        // of the variables selected by mask. Entry 7 is the cubic coefficient.
        for bit in 0..3 {
            for mask in 0..8 {
                if mask & (1 << bit) != 0 {
                    coefficients[mask] = add(coefficients[mask], -coefficients[mask ^ (1 << bit)])?;
                }
            }
        }
        if coefficients[7] != 0.0 {
            return None;
        }

        // Prepare every update before mutating. Rejected or inexact candidates
        // must leave earlier successful reductions and the current graph intact.
        let constant = add(self.constant, coefficients[0])?;
        let mut linear = [0.0; 3];
        for (bit, &j) in neighbors.iter().enumerate() {
            linear[bit] = add(self.linear[j], coefficients[1 << bit])?;
        }
        let pairs = [(0, 1), (0, 2), (1, 2)];
        let mut edges = [0.0; 3];
        for (index, &(a, b)) in pairs.iter().enumerate() {
            let existing = self.edges[neighbors[a]]
                .get(&neighbors[b])
                .copied()
                .unwrap_or(0.0);
            edges[index] = add(existing, coefficients[(1 << a) | (1 << b)])?;
        }
        self.constant = constant;
        for (bit, &j) in neighbors.iter().enumerate() {
            self.linear[j] = linear[bit];
            self.edges[j].remove(&i);
        }
        for (index, &(a, b)) in pairs.iter().enumerate() {
            let (j, k) = (neighbors[a], neighbors[b]);
            if edges[index] == 0.0 {
                self.edges[j].remove(&k);
                self.edges[k].remove(&j);
            } else {
                self.edges[j].insert(k, edges[index]);
                self.edges[k].insert(j, edges[index]);
            }
        }
        self.edges[i].clear();
        self.active[i] = false;
        Some(Elimination {
            variable: i,
            neighbors,
            choices,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{qubo::Qubo, solver_options::SolverOptions, FixedVarMap};
    use ndarray::Array1;
    use sprs::TriMat;
    use std::time::Duration;

    fn star(a: f64, weights: [f64; 3]) -> Qubo {
        let mut q = TriMat::new((4, 4));
        q.add_triplet(0, 0, 2.0);
        for (bit, weight) in weights.into_iter().enumerate() {
            q.add_triplet(bit + 1, 0, 2.0 * weight);
        }
        q.add_triplet(1, 2, 2.0);
        Qubo::new_with_c(q.to_csc(), ndarray::array![a - 1.0, -1.0, 2.0, -3.0])
    }

    #[test]
    fn degree_three_checks_all_eight_conditional_minima_and_cubic_coefficients() {
        let mut accepted = 0;
        let mut rejected = 0;
        for a in -6..=6 {
            for b in [-4.0, -2.0, 2.0, 4.0] {
                for c in [-4.0, -2.0, 2.0, 4.0] {
                    for d in [-4.0, -2.0, 2.0, 4.0] {
                        let qubo = star(f64::from(a), [b, c, d]);
                        let mut graph = Graph::new(&qubo).unwrap();
                        let table: Vec<_> = (0..8)
                            .map(|mask| {
                                (f64::from(a)
                                    + b * (mask & 1) as f64
                                    + c * ((mask >> 1) & 1) as f64
                                    + d * ((mask >> 2) & 1) as f64)
                                    .min(0.0)
                            })
                            .collect();
                        let cubic = table[7] - table[3] - table[5] - table[6]
                            + table[1]
                            + table[2]
                            + table[4]
                            - table[0];
                        let Some(step) = graph.try_eliminate_three(0) else {
                            assert_ne!(cubic, 0.0);
                            assert!(graph.active.iter().all(|&v| v));
                            rejected += 1;
                            continue;
                        };
                        assert_eq!(cubic, 0.0);
                        let (small, mapping) = graph.compact();
                        assert_eq!(mapping, vec![1, 2, 3]);
                        for mask in 0..8 {
                            let x = Array1::from_iter((0..3).map(|bit| (mask >> bit) & 1));
                            let mut full = ndarray::array![0, x[0], x[1], x[2]];
                            let zero = qubo.eval_usize(&full);
                            full[0] = 1;
                            let one = qubo.eval_usize(&full);
                            full[0] = step.choices[mask];
                            assert_eq!(qubo.eval_usize(&full), zero.min(one));
                            assert_eq!(small.eval_usize(&x) + graph.constant, zero.min(one));
                        }
                        accepted += 1;
                    }
                }
            }
        }
        assert!(accepted > 100 && rejected > 100);
    }

    #[test]
    fn degree_three_rejections_are_atomic_even_when_an_update_is_inexact() {
        for (a, weights, large_neighbor) in [
            (-1.0, [1.0; 3], false),          // Genuine cubic term.
            (-1e20, [1e20, 1.0, 2.0], false), // Inexact conditional sum.
            (-3.0, [2.0; 3], true),           // Exact table, inexact neighbor update.
        ] {
            let qubo = star(-3.0, weights);
            let mut graph = Graph::new(&qubo).unwrap();
            graph.linear[0] = a;
            if large_neighbor {
                graph.linear[3] = 1e20;
            }
            let before = (
                graph.linear.clone(),
                graph.edges.clone(),
                graph.active.clone(),
                graph.constant,
            );
            assert!(graph.try_eliminate_three(0).is_none());
            assert_eq!(
                (graph.linear, graph.edges, graph.active, graph.constant),
                before
            );
        }
    }

    #[test]
    fn degree_three_respects_enable_flags_and_root_time_budget() {
        let qubo = star(-3.0, [2.0; 3]);
        for low_degree in [false, true] {
            for enabled in [false, true] {
                let mut options = SolverOptions::new();
                options.root_dominant_edge_contraction = false;
                options.small_block_elimination = false;
                options.root_low_degree_elimination = low_degree;
                options.root_degree_three_elimination = enabled;
                let fixed = FixedVarMap::default();
                assert!(
                    super::super::reduce_root(&qubo, &fixed, &options, Duration::ZERO).is_none()
                );
                let r = super::super::reduce_root(&qubo, &fixed, &options, Duration::from_secs(5));
                if low_degree {
                    let r = r.unwrap();
                    assert_eq!(r.statistics.degree_three_eliminated, usize::from(enabled));
                    assert!(r.remaining.is_empty());
                    let expected = (0..16)
                        .map(|mask| {
                            qubo.eval_usize(&Array1::from_iter((0..4).map(|bit| (mask >> bit) & 1)))
                        })
                        .fold(f64::INFINITY, f64::min);
                    assert_eq!(qubo.eval_usize(&r.reconstruct(&Array1::zeros(0))), expected);
                } else {
                    assert!(r.is_none());
                }
            }
        }
    }

    #[test]
    fn degree_three_cascades_preserve_generated_conditioned_qubos() {
        use smolprng::{JsfLarge, PRNG};
        let mut rng = PRNG {
            generator: JsfLarge::from(99017u64),
        };
        let mut eliminated = 0;
        for sample in 0..96 {
            let mut qubo = Qubo::make_random_qubo(9, &mut rng, 0.4);
            qubo.q
                .data_mut()
                .iter_mut()
                .for_each(|v| *v = 4.0 * (*v * 8.0).round());
            qubo.c.fill(0.0);
            for (&value, (i, j)) in &qubo.q {
                if i == j {
                    qubo.c[i] -= 0.5 * value;
                } else {
                    qubo.c[i] -= 0.25 * value;
                    qubo.c[j] -= 0.25 * value;
                }
            }
            if sample % 3 == 0 {
                qubo.c[3] += 1.0;
            }
            if sample % 2 == 0 {
                qubo.q = qubo.q.to_csc();
            }
            for fixed in [FixedVarMap::default(), [(0, 1)].into_iter().collect()] {
                for dominant in [false, true] {
                    let mut options = SolverOptions::new();
                    options.root_dominant_edge_contraction = dominant;
                    let Some(r) = super::super::reduce(
                        &qubo,
                        &fixed,
                        &options,
                        Duration::from_secs(5),
                        false,
                    ) else {
                        continue;
                    };
                    eliminated += r.statistics.degree_three_eliminated;
                    let mut minima = vec![f64::INFINITY; 1 << r.remaining.len()];
                    for mask in 0..512 {
                        let x = Array1::from_iter((0..9).map(|bit| (mask >> bit) & 1));
                        if fixed.iter().any(|(&i, &v)| x[i] != v) {
                            continue;
                        }
                        let small_mask = r
                            .remaining
                            .iter()
                            .enumerate()
                            .fold(0, |m, (bit, &i)| m | (x[i] << bit));
                        minima[small_mask] = minima[small_mask].min(qubo.eval_usize(&x));
                    }
                    for (mask, expected) in minima.into_iter().enumerate() {
                        let x =
                            Array1::from_iter((0..r.remaining.len()).map(|bit| (mask >> bit) & 1));
                        let full = r.reconstruct(&x);
                        assert_eq!(r.qubo.eval_usize(&x) + r.constant, expected);
                        assert_eq!(qubo.eval_usize(&full), expected);
                        assert!(fixed.iter().all(|(&i, &v)| full[i] == v));
                    }
                }
            }
        }
        assert!(eliminated > 0);
    }
}
