//! Single-endpoint dominant-edge contractions. Certify on the current graph,
//! substitute immediately, and re-test affected rows before the next reduction.
use super::{add, Elimination, Graph};

impl Graph {
    pub(super) fn dominant_neighbor(&self, i: usize) -> Option<(usize, bool)> {
        // Native binary polynomial: a_i*x_i + sum_j b_ij*x_i*x_j.
        // Delta_i = a_i + sum_j b_ij*x_j. Include the residual linear term;
        // ignoring it would make this rule unsound after node/root fixings.
        let mut lo = self.linear[i];
        let mut hi = self.linear[i];
        for &weight in self.edges[i].values() {
            lo = add(lo, weight.min(0.0))?;
            hi = add(hi, weight.max(0.0))?;
        }
        for (&j, &weight) in &self.edges[i] {
            let bounds = if weight > 0.0 {
                // x_j=0: Delta_i<=0; x_j=1: Delta_i>=0 => x_i=1-x_j.
                add(hi, -weight).zip(add(lo, weight))
            } else {
                // x_j=1: Delta_i<=0; x_j=0: Delta_i>=0 => x_i=x_j.
                add(hi, weight).zip(add(lo, -weight))
            };
            if let Some((nonpositive, nonnegative)) = bounds {
                if nonpositive <= 0.0 && nonnegative >= 0.0 {
                    return Some((j, nonpositive == 0.0 || nonnegative == 0.0));
                }
            }
        }
        None
    }

    pub(super) fn contract(&mut self, i: usize, j: usize) -> Option<(Elimination, Vec<usize>)> {
        let weight = self.edges[i][&j];
        let complement = weight > 0.0;
        let affected: Vec<_> = self.edges[i].keys().copied().collect();
        if complement {
            // x_i=1-x_j: b_ij*x_i*x_j vanishes identically for binary x_j.
            self.constant = add(self.constant, self.linear[i])?;
            self.linear[j] = add(self.linear[j], -self.linear[i])?;
        } else {
            // x_i=x_j: both a_i*x_i and b_ij*x_i*x_j become linear in x_j.
            self.linear[j] = add(self.linear[j], add(self.linear[i], weight)?)?;
        }
        for (k, coefficient) in std::mem::take(&mut self.edges[i]) {
            self.edges[k].remove(&i);
            if k == j {
                continue;
            }
            if complement {
                self.linear[k] = add(self.linear[k], coefficient)?;
                self.add_edge(j, k, -coefficient)?;
            } else {
                self.add_edge(j, k, coefficient)?;
            }
        }
        self.active[i] = false;
        let choices = if complement {
            [1, 0, 0, 0, 0, 0, 0, 0]
        } else {
            [0, 1, 0, 0, 0, 0, 0, 0]
        };
        Some((
            Elimination {
                variable: i,
                neighbors: vec![j],
                choices,
            },
            affected,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preprocess::low_degree::{reduce, reduce_root};
    use crate::{qubo::Qubo, solver_options::SolverOptions, FixedVarMap};
    use ndarray::Array1;
    use sprs::TriMat;
    use std::time::Duration;

    fn point(n: usize, mask: usize) -> Array1<usize> {
        Array1::from_iter((0..n).map(|i| (mask >> i) & 1))
    }

    #[test]
    fn dominant_edge_certificates_preserve_every_conditional_minimum() {
        let mut checked = 0;
        // Include arbitrary fields, positive/negative interactions, zeros and ties.
        for a in -4..=4 {
            for b in -2..=2 {
                for c in -2..=2 {
                    for d in -2..=2 {
                        let mut q = TriMat::new((4, 4));
                        q.add_triplet(0, 0, 2.0);
                        for (j, weight) in [b, c, d].into_iter().enumerate() {
                            q.add_triplet(j + 1, 0, 2.0 * f64::from(weight));
                        }
                        q.add_triplet(1, 2, 4.0);
                        let qubo = Qubo::new_with_c(
                            q.to_csc(),
                            ndarray::array![f64::from(a) - 1.0, -1.0, 2.0, -3.0],
                        );
                        let mut graph = Graph::new(&qubo).unwrap();
                        let Some((j, _)) = graph.dominant_neighbor(0) else {
                            continue;
                        };
                        let (step, _) = graph.contract(0, j).unwrap();
                        let (small, mapping) = graph.compact();
                        for mask in 0..8 {
                            let reduced = point(3, mask);
                            let mut full = Array1::zeros(4);
                            for (i, &old) in mapping.iter().enumerate() {
                                full[old] = reduced[i];
                            }
                            full[0] = 0;
                            let zero = qubo.eval_usize(&full);
                            full[0] = 1;
                            let one = qubo.eval_usize(&full);
                            full[0] = step.choices[full[j]];
                            assert_eq!(qubo.eval_usize(&full), zero.min(one));
                            assert_eq!(small.eval_usize(&reduced) + graph.constant, zero.min(one));
                        }
                        checked += 1;
                    }
                }
            }
        }
        assert!(checked > 100);
    }

    fn triangle() -> Qubo {
        Qubo::from_vec(vec![0, 0, 1], vec![1, 2, 2], vec![4.0; 3], vec![-2.0; 3], 3)
    }

    #[test]
    fn dominant_edge_ties_are_applied_sequentially_not_as_global_relations() {
        let qubo = triangle();
        let graph = Graph::new(&qubo).unwrap();
        // All three edges of an antiferromagnetic triangle individually qualify,
        // but requiring all three pairs to differ would be inconsistent.
        for i in 0..3 {
            assert!(graph.dominant_neighbor(i).unwrap().1);
        }
        let mut options = SolverOptions::new();
        options.root_low_degree_elimination = false;
        let r = reduce_root(
            &qubo,
            &FixedVarMap::default(),
            &options,
            Duration::from_secs(5),
        )
        .unwrap();
        assert_eq!(r.statistics.contracted, 1);
        assert_eq!(r.statistics.weak_contractions, 1);
        assert_eq!(r.statistics.eliminated, 0);
        assert_eq!(qubo.eval_usize(&r.reconstruct(&Array1::zeros(0))), -2.0);
    }

    #[test]
    fn dominant_edge_respects_linear_fields_and_inexact_certificates() {
        let mut q = TriMat::new((2, 2));
        q.add_triplet(0, 1, 4.0);
        let qubo = Qubo::new_with_c(q.to_csr(), ndarray::array![10.0, 10.0]);
        let graph = Graph::new(&qubo).unwrap();
        assert!(graph.dominant_neighbor(0).is_none());
        assert!(graph.dominant_neighbor(1).is_none());
        // A single huge positive term is not a license to round away the field.
        let mut q = TriMat::new((3, 3));
        q.add_triplet(0, 1, 2e20);
        q.add_triplet(0, 2, 2.0);
        let qubo = Qubo::new_with_c(q.to_csr(), ndarray::array![-1e20, 0.0, 0.0]);
        assert!(Graph::new(&qubo).unwrap().dominant_neighbor(0).is_none());
    }

    #[test]
    fn dominant_edge_queue_closes_and_reconstructs_generated_conditional_problems() {
        use smolprng::{JsfLarge, PRNG};
        let mut rng = PRNG {
            generator: JsfLarge::from(190721u64),
        };
        let mut found = 0;
        for sample in 0..96 {
            let mut qubo = Qubo::make_random_qubo(8, &mut rng, 0.65);
            qubo.q
                .data_mut()
                .iter_mut()
                .for_each(|w| *w = (*w * 24.0).round());
            qubo.c.mapv_inplace(|a| (a * 12.0).round());
            if sample % 3 == 0 {
                qubo.c.fill(0.0);
                for (&value, (i, j)) in &qubo.q {
                    if i == j {
                        qubo.c[i] -= 0.5 * value;
                    } else {
                        qubo.c[i] -= 0.25 * value;
                        qubo.c[j] -= 0.25 * value;
                    }
                }
            }
            if sample % 2 == 0 {
                qubo.q = qubo.q.to_csc();
            }
            for fixed in [
                FixedVarMap::default(),
                [(0, 1), (3, 0)].into_iter().collect(),
            ] {
                for low_degree in [false, true] {
                    let mut options = SolverOptions::new();
                    options.root_low_degree_elimination = low_degree;
                    let Some(r) = reduce(&qubo, &fixed, &options, Duration::from_secs(5), false)
                    else {
                        continue;
                    };
                    found += r.statistics.contracted;
                    let mut minima = vec![f64::INFINITY; 1 << r.remaining.len()];
                    for mask in 0..256 {
                        let full = point(8, mask);
                        if fixed.iter().any(|(&i, &v)| full[i] != v) {
                            continue;
                        }
                        let index = r
                            .remaining
                            .iter()
                            .enumerate()
                            .fold(0, |m, (bit, &i)| m | (full[i] << bit));
                        minima[index] = minima[index].min(qubo.eval_usize(&full));
                    }
                    for (mask, expected) in minima.into_iter().enumerate() {
                        let small = point(r.remaining.len(), mask);
                        let full = r.reconstruct(&small);
                        assert_eq!(qubo.eval_usize(&full), expected);
                        assert_eq!(r.qubo.eval_usize(&small) + r.constant, expected);
                        assert!(fixed.iter().all(|(&i, &v)| full[i] == v));
                    }
                    let remaining_graph = Graph::new(&r.qubo).unwrap();
                    for i in 0..r.qubo.num_x() {
                        assert!(remaining_graph.dominant_neighbor(i).is_none());
                        if low_degree {
                            assert!(remaining_graph.edges[i].len() > 2);
                        }
                    }
                    // Exercise the same reductions with gradient/roof closure too.
                    for weak in [false, true] {
                        options.roof_dual_weak_persistencies = weak;
                        let closed =
                            reduce_root(&qubo, &fixed, &options, Duration::from_secs(5)).unwrap();
                        let expected = (0..256)
                            .filter(|mask| fixed.iter().all(|(&i, &v)| (mask >> i) & 1 == v))
                            .map(|mask| qubo.eval_usize(&point(8, mask)))
                            .fold(f64::INFINITY, f64::min);
                        let actual = (0..1 << closed.remaining.len())
                            .map(|mask| {
                                let x = point(closed.remaining.len(), mask);
                                let full = closed.reconstruct(&x);
                                assert_eq!(
                                    qubo.eval_usize(&full),
                                    closed.qubo.eval_usize(&x) + closed.constant
                                );
                                qubo.eval_usize(&full)
                            })
                            .fold(f64::INFINITY, f64::min);
                        assert_eq!(actual, expected);
                    }
                }
            }
        }
        assert!(found > 0, "generated cases must exercise contractions");
    }
}
