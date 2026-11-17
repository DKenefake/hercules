//! Condition an exact cost decomposition, not just its scalar lower bound.
use super::*;
use crate::branch_subproblem::ConditionalLowerBound;

enum Factor {
    Cycle {
        vertices: Vec<usize>,
        parity: Vec<bool>,
        weight: i128,
    },
    Edge {
        ends: [usize; 2],
        weight: i128,
    },
}

impl Factor {
    fn vertices(&self) -> &[usize] {
        match self {
            Self::Cycle { vertices, .. } => vertices,
            Self::Edge { ends, .. } => ends,
        }
    }

    fn minimum(&self) -> i128 {
        match self {
            Self::Cycle {
                vertices, weight, ..
            } => (2 - vertices.len() as i128) * weight,
            Self::Edge { weight, .. } => -weight.abs(),
        }
    }

    fn value(&self, spins: &[i8]) -> i128 {
        match self {
            Self::Edge {
                ends: [i, j],
                weight,
            } => {
                if spins[*i] == 0 || spins[*j] == 0 {
                    -weight.abs()
                } else {
                    weight * spins[*i] as i128 * spins[*j] as i128
                }
            }
            Self::Cycle {
                vertices,
                parity,
                weight,
            } => {
                let mut first = None;
                let mut last = 0;
                let mut broken = 0i128;
                for (position, &v) in vertices.iter().enumerate() {
                    if spins[v] == 0 {
                        continue;
                    }
                    if first.is_some() {
                        broken += i128::from(
                            (spins[v] != spins[vertices[last]])
                                != (parity[position] ^ parity[last]),
                        );
                    } else {
                        first = Some(position);
                    }
                    last = position;
                }
                // Each path between consecutive fixed vertices needs either
                // zero or one broken edge. The full cycle has odd parity.
                if let Some(first) = first {
                    broken += i128::from(
                        (spins[vertices[first]] != spins[vertices[last]])
                            != (parity[first] ^ parity[last] ^ true),
                    );
                } else {
                    broken = 1;
                }
                (-(vertices.len() as i128) + 2 * broken) * weight
            }
        }
    }
}

pub struct Certificate {
    size: usize,
    baseline: i128,
    factors: Vec<Factor>,
    incidence: Vec<Vec<usize>>,
}

impl Certificate {
    pub(super) fn new(prepared: &PreparedCycleBound) -> Option<Self> {
        let n = prepared.fields.len();
        let mut edges = prepared.edges.clone();
        for (i, &h) in prepared.fields.iter().enumerate() {
            if h != 0 {
                edges.push(((i, n), h));
            }
        }
        let mut trace = PackingTrace::default();
        pack(
            n,
            prepared.constant,
            &edges,
            Instant::now(),
            Duration::from_millis(5),
            f64::INFINITY,
            PackingStrategy::ExtendedGreedy,
            0,
            Some(&mut trace),
        )?;
        let mut factors = Vec::new();
        for (cycle, weight) in trace.cycles {
            let mut neighbors: BTreeMap<usize, Vec<(usize, bool)>> = BTreeMap::new();
            for e in cycle {
                let ((i, j), coupling) = edges[e];
                neighbors.entry(i).or_default().push((j, coupling > 0));
                neighbors.entry(j).or_default().push((i, coupling > 0));
            }
            if neighbors.values().any(|v| v.len() != 2) {
                return None;
            }
            let first = *neighbors.keys().next()?;
            let mut vertices = Vec::with_capacity(neighbors.len());
            let mut parity = Vec::with_capacity(neighbors.len());
            let (mut prev, mut current, mut sign) = (usize::MAX, first, false);
            loop {
                vertices.push(current);
                parity.push(sign);
                let links = &neighbors[&current];
                let &(next, edge_sign) = links.iter().find(|&&(v, _)| v != prev)?;
                prev = current;
                current = next;
                sign ^= edge_sign;
                if current == first {
                    break;
                }
                if vertices.len() >= neighbors.len() {
                    return None;
                }
            }
            if !sign || vertices.len() != neighbors.len() {
                return None;
            }
            factors.push(Factor::Cycle {
                vertices,
                parity,
                weight,
            });
        }
        for (e, weight) in trace.residual.into_iter().enumerate() {
            if weight != 0 {
                let ((i, j), sign) = edges[e];
                factors.push(Factor::Edge {
                    ends: [i, j],
                    weight: weight * sign.signum(),
                });
            }
        }
        let baseline = prepared.constant + factors.iter().map(Factor::minimum).sum::<i128>();
        let mut incidence = vec![Vec::new(); n + 1];
        for (k, factor) in factors.iter().enumerate() {
            for &v in factor.vertices() {
                incidence[v].push(k);
            }
        }
        Some(Self {
            size: n,
            baseline,
            factors,
            incidence,
        })
    }

    pub fn evaluate(&self, fixed: &FixedVarMap) -> Option<Evaluation<'_>> {
        let mut spins = vec![0i8; self.size + 1];
        spins[self.size] = 1;
        let mut affected = Vec::new();
        for (&i, &v) in fixed {
            if i >= self.size || v > 1 {
                return None;
            }
            spins[i] = 2 * v as i8 - 1;
            affected.extend_from_slice(&self.incidence[i]);
        }
        affected.sort_unstable();
        affected.dedup();
        let lower = self.baseline
            + affected
                .iter()
                .map(|&k| self.factors[k].value(&spins) - self.factors[k].minimum())
                .sum::<i128>();
        exact_float(lower)?;
        Some(Evaluation {
            certificate: self,
            spins,
            lower,
            affected,
        })
    }
}

pub struct Evaluation<'a> {
    certificate: &'a Certificate,
    spins: Vec<i8>,
    lower: i128,
    affected: Vec<usize>,
}

fn exact_float(value: i128) -> Option<f64> {
    (value.abs() <= (1i128 << 52)).then_some(value as f64 / 64.0)
}

impl Evaluation<'_> {
    pub fn lower_bound(&self) -> f64 {
        exact_float(self.lower).unwrap()
    }

    pub fn conditional(&mut self, variable: usize) -> Option<ConditionalLowerBound> {
        if variable >= self.certificate.size || self.spins[variable] != 0 {
            return None;
        }
        let mut values = [self.lower; 2];
        for &k in &self.certificate.incidence[variable] {
            let factor = &self.certificate.factors[k];
            let old = factor.value(&self.spins);
            for (x, value) in values.iter_mut().enumerate() {
                self.spins[variable] = 2 * x as i8 - 1;
                *value += factor.value(&self.spins) - old;
            }
            self.spins[variable] = 0;
        }
        Some(ConditionalLowerBound {
            variable,
            zero: exact_float(values[0])?,
            one: exact_float(values[1])?,
        })
    }

    pub fn probe(&mut self, limit: usize, budget: Duration) -> Vec<ConditionalLowerBound> {
        let start = Instant::now();
        let mut scores = vec![0i128; self.certificate.size];
        for &k in &self.affected {
            if start.elapsed() >= budget {
                break;
            }
            let factor = &self.certificate.factors[k];
            let fixed = factor
                .vertices()
                .iter()
                .filter(|&&v| self.spins[v] != 0)
                .count();
            let weight = match factor {
                Factor::Cycle { weight, .. } if fixed >= 2 => *weight,
                Factor::Edge { weight, .. } if fixed >= 1 => weight.abs(),
                _ => continue,
            };
            for &v in factor.vertices() {
                if self.spins[v] == 0 {
                    scores[v] += weight;
                }
            }
        }
        let mut candidates: Vec<_> = scores
            .into_iter()
            .enumerate()
            .filter(|&(_, s)| s > 0)
            .collect();
        candidates.sort_unstable_by_key(|&(v, s)| (std::cmp::Reverse(s), v));
        let mut bounds = Vec::new();
        for (v, _) in candidates.into_iter().take(limit) {
            if start.elapsed() >= budget {
                break;
            }
            if let Some(bound) = self.conditional(v) {
                bounds.push(bound);
            }
        }
        bounds
    }
}
