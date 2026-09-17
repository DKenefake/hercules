//! Fix one orientation of an exactly complement-symmetric component.
//! This preserves an optimum, not all optimal assignments.

use crate::{qubo::Qubo, FixedVarMap};

pub(crate) struct ComplementComponent {
    variables: Vec<usize>,
    anchor: usize,
}

impl ComplementComponent {
    pub(crate) fn anchor_if_unfixed(&self, fixed: &FixedVarMap) -> Option<usize> {
        (!self.variables.iter().any(|i| fixed.contains_key(i))).then_some(self.anchor)
    }
}

// Error-free expansion: a rounded sum of zero is not a symmetry certificate.
// Reject overflow rather than interpreting a nearly-zero field as exactly zero.
#[derive(Default)]
struct ExactSum {
    parts: Vec<f64>,
    invalid: bool,
}

impl ExactSum {
    fn add(&mut self, mut x: f64) {
        if !x.is_finite() || self.invalid {
            self.invalid = true;
            return;
        }
        let mut count = 0;
        for i in 0..self.parts.len() {
            let mut y = self.parts[i];
            if x.abs() < y.abs() {
                std::mem::swap(&mut x, &mut y);
            }
            let high = x + y;
            if !high.is_finite() {
                self.invalid = true;
                return;
            }
            let low = y - (high - x);
            if low != 0.0 {
                self.parts[count] = low;
                count += 1;
            }
            x = high;
        }
        self.parts.truncate(count);
        if x != 0.0 {
            self.parts.push(x);
        }
    }

    fn is_zero(&self) -> bool {
        !self.invalid && self.parts.is_empty()
    }
}

pub(crate) fn complement_components(qubo: &Qubo, min_size: usize) -> Vec<ComplementComponent> {
    let n = qubo.num_x();
    let mut fields: Vec<_> = (0..n).map(|_| ExactSum::default()).collect();
    let mut adjacency = vec![Vec::new(); n];
    let mut weights = vec![0.0; n];
    for (i, &c) in qubo.c.iter().enumerate() {
        fields[i].add(4.0 * c);
    }
    for (&q, (i, j)) in &qubo.q {
        if i == j {
            fields[i].add(2.0 * q);
        } else if q != 0.0 {
            // Scaled spin field, including asymmetric/triangular Q.
            fields[i].add(q);
            fields[j].add(q);
            adjacency[i].push(j);
            adjacency[j].push(i);
            weights[i] += q.abs();
            weights[j] += q.abs();
        }
    }
    let mut seen = vec![false; n];
    let mut result = Vec::new();
    let mut component = Vec::new();
    for start in 0..n {
        if seen[start] {
            continue;
        }
        component.clear();
        component.push(start);
        seen[start] = true;
        let mut cursor = 0;
        let mut anchor = start;
        let mut symmetric = true;
        while cursor < component.len() {
            let i = component[cursor];
            cursor += 1;
            symmetric &= fields[i].is_zero();
            if weights[i] > weights[anchor] || (weights[i] == weights[anchor] && i < anchor) {
                anchor = i;
            }
            for &j in &adjacency[i] {
                if !seen[j] {
                    seen[j] = true;
                    component.push(j);
                }
            }
        }
        if symmetric && component.len() >= min_size {
            result.push(ComplementComponent {
                variables: component.clone(),
                anchor,
            });
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn exact_zero_test_keeps_cancellation_residue_and_rejects_overflow() {
        let mut sum = ExactSum::default();
        for x in [1e20, 1.0, -1e20] {
            sum.add(x);
        }
        assert!(!sum.is_zero());
        sum.add(-1.0);
        assert!(sum.is_zero());
        sum.add(f64::MAX);
        sum.add(f64::MAX);
        assert!(!sum.is_zero());
    }

    #[test]
    fn component_anchors_preserve_a_minimizer_in_each_storage_convention() {
        for csc in [false, true] {
            let mut q = sprs::TriMat::new((8, 8));
            let mut c = Array1::zeros(8);
            for (i, j, weight) in [
                (0, 1, 2.0),
                (1, 2, -3.0),
                (0, 2, 1.5),
                (3, 4, 4.0),
                (4, 5, 2.0),
                (5, 6, -1.0),
                (6, 3, 0.5),
            ] {
                // Deliberately asymmetric; native binary edge coefficient = weight.
                q.add_triplet(i, j, 1.5 * weight);
                q.add_triplet(j, i, 0.5 * weight);
                c[i] -= 0.5 * weight;
                c[j] -= 0.5 * weight;
            }
            q.add_triplet(0, 0, 5.5);
            c[0] -= 2.75;
            let qubo = Qubo::new_with_c(if csc { q.to_csc() } else { q.to_csr() }, c);
            let components = complement_components(&qubo, 2);
            assert_eq!(components.len(), 2);
            let fixed: FixedVarMap = components.iter().map(|c| (c.anchor, 0)).collect();
            let mut best = f64::INFINITY;
            let mut kept = f64::INFINITY;
            for mask in 0..256 {
                let x = Array1::from_iter((0..8).map(|i| (mask >> i) & 1));
                let value = qubo.eval_usize(&x);
                best = best.min(value);
                if fixed.iter().all(|(&i, &v)| x[i] == v) {
                    kept = kept.min(value);
                }
                for component in &components {
                    let mut flipped = x.clone();
                    for &i in &component.variables {
                        flipped[i] ^= 1;
                    }
                    assert_eq!(value, qubo.eval_usize(&flipped));
                }
            }
            assert_eq!(best, kept);
            let incoming = [(1, 1)].into_iter().collect();
            assert!(components[0].anchor_if_unfixed(&incoming).is_none());
            assert!(components[1].anchor_if_unfixed(&incoming).is_some());
            let mut biased = qubo.clone();
            biased.c[0] += 2.0_f64.powi(-45);
            assert_eq!(complement_components(&biased, 2).len(), 1);
        }
    }
}
