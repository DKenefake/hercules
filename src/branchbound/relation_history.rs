//! Heuristic pseudocosts only: never used to certify pruning or fixings.
use super::*;
use std::collections::HashMap;

#[derive(Default)]
pub(super) struct RelationHistory {
    samples: HashMap<(usize, usize), ([usize; 2], [f64; 2])>,
    pub observations: usize,
    pub choices: usize,
    pub changed_choices: usize,
    pub construction_seconds: f64,
}
pub(super) struct RelationOrigin {
    key: (usize, usize),
    side: usize,
    bound: f64,
    gap: f64,
    distance: f64,
}
impl RelationHistory {
    pub fn observe(&mut self, origin: RelationOrigin, bound: f64) {
        if !bound.is_finite() || !origin.bound.is_finite() {
            return;
        }
        let reward = ((bound - origin.bound) / origin.gap).clamp(0.0, 1.0) / origin.distance;
        let (count, mean) = self.samples.entry(origin.key).or_default();
        let k = origin.side;
        count[k] += 1;
        mean[k] += (reward - mean[k]) / count[k] as f64;
        self.observations += 1;
    }
    fn prediction(&self, key: (usize, usize), rho: f64) -> f64 {
        let Some((counts, means)) = self.samples.get(&key) else {
            return 0.0;
        };
        let count = counts[0].min(counts[1]);
        if count < 2 {
            return 0.0;
        }
        let z = (1.0 - rho) * 0.5;
        let a = means[0] * z;
        let b = means[1] * (1.0 - z);
        (a.min(b) + 0.25 * a.max(b)) * count as f64 / (count as f64 + 4.0)
    }
}
impl BBSolver {
    pub fn relation_history_statistics(&self) -> (usize, usize, usize, f64) {
        let h = self.relation_history.lock().unwrap();
        (
            h.observations,
            h.choices,
            h.changed_choices,
            h.construction_seconds,
        )
    }
    pub(super) fn inherit_relation_history(&mut self, parent: &Self, remaining: &[usize]) {
        self.relation_history = parent.relation_history.clone();
        self.relation_ids = remaining.iter().map(|&i| parent.relation_ids[i]).collect();
    }
    fn relation_key(&self, i: usize, j: usize) -> (usize, usize) {
        let (i, j) = (self.relation_ids[i], self.relation_ids[j]);
        (i.min(j), i.max(j))
    }
    pub(super) fn relation_origin(
        &self,
        i: usize,
        j: usize,
        rho: f64,
        side: bool,
        bound: f64,
    ) -> RelationOrigin {
        let z = (1.0 - rho) * 0.5;
        let gap = (self.pruning_upper_bound() - bound).max(1.0);
        RelationOrigin {
            key: self.relation_key(i, j),
            side: usize::from(side),
            bound,
            gap: if gap.is_finite() { gap } else { 1.0 },
            distance: if side { 1.0 - z } else { z }.max(0.05),
        }
    }
    pub(crate) fn select_relation(
        &self,
        candidates: &[(usize, usize, f64, f64)],
    ) -> Option<(usize, usize, f64)> {
        let best = candidates
            .iter()
            .enumerate()
            .max_by(|a, b| a.1 .3.total_cmp(&b.1 .3).then_with(|| b.0.cmp(&a.0)))?
            .0;
        if !self.options.relation_pseudocosts {
            let c = candidates[best];
            return Some((c.0, c.1, c.2));
        }
        let mut h = self.relation_history.lock().unwrap();
        let predictions = candidates
            .iter()
            .map(|&(i, j, rho, _)| h.prediction(self.relation_key(i, j), rho))
            .collect::<Vec<_>>();
        let scale = predictions.iter().copied().fold(0.0, f64::max);
        let structural = candidates[best].3.max(f64::MIN_POSITIVE);
        let mut chosen = best;
        if scale > 0.0 {
            let score =
                |k: usize| 0.25 * candidates[k].3 / structural + 0.75 * predictions[k] / scale;
            for k in 0..candidates.len() {
                if score(k) > score(chosen) {
                    chosen = k;
                }
            }
        }
        h.choices += 1;
        h.changed_choices += usize::from(chosen != best);
        let c = candidates[chosen];
        Some((c.0, c.1, c.2))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn history_needs_both_directions_and_mapped_identity() {
        let mut o = SolverOptions::new();
        o.relation_pseudocosts = true;
        let parent = BBSolver::new(Qubo::new(sprs::CsMat::eye(4)), o.clone());
        let mut child = BBSolver::new(Qubo::new(sprs::CsMat::eye(3)), o);
        child.inherit_relation_history(&parent, &[3, 0, 2]);
        assert_eq!(child.relation_key(0, 1), (0, 3));
        let candidates = [(0, 2, 0.0, 10.0), (0, 1, 0.0, 8.0)];
        assert_eq!(child.select_relation(&candidates), Some((0, 2, 0.0)));
        for side in [false, true] {
            for _ in 0..2 {
                let origin = parent.relation_origin(0, 3, 0.0, side, -10.0);
                parent
                    .relation_history
                    .lock()
                    .unwrap()
                    .observe(origin, -5.0);
            }
        }
        assert_eq!(child.select_relation(&candidates), Some((0, 1, 0.0)));
        assert_eq!(parent.relation_history_statistics().0, 4);
    }
}
