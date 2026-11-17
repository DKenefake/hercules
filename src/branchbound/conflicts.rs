//! Cutoff conflicts are objective certificates, not infeasibility constraints.
//! Pools never cross a solver's coordinate/objective context.
use super::*;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};
use std::time::{Duration, Instant};

#[derive(Clone)]
struct Conflict {
    literals: Vec<(usize, usize)>,
    lower: f64,
}
#[derive(Default)]
pub(super) struct ConflictPool {
    entries: Mutex<Vec<Conflict>>,
    attempts: std::sync::Arc<AtomicUsize>,
    learned: std::sync::Arc<AtomicUsize>,
    fixed: std::sync::Arc<AtomicUsize>,
    pruned: std::sync::Arc<AtomicUsize>,
}
impl ConflictPool {
    pub(super) fn clear(&mut self) {
        self.entries.get_mut().unwrap().clear();
    }
    // Share only accounting and the attempt budget, NEVER the certificates.
    pub(super) fn share_statistics(&mut self, other: &Self) {
        self.attempts = other.attempts.clone();
        self.learned = other.learned.clone();
        self.fixed = other.fixed.clone();
        self.pruned = other.pruned.clone();
    }
}

impl Conflict {
    // Err: contradicted by this node or more than one literal is unassigned.
    // Ok(None): all match. Ok(Some): one remaining forbidden literal.
    fn applicable(&self, fixed: &FixedVarMap) -> Result<Option<(usize, usize)>, ()> {
        let mut missing = None;
        for &(i, v) in &self.literals {
            match fixed.get(&i) {
                Some(&x) if x != v => return Err(()),
                Some(_) => {}
                None if missing.is_some() => return Err(()),
                None => missing = Some((i, v)),
            }
        }
        Ok(missing)
    }
}

impl BBSolver {
    pub(super) fn can_learn_conflict(&self, fixed: &FixedVarMap) -> bool {
        self.options.conflict_learning
            && !fixed.is_empty()
            && fixed.len() <= 32
            && self.prepared_cycle_bound.is_some()
            && self.conflicts.attempts.load(Ordering::Relaxed) < 128
    }
    /// (attempts, learned, propagated fixings, pruned nodes), including subsearches.
    pub fn conflict_statistics(&self) -> (usize, usize, usize, usize) {
        let p = &self.conflicts;
        (
            p.attempts.load(Ordering::Relaxed).min(128),
            p.learned.load(Ordering::Relaxed),
            p.fixed.load(Ordering::Relaxed),
            p.pruned.load(Ordering::Relaxed),
        )
    }

    pub(super) fn learn_conflict(&self, fixed: &FixedVarMap) {
        if !self.options.conflict_learning || fixed.is_empty() || fixed.len() > 32 {
            return;
        }
        let Some(prepared) = &self.prepared_cycle_bound else {
            return;
        };
        let cutoff = self.pruning_upper_bound();
        if !cutoff.is_finite() {
            return;
        }
        if self.conflicts.attempts.fetch_add(1, Ordering::Relaxed) >= 128 {
            return;
        }
        let start = Instant::now();
        let budget = Duration::from_millis(2);
        let certify = |f: &FixedVarMap| {
            let remaining = budget.saturating_sub(start.elapsed());
            if remaining.is_zero() {
                return None;
            }
            let free = (0..self.input_qubo.num_x())
                .filter(|i| !f.contains_key(i))
                .collect::<Vec<_>>();
            prepared
                .conditioned_to(&free, f, remaining.min(Duration::from_micros(200)), cutoff)
                .map(|b| b.lower)
                .filter(|b| b.is_finite() && *b >= cutoff)
        };
        let mut core = fixed.clone();
        // Reprove independently: inherited node bounds and presolve relations
        // cannot justify removing assumptions from a conflict.
        let Some(mut lower) = certify(&core) else {
            return;
        };
        let mut keys = core.keys().copied().collect::<Vec<_>>();
        keys.sort_unstable();
        for i in keys {
            if start.elapsed() >= budget {
                break;
            }
            let v = core.remove(&i).unwrap();
            if let Some(bound) = certify(&core) {
                lower = bound;
            } else {
                core.insert(i, v);
            }
        }
        // Reject full-path certificates: they add no cross-branch information.
        if core.len() > 8 || core.len() == fixed.len() {
            return;
        }
        let mut literals = core.into_iter().collect::<Vec<_>>();
        literals.sort_unstable();
        let mut entries = self.conflicts.entries.lock().unwrap();
        if let Some(old) = entries.iter_mut().find(|c| c.literals == literals) {
            old.lower = old.lower.max(lower);
        } else if entries.len() < 128 {
            entries.push(Conflict { literals, lower });
            self.conflicts.learned.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub(super) fn apply_conflicts(&self, node: &mut QuboBBNode) -> (bool, bool) {
        if !self.options.conflict_learning {
            return (false, false);
        }
        let entries = self.conflicts.entries.lock().unwrap();
        let mut changed = false;
        loop {
            let mut pass = false;
            for c in entries.iter() {
                let Ok(missing) = c.applicable(&node.fixed_variables) else {
                    continue;
                };
                // Strict comparison first: never reuse a proof under a looser
                // cutoff. Existing pruning records externally discarded regions.
                if c.lower < self.pruning_upper_bound()
                    || !self.prunes_with_incumbent(c.lower, self.best_solution_value)
                {
                    continue;
                }
                if let Some((i, v)) = missing {
                    node.fixed_variables.insert(i, 1 - v);
                    node.solution[i] = (1 - v) as f64;
                    self.conflicts.fixed.fetch_add(1, Ordering::Relaxed);
                    changed = true;
                    pass = true;
                } else {
                    self.conflicts.pruned.fetch_add(1, Ordering::Relaxed);
                    return (true, changed);
                }
            }
            if !pass {
                return (false, changed);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn node() -> QuboBBNode {
        QuboBBNode {
            fixed_variables: FixedVarMap::default(),
            lower_bound: -10.0,
            solution: Array1::from_elem(2, 0.5),
            run_heuristic: false,
            subproblem_state: None,
            branch_origin: None,
            probe_bounds: None,
            probe_presolved: None,
        }
    }
    #[test]
    fn cutoff_scope_unit_propagation_and_external_proof() {
        let mut options = SolverOptions::new();
        options.conflict_learning = true;
        let mut s = BBSolver::new(Qubo::new(sprs::CsMat::eye(2)), options.clone());
        s.best_solution_value = 20.0;
        s.conflicts.entries.lock().unwrap().push(Conflict {
            literals: vec![(0, 1), (1, 0)],
            lower: 12.0,
        });
        let mut n = node();
        n.fixed_variables.insert(0, 1);
        assert_eq!(s.apply_conflicts(&mut n), (false, false));
        s.tighten_external_cutoff(10.0);
        assert_eq!(s.apply_conflicts(&mut n), (false, true));
        assert_eq!(n.fixed_variables[&1], 1);
        assert_eq!(n.lower_bound, -10.0);
        assert_eq!(s.cutoff.discarded_bound(), 12.0);
        let mut closed = node();
        closed.fixed_variables = [(0, 1), (1, 0)].into_iter().collect();
        assert_eq!(s.apply_conflicts(&mut closed), (true, false));
        let other = BBSolver::new(Qubo::new(sprs::CsMat::eye(2)), options);
        assert!(other.conflicts.entries.lock().unwrap().is_empty());
    }
    #[test]
    fn matching_requires_all_but_one_literal() {
        let c = Conflict {
            literals: vec![(0, 1), (2, 0)],
            lower: 4.0,
        };
        let mut f = FixedVarMap::default();
        assert!(c.applicable(&f).is_err());
        f.insert(0, 1);
        assert_eq!(c.applicable(&f), Ok(Some((2, 0))));
        f.insert(2, 0);
        assert_eq!(c.applicable(&f), Ok(None));
        f.insert(2, 1);
        assert!(c.applicable(&f).is_err());
    }
    #[test]
    fn enabled_search_matches_enumeration() {
        for seed in 0..8 {
            let n = 7;
            let mut m = sprs::TriMat::new((n, n));
            for i in 0..n {
                for j in i + 1..n {
                    m.add_triplet(
                        i,
                        j,
                        2.0 * (((seed * 13 + i * 7 + j * 11) % 9) as f64 - 4.0),
                    );
                }
            }
            let q = Qubo::new_with_c(
                m.to_csr(),
                Array1::from_iter((0..n).map(|i| (i % 3) as f64 - 2.0)),
            );
            let optimum = (0..1 << n)
                .map(|mask| q.eval_usize(&Array1::from_iter((0..n).map(|i| (mask >> i) & 1))))
                .fold(f64::INFINITY, f64::min);
            let mut options = SolverOptions::new();
            options.conflict_learning = true;
            options.verbose = 0;
            let mut s = BBSolver::new(q.clone(), options);
            let (x, value) = s.solve();
            assert_eq!(q.eval_usize(&x), optimum);
            assert!((value - optimum).abs() < 1e-9);
        }
    }
    #[test]
    fn extracted_conflict_is_valid_for_every_matching_assignment() {
        let q = Qubo::from_vec(vec![], vec![], vec![], vec![10.0, 1.0, 1.0, 1.0], 4);
        let mut options = SolverOptions::new();
        options.conflict_learning = true;
        let mut s = BBSolver::new(q.clone(), options);
        s.best_solution_value = 5.0;
        let fixed = [(0, 1), (1, 0), (2, 0)].into_iter().collect();
        s.learn_conflict(&fixed);
        let entries = s.conflicts.entries.lock().unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].literals, vec![(0, 1)]);
        for mask in 0..16 {
            let x = Array1::from_iter((0..4).map(|i| (mask >> i) & 1));
            if x[0] == 1 {
                assert!(q.eval_usize(&x) >= entries[0].lower);
            }
        }
    }
}
