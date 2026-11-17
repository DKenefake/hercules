//! Whole-node integration for exact bounded-width elimination.
use super::*;
use crate::preprocess::width_elimination::{solve, Declined, MAX_FREE};
use std::sync::atomic::Ordering;
use std::time::Duration;
impl BBSolver {
    pub(super) fn try_width_elimination(
        &self,
        node: &QuboBBNode,
        candidate: Option<&(Array1<usize>, f64)>,
    ) -> Option<ProcessNodeState> {
        if !self.options.bounded_width_elimination
            || self.qubo.num_x() - node.fixed_variables.len() > MAX_FREE
        {
            return None;
        }
        let remaining = self.time_start + self.options.max_time - get_current_time();
        if remaining <= 0.0 || !remaining.is_finite() {
            return None;
        }
        let counters = &self.work_counters;
        counters.width_attempts.fetch_add(1, Ordering::Relaxed);
        let result = solve(
            &self.input_qubo,
            &node.fixed_variables,
            Duration::from_secs_f64(remaining.min(0.005)),
        );
        match result {
            Ok(x) => {
                let value = self.qubo.eval_usize(&x);
                if !value.is_finite() {
                    counters.width_declined[2].fetch_add(1, Ordering::Relaxed);
                    return None;
                }
                counters.width_solved.fetch_add(1, Ordering::Relaxed);
                Some(ProcessNodeState::finished(
                    Self::better_candidate(candidate.cloned(), Some((x, value))),
                    NodeLoggingAction::Solved,
                ))
            }
            Err(reason) => {
                let index = match reason {
                    Declined::Width => 0,
                    Declined::Budget => 1,
                    Declined::Arithmetic => 2,
                };
                counters.width_declined[index].fetch_add(1, Ordering::Relaxed);
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn brute(q: &Qubo, fixed: &FixedVarMap) -> f64 {
        (0..1usize << q.num_x())
            .filter(|m| fixed.iter().all(|(&i, &v)| (m >> i) & 1 == v))
            .map(|m| q.eval_usize(&Array1::from_iter((0..q.num_x()).map(|i| (m >> i) & 1))))
            .fold(f64::INFINITY, f64::min)
    }

    #[test]
    fn asymmetric_conditioned_qubos_match_exhaustive_optima() {
        for seed in 0..24 {
            let mut q = sprs::TriMat::new((9, 9));
            for i in 0..9 {
                for j in 0..9 {
                    q.add_triplet(i, j, ((i * 17 + j * 7 + seed * 3) % 11) as f64 - 5.0);
                }
            }
            let q = Qubo::new_with_c(
                q.to_csr(),
                Array1::from_iter((0..9).map(|i| ((i * 3 + seed) % 7) as f64 - 3.0)),
            );
            for a in 0..2 {
                for b in 0..2 {
                    let fixed = [(1, a), (7, b)].into_iter().collect();
                    let x = solve(&q, &fixed, Duration::from_secs(1)).unwrap();
                    assert_eq!(x[1], a);
                    assert_eq!(x[7], b);
                    assert_eq!(q.eval_usize(&x), brute(&q, &fixed));
                }
            }
        }
    }

    #[test]
    fn handles_empty_and_full_bitset_boundary() {
        let q = Qubo::new(sprs::CsMat::zero((0, 0)));
        assert!(solve(&q, &FixedVarMap::default(), Duration::from_secs(1))
            .unwrap()
            .is_empty());
        let mut q = sprs::TriMat::new((128, 128));
        q.add_triplet(0, 127, 4.0);
        let mut c = Array1::zeros(128);
        c[0] = -1.0;
        c[127] = -1.0;
        let q = Qubo::new_with_c(q.to_csr(), c);
        let x = solve(&q, &FixedVarMap::default(), Duration::from_secs(1)).unwrap();
        assert_eq!(q.eval_usize(&x), -1.0);
        let fixed = (0..128).map(|i| (i, i % 2)).collect();
        let x = solve(&q, &fixed, Duration::from_secs(1)).unwrap();
        assert!(fixed.iter().all(|(&i, &v)| x[i] == v));
    }

    #[test]
    fn refusals_do_not_change_input() {
        let q = Qubo::new_with_c(sprs::CsMat::eye(10), Array1::from_elem(10, -1.0));
        assert_eq!(
            solve(&q, &FixedVarMap::default(), Duration::ZERO),
            Err(Declined::Budget)
        );
        let mut dense = sprs::TriMat::new((10, 10));
        for i in 0..10 {
            for j in i + 1..10 {
                dense.add_triplet(i, j, 2.0);
            }
        }
        assert_eq!(
            solve(
                &Qubo::new(dense.to_csr()),
                &FixedVarMap::default(),
                Duration::from_secs(1)
            ),
            Err(Declined::Width)
        );
        let q = Qubo::new_with_c(sprs::CsMat::eye(1), Array1::from_elem(1, 1e20));
        assert_eq!(
            solve(&q, &FixedVarMap::default(), Duration::from_secs(1)),
            Err(Declined::Arithmetic)
        );
        assert_eq!(q.c[0], 1e20);
        let q = Qubo::new(sprs::CsMat::zero((129, 129)));
        assert_eq!(
            solve(&q, &FixedVarMap::default(), Duration::from_secs(1)),
            Err(Declined::Width)
        );
    }

    #[test]
    fn enabled_solver_closes_an_odd_cycle_after_the_root_relaxation() {
        let mut q = sprs::TriMat::new((15, 15));
        for i in 0..15 {
            q.add_triplet(i, (i + 1) % 15, 4.0);
        }
        let q = Qubo::new_with_c(q.to_csr(), Array1::from_elem(15, -2.0));
        let mut options = SolverOptions::new();
        options.verbose = 0;
        assert!(options.bounded_width_elimination);
        // Isolate whole-node elimination from other exact root closure paths.
        options.objective_lattice_pruning = false;
        options.sdp_cycle_bound = false;
        options.width_region_elimination = false;
        options.root_roof_probe_candidates = 0;
        options.node_probe_candidates = 0;
        options.root_low_degree_elimination = false;
        options.root_dominant_edge_contraction = false;
        options.node_structural_reductions = false;
        options.root_complement_symmetry = false;
        options.component_decomposition = false;
        let mut solver = BBSolver::new(q.clone(), options);
        let (x, objective) = solver.solve();
        assert!((objective + 14.0).abs() < 1e-10);
        assert_eq!(q.eval_usize(&x), -14.0);
        assert!(solver.nodes.is_empty());
        assert!(solver.search_work_statistics().width_solved > 0);
        // Initialization still runs its existing root relaxation; no node
        // relaxation is needed after exact width elimination takes over.
        assert_eq!(solver.node_probing_statistics().subproblem_calls, 1);
    }
}
