use crate::branch_node::{QuboBBNode, SubProblemNodeState};
use crate::branch_subproblem::ConditionalLowerBound;
use crate::branch_subproblem::{SubProblemOptions, SubProblemResult, SubProblemSolver};
use crate::branchbound::BBSolver;
use crate::preprocess::make_sub_problem;
use crate::qubo::Qubo;
use mixingcut::sdp_solver::{solve_qubo_sdp_subproblem, SolveOptions, WarmStart};
use mixingcut::step_rules::StepRule;
use ndarray::{Array1, Array2};
use smolprng::{JsfLarge, PRNG};
use std::time::{Duration, Instant};
mod dual_fixing;

#[derive(Clone, Debug, Default)]
pub struct MixingCutSDPSolver {
    momentum: f64,
}

pub struct MixingCutSubProblemResult {
    pub lower_bound: f64,
    pub relaxed_solution: Array1<f64>,
    pub candidate_primal_solution: Option<Array1<usize>>,
    pub subproblem_state: Option<SubProblemNodeState>,
    pub conditional_bounds: Vec<ConditionalLowerBound>,
}

impl SubProblemResult for MixingCutSubProblemResult {
    fn take_conditional_bounds(&mut self) -> Vec<ConditionalLowerBound> {
        std::mem::take(&mut self.conditional_bounds)
    }
    fn lower_bound(&self) -> f64 {
        self.lower_bound
    }

    fn relaxed_solution(&self) -> Option<&Array1<f64>> {
        Some(&self.relaxed_solution)
    }

    fn candidate_primal_solution(&self) -> Option<&Array1<usize>> {
        self.candidate_primal_solution.as_ref()
    }

    fn subproblem_state(&self) -> Option<&SubProblemNodeState> {
        self.subproblem_state.as_ref()
    }

    fn into_parts(
        self: Box<Self>,
    ) -> (
        f64,
        Option<Array1<f64>>,
        Option<Array1<usize>>,
        Option<SubProblemNodeState>,
    ) {
        (
            self.lower_bound,
            Some(self.relaxed_solution),
            self.candidate_primal_solution,
            self.subproblem_state,
        )
    }
}

impl MixingCutSDPSolver {
    const NUM_HYPERPLANES: usize = 8;

    pub fn new(qubo: &Qubo) -> Self {
        let _ = qubo;
        Self::default()
    }

    /// Use MixingCut's coordinate momentum update, with zero retaining plain mixing.
    pub fn with_momentum(momentum: f64) -> Self {
        assert!(
            momentum.is_finite() && (0.0..1.0).contains(&momentum),
            "SDP momentum must be finite and in [0, 1)"
        );
        Self { momentum }
    }

    fn default_options(&self, num_free: usize, max_iterations: Option<usize>) -> SolveOptions {
        SolveOptions {
            rank: Some(((2.0 * (num_free + 1) as f64).sqrt().ceil() as usize).max(2)),
            seed: Some(7),
            max_iterations: max_iterations.unwrap_or(400),
            min_stationarity_iterations: 1,
            objective_tolerance: 1e-6,
            stationarity_tolerance: 1e-5,
            rounding_iterations: 0,
            beam_width: Some(0),
            // Without this, MixingCut 0.1.5 returns only its entrywise bound.
            compute_dual_bound: true,
            compute_rounding: false,
            step_rule: if self.momentum == 0.0 {
                StepRule::CoordNoStep
            } else {
                StepRule::CoordMomentum(self.momentum)
            },
            verbose: false,
            warm_start: WarmStart::Random,
        }
    }

    fn relaxed_solution_from_factor(factor_matrix: &Array2<f64>) -> Array1<f64> {
        let free_n = factor_matrix.nrows().saturating_sub(1);
        if free_n == 0 {
            return Array1::zeros(0);
        }

        let anchor = factor_matrix.row(free_n).to_owned();
        let mut reduced_solution = Array1::zeros(free_n);

        for i in 0..free_n {
            let sign_correlation = factor_matrix.row(i).dot(&anchor).clamp(-1.0, 1.0);
            reduced_solution[i] = 0.5 * (1.0 - sign_correlation);
        }

        reduced_solution
    }

    fn reduced_primal_solution_from_factor(
        factor_matrix: &Array2<f64>,
        reduced_qubo: &Qubo,
    ) -> Array1<usize> {
        let free_n = factor_matrix.nrows().saturating_sub(1);
        if free_n == 0 {
            return Array1::zeros(0);
        }

        let rank = factor_matrix.ncols();
        let mut prng = PRNG {
            generator: JsfLarge::from(7_u64),
        };
        let anchor = factor_matrix.row(free_n);
        let mut best_solution = Array1::zeros(free_n);
        let mut best_objective = f64::INFINITY;

        for _ in 0..Self::NUM_HYPERPLANES {
            let mut direction = Array1::zeros(rank);
            for component in &mut direction {
                *component = 2.0 * prng.gen_f64() - 1.0;
            }

            let anchor_dot = anchor.dot(&direction);
            let mut candidate = Array1::zeros(free_n);

            for i in 0..free_n {
                let same_side =
                    (factor_matrix.row(i).dot(&direction) >= 0.0) == (anchor_dot >= 0.0);
                candidate[i] = usize::from(!same_side);
            }

            let objective = reduced_qubo.eval_usize(&candidate);
            if objective < best_objective {
                best_objective = objective;
                best_solution = candidate;
            }
        }

        best_solution
    }
}

impl SubProblemSolver for MixingCutSDPSolver {
    fn for_reduced_qubo(&self, _: &Qubo) -> Option<Box<dyn SubProblemSolver + Sync>> {
        Some(Box::new(self.clone()))
    }

    fn solve_lower_bound(
        &self,
        bbsolver: &BBSolver,
        node: &QuboBBNode,
        sub_problem_options: Option<SubProblemOptions>,
    ) -> Box<dyn SubProblemResult> {
        let (sub_qubo, mapping, constant) = make_sub_problem(&bbsolver.qubo, &node.fixed_variables);

        if sub_qubo.num_x() == 0 {
            let mut solution = node.solution.clone();
            for (&index, &value) in &node.fixed_variables {
                solution[index] = value as f64;
            }
            let primal_solution = solution.mapv(|value| usize::from(value >= 0.5));
            return Box::new(MixingCutSubProblemResult {
                lower_bound: constant,
                relaxed_solution: solution,
                candidate_primal_solution: Some(primal_solution),
                subproblem_state: None,
                conditional_bounds: Vec::new(),
            });
        }

        let options = self.default_options(
            sub_qubo.num_x(),
            sub_problem_options.and_then(|opts| opts.max_iterations),
        );
        let result = solve_qubo_sdp_subproblem(&sub_qubo.q, &sub_qubo.c, &options);
        let mut conditional_bounds = Vec::new();
        let seconds = bbsolver.time_start + bbsolver.options.max_time
            - crate::branchbound_utils::get_current_time();
        if bbsolver.options.sdp_dual_fixing
            && sub_qubo.num_x() <= 128
            && seconds > 0.0
            && result.qubo_lower_bound + constant < bbsolver.pruning_upper_bound()
        {
            let start = Instant::now();
            conditional_bounds = dual_fixing::conditional_bounds(
                &sub_qubo,
                &result.dual_variables,
                result.qubo_lower_bound,
                start + Duration::from_secs_f64(seconds.min(0.005)),
            );
            let mut original = vec![0; sub_qubo.num_x()];
            for (&i, &j) in &mapping {
                original[j] = i;
            }
            for b in &mut conditional_bounds {
                b.variable = original[b.variable];
                b.zero = (b.zero + constant).next_down();
                b.one = (b.one + constant).next_down();
            }
            bbsolver.record_sdp_fixing(start.elapsed(), conditional_bounds.len());
        }
        let reduced_relaxed_solution = Self::relaxed_solution_from_factor(&result.factor_matrix);
        let reduced_primal_solution =
            Self::reduced_primal_solution_from_factor(&result.factor_matrix, &sub_qubo);

        let mut relaxed_solution = node.solution.clone();
        let mut primal_solution = Array1::zeros(node.solution.len());
        for (&original_index, &reduced_index) in &mapping {
            relaxed_solution[original_index] = reduced_relaxed_solution[reduced_index];
            primal_solution[original_index] = reduced_primal_solution[reduced_index];
        }
        for (&index, &value) in &node.fixed_variables {
            relaxed_solution[index] = value as f64;
            primal_solution[index] = value;
        }

        Box::new(MixingCutSubProblemResult {
            lower_bound: result.qubo_lower_bound + constant,
            relaxed_solution,
            candidate_primal_solution: Some(primal_solution),
            subproblem_state: None,
            conditional_bounds,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::MixingCutSDPSolver;
    use crate::branch_node::QuboBBNode;
    use crate::branch_subproblem::SubProblemSolver;
    use crate::branchbound::BBSolver;
    use crate::qubo::Qubo;
    use crate::solver_options::SolverOptions;
    use crate::FixedVarMap;
    use ndarray::Array1;
    use sprs::CsMat;

    #[test]
    fn conditional_bounds_include_node_constant_and_original_indices() {
        let mut q = sprs::TriMat::new((6, 6));
        for i in 0..6 {
            q.add_triplet(i, i, 2.0);
            for j in i + 1..6 {
                let w = ((i * 7 + j * 3) % 9) as f64 - 4.0;
                q.add_triplet(i, j, w);
                q.add_triplet(j, i, w);
            }
        }
        let qubo = Qubo::new_with_c(
            q.to_csr(),
            Array1::from_vec(vec![3.0, -5.0, 1.0, 2.0, -3.0, 4.0]),
        );
        let mut options = SolverOptions::new();
        options.sdp_dual_fixing = true;
        options.verbose = 0;
        let mut solver = BBSolver::new(qubo.clone(), options);
        solver.time_start = crate::branchbound_utils::get_current_time();
        let node = QuboBBNode {
            fixed_variables: [(1, 1), (4, 0)].into_iter().collect(),
            lower_bound: f64::NEG_INFINITY,
            solution: Array1::from_elem(6, 0.5),
            run_heuristic: false,
            subproblem_state: None,
        };
        let mut result =
            MixingCutSDPSolver::new(&solver.qubo).solve_lower_bound(&solver, &node, None);
        let bounds = result.take_conditional_bounds();
        assert_eq!(bounds.len(), 4);
        assert!(bounds.iter().all(|b| b.variable != 1 && b.variable != 4));
        for mask in 0..64 {
            if (mask >> 1) & 1 != 1 || (mask >> 4) & 1 != 0 {
                continue;
            }
            let x = Array1::from_iter((0..6).map(|i| (mask >> i) & 1));
            let value = qubo.eval_usize(&x);
            for b in &bounds {
                assert!(if x[b.variable] == 0 { b.zero } else { b.one } <= value + 1e-9);
            }
        }
    }

    #[test]
    fn momentum_requires_a_valid_coefficient_and_requests_dual_bounds() {
        for beta in [0.0, 0.5, 0.8] {
            let options = MixingCutSDPSolver::with_momentum(beta).default_options(7, Some(40));
            assert!(options.compute_dual_bound);
            assert_eq!(options.max_iterations, 40);
        }
        for beta in [-0.1, 1.0, f64::NAN, f64::INFINITY] {
            assert!(std::panic::catch_unwind(|| MixingCutSDPSolver::with_momentum(beta)).is_err());
        }
    }

    #[test]
    fn sdp_bounds_remain_valid_with_momentum_and_early_iteration_limits() {
        use crate::branch_subproblem::SubProblemOptions;
        use smolprng::{JsfLarge, PRNG};
        let mut prng = PRNG {
            generator: JsfLarge::from(827351u64),
        };
        for _ in 0..8 {
            let mut options = SolverOptions::new();
            options.verbose = 0;
            let solver = BBSolver::new(Qubo::make_random_qubo(7, &mut prng, 0.5), options);
            for fixed_variables in [
                FixedVarMap::default(),
                [(0, 1), (3, 0)].into_iter().collect(),
            ] {
                let optimum = (0..128)
                    .filter(|mask| fixed_variables.iter().all(|(&i, &v)| (mask >> i) & 1 == v))
                    .map(|mask| {
                        solver
                            .qubo
                            .eval_usize(&Array1::from_iter((0..7).map(|i| (mask >> i) & 1)))
                    })
                    .fold(f64::INFINITY, f64::min);
                let node = QuboBBNode {
                    lower_bound: f64::NEG_INFINITY,
                    solution: Array1::from_elem(7, 0.5),
                    fixed_variables,
                    run_heuristic: false,
                    subproblem_state: None,
                };
                for beta in [0.0, 0.5, 0.8] {
                    for limit in [0, 1, 40] {
                        let result = MixingCutSDPSolver::with_momentum(beta).solve_lower_bound(
                            &solver,
                            &node,
                            Some(SubProblemOptions::new(Some(limit))),
                        );
                        assert!(result.lower_bound().is_finite());
                        assert!(
                            result.lower_bound() <= optimum + 1e-8,
                            "beta={beta}, limit={limit}: bound={} optimum={optimum}",
                            result.lower_bound()
                        );
                        let primal = result.candidate_primal_solution().unwrap();
                        assert!(node.fixed_variables.iter().all(|(&i, &v)| primal[i] == v));
                        assert!(primal.iter().all(|&v| v <= 1));
                    }
                }
            }
        }
    }

    #[test]
    fn fully_fixed_sdp_node_returns_its_actual_completion() {
        let mut options = SolverOptions::new();
        options.verbose = 0;
        let solver = BBSolver::new(Qubo::new(CsMat::eye(3)), options);
        let node = QuboBBNode {
            lower_bound: f64::NEG_INFINITY,
            solution: Array1::from_elem(3, 0.5),
            fixed_variables: [(0, 1), (1, 0), (2, 1)].into_iter().collect(),
            run_heuristic: false,
            subproblem_state: None,
        };
        let result = MixingCutSDPSolver::with_momentum(0.8).solve_lower_bound(&solver, &node, None);
        let expected = Array1::from_vec(vec![1, 0, 1]);
        assert_eq!(result.candidate_primal_solution(), Some(&expected));
        assert!((result.lower_bound() - solver.qubo.eval_usize(&expected)).abs() < 1e-9);
    }

    #[test]
    fn mixingcut_backend_solves_small_node() {
        let q = CsMat::eye(3);
        let c = Array1::from_vec(vec![-1.0, -2.0, -3.0]);
        let qubo = Qubo::new_with_c(q, c);
        let solver = BBSolver::new(qubo, SolverOptions::new());
        let node = QuboBBNode {
            lower_bound: f64::NEG_INFINITY,
            solution: 0.5 * Array1::ones(3),
            fixed_variables: FixedVarMap::default(),
            run_heuristic: false,
            subproblem_state: None,
        };

        let mixingcut = MixingCutSDPSolver::new(&solver.qubo);
        let result = mixingcut.solve_lower_bound(&solver, &node, None);

        assert!(result.lower_bound().is_finite());
        let relaxed = result
            .relaxed_solution()
            .expect("expected relaxed solution");
        assert_eq!(relaxed.len(), 3);
        assert!(relaxed.iter().all(|value| (0.0..=1.0).contains(value)));
        let primal = result
            .candidate_primal_solution()
            .expect("expected candidate primal solution");
        assert!(primal.iter().all(|value| *value <= 1));
        assert!(result.subproblem_state().is_none());
    }
}
