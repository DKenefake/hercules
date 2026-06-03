use crate::branch_node::{QuboBBNode, SubProblemNodeState};
use crate::branch_subproblem::{SubProblemOptions, SubProblemResult, SubProblemSolver};
use crate::branchbound::BBSolver;
use crate::preprocess::make_sub_problem;
use crate::qubo::Qubo;
use mixingcut::sdp_solver::{solve_qubo_sdp_subproblem, SolveOptions, WarmStart};
use mixingcut::step_rules::StepRule;
use ndarray::{Array1, Array2};
use smolprng::{JsfLarge, PRNG};

#[derive(Clone, Debug, Default)]
pub struct MixingCutSDPSolver;

pub struct MixingCutSubProblemResult {
    pub lower_bound: f64,
    pub relaxed_solution: Array1<f64>,
    pub candidate_primal_solution: Option<Array1<usize>>,
    pub subproblem_state: Option<SubProblemNodeState>,
}

impl SubProblemResult for MixingCutSubProblemResult {
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
        Self
    }

    fn default_options(
        num_free: usize,
        max_iterations: Option<usize>,
    ) -> SolveOptions {
        SolveOptions {
            rank: Some(((2.0 * (num_free + 1) as f64).sqrt().ceil() as usize).max(2)),
            seed: Some(7),
            max_iterations: max_iterations.unwrap_or(400),
            min_stationarity_iterations: 1,
            objective_tolerance: 1e-6,
            stationarity_tolerance: 1e-5,
            rounding_iterations: 0,
            beam_width: Some(0),
            compute_dual_bound: false,
            compute_rounding: false,
            step_rule: StepRule::CoordNoStep,
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
                let same_side = (factor_matrix.row(i).dot(&direction) >= 0.0)
                    == (anchor_dot >= 0.0);
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
            return Box::new(MixingCutSubProblemResult {
                lower_bound: constant,
                relaxed_solution: solution,
                candidate_primal_solution: Some(node.solution.mapv(|value| usize::from(value >= 0.5))),
                subproblem_state: None,
            });
        }

        let options = Self::default_options(
            sub_qubo.num_x(),
            sub_problem_options.and_then(|opts| opts.max_iterations),
        );
        let result = solve_qubo_sdp_subproblem(&sub_qubo.q, &sub_qubo.c, &options);
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
        let relaxed = result.relaxed_solution().expect("expected relaxed solution");
        assert_eq!(relaxed.len(), 3);
        assert!(relaxed.iter().all(|value| (0.0..=1.0).contains(value)));
        let primal = result
            .candidate_primal_solution()
            .expect("expected candidate primal solution");
        assert!(primal.iter().all(|value| *value <= 1));
        assert!(result.subproblem_state().is_none());
    }
}
