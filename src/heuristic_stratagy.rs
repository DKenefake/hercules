use crate::branch_node::QuboBBNode;
use crate::branchbound::BBSolver;
use crate::{local_search_utils, utils};
use ndarray::Array1;

#[derive(Clone, Copy)]
pub enum HeuristicSelection {
    SimpleRounding,
    LocalSearch,
}

impl HeuristicSelection {
    fn apply_node_fixings(solution: &mut Array1<usize>, node: &QuboBBNode) {
        for (&index, &value) in &node.fixed_variables {
            solution[index] = value;
        }
    }

    fn unfixed_variables(solver: &BBSolver, node: &QuboBBNode) -> Vec<usize> {
        (0..solver.qubo.num_x())
            .filter(|index| !node.fixed_variables.contains_key(index))
            .collect()
    }

    pub fn make_heuristic(self, solver: &BBSolver, node: &QuboBBNode) -> (Array1<usize>, f64) {
        match self {
            Self::SimpleRounding => Self::simple_rounding(solver, node),
            Self::LocalSearch => Self::local_search(solver, node),
        }
    }

    pub fn simple_rounding(solver: &BBSolver, node: &QuboBBNode) -> (Array1<usize>, f64) {
        // round the solution to the nearest integer
        let mut rounded_solution = utils::rounded_vector(&node.solution);
        Self::apply_node_fixings(&mut rounded_solution, node);
        let objective = solver.qubo.eval_usize(&rounded_solution);

        (rounded_solution, objective)
    }

    pub fn local_search(solver: &BBSolver, node: &QuboBBNode) -> (Array1<usize>, f64) {
        // round the solution to the nearest integer
        let mut rounded_solution = utils::rounded_vector(&node.solution);
        Self::apply_node_fixings(&mut rounded_solution, node);

        let selected_vars = Self::unfixed_variables(solver, node);
        let (mut solution, _) = local_search_utils::two_step_local_search_descent(
            &solver.qubo,
            &rounded_solution,
            &selected_vars,
            100,
        );
        Self::apply_node_fixings(&mut solution, node);
        let objective = solver.qubo.eval_usize(&solution);
        (solution, objective)
    }
}

#[cfg(test)]
mod tests {
    use crate::branch_node::QuboBBNode;
    use crate::branchbound::BBSolver;
    use crate::heuristic_stratagy::HeuristicSelection;
    use crate::qubo::Qubo;
    use crate::solver_options::SolverOptions;
    use crate::FixedVarMap;
    use crate::utils;
    use ndarray::Array1;
    use smolprng::{JsfLarge, PRNG};

    #[test]
    fn test_local_search() {
        // generate a random QUBO
        let mut prng = PRNG {
            generator: JsfLarge::default(),
        };

        let p = Qubo::make_random_qubo(50, &mut prng, 0.2);
        let mut solver = BBSolver::new(p.clone(), SolverOptions::new());

        solver.options.heuristic = HeuristicSelection::LocalSearch;

        for _ in 0..100 {
            // generate a random point inside with x in [0, 1]^10 with
            let mut x_0 = Array1::zeros(p.num_x());
            (0..p.num_x()).for_each(|i| x_0[i] = prng.gen_f64());

            let obj_0 = solver.qubo.eval(&x_0);

            // make a dummy node
            let node = QuboBBNode {
                lower_bound: f64::NEG_INFINITY,
                solution: x_0.clone(),
                fixed_variables: FixedVarMap::default(),
                run_heuristic: false,
                subproblem_state: None,
            };

            // compute the next step
            let (_, obj_1) = solver.options.heuristic.make_heuristic(&solver, &node);

            // ensure that the objective has not increased
            assert!(obj_1 <= obj_0);
        }
    }

    #[test]
    fn test_random_search() {
        // generate a random QUBO
        let mut prng = PRNG {
            generator: JsfLarge::default(),
        };

        let p = Qubo::make_random_qubo(50, &mut prng, 0.2);
        let mut solver = BBSolver::new(p.clone(), SolverOptions::new());

        solver.options.heuristic = HeuristicSelection::SimpleRounding;

        for _ in 0..100 {
            // generate a random point inside with x in [0, 1]^10 with
            let mut x_0 = Array1::zeros(p.num_x());
            (0..p.num_x()).for_each(|i| x_0[i] = prng.gen_f64());

            // make a dummy node
            let node = QuboBBNode {
                lower_bound: f64::NEG_INFINITY,
                solution: x_0.clone(),
                fixed_variables: FixedVarMap::default(),
                run_heuristic: false,
                subproblem_state: None,
            };

            let rounded_sol = utils::rounded_vector(&x_0);

            let obj_0 = solver.qubo.eval_usize(&rounded_sol);

            // compute the next step
            let (_, obj_1) = solver.options.heuristic.make_heuristic(&solver, &node);

            println!("Obj 0: {}, Obj 1: {}", obj_0, obj_1);

            // ensure that the objective has not increased
            assert!(obj_1 <= obj_0);
        }
    }

    #[test]
    fn heuristics_respect_node_fixings() {
        let q = Qubo::new(sprs::CsMat::eye(4));
        let solver = BBSolver::new(q, SolverOptions::new());
        let mut fixed_variables = FixedVarMap::default();
        fixed_variables.insert(1, 1);
        fixed_variables.insert(3, 0);
        let node = QuboBBNode {
            lower_bound: f64::NEG_INFINITY,
            solution: Array1::from_vec(vec![0.2, 0.1, 0.8, 0.9]),
            fixed_variables,
            run_heuristic: false,
            subproblem_state: None,
        };

        let (rounded, _) = HeuristicSelection::SimpleRounding.make_heuristic(&solver, &node);
        assert_eq!(rounded[1], 1);
        assert_eq!(rounded[3], 0);

        let (searched, _) = HeuristicSelection::LocalSearch.make_heuristic(&solver, &node);
        assert_eq!(searched[1], 1);
        assert_eq!(searched[3], 0);
    }
}
