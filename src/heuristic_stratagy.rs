use crate::branch_node::QuboBBNode;
use crate::branchbound::BBSolver;
use crate::{local_search_utils, utils};
use ndarray::Array1;

pub enum HeuristicSelection {
    SimpleRounding,
    LocalSearch,
}

impl HeuristicSelection {
    pub fn make_heuristic(&self, solver: &BBSolver, node: &QuboBBNode) -> (Array1<usize>, f64) {
        match self {
            Self::SimpleRounding => Self::simple_rounding(solver, node),
            Self::LocalSearch => Self::local_search(solver, node),
        }
    }

    pub fn simple_rounding(solver: &BBSolver, node: &QuboBBNode) -> (Array1<usize>, f64) {
        // round the solution to the nearest integer
        let rounded_solution = utils::rounded_vector(&node.solution);
        let objective = solver.qubo.eval_usize(&rounded_solution);

        (rounded_solution, objective)
    }

    pub fn local_search(solver: &BBSolver, node: &QuboBBNode) -> (Array1<usize>, f64) {
        // round the solution to the nearest integer
        let rounded_solution = utils::rounded_vector(&node.solution);

        let mut x = rounded_solution;

        let mut x_1 =
            local_search_utils::two_step_local_search_improved(&solver.qubo, &x,);

        let mut steps = 0;

        while x_1 != x  && steps < 100 {
            x.clone_from(&x_1);
            x_1 = local_search_utils::two_step_local_search_improved(&solver.qubo, &x);
            steps += 1
        }

        let objective = solver.qubo.eval_usize(&x_1);

        (x_1, objective)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array1;
    use smolprng::{JsfLarge, PRNG};
    use crate::branchbound::BBSolver;
    use crate::heuristic_stratagy::HeuristicSelection;
    use crate::branch_node::QuboBBNode;
    use crate::qubo::Qubo;
    use crate::solver_options::SolverOptions;
    use crate::utils;

    #[test]
    fn test_local_search() {

        // generate a random QUBO
        let mut prng = PRNG {
            generator: JsfLarge::default(),
        };

        let p = Qubo::make_random_qubo(50, &mut prng, 0.2);
        let mut solver = BBSolver::new(p.clone(), SolverOptions::new());

        solver.options.heuristic = HeuristicSelection::LocalSearch;

        let selected_vars: Vec<usize> = (0..p.num_x()).collect();


        for _ in 0..100 {
            // generate a random point inside with x in [0, 1]^10 with
            let mut x_0 = Array1::zeros(p.num_x());
            (0..p.num_x()).for_each(|i| x_0[i] = prng.gen_f64());

            let obj_0 = p.eval(&x_0);


            // make a dummy node
            let node = QuboBBNode{
                lower_bound: f64::NEG_INFINITY,
                solution: x_0.clone(),
                fixed_variables: std::collections::HashMap::new(),
            };

            // compute the next step
            let ( _, obj_1) = solver.options.heuristic.make_heuristic(&solver, &node);

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

        let selected_vars: Vec<usize> = (0..p.num_x()).collect();


        for _ in 0..100 {
            // generate a random point inside with x in [0, 1]^10 with
            let mut x_0 = Array1::zeros(p.num_x());
            (0..p.num_x()).for_each(|i| x_0[i] = prng.gen_f64());

            // make a dummy node
            let node = QuboBBNode{
                lower_bound: f64::NEG_INFINITY,
                solution: x_0.clone(),
                fixed_variables: std::collections::HashMap::new(),
            };

            let rounded_sol = utils::rounded_vector(&x_0);

            let obj_0 = p.eval_usize(&rounded_sol);

            // compute the next step
            let ( _, obj_1) = solver.options.heuristic.make_heuristic(&solver, &node);

            // ensure that the objective has not increased
            assert!(obj_1 <= obj_0);
        }
    }
}
