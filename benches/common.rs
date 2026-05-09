use hercules::branch_stratagy::BranchStrategy;
use hercules::branch_node::QuboBBNode;
use hercules::branchbound::BBSolver;
use hercules::initial_points::generate_random_binary_point;
use hercules::qubo::Qubo;
use hercules::solver_options::SolverOptions;
use ndarray::Array1;
use smolprng::{JsfLarge, PRNG};
use std::collections::HashMap;

#[allow(dead_code)]
pub struct BenchData {
    pub qubo_small: Qubo,
    pub qubo_medium: Qubo,
    pub qubo_enum: Qubo,
    pub qubo_solve: Qubo,
    pub qubo_gka6a: Qubo,
    pub qubo_test_large: Qubo,
    pub x_small: Array1<usize>,
    pub x_medium: Array1<usize>,
    pub selected_small: Vec<usize>,
    pub empty_fixed: HashMap<usize, usize>,
    pub process_solver: BBSolver,
    pub process_node: QuboBBNode,
}

pub fn make_solver_options() -> SolverOptions {
    let mut options = SolverOptions::new();
    options.branch_strategy = BranchStrategy::MostFixed;
    options.verbose = 0;
    options.threads = 1;
    options.max_time = f64::INFINITY;
    options
}

pub fn make_bench_data() -> BenchData {
    let mut prng = PRNG {
        generator: JsfLarge::from(12_345_679_u64),
    };
    let mut enum_prng = PRNG {
        generator: JsfLarge::from(22_345_679_u64),
    };
    let mut solve_prng = PRNG {
        generator: JsfLarge::from(32_345_679_u64),
    };

    let qubo_small = Qubo::make_random_qubo(64, &mut prng, 0.1);
    let qubo_medium = Qubo::make_random_qubo(128, &mut prng, 0.08);
    let qubo_enum = Qubo::make_random_qubo(10, &mut enum_prng, 0.25);
    let qubo_solve = Qubo::make_random_qubo(96, &mut solve_prng, 0.08);
    let qubo_gka6a = Qubo::read_qubo("test_data/gka6a.qubo");
    let qubo_test_large = Qubo::read_qubo("test_data/test_large.qubo");
    let x_small = generate_random_binary_point(qubo_small.num_x(), &mut prng, 0.5);
    let x_medium = generate_random_binary_point(qubo_medium.num_x(), &mut prng, 0.5);
    let selected_small: Vec<usize> = (0..qubo_small.num_x()).collect();
    let empty_fixed = HashMap::new();

    let process_solver = BBSolver::new(qubo_small.clone(), make_solver_options());
    let process_node = QuboBBNode {
        lower_bound: f64::NEG_INFINITY,
        solution: 0.5 * Array1::ones(process_solver.qubo.num_x()),
        fixed_variables: process_solver.options.fixed_variables.clone(),
    };

    BenchData {
        qubo_small,
        qubo_medium,
        qubo_enum,
        qubo_solve,
        qubo_gka6a,
        qubo_test_large,
        x_small,
        x_medium,
        selected_small,
        empty_fixed,
        process_solver,
        process_node,
    }
}
