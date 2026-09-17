//! Compare MixingCut update rules on identical root/conditional SDP matrices.
//! cargo run --release --example benchmark_sdp_momentum -- test_data/mk487a.qubo [iterations]
//! For the public BQP fixtures, append `--public-bqp` after the iteration limit.
use hercules::branchbound::BBSolver;
use hercules::preprocess::make_sub_problem;
use hercules::qubo::Qubo;
use hercules::solver_options::SolverOptions;
use hercules::subproblemsolvers::roofdual::iterative_roof_duality_presolve;
use hercules::FixedVarMap;
use mixingcut::sdp_solver::{
    absorb_linear_terms_into_hessian, qubo_hessian_to_sign_matrix, solve_maxcut_sdp_profiled,
    SolveOptions, WarmStart,
};
use mixingcut::step_rules::StepRule;
use std::time::Instant;

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "test_data/mk487a.qubo".into());
    let max_iterations = std::env::args()
        .nth(2)
        .map(|value| value.parse().expect("invalid iteration limit"))
        .unwrap_or(400);
    let mut qubo = Qubo::read_qubo(&path);
    if std::env::args().nth(3).as_deref() == Some("--public-bqp") {
        // Match the existing public BQP objective tests, not the MK convention.
        let mut quadratic = sprs::TriMat::new((qubo.num_x(), qubo.num_x()));
        for (&value, (i, j)) in &qubo.q {
            quadratic.add_triplet(i, j, value * if i == j { 2.0 } else { 4.0 });
        }
        qubo.q = quadratic.to_csr();
    } else {
        qubo.q = &qubo.q * 2.0;
    }
    let mut options = SolverOptions::new();
    options.verbose = 0;
    let solver = BBSolver::new(qubo, options);
    let initial = solver.preprocess_fixed_variables(&FixedVarMap::default());
    let mut root = initial.clone();
    root.extend(
        iterative_roof_duality_presolve(&solver.qubo_pp_form, &initial, solver.qubo.num_x())
            .fixed_variables,
    );
    let free: Vec<_> = (0..solver.qubo.num_x())
        .filter(|i| !root.contains_key(i))
        .collect();
    for percent in [0, 25, 50] {
        let mut fixed = root.clone();
        for (position, &i) in free.iter().enumerate().take(free.len() * percent / 100) {
            fixed.insert(i, position % 2);
        }
        let mut fixed = solver.preprocess_fixed_variables(&fixed);
        let roof =
            iterative_roof_duality_presolve(&solver.qubo_pp_form, &fixed, solver.qubo.num_x())
                .fixed_variables;
        fixed.extend(roof);
        let (reduced, _, constant) = make_sub_problem(&solver.qubo, &fixed);
        if reduced.num_x() == 0 {
            println!("FIXTURE additional_fixed_percent={percent} free=0 solved_by_presolve=true");
            continue;
        }
        let matrix =
            qubo_hessian_to_sign_matrix(&absorb_linear_terms_into_hessian(&reduced.q, &reduced.c));
        println!("FIXTURE path={path} additional_fixed_percent={percent} free={} nnz={} constant={constant:.9}", reduced.num_x(), matrix.nnz());
        // Alternate rule order to reduce warmup/order bias.
        for repeat in 0..3 {
            let rules = if repeat % 2 == 0 {
                [0.0, 0.5, 0.8]
            } else {
                [0.8, 0.5, 0.0]
            };
            for momentum in rules {
                let options = SolveOptions {
                    rank: Some(
                        ((2.0 * (reduced.num_x() + 1) as f64).sqrt().ceil() as usize).max(2),
                    ),
                    seed: Some(7),
                    max_iterations,
                    min_stationarity_iterations: 1,
                    objective_tolerance: 1e-6,
                    stationarity_tolerance: 1e-5,
                    rounding_iterations: 0,
                    beam_width: Some(0),
                    compute_dual_bound: true,
                    compute_rounding: false,
                    step_rule: if momentum == 0.0 {
                        StepRule::CoordNoStep
                    } else {
                        StepRule::CoordMomentum(momentum)
                    },
                    verbose: false,
                    warm_start: WarmStart::Random,
                };
                let start = Instant::now();
                let result = solve_maxcut_sdp_profiled(&matrix, &options);
                println!("SDP repeat={repeat} momentum={momentum} milliseconds={:.6} iteration_ms={:.6} finalization_ms={:.6} iterations={} rank={} lower_bound={:.9} relaxed_objective={:.9} status={:?}",
                    start.elapsed().as_secs_f64() * 1000.0,
                    result.iteration_seconds * 1000.0,
                    result.finalization_seconds * 1000.0,
                    result.solve_result.iterations,
                    result.solve_result.rank,
                    result.solve_result.dual_bound.expect("requested dual bound") + constant,
                    result.solve_result.relaxed_objective + constant,
                    result.solve_result.status);
            }
        }
    }
}
