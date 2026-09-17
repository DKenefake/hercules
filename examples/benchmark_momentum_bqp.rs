//! Compare MixingCut modes on the public BQP fixtures using the solver tests' convention.
//! Arguments: [repeats] [node_probe_candidates] [node_probe_max_free] [weak_roof]
//! [relation_penalties] [pair_dominance] [root_roof_probes] [root_roof_seconds]
//! [symmetry] [low_degree] [dominant_edge] [degree_three] [components].
use hercules::branch_stratagy::BranchStrategy;
use hercules::branch_subproblem::SubProblemSelection;
use hercules::branchbound::BBSolver;
use hercules::qubo::Qubo;
use hercules::solver_options::SolverOptions;
use hercules::subproblemsolvers::mixingcut_sdp::MixingCutSDPSolver;
use std::time::Instant;

fn load_public_bqp(path: &str) -> Qubo {
    let mut qubo = Qubo::read_qubo(path);
    // Match test_bqp50_objectives/test_bqp100_objectives: these fixtures are not
    // stored in the same coefficient convention as the MK runner's input.
    let mut quadratic = sprs::TriMat::new((qubo.num_x(), qubo.num_x()));
    for (&value, (i, j)) in &qubo.q {
        quadratic.add_triplet(i, j, value * if i == j { 2.0 } else { 4.0 });
    }
    qubo.q = quadratic.to_csr();
    qubo
}

fn main() {
    let repeats = std::env::args()
        .nth(1)
        .map(|value| value.parse::<usize>().expect("invalid repeat count"))
        .unwrap_or(3);
    assert!(repeats > 0);
    // Published optimal objectives, as used by the existing regression tests.
    let suites = [
        (
            50,
            [
                -2098., -3702., -4626., -3544., -4012., -3693., -4520., -4216., -3780., -3507.,
            ],
        ),
        (
            100,
            [
                -7970., -11036., -12723., -10368., -9083., -10210., -10125., -11435., -11455.,
                -12565.,
            ],
        ),
    ];
    let momenta = [0.0, 0.5, 0.8];
    println!("CONFIG mixingcut=0.1.5 max_iterations=400 stationarity_tolerance=1e-5 batch_size=64 seconds_per_solve=10 repeats={repeats} input=public_bqp_test_convention");
    for repeat in 0..repeats {
        for (size, objectives) in &suites {
            for (index, expected) in objectives.iter().enumerate() {
                let instance = format!("bqp{size}-{}", index + 1);
                let path = format!("test_data/bqp/{instance}.qubo");
                for offset in 0..momenta.len() {
                    let momentum = momenta[(repeat + index + offset) % momenta.len()];
                    let qubo = load_public_bqp(&path);
                    let mut options = SolverOptions::new();
                    options.sub_problem_solver = SubProblemSelection::MixingCutSDP;
                    options.branch_strategy = BranchStrategy::LargestEdges;
                    options.threads = 64;
                    options.max_time = 10.0;
                    options.verbose = 0;
                    options.node_probe_candidates = std::env::args()
                        .nth(2)
                        .map(|value| value.parse().expect("invalid probe budget"))
                        .unwrap_or(options.node_probe_candidates);
                    options.node_probe_max_free = std::env::args()
                        .nth(3)
                        .map(|value| value.parse().expect("invalid probe size limit"))
                        .unwrap_or(options.node_probe_max_free);
                    options.roof_dual_weak_persistencies = std::env::args()
                        .nth(4)
                        .map(|value| value.parse().expect("invalid weak roof flag (true/false)"))
                        .unwrap_or(options.roof_dual_weak_persistencies);
                    options.roof_dual_relation_penalties = std::env::args()
                        .nth(5)
                        .map(|value| {
                            value
                                .parse()
                                .expect("invalid relation penalty flag (true/false)")
                        })
                        .unwrap_or(options.roof_dual_relation_penalties);
                    options.root_pair_dominance = std::env::args()
                        .nth(6)
                        .map(|value| {
                            value
                                .parse()
                                .expect("invalid pair dominance flag (true/false)")
                        })
                        .unwrap_or(options.root_pair_dominance);
                    options.root_roof_probe_candidates = std::env::args()
                        .nth(7)
                        .map(|value| value.parse().expect("invalid root roof probe count"))
                        .unwrap_or(options.root_roof_probe_candidates);
                    options.root_roof_probe_max_seconds = std::env::args()
                        .nth(8)
                        .map(|value| value.parse().expect("invalid root roof probe time budget"))
                        .unwrap_or(options.root_roof_probe_max_seconds);
                    options.root_complement_symmetry = std::env::args()
                        .nth(9)
                        .map(|value| value.parse().expect("invalid complement symmetry flag"))
                        .unwrap_or(options.root_complement_symmetry);
                    options.root_low_degree_elimination = std::env::args()
                        .nth(10)
                        .map(|value| value.parse().expect("invalid elimination flag"))
                        .unwrap_or(options.root_low_degree_elimination);
                    options.root_dominant_edge_contraction = std::env::args()
                        .nth(11)
                        .map(|value| value.parse().expect("invalid contraction flag"))
                        .unwrap_or(options.root_dominant_edge_contraction);
                    options.root_degree_three_elimination = std::env::args()
                        .nth(12)
                        .map(|value| value.parse().expect("invalid degree-three flag"))
                        .unwrap_or(options.root_degree_three_elimination);
                    options.component_decomposition = std::env::args()
                        .nth(13)
                        .map(|value| value.parse().expect("invalid component decomposition flag"))
                        .unwrap_or(options.component_decomposition);
                    let start = Instant::now();
                    let mut solver = BBSolver::new(qubo, options);
                    solver.subproblem_solver =
                        Box::new(MixingCutSDPSolver::with_momentum(momentum));
                    let setup_seconds = start.elapsed().as_secs_f64();
                    let (_, objective) = solver.solve();
                    let seconds = start.elapsed().as_secs_f64();
                    let lower_bound = solver
                        .nodes
                        .iter()
                        .map(|node| node.lower_bound)
                        .fold(solver.best_solution_value, f64::min);
                    let optimal = solver.nodes.is_empty();
                    let matches_expected = (objective - expected).abs() <= 1e-5;
                    let status = if optimal { "Optimal" } else { "Incomplete" };
                    let stats = solver.node_probing_statistics();
                    println!(
                        "REDUCTION instance={instance} momentum={momentum} {:?}",
                        solver.root_reduction_statistics
                    );
                    println!(
                        "BQP instance={instance} repeat={} momentum={momentum} seconds={seconds:.6} setup_seconds={setup_seconds:.6} solved={} visited={} objective={objective:.8} lower_bound={lower_bound:.8} expected={expected} remaining={} status={status} matches_expected={matches_expected}",
                        repeat + 1, solver.nodes_solved, solver.nodes_visited, solver.nodes.len(),
                    );
                    println!(
                        "PROBING instance={instance} repeat={} momentum={momentum} {stats:?}",
                        repeat + 1
                    );
                    assert!(
                        lower_bound <= expected + 1e-5,
                        "invalid bound on {instance}"
                    );
                    assert!(
                        objective >= expected - 1e-5,
                        "invalid objective on {instance}"
                    );
                    assert!(
                        !optimal || matches_expected,
                        "false optimal status on {instance}"
                    );
                }
            }
        }
    }
}
