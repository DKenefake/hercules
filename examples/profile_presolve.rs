//! Run with `cargo run --release --example profile_presolve -- [qubo paths...]`.
//! Solve arguments: --solve path [momentum] [seconds] [probes] [probe_max_free]
//! [probe_seconds] [weak_roof] [relations] [pairs] [root_probes] [root_seconds]
//! [symmetry] [low_degree] [dominant_edge] [degree_three] [components]
//! [branch_strategy: any non-strong rule, default LargestEdges].
use hercules::branch_subproblem::SubProblemSelection;
use hercules::branchbound::BBSolver;
use hercules::preprocess::{make_sub_problem, preprocess_qubo, shift_qubo};
use hercules::qubo::Qubo;
use hercules::solver_options::SolverOptions;
use hercules::subproblemsolvers::mixingcut_sdp::MixingCutSDPSolver;
use hercules::subproblemsolvers::roofdual::{roof_duality_presolve, PreparedRoofDual};
use hercules::variable_reduction::probe_limited;
use hercules::FixedVarMap;
use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

struct CountingAllocator;
static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);

fn record(size: usize) {
    if COUNTING.load(Ordering::Relaxed) {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(size, Ordering::Relaxed);
    }
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        record(size);
        unsafe { System.realloc(ptr, layout, size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn fingerprint(fixed: &FixedVarMap) -> u64 {
    fixed.iter().fold(0, |hash, (&i, &v)| {
        hash ^ ((i as u64 + 1).wrapping_mul(0x9e3779b97f4a7c15) ^ v as u64)
    })
}

fn measure(label: &str, mut operation: impl FnMut() -> u64) {
    black_box(operation());
    let start = Instant::now();
    let mut iterations = 0;
    while iterations < 10 || start.elapsed() < Duration::from_millis(250) {
        black_box(operation());
        iterations += 1;
    }
    let micros = start.elapsed().as_secs_f64() * 1e6 / f64::from(iterations);
    ALLOCATIONS.store(0, Ordering::Relaxed);
    BYTES.store(0, Ordering::Relaxed);
    COUNTING.store(true, Ordering::Relaxed);
    let checksum = black_box(operation());
    COUNTING.store(false, Ordering::Relaxed);
    println!(
        "{label}: us={micros:.2} allocations={} requested_bytes={} checksum={checksum}",
        ALLOCATIONS.load(Ordering::Relaxed),
        BYTES.load(Ordering::Relaxed)
    );
}

fn profile(path: &str) {
    let qubo = Qubo::read_qubo(path).make_symmetric();
    let shifted = shift_qubo(&qubo);
    let empty = FixedVarMap::default();
    let fixed: FixedVarMap = (0..qubo.num_x()).step_by(5).map(|i| (i, i % 2)).collect();
    println!("INSTANCE {path} n={} nnz={}", qubo.num_x(), qubo.q.nnz());
    measure("projection_20pct", || {
        let (sub, mapping, constant) = make_sub_problem(black_box(&qubo), &fixed);
        black_box((&sub, &mapping));
        ((sub.q.data().iter().sum::<f64>() + sub.c.sum() + constant) * 1e6).round() as i64 as u64
    });
    measure("roof_20pct", || {
        let result = roof_duality_presolve(black_box(&shifted), &fixed);
        fingerprint(&result.fixed_variables)
            ^ (result.lower_bound.unwrap() * 1e6).round() as i64 as u64
    });
    let prepared_roof = PreparedRoofDual::new(&shifted);
    measure("prepared_roof_20pct", || {
        let result = prepared_roof.solve(black_box(&fixed));
        fingerprint(&result.fixed_variables)
            ^ (result.lower_bound.unwrap() * 1e6).round() as i64 as u64
    });
    measure("preprocess_root", || {
        fingerprint(&preprocess_qubo(black_box(&shifted), &empty, true))
    });
    let mut options = SolverOptions::new();
    options.verbose = 0;
    let solver = BBSolver::new(qubo, options);
    measure("prepared_node_20pct", || {
        fingerprint(&solver.preprocess_fixed_variables(black_box(&fixed)))
    });
    measure("probe25_root", || {
        let (relations, fixes) = probe_limited(black_box(&shifted), &empty, true, 25);
        fingerprint(&fixes) ^ relations.constraints.len() as u64
    });
}

fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.first().is_some_and(|arg| arg == "--solve") {
        let path = args.get(1).expect("--solve requires a QUBO path");
        let mut qubo = Qubo::read_qubo(path);
        qubo.q = &qubo.q * 2.0;
        let mut options = SolverOptions::new();
        options.sub_problem_solver = SubProblemSelection::MixingCutSDP;
        let branch_strategy = args.get(17).map(String::as_str).unwrap_or("LargestEdges");
        assert!(
            matches!(
                branch_strategy,
                "FirstNotFixed"
                    | "MostViolated"
                    | "Random"
                    | "WorstApproximation"
                    | "WorstApproximation2"
                    | "MostEdges"
                    | "LargestEdges"
                    | "MostFixed"
                    | "RoundRobin"
                    | "LargestDiag"
                    | "MovingEdges"
                    | "ConnectedComponents"
            ),
            "expected a supported non-strong branching rule"
        );
        options.set_branch_strategy(Some(branch_strategy.to_owned()));
        options.threads = 64;
        options.node_probe_candidates = args
            .get(4)
            .map(|value| value.parse().expect("invalid probe budget"))
            .unwrap_or(options.node_probe_candidates);
        options.node_probe_max_free = args
            .get(5)
            .map(|value| value.parse().expect("invalid probe free limit"))
            .unwrap_or(options.node_probe_max_free);
        options.node_probe_max_seconds = args
            .get(6)
            .map(|value| value.parse().expect("invalid probe time budget"))
            .unwrap_or(options.node_probe_max_seconds);
        options.roof_dual_weak_persistencies = args
            .get(7)
            .map(|value| value.parse().expect("invalid weak roof flag (true/false)"))
            .unwrap_or(options.roof_dual_weak_persistencies);
        options.roof_dual_relation_penalties = args
            .get(8)
            .map(|value| {
                value
                    .parse()
                    .expect("invalid relation penalty flag (true/false)")
            })
            .unwrap_or(options.roof_dual_relation_penalties);
        options.root_pair_dominance = args
            .get(9)
            .map(|value| {
                value
                    .parse()
                    .expect("invalid pair dominance flag (true/false)")
            })
            .unwrap_or(options.root_pair_dominance);
        options.root_roof_probe_candidates = args
            .get(10)
            .map(|value| value.parse().expect("invalid root roof probe count"))
            .unwrap_or(options.root_roof_probe_candidates);
        options.root_roof_probe_max_seconds = args
            .get(11)
            .map(|value| value.parse().expect("invalid root roof probe time budget"))
            .unwrap_or(options.root_roof_probe_max_seconds);
        options.root_complement_symmetry = args
            .get(12)
            .map(|value| value.parse().expect("invalid complement symmetry flag"))
            .unwrap_or(options.root_complement_symmetry);
        options.root_low_degree_elimination = args
            .get(13)
            .map(|value| value.parse().expect("invalid elimination flag"))
            .unwrap_or(options.root_low_degree_elimination);
        options.root_dominant_edge_contraction = args
            .get(14)
            .map(|value| value.parse().expect("invalid contraction flag"))
            .unwrap_or(options.root_dominant_edge_contraction);
        options.root_degree_three_elimination = args
            .get(15)
            .map(|value| value.parse().expect("invalid degree-three flag"))
            .unwrap_or(options.root_degree_three_elimination);
        options.component_decomposition = args
            .get(16)
            .map(|value| value.parse().expect("invalid component decomposition flag"))
            .unwrap_or(options.component_decomposition);
        options.node_structural_reductions = args
            .get(18)
            .map(|value| value.parse().expect("invalid node reduction flag"))
            .unwrap_or(options.node_structural_reductions);
        options.small_block_elimination = args
            .get(19)
            .map(|value| value.parse().expect("invalid small-block flag"))
            .unwrap_or(options.small_block_elimination);
        options.component_cutoff_propagation = args
            .get(20)
            .map(|value| value.parse().expect("invalid component cutoff flag"))
            .unwrap_or(options.component_cutoff_propagation);
        options.root_cut_dominance = args
            .get(21)
            .map(|value| value.parse().expect("invalid root cut dominance flag"))
            .unwrap_or(options.root_cut_dominance);
        options.sdp_dual_fixing = args
            .get(22)
            .map(|value| value.parse().expect("invalid SDP dual fixing flag"))
            .unwrap_or(options.sdp_dual_fixing);
        println!("SDP_FIXING_CONFIG enabled={}", options.sdp_dual_fixing);
        options.max_time = args
            .get(3)
            .map(|value| value.parse().expect("invalid time limit"))
            .unwrap_or(60.0);
        let momentum = args
            .get(2)
            .map(|value| value.parse().expect("invalid momentum"))
            .unwrap_or(0.0);
        println!("CONFIG mixingcut=0.1.5 momentum={momentum} branch_strategy={branch_strategy} certified_dual=true max_iterations=400 stationarity_tolerance=1e-5 batch_size=64 probes={} probe_max_free={} probe_seconds={} weak_roof={} relation_penalties={} pair_dominance={}", options.node_probe_candidates, options.node_probe_max_free, options.node_probe_max_seconds, options.roof_dual_weak_persistencies, options.roof_dual_relation_penalties, options.root_pair_dominance);
        println!(
            "ROOT_PROBE_CONFIG candidates={} seconds={} symmetry={}",
            options.root_roof_probe_candidates,
            options.root_roof_probe_max_seconds,
            options.root_complement_symmetry
        );
        println!(
            "ROOT_REDUCTION_CONFIG low_degree={} dominant_edge={} degree_three={} components={} node_reductions={} small_blocks={} cutoff_propagation={} cut_dominance={}",
            options.root_low_degree_elimination,
            options.root_dominant_edge_contraction,
            options.root_degree_three_elimination,
            options.component_decomposition,
            options.node_structural_reductions,
            options.small_block_elimination,
            options.component_cutoff_propagation,
            options.root_cut_dominance
        );
        let start = Instant::now();
        let original_qubo = qubo.clone();
        let mut solver = BBSolver::new(qubo, options);
        solver.subproblem_solver = Box::new(MixingCutSDPSolver::with_momentum(momentum));
        let (solution, objective) = solver.solve();
        let seconds = start.elapsed().as_secs_f64();
        assert_eq!(solution.len(), original_qubo.num_x());
        assert!(solution.iter().all(|&value| value <= 1));
        let evaluated = original_qubo.eval_usize(&solution);
        assert!((evaluated - objective).abs() <= 1e-5);
        println!("VALIDATION binary=true original_objective={evaluated:.8}");
        println!(
            "ROOT relations={} fixed={}",
            solver.root_constraints.len(),
            solver.options.fixed_variables.len()
        );
        println!("PROBING {:?}", solver.node_probing_statistics());
        println!("REDUCTION {:?}", solver.root_reduction_statistics);
        println!("NODE_REDUCTION {:?}", solver.node_reduction_statistics());
        println!("DECOMPOSITION {:?}", solver.component_statistics());
        println!("CUTOFF {:?}", solver.cutoff_statistics());
        println!("SDP_FIXING {:?}", solver.sdp_fixing_statistics());
        let lower_bound = solver
            .nodes
            .iter()
            .map(|node| node.lower_bound)
            .fold(solver.best_solution_value, f64::min);
        let status = if solver.nodes.is_empty() {
            "Optimal"
        } else {
            "Incomplete"
        };
        println!(
            "END_TO_END seconds={:.6} solved={} visited={} objective={:.8} stopped={} lower_bound={:.8} remaining={} status={status}",
            seconds,
            solver.nodes_solved,
            solver.nodes_visited,
            solver.best_solution_value,
            solver.early_stop,
            lower_bound,
            solver.nodes.len(),
        );
    } else if args.is_empty() {
        for path in [
            "test_data/mk487a.qubo",
            "test_data/mk487b.qubo",
            "test_data/bqp/bqp100-1.qubo",
            "test_data/sg3dl101000.qubo",
        ] {
            profile(path);
        }
    } else {
        for path in &args {
            profile(path);
        }
    }
}
