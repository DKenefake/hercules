use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use hercules::branch_node::QuboBBNode;
use hercules::branchbound::BBSolver;
use hercules::graph_utils::get_all_disconnected_graphs;
use hercules::initial_points::generate_random_binary_point;
use hercules::local_search_utils;
use hercules::persistence::compute_iterative_persistence;
use hercules::preprocess::{preprocess_qubo, solve_small_components};
use hercules::qubo::Qubo;
use hercules::solver_options::SolverOptions;
use hercules::subproblemsolvers::enumerate_qubo::enumerate_solve;
use ndarray::Array1;
use smolprng::{JsfLarge, PRNG};
use std::collections::HashMap;
use std::time::Duration;

struct BenchData {
    qubo_small: Qubo,
    qubo_medium: Qubo,
    qubo_enum: Qubo,
    qubo_solve: Qubo,
    x_small: Array1<usize>,
    x_medium: Array1<usize>,
    selected_small: Vec<usize>,
    empty_fixed: HashMap<usize, usize>,
    process_solver: BBSolver,
    process_node: QuboBBNode,
}

fn make_bench_data() -> BenchData {
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
    let x_small = generate_random_binary_point(qubo_small.num_x(), &mut prng, 0.5);
    let x_medium = generate_random_binary_point(qubo_medium.num_x(), &mut prng, 0.5);
    let selected_small: Vec<usize> = (0..qubo_small.num_x()).collect();
    let empty_fixed = HashMap::new();

    let mut options = SolverOptions::new();
    options.verbose = 0;
    options.threads = 1;
    options.max_time = 1.0;
    let process_solver = BBSolver::new(qubo_small.clone(), options);
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
        x_small,
        x_medium,
        selected_small,
        empty_fixed,
        process_solver,
        process_node,
    }
}

fn helper_benches(c: &mut Criterion) {
    let data = make_bench_data();
    let mut group = c.benchmark_group("helpers");
    group.warm_up_time(Duration::from_millis(250));

    group.bench_function(BenchmarkId::new("eval_usize", 128), |b| {
        b.iter(|| data.qubo_medium.eval_usize(black_box(&data.x_medium)));
    });

    group.bench_function(BenchmarkId::new("eval_grad_usize", 128), |b| {
        b.iter(|| data.qubo_medium.eval_grad_usize(black_box(&data.x_medium)));
    });

    group.bench_function(BenchmarkId::new("one_flip_objective", 64), |b| {
        b.iter(|| {
            local_search_utils::one_flip_objective(
                black_box(&data.qubo_small),
                black_box(&data.x_small),
            )
        });
    });

    group.bench_function(BenchmarkId::new("one_step_local_search", 64), |b| {
        b.iter(|| {
            local_search_utils::one_step_local_search_improved(
                black_box(&data.qubo_small),
                black_box(&data.x_small),
                black_box(&data.selected_small),
            )
        });
    });

    group.finish();
}

fn preprocess_benches(c: &mut Criterion) {
    let data = make_bench_data();
    let mut group = c.benchmark_group("preprocess");
    group.warm_up_time(Duration::from_millis(250));

    group.bench_function(BenchmarkId::new("iterative_persistence", 64), |b| {
        b.iter(|| {
            compute_iterative_persistence(
                black_box(&data.qubo_small),
                black_box(&data.empty_fixed),
                black_box(data.qubo_small.num_x()),
            )
        });
    });

    group.bench_function(BenchmarkId::new("disconnected_graphs", 64), |b| {
        b.iter(|| {
            get_all_disconnected_graphs(
                black_box(&data.qubo_small),
                black_box(&data.empty_fixed),
            )
        });
    });

    group.bench_function(BenchmarkId::new("solve_small_components", 64), |b| {
        b.iter(|| {
            solve_small_components(
                black_box(&data.qubo_small),
                black_box(&data.empty_fixed),
                black_box(10),
            )
        });
    });

    group.bench_function(BenchmarkId::new("enumerate_solve", 10), |b| {
        b.iter(|| enumerate_solve(black_box(&data.qubo_enum)));
    });

    group.bench_function(BenchmarkId::new("preprocess_qubo", 64), |b| {
        b.iter(|| {
            preprocess_qubo(
                black_box(&data.process_solver.qubo_pp_form),
                black_box(&data.empty_fixed),
                black_box(true),
            )
        });
    });

    group.finish();
}

fn solver_benches(c: &mut Criterion) {
    let data = make_bench_data();
    let mut group = c.benchmark_group("solver");
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_secs(5));

    group.bench_function(BenchmarkId::new("process_node", 64), |b| {
        b.iter(|| {
            data.process_solver
                .process_node(black_box(&data.process_node))
        });
    });

    group.bench_function(BenchmarkId::new("convex_symmetric_form", 64), |b| {
        b.iter(|| data.qubo_small.convex_symmetric_form());
    });

    group.bench_function(BenchmarkId::new("branch_bound_solve", 96), |b| {
        b.iter_batched(
            || {
                let mut options = SolverOptions::new();
                options.verbose = 0;
                options.threads = 1;
                options.max_time = 5.0;
                BBSolver::new(data.qubo_solve.clone(), options)
            },
            |mut solver| {
                solver.solve();
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

criterion_group!(benches, helper_benches, preprocess_benches, solver_benches);
criterion_main!(benches);
