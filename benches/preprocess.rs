mod common;

use common::make_bench_data;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use hercules::graph_utils::get_all_disconnected_graphs;
use hercules::persistence::compute_iterative_persistence;
use hercules::preprocess::{preprocess_qubo, preprocess_qubo_heavy, solve_small_components};
use hercules::subproblemsolvers::enumerate_qubo::enumerate_solve;
use hercules::variable_reduction::probe_limited;
use std::time::Duration;

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

    let mut large_probe_group = c.benchmark_group("preprocess");
    large_probe_group.warm_up_time(Duration::from_millis(250));
    large_probe_group.sample_size(10);

    for max_candidates in [25usize, 50, 100] {
        large_probe_group.bench_function(
            BenchmarkId::new("probe_limited/test_large", max_candidates),
            |b| {
                b.iter(|| {
                    probe_limited(
                        black_box(&data.qubo_test_large),
                        black_box(&data.empty_fixed),
                        black_box(false),
                        black_box(max_candidates),
                    )
                });
            },
        );
    }

    large_probe_group.bench_function(BenchmarkId::new("preprocess_qubo_heavy", "test_large"), |b| {
        b.iter(|| {
            preprocess_qubo_heavy(
                black_box(&data.qubo_test_large),
                black_box(&data.empty_fixed),
                black_box(false),
            )
        });
    });

    large_probe_group.finish();
}

criterion_group!(benches, preprocess_benches);
criterion_main!(benches);
