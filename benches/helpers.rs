mod common;

use common::make_bench_data;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use hercules::local_search_utils;
use std::time::Duration;

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

criterion_group!(benches, helper_benches);
criterion_main!(benches);
