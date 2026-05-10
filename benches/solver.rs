mod common;

use common::{make_bench_data, make_solver_options};
use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use hercules::branchbound::BBSolver;
use hercules::qubo::Qubo;
use std::time::Duration;

fn bench_solve_case(group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>, name: &str, qubo: &Qubo) {
    group.bench_function(BenchmarkId::new("branch_bound_solve", name), |b| {
        b.iter_batched(
            || BBSolver::new(qubo.clone(), make_solver_options()),
            |mut solver| {
                solver.solve();
            },
            BatchSize::SmallInput,
        );
    });
}

fn solver_benches(c: &mut Criterion) {
    let data = make_bench_data();
    let mut group = c.benchmark_group("solver");
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_secs(8));

    group.bench_function(BenchmarkId::new("process_node", 64), |b| {
        b.iter(|| {
            data.process_solver
                .process_node(black_box(&data.process_node))
        });
    });

    group.bench_function(BenchmarkId::new("convex_symmetric_form", "random64"), |b| {
        b.iter(|| data.qubo_small.convex_symmetric_form());
    });

    group.bench_function(BenchmarkId::new("convex_symmetric_form", "gka6a"), |b| {
        b.iter(|| data.qubo_gka6a.convex_symmetric_form());
    });

    bench_solve_case(&mut group, "random96", &data.qubo_solve);
    bench_solve_case(&mut group, "gka1b", &data.qubo_gka1b);
    bench_solve_case(&mut group, "gka2b", &data.qubo_gka2b);
    bench_solve_case(&mut group, "gka6a", &data.qubo_gka6a);
    bench_solve_case(&mut group, "gka7a", &data.qubo_gka7a);
    bench_solve_case(&mut group, "bqp50", &data.qubo_bqp50);

    group.finish();
}

criterion_group!(benches, solver_benches);
criterion_main!(benches);
