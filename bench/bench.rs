#![feature(test)]
extern crate test;

use std::time::SystemTime;

#[bench]
fn bench_local_search(b: &mut test::Bencher){

    let current_time = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs();

    let mut prng = smolprng::PRNG {
        generator: smolprng::JsfLarge::new(current_time),
    };

    const SIZE: usize = 1024;

    let p = make_solver_qubo();

    let x_0 = initial_points::generate_random_binary_point(&p, &mut prng, 0.5);

    b.iter(|| {
        (0..1000).map(local_search::simple_local_search(&p, &x_0, 10))
    })
}