# Hurricane: QUBO Heuristic Solver Toolkit

Hurricane is a Rust library for (heuristically) solving Quadratic Unconstrained Binary Optimization (QUBO) problems. Hurricane is designed to be used as a library for implementing and testing QUBO heuristics. It is mostly a sideproject for me during my PhD, so please be understanding of this.

## What is this library for?

Hurricane is designed as a simple, easy-to-use library for solving QUBO problems. It is not necessarily designed to be a state-of-the-art tool but a toolkit for quickly prototyping and testing new heuristics. That said, Hurricane is designed to be fast and written in Rust, a high-performance systems programming language.

## Progress

Hurricane is currently in the early stages of development. The following features are currently implemented:

- [x] QUBO data structure
- [x] QUBO problem generation
- [x] 1-opt heuristic
- [x] Gain heuristic (based on boros2007)
- [ ] Tabu search
- [ ] Discrete Particle Swarm Optimization
- [ ] Simulated Annealing
- [ ] Parallel Tempering

## A Simple Example: Mixing two local search heuristics

The subcomponents of Hurricane can be used independently. For example, the following code shows how to make a new local search function based on 1-opt and gain search. Each iteration of the algorithm is defined as finding the lowest energy point in the neighborhood of the current point and then doing a large-scale flipping operation, flipping all bits based on the gains of the function. This allows for simple, easy-to-understand, and easy-to-implement local search algorithms (amongst other ideas).
    
```rust no_run
use hurricane::qubo::Qubo;
use hurricane::local_search::*;
use hurricane::local_search_utils::*;
use ndarray::Array1;
use rayon::prelude::*;
use hurricane::initial_points;
use smolprng::{PRNG, JsfLarge};

// A simple local search heuristic that uses 1-opt and gain-criteria search
pub fn simple_mixed_search(qubo: &Qubo, x_0: &Array1<f64>, max_steps:usize) -> Array1<f64>{
    // create a mutable copy of the initial point
    let mut x = x_0.clone();
    // flip the bits maximize the 1D gains
    let mut x_1 = get_gain_criteria(qubo, &x);
    let mut steps = 0;
    
    // run the local search until we reach a local minimum or we reach the maximum number of steps
    while x_1 != x && steps <= max_steps {
        x = x_1.clone();
        // find the lowest energy point in the neighborhood of x (can be x itself)
        x_1 = one_step_local_search(qubo, &x, &(0..qubo.num_x()).collect());
        // flip the bits to better satisfy the stationary conditions
        x_1 = get_gain_criteria(qubo, &x_1);
        // increment the number of steps
        steps += 1;
    }
    // return the best point found
    x_1
}

// create a random QUBO problem
let mut prng = PRNG {
    generator: JsfLarge::default(),
};
let p = Qubo::make_random_qubo(1000, &mut prng, 0.1);

// generate 8 random starting points
let mut x_s = initial_points::generate_random_starting_points(&p, 8, &mut prng);

// solve each initial point, in parallel
let x_sols: Vec<_> = x_s
    .par_iter()
    .map(|x| simple_mixed_search(&p, &x, 1000))
    .collect();

// find the best solution
let min_obj = x_sols
    .iter()
    .map(|x| p.eval(&x))
    .min_by(|a, b| a.partial_cmp(b).unwrap())
    .unwrap();
```

This is actually a quite effective simple local search heuristic, and can be used as a starting point for more complex heuristics. Here, we can solve a randomly generated sparse 1000x1000 QUBO problem to within 0.5% of the optimal solution in about half second on a laptop. 
