# Hurricane: QUBO Heuristic Solver Toolkit

Hurricane is a Rust library for (heuristically) solving Quadratic Unconstrained Binary Optimization (QUBO) problems. Hurricane is designed to be used as a library, for the implimentation and testing of QUBO heuristics.



## Overview
The subcomponents of Hurricane can be used independently. For example, the following code shows how to make a new local search function based on 1-opt and opt-criteria search. Where each iteration of the algorithm is defined as finding the lowest energy point in the neighborhood of the current point, and then doing a large scale flipping operation based on trying to approximately satisfy the stationary conditions of the problem. This allows for a simple, easy to understand, and easy to implement local search algorithms (amongst other ideas). 
    
```rust

extern crate hurricane;

use hurricane::qubo::QUBO;
use hurricane::qubo_heuristic::{one_step_local_search, get_opt_criteria};

pub fn simple_mixed_search(qubo: &Qubo, x_0: &Array1<f64>, max_steps:usize) -> Array1<f64>{

    let mut x = x_0.clone();
    let mut x_1 = get_opt_criteria(qubo, &x);
    let mut steps = 0;

    while x_1 != x && steps <= max_steps {
        x = x_1.clone();
        x_1 = one_step_local_search(qubo, &x, &(0..qubo.num_x()).collect());
        x_1 = get_opt_criteria(qubo, &x_1);
        steps += 1;
    }

    x_1
}
```

This is actually a quite effective simple local search heuristic, and can be used as a starting point for more complex heuristics. Here, we can solve a randomly generated sparse 1000x1000 QUBO problem to within 0.5% of the optimal solution in about half second on a laptop. 



## What is this library for?

Hurricane is designed to be a simple, easy-to-use library for solving QUBO problems. It is not designed to be a high-performance solver, but rather a tool for quickly prototyping and testing new heuristics.


## Progress

Hurricane is currently in the early stages of development. The following features are currently implemented:

- [x] QUBO data structure
- [x] QUBO problem generation
- [x] 1-opt heuristic
- [x] Opt-Cond heuristic (based on boros2007)
- [ ] Tabu search
- [ ] Discrete Particle Swarm Optimization
- [ ] Simulated Annealing
- [ ] Parallel Tempering