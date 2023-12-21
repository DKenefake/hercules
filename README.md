# Hurricane: QUBO Heuristic Solver Toolkit

Hurricane is a Rust library for (heuristically) solving Quadratic Unconstrained Binary Optimization (QUBO) problems.

## Installation

Hurricane is available on crates.io and can be added to a project's `Cargo.toml` as follows:

```toml
[dependencies]
hurricane = "0.1.0"
```

## Usage

Hurricane is designed to be used as a library. The following example shows how to use Hurricane to (Heuristically) solve a QUBO problem:

```rust
extern crate hurricane;

use hurricane::qubo::QUBO;

fn main() {
    // Create a new QUBO problem
    let qubo = QUBO::make_random_qubo(10, 0.5);

    // generate a set of initial points, based of the alpha heuristic
    let x_0 = qubo_heuristic::generate_alpha_starting_point(&qubo);

    // do iterated search of flipping variables based on simple optimality criteria
    let x_sol = qubo_heuristic::simple_opt_criteria_search(&qubo, &x_0);

    // print the solution objective value
    println!("Solution: {:?}", qubo.eval_sol(x_sol));
}
```

The subcomponents of Hurricane can also be used independently. For example, the following code shows how to make a new local search function based on 1-opt and opt-criterial search:
    
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

What this does, is a mixture of the 1-opt and opt-criterial search. It starts with a random point, and then does 1-opt search until it finds a local minimum. Then it does opt-criterial search until it finds a local minimum. It then repeats this process until it reaches a maximum number of steps. This is a simple example of how to combine the subcomponents of Hurricane to create new heuristics.


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