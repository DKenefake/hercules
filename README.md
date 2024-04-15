# Hercules: QUBO Solver and Heuristics Toolkit

Hercules is a Rust library for analysing, finding approximate solutions, and finding exact solutions for Quadratic Unconstrained Binary Optimization (QUBO) problems. It is mostly a side project for me during my PhD, so please be understanding of this.

## What is this library for?

Hercules is designed as a simple, easy-to-use library for solving QUBO problems. It is not necessarily designed to be a state-of-the-art tool but a toolkit for quickly prototyping and testing QUBO methods. That said, Hercules is designed to be fast and written in Rust, a high-performance systems programming language.

## Progress

Hercules is currently in the early stages of development. The following features are currently implemented:

- [x] QUBO data structure
- [x] QUBO problem generation
- [x] QUBO Heuristics
- [x] Initial Branch & Bound Solver

When refereing to the solver, there is a world of a difference between naive implimentations and useful for real world implimentations. I am trying to oterativley move the solver to the category of usefull for real world problems, without punting to much of the responsibilities to dependencies. This is documented in a very high level on my [personal blog](https://dkenefake.github.io/blog/bb1). As it stands, it can generally solve dense and sparse problems below 80 binaries. But I hope to push the capabilities to larger problem sizes, and solve the problems we can much faster. 

- [x] Initial Branch and Bound
- [x] Initial Presolver
- [x] Warm Starting
- [x] Variable Branching Rules
- [x] Multithreaded B&B solver
- [ ] Modern Presolver
- [ ] Warm starting subproblems


## Simple: Read and solve QUBO example

This can be used to generate get and generate high quality (depending on the search heuristic) solutions to the QUBO problem being considered. For example, the following code shows how to use the gain criteria search to find a local minimum of a QUBO problem.

```rust no_run
use hercules::qubo::Qubo;
use hercules::local_search::simple_gain_criteria_search;
use hercules::initial_points::generate_central_starting_points;

// read in a QUBO problem from a file
let p = Qubo::read_qubo("test.qubo");

// generate an initial point of 0.5 for each variable
let x_0 = generate_central_starting_points(&p);

// use the gain criteria search to find a local minimum with an upper bound of 1000 iterations
let x_1 = simple_gain_criteria_search(&p, &x_0, 1000);
```

## Advanced: Mixing two local search heuristics

The subcomponents of Hercules can be used independently and interchangeably allowing us to create new heuristics on the fly. For example, the following code shows how to make a new local search function based on 1-opt and gain search. Each iteration of the algorithm is defined as finding the lowest energy point in the neighborhood of the current point and then doing a large-scale flipping operation, flipping bits based on the gains of the function. This allows for simple, easy-to-understand, and easy-to-implement local search algorithms (amongst other ideas).
    
```rust no_run
use hercules::qubo::Qubo;
use hercules::local_search::*;
use hercules::local_search_utils::*;
use ndarray::Array1;
use rayon::prelude::*;
use hercules::initial_points;
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

This is actually a quite effective simple local search heuristic, and can be used as a starting point for more complex heuristics. Here, we can solve a randomly generated sparse 1000x1000 QUBO problem to within 0.5% of the optimal solution in about 15 ms on a laptop. 

## Python Interface: Using Hercules from Python

Hercules can be used from Python using the Py03 crate. This is currently a work in progress, and a standalone package is not currently available. This can be built via maturin via the ``maturin develop`` command.  However, the following code shows how to use Hercules from Python.

Here we can use the python interface to use the simple mixed search heuristic we developed above to solve a randomly generated QUBO problem. This is a 1000x1000 QUBO problem with 1000 nonzero entries, and we solve it from a random initial point for 50 iterations. This takes about the same about of time as the pure Rust version, and we get the same solution.

```python
import hercules
import random
import time

# set a random seed
random.seed(time.time())

# read in the qubo problem
problem = hercules.read_qubo('test_large.qubo')
num_x = problem[-1]

# create a random initial point
x_0 = [random.randint(0, 1) for i in range(num_x)]

# start timing
start = time.time()

# solve the QUBO problem via the mixed local search heuristic with, initial point x_0, for 50 iterations
x_soln, obj = hercules.mls(problem, x_0, 50)

# stop timing
end = time.time()

# print the objective value of the solution
print('Solution: ', obj, ' in ', end - start, ' seconds')

```
