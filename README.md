# Hercules: QUBO Solver and Heuristics Toolkit

Hercules is a Rust library for analysing, finding approximate solutions, and finding exact solutions for Quadratic Unconstrained Binary Optimization (QUBO) problems. It is mostly a side project, so please be understanding of this.

## What is this library for?

Hercules is designed as a simple, easy-to-use library for exactly or approximately solving QUBO problems. It is not 
necessarily designed to be a state-of-the-art tool but a toolkit for quickly prototyping and testing QUBO methods. 
That said, Hercules is designed to be fast and written in Rust, a high-performance systems programming language.

## Progress

Hercules is currently in the early stages of development. The following features are currently implemented:

- [x] QUBO data structure
- [x] QUBO problem generation
- [x] QUBO Heuristics
- [x] Initial Branch & Bound Solver
- [x] Python interface (via PyO3)

When referring to the solver, there is a world of a difference between naive implementations and useful for real 
world implementations. I am trying to iteratively move the solver to the category of useful for real world problems, 
without punting too much of the responsibilities to dependencies. This is documented on a very high level on my [personal blog](https://dkenefake.github.io/blog/bb1). As it stands, it can generally solve dense and sparse problems below 80 binaries. But I hope to push the capabilities to larger problem sizes, and solve the problems we can much faster. 

- [x] Initial Branch and Bound
- [x] Initial Presolver
- [x] Warm Starting
- [x] Variable Branching Rules
- [x] Multithreaded B&B solver
- [ ] Problem Reformulation
- [ ] Modern Presolver
- [x] Warm starting subproblems
- [x] Beck Optimality Proof
- [x] Variable Probing

## Example: Approximately solve a QUBO

This can be used to generate get and generate high quality (depending on the search heuristic) solutions to the QUBO problem being considered. For example, the following code shows how to use the gain criteria search to find a local minimum of a QUBO problem.

```rust
use hercules::qubo::Qubo;
use hercules::local_search::simple_gain_criteria_search;
use hercules::initial_points::generate_random_binary_point;
use smolprng::{PRNG, JsfLarge};

// generate a make a random number generator
let mut prng = PRNG {
    generator: JsfLarge::default(),
};

// read a QUBO problem from a file
let p = Qubo::read_qubo("test_data/test_large.qubo");

// generate an initial point of 0.5 for each variable
let x_0 = generate_random_binary_point(p.num_x(), &mut prng, 0.5);

// use the gain criteria search to find a local minimum with an upper bound of 1000 iterations
let x_1 = simple_gain_criteria_search(&p, &x_0, 1000);
```

This can be accomplished in using the python interface as well, as shown below.

```python
import hercules
import random

# read in the qubo problem
problem = hercules.read_qubo('test_data/test_large.qubo')

# generate a random point
x_0 = [random.randint(0,1) for i in range(problem[-1])]

# solve the QUBO problem via the gain criteria search heuristic with initial point x_0 for at most 1000 iterations
x_heur, obj_heur = hercules.gain_criteria(problem, x_0, 1000)
```

## Example: Solve a QUBO via Branch and Bound

Hercules can also be used to find global solutions to QUBO problems.
This is the code to read in a QUBO problem from 
a file, set up the solver options, and solve the problem via branch and bound. Here, we are solving a QUBO problem 
from the file ``test_large.qubo``, and we are using the LP relaxation as the subproblem solver. This QUBO has 1000 
variables and 5000 nonzero entries in the upper triangle. This is solved quite quickly (in under a few seconds) on a 
modern desktop. That being said, the performance of the solver is highly dependent on the problem being solved, and the
solver options being used.

```rust
use hercules::qubo::Qubo;
use hercules::branchbound::BBSolver;
use hercules::solver_options::SolverOptions;
use hercules::branch_subproblem::SubProblemSelection

// read in the QUBO problem
let p = Qubo::read_qubo("test_data/test_large.qubo");

// set up the solver options
let mut options = SolverOptions::new();

// use the LP relaxation as proposed by Glover
options.sub_problem_solver = SubProblemSelection::ClarabelLP;

// set up the solver
let solver = BBSolver::new(p, options);

// solve the QUBO problem
let (x_soln, obj) = solver.solve();
```

The branch and bound solver can also be used from Python, as shown below.

```python
import hercules

# read in the qubo problem
problem = hercules.read_qubo('test_data/test_large.qubo')

# the python interface requires a specified timeout 
x_soln, obj = hercules.solve_branch_bound(problem, timeout=20.0, sub_problem_solver = "clarabel_lp")
```

## Python Interface: Using Hercules from Python

Hercules can be used from Python using the Py03 crate. This is currently a work in progress, and a standalone package is not currently available. This can be built via maturin via the ``maturin develop`` command. 

## Docker

A Docker image is available [here](https://hub.docker.com/repository/docker/dkenefake/hercules/general).
To run this, pull down the image and run the following:

```bash
docker run --platform linux/amd64 dkenefake/hercules:test python pyhercules.py
```
