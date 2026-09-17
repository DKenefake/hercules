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
use hercules::branch_subproblem::SubProblemSelection;

// read in the QUBO problem
let p = Qubo::read_qubo("test_data/test_large.qubo");

// set up the solver options
let mut options = SolverOptions::new();

// use the LP relaxation as proposed by Glover
options.sub_problem_solver = SubProblemSelection::ClarabelQP;

// set up the solver
let mut solver = BBSolver::new(p, options);

// solve the QUBO problem
let (x_soln, obj) = solver.solve();
```

The branch and bound solver can also be used from Python, as shown below.

```python
import hercules

# read in the qubo problem
problem = hercules.read_qubo('test_data/test_large.qubo')

# the python interface requires a specified timeout 
x_soln, obj = hercules.solve_branch_bound(problem, timeout=20.0, sub_problem_solver = "clarabel_qp")
```

### MixingCut SDP and momentum

Use `SubProblemSelection::MixingCutSDP` for plain coordinate mixing, or
`SubProblemSelection::MixingCutSDPMomentum` for coordinate momentum with beta 0.8.
The Python `sub_problem_solver` names are `"mixingcut_sdp"` and
`"mixingcut_sdp_momentum"`, respectively. Both use MixingCut 0.1.5's repaired
dual lower bound, not its primal relaxation objective. Plain mixing remains the
default SDP mode; momentum is problem-dependent.

For a custom coefficient in Rust, assign
`Box::new(MixingCutSDPSolver::with_momentum(beta))` to `solver.subproblem_solver`.
The coefficient must be finite and in `[0, 1)`; zero selects plain mixing.

The Docker benchmark runners can be reproduced with:

```bash
# Full mk487a solve: momentum coefficient, then time limit in seconds.
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 cargo run --release --example profile_presolve -- --solve test_data/mk487a.qubo 0.8 60
# Identical root/conditional SDP matrices, comparing beta 0, 0.5 and 0.8.
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 cargo run --release --example benchmark_sdp_momentum -- test_data/mk487a.qubo
```

These runners double the stored MK Hessian, use a batch size of 64 for the full
solve, and keep the SDP iteration limit at 400 and stationarity tolerance at 1e-5.

For easier instances, `benchmark_momentum_bqp` compares all BQP50/BQP100 test
instances with beta 0, 0.5 and 0.8, checking the known optimal objectives. It uses
the same input conversion as the public BQP regression tests, rotates method
order, and caps each solve at 10 seconds:

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 cargo run --release --example benchmark_momentum_bqp -- 21
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 cargo run --release --example benchmark_sdp_momentum -- test_data/bqp/bqp100-1.qubo 400 --public-bqp
```

### Presolve propagation

The B&B contract is to return **one global minimizer**, not every tied solution.
Optimum-preserving choices are therefore allowed, provided the complete batch
retains at least one optimum. They are not treated as all-optima facts.

`root_complement_symmetry=true` (Rust and Python) chooses one orientation for each
eligible root component. For the binary polynomial
`f(x) = sum(a_i*x_i) + sum(w_ij*x_i*x_j)`, a component is complement-symmetric when
`2*a_i + sum_j(w_ij) = 0` for every variable in it. Flipping all its bits leaves
the objective unchanged, so fixing one high-interaction variable to zero retains
an optimum. Incoming caller fixings are never overwritten: a component with any
incoming fixation is skipped. Components of at most ten variables already use
exact enumeration and do not need this symmetry pass.

The symmetry check uses the original input coefficients before convexification,
with error-free summation rather than a near-zero tolerance. Arbitrary biased
QUBOs are not assumed to have MaxCut symmetry. Set the option to `false` for an
otherwise identical comparison. `root_symmetry_fixings` counts the explicit
orientation choices; further gradient/roof reductions follow normally.

The root keeps common fixings from its bounded probing pass and indexes the
generated two-variable relations for propagation at child nodes. Cheap gradient,
component and implication passes consume roof-dual fixings before the SDP solve;
roof duality is rerun only when those passes produce further fixings.
Roof-dual binary coefficients are prepared once per solver and projected directly
at each node, avoiding repeated sparse-matrix construction and pair aggregation.

Root probing uses strict gradient deductions and rolls back changed entries between
assumptions. It does not turn arbitrary tied component solutions into relations.
Small-component enumeration remains part of ordinary node presolve, including
when incoming fixings disconnect a component without new gradient fixings.
Components of at most ten variables are enumerated directly from their incident
terms without constructing a sparse subproblem or variable mapping. Once the
gradient queue reaches closure, solving whole disconnected components requires
no further gradient sweep over the remaining components.

Node batches transfer ownership into processing instead of cloning each popped
node. A newly computed bound is checked before branching or running another
heuristic, retaining any returned primal improvement when the node is closed.

### Incumbent-aware node probing

Before the main subproblem solve, eligible nodes probe one high-interaction
variable at both binary values. Each side runs the existing gradient, component,
implication and cheap lower-bound closure. A side whose bound cannot beat the
incumbent is discarded; surviving/common fixings stay local to that node.
Complete probe solutions are retained as primal candidates. The parent bound
can be strengthened by `min(L0, L1)`, and conditional child information is reused
if branching selects the same variable. This does not change SDP settings.

`SolverOptions` and the Python `solve_branch_bound` keyword arguments share:

- `node_probe_candidates=1`: variables probed per eligible node; zero disables it.
- `node_probe_max_free=256`: skip larger residual nodes.
- `node_probe_max_seconds=0.01`: soft per-node budget checked between conditional
  presolve calls. An unfinished pair makes no fixing or disjunction-bound deduction.

Unlike strict root probing, this lookahead may choose tied component optima and
discard incumbent-dominated assignments. Its fixings are not exported as global
persistency relations. Rust callers can inspect `solver.node_probing_statistics()`;
the benchmark runners report conditional work and main subproblem calls separately.

For example, compare otherwise identical MK solves in Docker:

```bash
# Arguments after the path: momentum, time limit, probes, max free, probe seconds.
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 cargo run --release --example profile_presolve -- --solve test_data/mk487a.qubo 0 60 0 256 0.01
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 cargo run --release --example profile_presolve -- --solve test_data/mk487a.qubo 0 60 1 256 0.01
```

### Optional SCC roof-dual reductions

`SolverOptions::roof_dual_weak_persistencies` and the same Python keyword enable
SCC-based weak persistency extraction in ordinary root/node presolve and node
lookahead. It defaults to `false`: initial experiments found modest BQP node
reductions, no additional MK487A fixings, and mixed MK487B bound/incumbent results.

Following [Boros, Hammer and Tavares, Section 3.3.2](https://users.cecs.anu.edu.au/~pcarr/qpbo/BorosRRR102006.pdf),
the pass symmetrizes the completed flow's residual graph and chooses compatible
complementary SCCs in topological order. Self-complementary SCCs remain unresolved.
This is linear graph work after max-flow, not another optimization subproblem.

Weak fixings preserve at least one optimum **jointly**, not every optimum. They
remain local and are never exported as global implications. The existing roof-dual
entry points still return only strong fixings; the prepared API
`solve_iterative_with_weak_persistencies` returns the additional reductions in
`weak_fixed_variables`. Deductions conditional on a weak choice stay in that field.
`node_probing_statistics().weak_roof_fixings` counts these reductions across all
presolve calls, including conditional probes, not unique original variables.

After the node probe time budget, `profile_presolve --solve` accepts flags for
weak SCC reductions, relation penalties and pair dominance, then root-probe
budgets and the complement-symmetry flag (example below).

### Optional root roof probing

`root_roof_probe_candidates` controls candidates per root pass; zero (the default)
disables it. `root_roof_probe_max_seconds=0.25` is a shared soft time budget across
all passes, checked between conditional presolve calls. Unlike node probing,
this pass is not restricted by `node_probe_max_free`. It requires the roof-dual
cheap bound and always allows compatible SCC choices within its conditional sides.

Both values of each candidate are tested. Common fixings retain an optimum in
both sides, and `min(L0, L1)` bounds their union. An interrupted pair makes no
disjunction deduction. Complete candidates are retained, and a closed root skips
the main subproblem solver. After a pass fixes variables, candidates are reranked
and another pass runs while time remains. These local choices never become new
global persistency relations.

The pass remains opt-in: initial BQP/MK tests did not show a useful node-count
improvement. The statistics `root_rounds`, `root_assignments`, and
`root_fixed_variables` separate its work from ordinary node probing.

```bash
# After the three previous flags: root roof candidates, root budget, symmetry.
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 cargo run --release --example profile_presolve -- --solve test_data/mk487a.qubo 0 30 1 256 0.01 false false false 0 0.25 true
```

### Experimental pair persistencies

Two additional `SolverOptions` fields and matching Python keywords default to
`false`; current BQP50/100 and MK487A comparisons found no node-count benefit.

- `root_pair_dominance`: compare the 00/11 and 01/10 assignments jointly, combining
  shared-neighbor coefficients before bounding their objective difference. Only
  strict dominance generates a relation. This root-only pass has a 50 ms soft
  budget and a 250,000-pair work limit.
- `roof_dual_relation_penalties`: encode forbidden assignments from strict root
  relations as nonnegative quadratic penalties, combined with the original binary
  coefficients once and reused by the prepared roof-dual solver at child nodes.

These follow the derivative/co-derivative and relation-strengthening ideas in
[Boros, Hammer and Tavares, Sections 3.2 and 4.1](https://users.cecs.anu.edu.au/~pcarr/qpbo/BorosRRR102006.pdf).
The penalty is zero on all assignments respecting the relations and exceeds the
original binary objective range for a forbidden assignment. These are optimality
relations, not constraints satisfied by every binary point: the resulting bound
is for the relation-restricted node problem. Weak or incumbent-dependent probe
deductions are never exported as these root relations. The primal objective and
SDP input remain unchanged.

```bash
# After probe seconds: weak SCC roof reductions, relation penalties, pair dominance.
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 cargo run --release --example profile_presolve -- --solve test_data/mk487a.qubo 0 60 1 256 0.01 false true true
```

### Root structural reductions

`root_low_degree_elimination` (Rust option and Python keyword, default `true`)
eliminates variables with at most two free neighbors after ordinary root presolve.
For each neighbor assignment it minimizes over the removed bit, replacing the
result with an exact quadratic table and an objective offset. A reverse stack
reconstructs the original binary vector. Removed variables are not reported as
unconditional persistencies.

`root_degree_three_elimination` (default `true`, requires
`root_low_degree_elimination`) extends the queue to variables with three free
neighbors. It evaluates all eight conditional minima and their exact Boolean
polynomial coefficients. Only a zero cubic coefficient permits elimination;
otherwise the variable stays in the QUBO. All updates are checked before any
mutation, so inexact arithmetic also skips the candidate safely. This implements
the star-triangle case of
[separator-based MaxCut reduction](https://drops.dagstuhl.de/storage/00lipics/lipics-vol301-sea2024/LIPIcs.SEA.2024.4/LIPIcs.SEA.2024.4.pdf),
with an explicit validity check for general QUBOs with nonzero fields.

`root_dominant_edge_contraction` (also default `true`) independently enables
dominant-edge substitutions `x_i=x_j` or `x_i=1-x_j`. The test bounds the binary
flip gain conditional on each value of the surviving variable. It includes the
residual linear coefficient, so it remains valid after root fixings and for QUBOs
with nonzero spin fields. This is the single-endpoint dominant-edge rule from
[Rehfeldt, Koch and Shinano, Section 3.1](https://link.springer.com/article/10.1007/s12532-023-00236-6),
expressed directly in native QUBO coefficients.

Each contraction is immediately substituted before testing another relation.
Affected rows are requeued; independent weak certificates are never combined in
a batch. Ties retain one optimum and only enter the reconstruction stack, not the
global implication graph. Equality contractions add incident coefficients;
complemented contractions also update the linear terms and objective constant.

The reductions alternate with gradient/component presolve and, when selected, roof
duality until no further reduction is available. It runs only at the root with a
one-second soft budget, also limited by remaining solve time. Coefficient updates
must be exactly representable in `f64`; otherwise the attempted reduction is
discarded and the original solve path is used. This is conservative for general
floating-point inputs, and avoids silently dropping small terms.

The smaller problem gets a freshly prepared backend, preserving MixingCut's
momentum setting. Custom Rust backends opt in via `for_reduced_qubo`; the default
implementation declines reduction. Logged objectives/bounds and returned vectors
remain in original units/coordinates, including on a timeout. Rust exposes
`solver.root_reduction_statistics` (eliminated, degree-three eliminated/skipped,
contracted, weak contractions, additional fixed, remaining, passes).
`degree_three_eliminated` is a subset of `eliminated`, not an additional count;
`degree_three_skipped` counts rejected attempts, including retries on changed rows.
`contracted` counts edge substitutions. The profile example accepts the low-degree,
contraction, then degree-three flags after its symmetry flag. Disable low-degree
elimination and contraction to disable structural root reduction entirely.

```python
x, objective, seconds, visited, processed = hercules.solve_branch_bound(
    problem, timeout=30.0, sub_problem_solver="mixingcut_sdp",
    root_low_degree_elimination=True,
    root_dominant_edge_contraction=True,
    root_degree_three_elimination=True,
)
```

### Dynamic component searches

The default branching rule is `LargestEdges` in Rust and Python. Component
searches inherit the parent rule; an explicit `branch_strategy` overrides it.

`component_decomposition=True` (Rust/Python, default enabled) detects disconnected
free-variable components after node presolve/probing, including at the root.
Components of at most ten variables still use the existing exact enumerator.
Larger components get independent B&B frontiers, primal solutions, and bounds.
Their backend is re-prepared through `SubProblemSolver::for_reduced_qubo`;
unsupported custom backends retain the ordinary connected-node path. Backends
must be `Send + Sync` because a component context can move between workers.

An independent split is an AND node, not an ordinary binary branch. Its bound
is the fixed objective constant plus the **sum** of component lower bounds.
Within each component the bound is still the **minimum** over its open branches
and incumbent. Component solutions are combined in original coordinates and
evaluated against the parent QUBO. The parent's stronger inherited bound is
retained. Fixed-only objective terms are counted once.
Splits requiring inexact fixed-term projection are skipped rather than silently
discarding coefficients; aggregate lower-bound sums are rounded downwards.

Each scheduler visit advances one unfinished component by a bounded batch
(up to 16 local nodes). Frontiers persist between visits; completed components
are not restarted. Component roots use their presolve bound until their first
processing slice, avoiding a duplicate initial relaxation. Components may themselves split recursively. All searches
use the existing Rayon pool and share the parent's deadline; no thread pool is
created per component. Timeouts are soft between batches/initialization calls.
Unfinished AND nodes stay in the normal frontier with aggregate bounds, even
through root-reduction postsolve. A complete parent can also be pruned by the
global incumbent without finishing every component.

The first implementation conservatively skips splits crossed by root relations.
Component-local presolve rebuilds its own QUBO optimality deductions; it does not
reuse a parent-wide relaxation bound as an individual component bound. Structural
root compaction is disabled inside component contexts because that older path
runs an inner solve to completion. The bounded node-compaction path described
below, ordinary presolve, probing, and further component splits remain enabled.
There is no cross-branch component cache yet.

`solver.component_statistics()` and verbose exit output report checks, splits,
created/solved components, batches, skipped splits, and maximum nesting depth.
Visited-node totals include component work and AND-node scheduling visits, so
compare SDP-call counts as well as node counts. An open AND node counts as one
entry in the parent frontier, not as all its descendant leaves. Only an empty
frontier is logged as `Optimal`; the final lower bound is printed explicitly.

Use `examples/benchmark_components.rs` for analytically solvable, synthetic
independent-clique comparisons. `profile_presolve --solve` accepts the component
toggle after its degree-three toggle for paired real-instance benchmarks.

### Structural node and block reductions

`node_structural_reductions=True` (Rust/Python, enabled by default) applies low-degree elimination,
exact quadratic degree-three elimination, and sequential dominant-edge
contractions after ordinary node presolve/probing. It also runs within dynamic
component searches. A pass has a 5 ms soft budget, capped by the shared solve
deadline. Re-preparing a backend requires a reduction of at least
`max(4, free_variables / 8)` variables, or complete elimination. Smaller gains
fall back to the original node without changing it.

A compacted node holds one persistent reduced search and its postsolve map,
rather than recursively calling `solve()` to completion. Subsequent scheduler
visits advance bounded batches, retain the inherited lower bound, add objective
constants with outward rounding, and reconstruct primal candidates in original
coordinates. Ordinary presolve/roof duality runs again after reductions. A newly
compacted empty-fixing root is not immediately compacted again. Unsupported
custom backends fall back to the unreduced node through `for_reduced_qubo`.

`small_block_elimination=True` (opt-in; disabled by default) enables exact elimination of connected pockets
with at most eight interior variables and at most two boundary variables, at
the root and during node compaction. Articulation searches, also after removing
one vertex, discover separators. Every interior assignment is evaluated for each
of the 2 or 4 boundary assignments. Only interior-dependent costs are included;
boundary-only terms stay in the graph. The conditional minimum is replaced by
a constant, boundary linear coefficients, and at most one boundary edge.
One minimizing interior assignment per boundary case is saved for postsolve.
Ties preserve one optimum; no simultaneous weak-relation assumptions are made.
Inexact arithmetic or an expired budget declines an uncommitted block atomically.
This is a budgeted search for small separators, not complete separator enumeration.

`node_reduction_statistics()` reports accepted node compactions, eliminated and
contracted variables, block counts, additional fixings, and skipped small gains.
Counts sum across node contexts; they are not distinct original variables.
Scheduling visits are included in node counts, so also compare SDP calls,
elapsed time, the final incumbent, and the lower bound.

Both options can be disabled independently for ablation. The profile example
accepts their two flags after the branching-rule argument.
`examples/benchmark_node_reductions.ps1` compares baseline, node reductions, and
node reductions plus blocks on the three MK instances, serially, with soft and
external hard timeouts. These reductions do not guarantee fewer search nodes or
lower runtime; rebuilding models and changed branching order can cost more.
The first MK comparison reduced SDP calls with node compaction, but adding block
search increased runtime on both completed instances, so blocks remain opt-in.

### Cutoff propagation and whole-cut dominance

`component_cutoff_propagation=True` (Rust/Python, default enabled) passes the
current pruning threshold into every reduced/component search before its next
batch. For a single reduced problem the threshold is `U - constant`. For
component `k` it is `U - constant - sum(other_component_lower_bounds)`.
Using the other components' incumbents instead would be unsafe. Unknown lower
bounds disable the translation until finite certificates are available. All
threshold subtractions are rounded upwards.

An external threshold never replaces a feasible local incumbent. Each subsearch
retains the minimum certified lower bound of regions discarded by that threshold,
including failed probing sides. A finished, cutoff-pruned component is not
reported as exactly optimized; its retained certificate is used in the parent's
sum. External-cutoff pruning does not use a relative tolerance in the child's
shifted objective units. `cutoff_statistics()` and verbose output report cutoff
updates and pruning events (including probing sides, not distinct B&B nodes).

`root_cut_dominance=True` (Rust/Python, opt-in) searches for whole-cut dominance
after the existing cheap structural reductions. The signed MaxCut analysis graph
includes linear fields as anchor edges: for the native polynomial
`a_i*x_i + sum(b_ij*x_i*x_j)`, use `w_ij=b_ij` and
`w_i,anchor=-2*a_i-sum_j(b_ij)`, with the anchor fixed at zero.
Thus `2*(f-constant)` equals the negative cut weight.

For candidate edge `e`, a cut separating its endpoints certifies its preferred
parity when `abs(w_e) >= sum(abs(w_other_crossing_edges))`. A bounded max-flow
search proposes a partition; the actual original crossing weights are summed
with exactness checks to certify it. Accepting a partition does not depend on
the numerical accuracy of the reported flow value. Positive edges imply opposite
bits, negative edges equal bits, and anchor edges yield fixings. Apply only one
certificate before rebuilding/rechecking, preserving one optimum even with ties.
All substitutions and solutions remain in native QUBO coordinates.

The rule is Proposition 3.3 in
[Separator Based Data Reduction for the Maximum Cut Problem](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.SEA.2024.4).
This implementation prioritizes heavy edges and allows at most 64 flow trials
and a 50 ms soft budget within the root reduction deadline. It is not a complete
Gomory-Hu search, and it never runs inside tree or component contexts.
`root_reduction_statistics.cut_dominance` reports trials, accepted contractions,
fixings, and budget exhaustion, even if no reduction is accepted.

`examples/benchmark_cutoff_dominance.ps1` compares the previous node-reduction
path, cutoff propagation, and cutoff plus whole-cut dominance sequentially on
the MK instances. `profile_presolve --solve` accepts the two new flags after
its node-reduction and small-block flags. Whole-cut dominance remains opt-in
until its incremental reduction benefit justifies enabling it generally.

### Experimental SDP Dual Fixing

`sdp_dual_fixing=True` (Rust/Python, default off) extracts assignment bounds from
the MixingCut result before discarding its dual information. It runs only on
residual problems with at most 128 free variables, with a 5 ms soft per-call
budget and the solver deadline also respected. It skips nodes whose ordinary
SDP bound already reaches the current pruning threshold. No additional SDP
solve, persistent factor cache, or change to LargestEdges is involved.

For sign variables `s` including the anchor, write the native binary objective
as `s^T C s`. The reported scalar bound and raw dual variables propose a repaired
slack matrix. A Cholesky factor `L` is only a proposal: outward-rounded arithmetic
encloses the actual residual `E = C - L L^T`, including QUBO conversion error.
For binary spins this gives the independently checked inequality

```text
base = sum_i lower(E_ii) - 2 sum_{i>j} upper(abs(E_ij))
f(s) >= base + ||L^T s||^2
```

For assignment `s_i = t*s_anchor`, let `v=e_i+t*e_anchor`. Cauchy-Schwarz gives
`f(s) >= base + 4/||L^-1 v||^2`. Outward triangular solves bound the denominator
from above, and lower bounds are rounded down. `t=1` means `x_i=0`, and `t=-1`
means `x_i=1`. A failed factorization, invalid data, or expired budget cannot
produce a fixing. This does not trust the uncorrected dual multipliers as a PSD
certificate, and does not require changing MixingCut 0.1.5.

The adapter maps bounds back to original node indices and adds the node
constant. `SubProblemResult::take_conditional_bounds` defaults to an empty list
for other backends. The B&B layer compares each side with the current incumbent
and external component cutoff, using the existing discarded-region proof ledger.
These are local, incumbent-dependent deductions, not global persistencies.
After fixing, the node returns through presolve and structural reductions before
branching; stale probed children are not reused.

`sdp_fixing_statistics()` and verbose logging report attempted passes, variable
bound pairs produced, accepted fixings, direct node closures, and summed worker
seconds (not wall time). `examples/benchmark_sdp_fixing.ps1` performs a capped
on/off MK comparison. The profile runner accepts the switch after the whole-cut
dominance flag. Initial A/B measurements found a few additional fixings but no
meaningful reduction in SDP calls, so the feature remains opt-in.

## Packaging

Hercules can now be used as either a normal Rust library or a Python extension module from the same repository.

### Rust library

Add Hercules as a normal Rust dependency:

```toml
[dependencies]
hercules = "0.5"
```

Then build as usual:

```bash
cargo build
```

Useful checks during development:

```bash
cargo check
cargo test --lib
```

### Python package

The Python bindings live behind the `python` feature and are configured for `maturin` via `pyproject.toml`.

From an activated virtual environment, install an editable build with:

```bash
maturin develop
```

Or build a wheel with:

```bash
maturin build
```

If you want the optimized wheel locally, use:

```bash
maturin build --release
```
