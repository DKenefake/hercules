"""Check a freshly built extension without replacing the user's installed wheel."""
import importlib.util
import sys
import time

spec = importlib.util.spec_from_file_location("hercules", sys.argv[1])
hercules = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hercules)


def check(problem, expected, enabled, momentum=False, verbose=0):
    started = time.perf_counter()
    x, objective, _seconds, visited, _processed = hercules.solve_branch_bound(
        problem, timeout=30.0, seed=0, verbose=verbose, threads=64,
        branch_strategy="LargestEdges",
        sub_problem_solver="mixingcut_sdp_momentum" if momentum else "mixingcut_sdp",
        cheap_lower_bound_problem="roof_dual",
        root_low_degree_elimination=enabled,
        root_dominant_edge_contraction=False,
    )
    if len(x) != problem[-1] or any(v not in (0, 1) for v in x):
        raise RuntimeError("Solver returned an invalid binary solution")
    actual = sum(0.5*q*x[i]*x[j] for i, j, q in zip(*problem[:3]))
    actual += sum(c*v for c, v in zip(problem[3], x))
    if not abs(actual-objective) < 1e-5:
        raise RuntimeError(f"Objective mismatch: evaluated={actual}, reported={objective}")
    if not abs(objective-expected) < 1e-5:
        raise RuntimeError(f"Known objective mismatch: expected={expected}, reported={objective}")
    return f"elimination={enabled} momentum={momentum} objective={objective:.8f} visited={visited} seconds={time.perf_counter()-started:.6f}"


objectives = {
    50: [-2098, -3702, -4626, -3544, -4012, -3693, -4520, -4216, -3780, -3507],
    100: [-7970, -11036, -12723, -10368, -9083, -10210, -10125, -11435, -11455, -12565],
}
for size, expected_values in objectives.items():
    for index, expected in enumerate(expected_values, 1):
        name = f"bqp{size}-{index}"
        problem = list(hercules.read_qubo(f"test_data/bqp/{name}.qubo"))
        problem[2] = [q*(2 if i == j else 4) for i, j, q in zip(*problem[:3])]
        for enabled in (False, True):
            for momentum in (False, True):
                print(name, check(tuple(problem), expected, enabled, momentum), flush=True)

problem = list(hercules.read_qubo("test_data/mk487a.qubo"))
problem[2] = [2*q for q in problem[2]]
for enabled in (False, True):
    print("mk487a", check(tuple(problem), -1110926, enabled, verbose=1), flush=True)
print("82 Python solves matched independently confirmed objectives and reconstructed objectives.")
