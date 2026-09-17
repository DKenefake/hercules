"""Exercise the degree-three toggle on a freshly built, uninstalled extension."""
import importlib.util
import itertools
import sys

spec = importlib.util.spec_from_file_location("hercules", sys.argv[1])
hercules = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hercules)

objectives = {
    50: [-2098, -3702, -4626, -3544, -4012, -3693, -4520, -4216, -3780, -3507],
    100: [-7970, -11036, -12723, -10368, -9083, -10210, -10125, -11435, -11455, -12565],
}


def check(name, problem, expected, enabled, momentum=False):
    x, objective, seconds, visited, _ = hercules.solve_branch_bound(
        tuple(problem), timeout=30.0, seed=0, verbose=0, threads=64,
        branch_strategy="LargestEdges", cheap_lower_bound_problem="roof_dual",
        sub_problem_solver="mixingcut_sdp_momentum" if momentum else "mixingcut_sdp",
        root_low_degree_elimination=True, root_dominant_edge_contraction=True,
        root_degree_three_elimination=enabled,
    )
    assert len(x) == problem[-1] and all(v in (0, 1) for v in x)
    actual = sum(0.5*q*x[i]*x[j] for i, j, q in zip(*problem[:3]))
    actual += sum(c*v for c, v in zip(problem[3], x))
    assert abs(actual-objective) < 1e-5, (name, actual, objective)
    assert abs(expected-objective) < 1e-5, (name, expected, objective)
    print(f"{name} degree_three={enabled} momentum={momentum} "
          f"objective={objective:.8f} seconds={seconds:.6f} visited={visited}", flush=True)


for size, values in objectives.items():
    for index, expected in enumerate(values, 1):
        name = f"bqp{size}-{index}"
        problem = list(hercules.read_qubo(f"test_data/bqp/{name}.qubo"))
        problem[2] = [q*(2 if i == j else 4) for i, j, q in zip(*problem[:3])]
        for enabled, momentum in itertools.product((False, True), repeat=2):
            check(name, problem, expected, enabled, momentum)

# MK values are regression references from completed solves, not a new external oracle.
for name, expected in [("mk487a", -1110926), ("mk487b", -3655475)]:
    problem = list(hercules.read_qubo(f"test_data/{name}.qubo"))
    problem[2] = [2*q for q in problem[2]]
    for enabled in (False, True):
        check(name, problem, expected, enabled)
print("84 Python solves passed objective and original-coordinate reconstruction checks.")
