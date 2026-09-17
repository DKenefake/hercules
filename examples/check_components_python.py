"""Validate fresh component-search Python bindings without installing a wheel."""
import importlib.util
import sys

spec = importlib.util.spec_from_file_location("hercules", sys.argv[1])
hercules = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hercules)

for count in (2, 4):
    size = 13
    n = count * size
    rows, columns, values = [], [], []
    linear = [0.0] * n
    for block in range(count):
        start = block * size
        for i in range(start, start + size):
            for j in range(i + 1, start + size):
                rows.append(i)
                columns.append(j)
                values.append(4.0)
                linear[i] -= 1.0
                linear[j] -= 1.0
    expected = -count * (size * size // 4)
    for enabled in (False, True):
        for backend in ("mixingcut_sdp", "mixingcut_sdp_momentum"):
            x, objective, seconds, visited, _ = hercules.solve_branch_bound(
                (rows, columns, values, linear, n), timeout=10.0,
                verbose=1, threads=64, branch_strategy="LargestEdges",
                sub_problem_solver=backend, cheap_lower_bound_problem="roof_dual",
                node_probe_candidates=0, component_decomposition=enabled,
            )
            if len(x) != n or any(v not in (0, 1) for v in x):
                raise RuntimeError("Solver returned an invalid binary solution")
            actual = sum(0.5*q*x[i]*x[j] for i, j, q in zip(rows, columns, values))
            actual += sum(c*v for c, v in zip(linear, x))
            if not abs(actual-objective) < 1e-7:
                raise RuntimeError(f"Objective mismatch: evaluated={actual}, reported={objective}")
            if not abs(objective-expected) < 1e-7:
                raise RuntimeError(f"Known objective mismatch: expected={expected}, reported={objective}")
            print(f"PY_COMPONENTS count={count} enabled={enabled} backend={backend} "
                  f"objective={objective:.8f} visited={visited} seconds={seconds:.6f}", flush=True)
print("8 Python component solves matched analytical optima and original-polynomial objectives.")
