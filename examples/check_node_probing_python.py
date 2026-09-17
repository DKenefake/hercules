"""
Smoke-test fresh Python bindings.

Usage: python3 examples/check_node_probing_python.py /target/debug/libhercules.so
"""

import importlib.util
from itertools import product
import sys


spec = importlib.util.spec_from_file_location("hercules", sys.argv[1])
hercules = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hercules)

objectives = {
    50: [-2098, -3702, -4626, -3544, -4012, -3693, -4520, -4216, -3780, -3507],
    100: [-7970, -11036, -12723, -10368, -9083, -10210, -10125, -11435, -11455, -12565],
}

checked = 0
for size, expected_values in objectives.items():
    for index, expected in enumerate(expected_values, 1):
        problem = list(hercules.read_qubo(f"test_data/bqp/bqp{size}-{index}.qubo"))
        # Match the existing public-objective tests' stored fixture convention.
        problem[2] = [
            value * (2.0 if i == j else 4.0)
            for i, j, value in zip(problem[0], problem[1], problem[2])
        ]
        problem = tuple(problem)
        for probes, weak_roof, relations, pairs, root_probes, symmetry in product(
            (0, 1), (False, True), (False, True), (False, True), (0, 8), (False, True)
        ):
            solution, objective, _seconds, visited, _processed = hercules.solve_branch_bound(
                problem,
                timeout=10.0,
                branch_strategy="LargestEdges",
                sub_problem_solver="mixingcut_sdp",
                threads=64,
                verbose=0,
                node_probe_candidates=probes,
                node_probe_max_free=256,
                node_probe_max_seconds=0.01,
                roof_dual_weak_persistencies=weak_roof,
                roof_dual_relation_penalties=relations,
                root_pair_dominance=pairs,
                root_roof_probe_candidates=root_probes,
                root_roof_probe_max_seconds=0.25,
                root_complement_symmetry=symmetry,
            )
            # Stored fixtures may include an unused zero-index padding variable.
            if len(solution) != problem[-1] or any(value not in (0, 1) for value in solution):
                raise RuntimeError("Solver returned an invalid binary solution")
            if not abs(objective - expected) < 1e-5:
                raise RuntimeError(
                    f"bqp{size}-{index}: known objective mismatch: "
                    f"expected={expected}, reported={objective}; probes={probes}, "
                    f"weak_roof={weak_roof}, relations={relations}, pairs={pairs}, "
                    f"root_probes={root_probes}, symmetry={symmetry}"
                )
            checked += 1
            print(
                f"bqp{size}-{index} probes={probes} weak_roof={weak_roof} "
                f"relations={relations} pairs={pairs} root_probes={root_probes} "
                f"symmetry={symmetry} objective={objective} visited={visited}"
            )

for option in ("node_probe_max_seconds", "root_roof_probe_max_seconds"):
    for invalid in (-1.0, float("inf"), float("nan")):
        try:
            hercules.solve_branch_bound(problem, timeout=1.0, **{option: invalid})
        except ValueError:
            pass
        else:
            raise RuntimeError(f"accepted invalid {option}={invalid}")

print(f"{checked} Python solves matched published objectives; invalid budgets rejected.")
