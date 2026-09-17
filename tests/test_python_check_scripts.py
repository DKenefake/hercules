"""Keep standalone binding checks effective even with optimized Python bytecode."""

import importlib.util
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
SCRIPTS = (
    "check_components_python.py",
    "check_degree_three_python.py",
    "check_dominant_edge_python.py",
    "check_low_degree_python.py",
    "check_node_probing_python.py",
)


class CheckScriptValidationTests(unittest.TestCase):
    """Exercise each script's first solve with deliberately invalid results."""

    def test_invalid_results_raise_with_and_without_optimization(self):
        """Run this suite normally and with -O; runpy uses that optimization mode."""
        for script in SCRIPTS:
            for failure in ("dimension", "binary", "nan", "infinity", "reported", "optimum"):
                with self.subTest(script=script, failure=failure):
                    def solve(problem, *, failure=failure, **_options):
                        rows, columns, values, linear, size = problem
                        solution = [(i % 13) % 2 for i in range(size)] if size > 1 else [1]
                        if failure == "optimum":
                            solution = [0] * size
                        objective = sum(
                            0.5 * q * solution[i] * solution[j]
                            for i, j, q in zip(rows, columns, values)
                        ) + sum(c * v for c, v in zip(linear, solution))
                        if failure == "dimension":
                            solution = []
                        elif failure == "binary":
                            solution[0] = 2
                        elif failure == "nan":
                            objective = float("nan")
                        elif failure == "infinity":
                            objective = float("inf")
                        elif failure == "reported":
                            objective += 1.0
                        return solution, objective, 0.0, 1, 1

                    fake = SimpleNamespace(
                        read_qubo=lambda _path: ([0], [0], [-2098.0], [0.0], 1),
                        solve_branch_bound=solve,
                    )
                    spec = SimpleNamespace(loader=SimpleNamespace(exec_module=lambda _module: None))
                    message = "invalid binary solution" if failure in ("dimension", "binary") else "[Oo]bjective mismatch"
                    with patch.object(importlib.util, "spec_from_file_location", return_value=spec), \
                            patch.object(importlib.util, "module_from_spec", return_value=fake), \
                            patch.object(sys, "argv", [script, "mock-extension"]), \
                            self.assertRaisesRegex(RuntimeError, message):
                        runpy.run_path(str(EXAMPLES / script))


if __name__ == "__main__":
    unittest.main()
