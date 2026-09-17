//! Synthetic separable QUBOs with analytically known optima; not MK fixtures.
use hercules::branch_stratagy::BranchStrategy;
use hercules::branch_subproblem::SubProblemSelection;
use hercules::branchbound::BBSolver;
use hercules::qubo::Qubo;
use hercules::solver_options::SolverOptions;
use ndarray::Array1;
use std::time::Instant;

fn main() {
    for count in [2, 4, 8] {
        let size = 13;
        let n = count * size;
        let mut q = sprs::TriMat::new((n, n));
        let mut c = Array1::zeros(n);
        for block in 0..count {
            let start = block * size;
            for i in start..start + size {
                for j in i + 1..start + size {
                    q.add_triplet(i, j, 4.0);
                    c[i] -= 1.0;
                    c[j] -= 1.0;
                }
            }
        }
        let qubo = Qubo::new_with_c(q.to_csr(), c);
        let expected = -(count as f64) * ((size * size / 4) as f64);
        for enabled in [false, true] {
            let mut options = SolverOptions::new();
            options.component_decomposition = enabled;
            options.sub_problem_solver = SubProblemSelection::MixingCutSDP;
            options.branch_strategy = BranchStrategy::LargestEdges;
            options.node_probe_candidates = 0;
            options.threads = 64;
            options.max_time = 5.0;
            options.verbose = 0;
            let start = Instant::now();
            let mut solver = BBSolver::new(qubo.clone(), options);
            let (x, objective) = solver.solve();
            let lower = solver
                .nodes
                .iter()
                .map(|n| n.lower_bound)
                .fold(solver.best_solution_value, f64::min);
            assert!((qubo.eval_usize(&x) - objective).abs() < 1e-6);
            assert!(lower <= expected + 1e-6 && objective >= expected - 1e-6);
            assert!(!solver.nodes.is_empty() || (objective - expected).abs() < 1e-6);
            println!("SYNTHETIC cliques={count} n={n} components={enabled} seconds={:.6} visited={} sdp={} objective={objective:.8} lower_bound={lower:.8} remaining={} optimal={} stats={:?}",
                start.elapsed().as_secs_f64(), solver.nodes_visited, solver.node_probing_statistics().subproblem_calls,
                solver.nodes.len(), solver.nodes.is_empty(), solver.component_statistics());
        }
    }
}
