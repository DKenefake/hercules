//! Inspect stronger root probing on MK inputs, using the MK runner's 2x Hessian.
use hercules::branchbound::BBSolver;
use hercules::qubo::Qubo;
use hercules::solver_options::SolverOptions;
use hercules::subproblemsolvers::roofdual::iterative_roof_duality_presolve;
use hercules::variable_reduction::probe_limited;
use hercules::FixedVarMap;
use std::time::Instant;

fn close(solver: &BBSolver, mut fixed: FixedVarMap) -> FixedVarMap {
    loop {
        let before = fixed.len();
        fixed = solver.preprocess_fixed_variables(&fixed);
        fixed.extend(
            iterative_roof_duality_presolve(&solver.qubo_pp_form, &fixed, solver.qubo.num_x())
                .fixed_variables,
        );
        if fixed.len() == before {
            return fixed;
        }
    }
}

fn main() {
    let paths: Vec<_> = std::env::args().skip(1).collect();
    for path in paths {
        let mut qubo = Qubo::read_qubo(&path);
        qubo.q = &qubo.q * 2.0;
        let mut options = SolverOptions::new();
        options.verbose = 0;
        let solver = BBSolver::new(qubo, options);
        let base = close(&solver, FixedVarMap::default());
        println!(
            "ROOT path={path} n={} fixed={}",
            solver.qubo.num_x(),
            base.len()
        );
        for limit in [25, 100, solver.qubo.num_x()] {
            let start = Instant::now();
            let mut fixed = base.clone();
            for round in 0..3 {
                let (relations, added) = probe_limited(&solver.qubo_pp_form, &fixed, true, limit);
                let found = added.len();
                fixed.extend(added);
                fixed = close(&solver, fixed);
                println!("PROBE limit={limit} round={round} found={found} fixed={} relations={} cumulative_ms={:.3}",
                    fixed.len(), relations.constraints.len(), start.elapsed().as_secs_f64() * 1000.0);
                if found == 0 {
                    break;
                }
            }
        }
    }
}
