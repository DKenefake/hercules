//! Compare conditional roof-dual deductions after ordinary root presolve.
use hercules::branchbound::BBSolver;
use hercules::qubo::Qubo;
use hercules::solver_options::SolverOptions;
use hercules::subproblemsolvers::roofdual::PreparedRoofDual;
use hercules::FixedVarMap;
use std::time::Instant;

fn main() {
    for path in std::env::args().skip(1) {
        let mut qubo = Qubo::read_qubo(&path);
        if path.contains("bqp") {
            let mut terms = sprs::TriMat::new((qubo.num_x(), qubo.num_x()));
            for (&v, (i, j)) in &qubo.q {
                terms.add_triplet(i, j, v * if i == j { 2.0 } else { 4.0 });
            }
            qubo.q = terms.to_csr();
        } else {
            qubo.q = &qubo.q * 2.0;
        }
        let mut options = SolverOptions::new();
        options.verbose = 0;
        let solver = BBSolver::new(qubo, options);
        let roof = PreparedRoofDual::new(&solver.qubo_pp_form);
        let mut fixed = FixedVarMap::default();
        loop {
            let before = fixed.len();
            fixed = solver.preprocess_fixed_variables(&fixed);
            fixed.extend(
                roof.solve_iterative(&fixed, solver.qubo.num_x())
                    .fixed_variables,
            );
            if fixed.len() == before {
                break;
            }
        }
        let mut scores = vec![0.0; solver.qubo.num_x()];
        for (&v, (i, j)) in &solver.qubo_pp_form.q {
            if !fixed.contains_key(&i) && !fixed.contains_key(&j) {
                scores[i] += v.abs();
                scores[j] += v.abs();
            }
        }
        let mut candidates: Vec<_> = (0..scores.len())
            .filter(|i| !fixed.contains_key(i))
            .collect();
        candidates.sort_by(|&i, &j| scores[j].total_cmp(&scores[i]).then(i.cmp(&j)));
        let start = Instant::now();
        let mut common = FixedVarMap::default();
        let mut relations = 0;
        println!(
            "ROOT path={path} n={} fixed={} bound={:.8}",
            solver.qubo.num_x(),
            fixed.len(),
            roof.solve(&fixed).lower_bound.unwrap()
        );
        for (position, &i) in candidates.iter().enumerate() {
            let sides: Vec<_> = [0, 1]
                .into_iter()
                .map(|v| {
                    let mut assumed = fixed.clone();
                    assumed.insert(i, v);
                    roof.solve_iterative(&assumed, solver.qubo.num_x())
                })
                .collect();
            for (&j, &v) in &sides[0].fixed_variables {
                if sides[1].fixed_variables.get(&j) == Some(&v) {
                    common.insert(j, v);
                } else {
                    relations += 1;
                }
            }
            relations += sides[1]
                .fixed_variables
                .iter()
                .filter(|(j, _)| !sides[0].fixed_variables.contains_key(j))
                .count();
            if [4, 8, 25, 100, candidates.len()].contains(&(position + 1)) {
                println!(
                    "PROBE count={} common={} relations={} elapsed_ms={:.3}",
                    position + 1,
                    common.len(),
                    relations,
                    start.elapsed().as_secs_f64() * 1000.0
                );
            }
            if start.elapsed().as_secs_f64() > 10.0 {
                println!("TIMEOUT");
                break;
            }
        }
    }
}
