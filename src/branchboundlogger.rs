use crate::branchbound::BBSolver;

pub fn output_header(solver_instance: &BBSolver) {
    let version_number = env!("CARGO_PKG_VERSION");
    let num_variables = solver_instance.qubo.num_x();
    let fixed_vars = solver_instance.options.fixed_variables.len();

    println!("Hercules: A Rust-based Branch and Bound Solver for QUBO");
    println!("Version number {version_number}");
    println!("Problem size: {num_variables}");
    println!("Fixed variables: {fixed_vars}");

    println!("------------------------------------------------------");
    println!("Nodes Visited | Best Solution | Lower Bound | Gap (%)");
}

pub fn generate_output_line(solver_instance: &BBSolver) {
    let num_nodes = solver_instance.nodes_visited;
    let upper_bound = solver_instance.best_solution_value;
    let lower_bound = solver_instance.nodes.iter().map(|x| x.lower_bound).fold(f64::INFINITY, |a, b| a.min(b));
    let gap = (upper_bound - lower_bound) / upper_bound * 100.0;
    println!("{num_nodes} | {upper_bound} | {lower_bound} | {gap}");
}

#[cfg(test)]
mod tests {
    use crate::branchbound::BBSolver;
    use crate::solver_options::SolverOptions;
    use crate::branchboundlogger::{generate_output_line, output_header};
    use crate::qubo::Qubo;
    use ndarray::Array1;
    use sprs::CsMat;

    #[test]
    fn test_output_header() {
        let mut solver = BBSolver::new(
            Qubo::new_with_c(CsMat::eye(3), Array1::from_vec(vec![1.0, -2.0, 3.0])),
            SolverOptions::new(),
        );

        solver.preprocess_initial();

        let s = solver.solve();

        output_header(&solver);
        generate_output_line(&solver);
    }
}
