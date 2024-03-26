use crate::branchbound::BBSolver;

pub fn output_header(solver_instance: &BBSolver) {
    let version_number = env!("CARGO_PKG_VERSION");
    let num_variables = solver_instance.qubo.num_x();
    let fixed_vars = solver_instance.options.fixed_variables.len();

    println!("Hercules: A Rust-based Branch and Bound Solver for QUBO");
    println!("Version number {version_number}");
    println!("Problem size: {num_variables}");
    println!("Fixed variables: {fixed_vars}");
}

#[cfg(test)]
mod tests {
    use crate::branchbound::BBSolver;
    use crate::branchbound_utils::SolverOptions;
    use crate::branchboundlogger::output_header;
    use crate::qubo::Qubo;
    use ndarray::Array1;
    use sprs::CsMat;

    #[test]
    fn test_output_header() {
        let solver = BBSolver::new(
            Qubo::new_with_c(CsMat::eye(3), Array1::from_vec(vec![1.0, 2.0, 3.0])),
            SolverOptions::new(),
        );

        output_header(&solver);
    }
}
