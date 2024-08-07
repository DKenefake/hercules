use crate::branchbound::BBSolver;
use crate::branchbound_utils::get_current_time;

/// This is the main logic behind the solver output
///
/// It has varying levels of output, where 0 means nothing is displayed to the screen, and each
/// additional level includes everything previous
///
/// 0 - Nothing
/// 1 - Header, Iteration Log, and Finish
/// 2 - Each New Feasible Solution
///
pub struct SolverOutputLogger {
    pub output_level: usize,
}

impl SolverOutputLogger {
    pub const fn new(level: usize) -> Self {
        Self {
            output_level: level,
        }
    }

    pub fn output_header(&self, solver_instance: &BBSolver) {
        if self.output_level < 1 {
            return;
        }

        let version_number = env!("CARGO_PKG_VERSION");
        let num_variables = solver_instance.qubo.num_x();
        let fixed_vars = solver_instance.options.fixed_variables.len();

        println!("Hercules: A Rust-based Branch and Bound Solver for QUBO");
        println!("Version number {version_number}");
        println!("Problem size: {num_variables}");
        println!("Fixed variables: {fixed_vars}");

        println!("----------------------------------------------------------------------------");
        println!("Nodes Visited |  Best Solution |   Lower Bound  |    Gap (%)    | Time (sec)");
    }

    pub fn generate_output_line(&self, solver_instance: &BBSolver) {
        if self.output_level < 1 {
            return;
        }

        let num_nodes = solver_instance.nodes_solved;
        let upper_bound = solver_instance.best_solution_value;
        let lower_bound = solver_instance
            .nodes
            .iter()
            .map(|x| x.lower_bound)
            .fold(f64::INFINITY, f64::min);
        let gap = 100.0 * (upper_bound - lower_bound) / (upper_bound + 1E-5).abs();
        let gap = gap.max(0.0);
        let lower_bound = lower_bound.min(upper_bound);
        let current_time = get_current_time() - solver_instance.time_start;
        println!("{num_nodes:<13} | {upper_bound:<13.8} | {lower_bound:<11.8} | {gap:<13.8} | {current_time:<13.8}");
    }

    pub fn generate_exit_line(&self, solver_instance: &BBSolver) {
        if self.output_level < 1 {
            return;
        }

        let solution = solver_instance.best_solution.clone();
        let solution_value = solver_instance.best_solution_value;

        let nodes_solved = solver_instance.nodes_solved;
        let nodes_processed = solver_instance.nodes_processed;
        let nodes_visited = solver_instance.nodes_visited;

        let current_time = get_current_time();
        let time_passed = current_time - solver_instance.time_start;

        let upper_bound = solver_instance.best_solution_value;
        let lower_bound = solver_instance
            .nodes
            .iter()
            .map(|x| x.lower_bound)
            .fold(f64::INFINITY, f64::min);
        let gap = 100.0 * (upper_bound - lower_bound) / (upper_bound + 1E-5).abs();
        let gap = gap.max(0.0);

        let status = if gap < 1E-5 { "Optimal" } else { "Suboptimal" };

        println!("----------------------------------------------------------------------------");
        println!("Branch and Bound Solver Finished");
        println!("Best Solution: {solution}");
        println!("Best Solution Value: {solution_value}");
        println!("Nodes Solved: {nodes_solved}");
        println!("Nodes Processed: {nodes_processed}");
        println!("Nodes Visited: {nodes_visited}");
        println!("Time to Solve: {time_passed}");
        println!("Solver Status: {status}");
        println!("----------------------------------------------------------------------------");
    }

    pub fn output_warm_start_info(&self, solver_instance: &BBSolver) {
        if self.output_level < 1 {
            return;
        }

        let solution_value = solver_instance.best_solution_value;
        println!("----------------------------------------------------------------------------");
        println!("Warm Start Information");
        println!("Warm started objective: {solution_value}");
        println!("----------------------------------------------------------------------------");
    }

    pub fn early_termination(&self) {
        if self.output_level < 1 {
            return;
        }
        println!("----------------------------------------------------------------------------");
        println!("Beck Proof of Optimality Found!");
        println!("Early Termination");
        println!("----------------------------------------------------------------------------");
    }
}

#[cfg(test)]
mod tests {
    use crate::branchbound::BBSolver;
    use crate::branchboundlogger::SolverOutputLogger;
    use crate::qubo::Qubo;
    use crate::solver_options::SolverOptions;
    use ndarray::Array1;
    use sprs::CsMat;

    #[test]
    fn test_output_header() {
        let mut solver = BBSolver::new(
            Qubo::new_with_c(CsMat::eye(3), Array1::from_vec(vec![1.0, -2.0, 3.0])),
            SolverOptions::new(),
        );

        let solver_logger = SolverOutputLogger { output_level: 1 };

        let _ = solver.solve();

        solver_logger.output_warm_start_info(&solver);
        solver_logger.generate_exit_line(&solver);
    }
}
