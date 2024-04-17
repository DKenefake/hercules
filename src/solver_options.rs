use crate::branch_stratagy::BranchStrategySelection;
use crate::branch_subproblem::SubProblemSelection;
use std::collections::HashMap;

/// Options for the B&B solver for run time
pub struct SolverOptions {
    pub fixed_variables: HashMap<usize, usize>,
    pub branch_strategy: BranchStrategySelection,
    pub sub_problem_solver: SubProblemSelection,
    pub max_time: f64,
    pub seed: usize,
    pub verbose: usize,
    pub threads: usize,
}

impl SolverOptions {
    pub fn new() -> Self {
        Self {
            fixed_variables: HashMap::new(),
            branch_strategy: BranchStrategySelection::MostViolated,
            sub_problem_solver: SubProblemSelection::Clarabel,
            max_time: 100.0,
            seed: 0,
            verbose: 1,
            threads: 1,
        }
    }

    pub fn set_branch_strategy(&mut self, strategy: Option<String>) {
        if let Some(s) = strategy {
            match s.as_str() {
                "FirstNotFixed" => self.branch_strategy = BranchStrategySelection::FirstNotFixed,
                "MostViolated" => self.branch_strategy = BranchStrategySelection::MostViolated,
                "Random" => self.branch_strategy = BranchStrategySelection::Random,
                "WorstApproximation" => {
                    self.branch_strategy = BranchStrategySelection::WorstApproximation;
                }
                "BestApproximation" => {
                    self.branch_strategy = BranchStrategySelection::BestApproximation;
                }
                _ => {}
            }
        }
    }

    pub fn set_sub_problem_strategy(&mut self, strategy: Option<String>) {
        // currently only one strategy is implemented but the structure is left for extension
        if let Some(_) = strategy {
            self.sub_problem_solver = SubProblemSelection::Clarabel;
        }
    }
}
