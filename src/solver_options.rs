use std::collections::HashMap;
use crate::branch_stratagy::{BranchStrategy, BranchStrategySelection};
use crate::branch_subproblem::{ClarabelSubProblemSolver, SubProblemSelection, SubProblemSolver};

/// Options for the B&B solver for run time
pub struct SolverOptions {
    pub fixed_variables: HashMap<usize, f64>,
    pub branch_strategy: BranchStrategySelection,
    pub sub_problem_solver: SubProblemSelection,
    pub max_time: f64,
    pub seed: usize,
    pub verbose: bool,
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
            verbose: true,
            threads: 1,
        }
    }

    pub fn set_branch_strategy(&mut self, strategy: Option<String>) {
        match strategy {
            Some(s) => match s.as_str() {
                "FirstNotFixed" => self.branch_strategy = BranchStrategySelection::FirstNotFixed,
                "MostViolated" => self.branch_strategy = BranchStrategySelection::MostViolated,
                "Random" => self.branch_strategy = BranchStrategySelection::Random,
                "WorstApproximation" => self.branch_strategy = BranchStrategySelection::WorstApproximation,
                "BestApproximation" => self.branch_strategy = BranchStrategySelection::BestApproximation,
                _ => self.branch_strategy = BranchStrategySelection::MostViolated,
            },
            None => (),
        }
    }

    pub fn set_sub_problem_strategy(&mut self, strategy: Option<String>) {
        match strategy {
            Some(s) => match s.as_str() {
                "QP" => self.sub_problem_solver = SubProblemSelection::Clarabel,
                _ => self.sub_problem_solver = SubProblemSelection::Clarabel,
            },
            None => (),
        }
    }
}