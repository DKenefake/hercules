use crate::FixedVarMap;
use crate::branch_stratagy::BranchStrategy;
use crate::branch_subproblem::SubProblemSelection;
use crate::heuristic_stratagy::HeuristicSelection;

#[derive(Clone, Copy)]
pub enum NodeLowerBoundSelection {
    Li,
    RoofDual,
}

/// Options for the B&B solver for run time
pub struct SolverOptions {
    pub fixed_variables: FixedVarMap,
    pub branch_strategy: BranchStrategy,
    pub sub_problem_solver: SubProblemSelection,
    pub node_lower_bound: NodeLowerBoundSelection,
    pub heuristic: HeuristicSelection,
    pub max_time: f64,
    pub seed: usize,
    pub verbose: usize,
    pub threads: usize,
}

impl SolverOptions {
    pub fn new() -> Self {
        Self {
            fixed_variables: FixedVarMap::default(),
            branch_strategy: BranchStrategy::MostViolated,
            sub_problem_solver: SubProblemSelection::HerculesABQP,
            node_lower_bound: NodeLowerBoundSelection::RoofDual,
            heuristic: HeuristicSelection::LocalSearch,
            max_time: 100.0,
            seed: 0,
            verbose: 1,
            threads: 256,
        }
    }

    pub fn set_branch_strategy(&mut self, strategy: Option<String>) {
        if let Some(s) = strategy {
            match s.as_str() {
                "FirstNotFixed" => self.branch_strategy = BranchStrategy::FirstNotFixed,
                "MostViolated" => self.branch_strategy = BranchStrategy::MostViolated,
                "Random" => self.branch_strategy = BranchStrategy::Random,
                "WorstApproximation" => {
                    self.branch_strategy = BranchStrategy::WorstApproximation;
                }
                "WorstApproximation2" => {
                    self.branch_strategy = BranchStrategy::WorstApproximation2;
                }
                "MostEdges" => self.branch_strategy = BranchStrategy::MostEdges,
                "LargestEdges" => self.branch_strategy = BranchStrategy::LargestEdges,
                "MostFixed" => self.branch_strategy = BranchStrategy::MostFixed,
                "FullStrongBranching" => self.branch_strategy = BranchStrategy::FullStrongBranching,
                "PartialStrongBranching" => {
                    self.branch_strategy = BranchStrategy::PartialStrongBranching;
                }
                "RoundRobin" => self.branch_strategy = BranchStrategy::RoundRobin,
                "LargestDiag" => {
                    self.branch_strategy = BranchStrategy::LargestDiag;
                }
                "MovingEdges" => {
                    self.branch_strategy = BranchStrategy::MoveingEdges;
                }
                "ConnectedComponents" => {
                    self.branch_strategy = BranchStrategy::ConnectedComponents;
                }
                _ => {}
            }
        }
    }

    pub fn set_sub_problem_strategy(&mut self, strategy: Option<String>) {
        // currently only one strategy is implemented but the structure is left for extension
        #[allow(clippy::redundant_pattern_matching)]
        if let Some(s) = strategy {
            match s.as_str() {
                "hercules_cd" => {
                    self.sub_problem_solver = SubProblemSelection::HerculesCDQP;
                }
                "hercules_abqp" => {
                    self.sub_problem_solver = SubProblemSelection::HerculesABQP;
                }
                "clarabel_lp" => {
                    self.sub_problem_solver = SubProblemSelection::ClarabelLP;
                }
                "roof_dual" => {
                    self.sub_problem_solver = SubProblemSelection::RoofDualQPBO;
                }
                "mixingcut_sdp" => {
                    self.sub_problem_solver = SubProblemSelection::MixingCutSDP;
                }
                _ => self.sub_problem_solver = SubProblemSelection::HerculesABQP,
            }
        }
    }

    pub fn set_node_lower_bound_strategy(&mut self, strategy: Option<String>) {
        if let Some(s) = strategy {
            match s.as_str() {
                "li" => self.node_lower_bound = NodeLowerBoundSelection::Li,
                _ => self.node_lower_bound = NodeLowerBoundSelection::RoofDual,
            }
        }
    }

    pub fn set_heuristic_strategy(&mut self, strategy: Option<String>) {
        if let Some(s) = strategy {
            match s.as_str() {
                "SimpleRounding" => self.heuristic = HeuristicSelection::SimpleRounding,
                _ => self.heuristic = HeuristicSelection::LocalSearch,
            }
        }
    }
}

impl Default for SolverOptions {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use crate::branch_stratagy::BranchStrategy;
    use crate::branch_subproblem::SubProblemSelection;
    use crate::heuristic_stratagy::HeuristicSelection;
    use crate::solver_options::{NodeLowerBoundSelection, SolverOptions};

    #[test]
    fn test_solver_options_set_branch_strat() {
        let mut options = SolverOptions::new();
        options.set_branch_strategy(Some("Random".to_string()));
        assert!(matches!(options.branch_strategy, BranchStrategy::Random));
    }

    #[test]
    fn test_solver_options_set_sub_problem_strat_2() {
        let mut options = SolverOptions::new();
        options.set_sub_problem_strategy(Some("hercules_cd".to_string()));
        assert!(matches!(
            options.sub_problem_solver,
            SubProblemSelection::HerculesCDQP
        ));
    }

    #[test]
    fn test_solver_options_set_sub_problem_strat_abqp() {
        let mut options = SolverOptions::new();
        options.set_sub_problem_strategy(Some("hercules_abqp".to_string()));
        assert!(matches!(
            options.sub_problem_solver,
            SubProblemSelection::HerculesABQP
        ));
    }

    #[test]
    fn test_solver_options_set_sub_problem_strat_3() {
        let mut options = SolverOptions::new();
        options.set_sub_problem_strategy(Some("qweqwe".to_string()));
        assert!(matches!(
            options.sub_problem_solver,
            SubProblemSelection::HerculesABQP
        ));
    }

    #[test]
    fn test_solver_options_set_sub_problem_strat_roof_dual() {
        let mut options = SolverOptions::new();
        options.set_sub_problem_strategy(Some("roof_dual".to_string()));
        assert!(matches!(
            options.sub_problem_solver,
            SubProblemSelection::RoofDualQPBO
        ));
    }

    #[test]
    fn test_solver_options_set_sub_problem_strat_mixingcut() {
        let mut options = SolverOptions::new();
        options.set_sub_problem_strategy(Some("mixingcut_sdp".to_string()));
        assert!(matches!(
            options.sub_problem_solver,
            SubProblemSelection::MixingCutSDP
        ));
    }

    #[test]
    fn test_solver_options_default_node_lower_bound() {
        let options = SolverOptions::new();
        assert!(matches!(
            options.node_lower_bound,
            NodeLowerBoundSelection::RoofDual
        ));
    }

    #[test]
    fn test_solver_options_set_node_lower_bound_li() {
        let mut options = SolverOptions::new();
        options.set_node_lower_bound_strategy(Some("li".to_string()));
        assert!(matches!(
            options.node_lower_bound,
            NodeLowerBoundSelection::Li
        ));
    }

    #[test]
    fn test_solver_options_set_node_lower_bound_roof_dual() {
        let mut options = SolverOptions::new();
        options.set_node_lower_bound_strategy(Some("roof_dual".to_string()));
        assert!(matches!(
            options.node_lower_bound,
            NodeLowerBoundSelection::RoofDual
        ));
    }
    #[test]
    fn test_solver_options_set_heuristic_strat_1() {
        let mut options = SolverOptions::new();
        options.set_heuristic_strategy(Some("SimpleRounding".to_string()));
        assert!(matches!(
            options.heuristic,
            HeuristicSelection::SimpleRounding
        ));
    }

    #[test]
    fn test_solver_options_set_heuristic_strat_2() {
        let mut options = SolverOptions::new();
        options.set_heuristic_strategy(Some("LocalSearch".to_string()));
        assert!(matches!(options.heuristic, HeuristicSelection::LocalSearch));
    }
}
