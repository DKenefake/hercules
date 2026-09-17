use crate::branch_stratagy::BranchStrategy;
use crate::branch_subproblem::SubProblemSelection;
use crate::heuristic_stratagy::HeuristicSelection;
use crate::FixedVarMap;

#[derive(Clone, Copy)]
pub enum NodeLowerBoundSelection {
    Li,
    RoofDual,
}

/// Options for the B&B solver for run time
#[derive(Clone)]
pub struct SolverOptions {
    pub fixed_variables: FixedVarMap,
    pub branch_strategy: BranchStrategy,
    pub sub_problem_solver: SubProblemSelection,
    pub node_lower_bound: NodeLowerBoundSelection,
    /// Use compatible weak SCC fixings in roof-dual node presolve.
    pub roof_dual_weak_persistencies: bool,
    /// Strengthen roof duality with nonnegative penalties for strict root relations.
    pub roof_dual_relation_penalties: bool,
    /// Root-only joint pair dominance, with a 50 ms soft budget and 250,000-pair limit.
    pub root_pair_dominance: bool,
    pub heuristic: HeuristicSelection,
    pub max_time: f64,
    pub seed: usize,
    pub verbose: usize,
    pub threads: usize,
    /// Conditional presolve candidates per eligible node (default one); zero disables it.
    pub node_probe_candidates: usize,
    /// Limit lookahead to residual nodes with at most this many free variables.
    pub node_probe_max_free: usize,
    /// Per-node wall-time budget, checked between conditional presolve calls.
    pub node_probe_max_seconds: f64,
    /// Roof-dual candidates per root pass; repeat after reductions. Zero disables it.
    pub root_roof_probe_candidates: usize,
    /// Soft budget for all root roof-probing passes together.
    pub root_roof_probe_max_seconds: f64,
    /// Choose one orientation per exactly complement-symmetric root component.
    pub root_complement_symmetry: bool,
    /// Eliminate degree-zero, one and two variables once at the root, with postsolve.
    pub root_low_degree_elimination: bool,
    /// Also eliminate degree-three variables when their exact table is quadratic.
    /// Requires `root_low_degree_elimination`.
    pub root_degree_three_elimination: bool,
    /// Sequential optimum-preserving dominant-edge substitutions at the root.
    pub root_dominant_edge_contraction: bool,
    /// Split disconnected residual QUBOs into independently searched components.
    pub component_decomposition: bool,
    /// Compact reduced child nodes, with bounded work and persistent postsolve.
    pub node_structural_reductions: bool,
    /// Opt-in elimination of blocks of at most eight variables with at most two boundary variables.
    pub small_block_elimination: bool,
    /// Propagate the incumbent cutoff into reduced/component searches.
    pub component_cutoff_propagation: bool,
    /// Root-only whole-cut dominance: at most 64 flow trials and 50 ms.
    pub root_cut_dominance: bool,
    /// Experimental SDP dual-slack fixing, capped at 128 free variables and 5 ms.
    pub sdp_dual_fixing: bool,
}

impl SolverOptions {
    pub fn new() -> Self {
        Self {
            fixed_variables: FixedVarMap::default(),
            branch_strategy: BranchStrategy::LargestEdges,
            sub_problem_solver: SubProblemSelection::HerculesABQP,
            node_lower_bound: NodeLowerBoundSelection::RoofDual,
            roof_dual_weak_persistencies: false,
            roof_dual_relation_penalties: false,
            root_pair_dominance: false,
            heuristic: HeuristicSelection::LocalSearch,
            max_time: 100.0,
            seed: 0,
            verbose: 1,
            threads: 256,
            node_probe_candidates: 1,
            node_probe_max_free: 256,
            node_probe_max_seconds: 0.01,
            root_roof_probe_candidates: 0,
            root_roof_probe_max_seconds: 0.25,
            root_complement_symmetry: true,
            root_low_degree_elimination: true,
            root_degree_three_elimination: true,
            root_dominant_edge_contraction: true,
            component_decomposition: true,
            node_structural_reductions: true,
            small_block_elimination: false,
            component_cutoff_propagation: true,
            root_cut_dominance: false,
            sdp_dual_fixing: false,
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
                "mixingcut_sdp_momentum" => {
                    self.sub_problem_solver = SubProblemSelection::MixingCutSDPMomentum;
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
    fn test_solver_options_default_branch_strategy() {
        for mut options in [SolverOptions::new(), SolverOptions::default()] {
            assert!(matches!(
                options.branch_strategy,
                BranchStrategy::LargestEdges
            ));
            // Python's omitted branch_strategy follows this same setter path.
            options.set_branch_strategy(None);
            assert!(matches!(
                options.branch_strategy,
                BranchStrategy::LargestEdges
            ));
            options.set_branch_strategy(Some("MostViolated".to_string()));
            assert!(matches!(
                options.branch_strategy,
                BranchStrategy::MostViolated
            ));
        }
    }

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
    fn test_solver_options_set_sub_problem_strat_mixingcut_momentum() {
        let mut options = SolverOptions::new();
        options.set_sub_problem_strategy(Some("mixingcut_sdp_momentum".to_string()));
        assert!(matches!(
            options.sub_problem_solver,
            SubProblemSelection::MixingCutSDPMomentum
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
