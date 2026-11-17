use crate::FixedVarMap;
use ndarray::Array1;
use std::cmp::Ordering;

#[derive(Clone)]
pub enum SubProblemNodeState {
    /// SDP proposal only; neither parity is a certified fixing.
    RelationCandidate(usize, usize, f64),
    /// An AND node: component frontiers are retained between scheduler batches.
    Components(std::sync::Arc<std::sync::Mutex<crate::branchbound::ComponentSearch>>),
}

/// Branch provenance for heuristic observations, never a proof of a bound.
#[derive(Clone, Copy, Debug)]
pub struct BranchOrigin {
    pub(crate) variable: usize,
    pub(crate) value: usize,
    pub(crate) parent_bound: f64,
    pub(crate) parent_free: usize,
    pub(crate) gap_scale: f64,
    pub(crate) size: usize,
}

/// A branch-and-bound tree node.
#[derive(Clone)]
pub struct QuboBBNode {
    pub lower_bound: f64,
    pub solution: Array1<f64>,
    pub fixed_variables: FixedVarMap,
    pub run_heuristic: bool,
    pub subproblem_state: Option<SubProblemNodeState>,
    /// Heuristic-only provenance, consumed on the first processing of this child.
    pub branch_origin: Option<BranchOrigin>,
    /// Conditional bounds valid only in this node's coordinate/objective context.
    /// Ordinary children share storage; remapped subproblems start empty.
    pub probe_bounds: Option<std::sync::Arc<Vec<crate::branch_subproblem::ConditionalLowerBound>>>,
    /// Completed probe presolve; validated and consumed on first processing.
    pub probe_presolved: Option<std::sync::Arc<ProbePresolve>>,
}

/// Opaque certificate tied to one solver's prepared problem and exact fixings.
pub struct ProbePresolve {
    pub(crate) context: std::sync::Arc<()>,
    pub(crate) fixed: FixedVarMap,
    pub(crate) bound: f64,
    pub(crate) weak_roof: bool,
    pub(crate) weak_gradient: bool,
    pub(crate) lower_bound: crate::solver_options::NodeLowerBoundSelection,
}

impl Eq for QuboBBNode {}

impl PartialEq<Self> for QuboBBNode {
    fn eq(&self, other: &Self) -> bool {
        self.fixed_variables == other.fixed_variables
    }
}

impl PartialOrd<Self> for QuboBBNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for QuboBBNode {
    fn cmp(&self, other: &Self) -> Ordering {
        other.lower_bound.partial_cmp(&self.lower_bound).unwrap()
    }
}
