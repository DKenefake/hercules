use ndarray::Array1;
use std::cmp::Ordering;
use std::collections::HashMap;

/// Struct the describes the branch and bound tree nodes
#[derive(Clone)]
pub struct QuboBBNode {
    pub lower_bound: f64,
    pub solution: Array1<f64>,
    pub fixed_variables: HashMap<usize, usize>,
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
        self.fixed_variables.len().partial_cmp(&other.fixed_variables.len()).unwrap()
    }
}
