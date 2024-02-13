use std::collections::HashMap;

pub enum ConstraintType {
    AtLeastOne,
    ExactlyOne,
    NoMoreThanOne,
    GreaterThan,
    LessThan,
    Equal,
}

pub struct Constraint {
    pub(crate) x_i: usize,
    pub(crate) x_j: usize,
    constraint_type: ConstraintType,
}

impl Constraint {
    /// Creates a new constraint with the given indices and type
    pub const fn new(x_i: usize, x_j: usize, constraint_type: ConstraintType) -> Self {
        Self {
            x_i,
            x_j,
            constraint_type,
        }
    }

    /// Checks if the persistent variables are consistent with the constraint
    pub fn check(&self, persistent: &HashMap<usize, f64>) -> bool {
        // can only be computed if both variables are fixed
        if self.how_many_fixed(persistent) != 2 {
            return true;
        }

        // check if we abide by the constraint
        match self.constraint_type {
            ConstraintType::NoMoreThanOne => self.no_more_than_one(persistent),
            ConstraintType::AtLeastOne => self.at_least_one(persistent),
            ConstraintType::ExactlyOne => self.exactly_one(persistent),
            ConstraintType::GreaterThan => self.greater_than(persistent),
            ConstraintType::LessThan => self.less_than(persistent),
            ConstraintType::Equal => self.equal(persistent),
        }
    }

    /// Given a set of persistent variables, computes of an inference can be made and if so
    /// returns the index and value of the fixed variable
    pub fn make_inference(&self, persistent: &HashMap<usize, f64>) -> Option<(usize, f64)> {
        // count how many fixed variables we have
        let num_fixed = self.how_many_fixed(persistent);

        // if both are fixed or unfixed then we can't make any inferences
        if num_fixed == 2 || num_fixed == 0 {
            return None;
        }

        // give we have one fixed, then we can always make the standard form
        let (_, free, fixed_value) = self.get_standard_form(persistent).unwrap();

        // if we have only one fixed, then we can make an inference
        match self.constraint_type {
            ConstraintType::NoMoreThanOne => self.no_more_then_one_inference(free, fixed_value),
            ConstraintType::ExactlyOne => self.exactly_one_inference(free, fixed_value),
            ConstraintType::AtLeastOne => self.at_least_one_inference(free, fixed_value),
            ConstraintType::GreaterThan => self.greater_than_inference(persistent),
            ConstraintType::LessThan => self.less_than_inference(persistent),
            ConstraintType::Equal => self.equal_inference(free, fixed_value),
        }
    }

    pub fn no_more_than_one(&self, persistent: &HashMap<usize, f64>) -> bool {
        // If we have explicitly checked both keys to be fixed so this is safe
        let x_i_val = *persistent.get(&self.x_i).unwrap();
        let x_j_val = *persistent.get(&self.x_j).unwrap();

        // We can't have both
        !(x_i_val == 1.0 && x_j_val == 1.0)
    }

    pub fn at_least_one(&self, persistent: &HashMap<usize, f64>) -> bool {
        // If we have explicitly checked both keys to be fixed so this is safe
        let x_i_val = *persistent.get(&self.x_i).unwrap();
        let x_j_val = *persistent.get(&self.x_j).unwrap();

        // If either is 1, then we are consistent
        x_i_val == 1.0 || x_j_val == 1.0
    }

    pub fn exactly_one(&self, persistent: &HashMap<usize, f64>) -> bool {
        // If we have explicitly checked both keys to be fixed so this is safe
        let x_i_val = *persistent.get(&self.x_i).unwrap();
        let x_j_val = *persistent.get(&self.x_j).unwrap();

        // if we can only have one, then we can't have both
        x_i_val + x_j_val == 1.0
    }

    pub fn greater_than(&self, persistent: &HashMap<usize, f64>) -> bool {
        // If we have explicitly checked both keys to be fixed so this is safe
        let x_i_val = *persistent.get(&self.x_i).unwrap();
        let x_j_val = *persistent.get(&self.x_j).unwrap();

        // if x_i is 1, then x_j must be 0
        x_i_val >= x_j_val
    }

    pub fn less_than(&self, persistent: &HashMap<usize, f64>) -> bool {
        // If we have explicitly checked both keys to be fixed so this is safe
        let x_i_val = *persistent.get(&self.x_i).unwrap();
        let x_j_val = *persistent.get(&self.x_j).unwrap();

        // if x_i is 1, then x_j must be 0
        x_i_val <= x_j_val
    }

    pub fn equal(&self, persistent: &HashMap<usize, f64>) -> bool {
        // If we have explicitly checked both keys to be fixed, so this is safe
        let x_i_val = *persistent.get(&self.x_i).unwrap();
        let x_j_val = *persistent.get(&self.x_j).unwrap();

        // if x_i is 1, then x_j must be 0
        x_i_val == x_j_val
    }

    pub fn how_many_fixed(&self, persistent: &HashMap<usize, f64>) -> usize {
        let mut count = 0;
        if Self::is_fixed(persistent, self.x_i){
            count += 1;
        }
        if Self::is_fixed(persistent, self.x_i){
            count += 1;
        }
        count
    }

    pub fn is_fixed(persistent: &HashMap<usize, f64>, index: usize) -> bool {
        persistent.contains_key(&index)
    }

    pub fn get_standard_form(
        &self,
        persistent: &HashMap<usize, f64>,
    ) -> Option<(usize, usize, f64)> {

        if self.how_many_fixed(persistent) != 1 {
            return None;
        }

        match persistent.get(&self.x_i) {
            Some(x_i_value) => Some((self.x_i, self.x_j, *x_i_value)),
            None => Some((self.x_j, self.x_i, *persistent.get(&self.x_j).unwrap())),
        }
    }

    /// Given a constraint of the type x_i + x_j <= 1, computes if we can make an inference on
    /// either x_i or x_j, given some fixed variables. If so, returns the index and value of the
    /// fixed value, otherwise returns None
    pub fn no_more_then_one_inference(
        &self,
        free_var: usize,
        fixed_value: f64,
    ) -> Option<(usize, f64)> {
        // if the fixed value is 1, then the free variable must be 0
        return match fixed_value == 1.0 {
            true => Some((free_var, 0.0)),
            false => Some((free_var, 1.0)),
        };
    }

    pub fn exactly_one_inference(&self, free_var: usize, fixed_value: f64) -> Option<(usize, f64)> {
        // if the fixed value is 1, then the free variable must be 0
        return match fixed_value == 1.0 {
            true => Some((free_var, 0.0)),
            false => Some((free_var, 1.0)),
        };
    }

    pub fn at_least_one_inference(
        &self,
        free_var: usize,
        fixed_value: f64,
    ) -> Option<(usize, f64)> {
        // if the fixed value is 0, then the free variable must be 1
        return match fixed_value == 1.0 {
            true => None,
            false => Some((free_var, 1.0)),
        };
    }

    pub fn greater_than_inference(&self, persistent: &HashMap<usize, f64>) -> Option<(usize, f64)> {
        // examines the constraint x_i >= x_j, to see if we can make a logical implication

        if persistent.contains_key(&self.x_i) {
            // we have x_i defined so we can check if we can make an inference on x_j
            let x_i_value = *persistent.get(&self.x_i).unwrap();

            // if x_i is 0, then x_j must be 0
            if x_i_value == 0.0 {
                return Some((self.x_j, 0.0));
            }
        }

        if persistent.contains_key(&self.x_j) {
            // we have x_j defined, so we can check if we can make an inference on x_i
            let x_j_value = *persistent.get(&self.x_j).unwrap();

            // if x_i is 0, then x_j must be 0
            if x_j_value == 1.0 {
                return Some((self.x_i, 1.0));
            }
        }

        // if we can't make any inferences, then return None
        None
    }

    pub fn less_than_inference(&self, persistent: &HashMap<usize, f64>) -> Option<(usize, f64)> {
        // examines the constraint x_i <= x_j, to see if we can make a logical implication

        if persistent.contains_key(&self.x_i) {
            // we have x_i defined, so we can check if we can make an inference on x_j
            let x_i_value = *persistent.get(&self.x_i).unwrap();

            // if x_i is 0, then x_j must be 0
            if x_i_value == 1.0 {
                return Some((self.x_j, 1.0));
            }
        }

        if persistent.contains_key(&self.x_j) {
            // we have x_j defined, so we can check if we can make an inference on x_i
            let x_j_value = *persistent.get(&self.x_j).unwrap();

            // if x_i is 0, then x_j must be 0
            if x_j_value == 0.0 {
                return Some((self.x_i, 0.0));
            }
        }

        // if we can't make any inferences, then return None
        None
    }

    pub fn equal_inference(&self, free_var: usize, fixed_value: f64) -> Option<(usize, f64)> {
        Some((free_var, fixed_value))
    }
}

#[cfg(test)]
mod tests {
    use crate::constraint::{Constraint, ConstraintType};
    use std::collections::HashMap;

    #[test]
    fn test_constraints_10() {
        let mut persistent = HashMap::new();
        persistent.insert(0, 1.0);
        persistent.insert(1, 0.0);

        let c_at_least = Constraint::new(0, 1, ConstraintType::AtLeastOne);
        let c_no_more = Constraint::new(0, 1, ConstraintType::NoMoreThanOne);
        let c_exactly = Constraint::new(0, 1, ConstraintType::ExactlyOne);
        let c_greater = Constraint::new(0, 1, ConstraintType::GreaterThan);
        let c_less = Constraint::new(0, 1, ConstraintType::LessThan);
        let c_equal = Constraint::new(0, 1, ConstraintType::Equal);

        assert_eq!(c_at_least.check(&persistent), true);
        assert_eq!(c_no_more.check(&persistent), true);
        assert_eq!(c_exactly.check(&persistent), true);
        assert_eq!(c_greater.check(&persistent), true);
        assert_eq!(c_less.check(&persistent), false);
        assert_eq!(c_equal.check(&persistent), false);
    }

    #[test]
    fn test_constraints_11() {
        let mut persistent = HashMap::new();
        persistent.insert(0, 1.0);
        persistent.insert(1, 1.0);

        let c_at_least = Constraint::new(0, 1, ConstraintType::AtLeastOne);
        let c_no_more = Constraint::new(0, 1, ConstraintType::NoMoreThanOne);
        let c_exactly = Constraint::new(0, 1, ConstraintType::ExactlyOne);
        let c_greater = Constraint::new(0, 1, ConstraintType::GreaterThan);
        let c_less = Constraint::new(0, 1, ConstraintType::LessThan);
        let c_equal = Constraint::new(0, 1, ConstraintType::Equal);

        assert_eq!(c_at_least.check(&persistent), true);
        assert_eq!(c_no_more.check(&persistent), false);
        assert_eq!(c_exactly.check(&persistent), false);
        assert_eq!(c_greater.check(&persistent), true);
        assert_eq!(c_less.check(&persistent), true);
        assert_eq!(c_equal.check(&persistent), true);
    }

    #[test]
    fn test_constraints_01() {
        let mut persistent = HashMap::new();
        persistent.insert(0, 0.0);
        persistent.insert(1, 1.0);

        let c_at_least = Constraint::new(0, 1, ConstraintType::AtLeastOne);
        let c_no_more = Constraint::new(0, 1, ConstraintType::NoMoreThanOne);
        let c_exactly = Constraint::new(0, 1, ConstraintType::ExactlyOne);
        let c_greater = Constraint::new(0, 1, ConstraintType::GreaterThan);
        let c_less = Constraint::new(0, 1, ConstraintType::LessThan);
        let c_equal = Constraint::new(0, 1, ConstraintType::Equal);

        assert_eq!(c_at_least.check(&persistent), true);
        assert_eq!(c_no_more.check(&persistent), true);
        assert_eq!(c_exactly.check(&persistent), true);
        assert_eq!(c_greater.check(&persistent), false);
        assert_eq!(c_less.check(&persistent), true);
        assert_eq!(c_equal.check(&persistent), false);
    }
    #[test]
    fn test_constraints_00() {
        let mut persistent = HashMap::new();
        persistent.insert(0, 0.0);
        persistent.insert(1, 0.0);

        let c_at_least = Constraint::new(0, 1, ConstraintType::AtLeastOne);
        let c_no_more = Constraint::new(0, 1, ConstraintType::NoMoreThanOne);
        let c_exactly = Constraint::new(0, 1, ConstraintType::ExactlyOne);
        let c_greater = Constraint::new(0, 1, ConstraintType::GreaterThan);
        let c_less = Constraint::new(0, 1, ConstraintType::LessThan);
        let c_equal = Constraint::new(0, 1, ConstraintType::Equal);

        assert_eq!(c_at_least.check(&persistent), false);
        assert_eq!(c_no_more.check(&persistent), true);
        assert_eq!(c_exactly.check(&persistent), false);
        assert_eq!(c_greater.check(&persistent), true);
        assert_eq!(c_less.check(&persistent), true);
        assert_eq!(c_equal.check(&persistent), true);
    }
}
