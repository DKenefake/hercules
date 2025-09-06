use std::collections::HashMap;

/// Enum for the type of constraint that is being used in the Constraint struct
pub enum ConstraintType {
    AtLeastOne,
    ExactlyOne,
    NoMoreThanOne,
    GreaterThan,
    LessThan,
    Equal,
}

/// The Constraint struct, that is for storing constraint information from the preprocessor
pub struct Constraint {
    pub(crate) x_i: usize,
    pub(crate) x_j: usize,
    constr_type: ConstraintType,
}

impl Constraint {
    /// Creates a new constraint with the given indices and type
    pub const fn new(x_i: usize, x_j: usize, constraint_type: ConstraintType) -> Self {
        Self {
            x_i,
            x_j,
            constr_type: constraint_type,
        }
    }

    /// Checks if the persistent variables are consistent with the constraint
    pub fn check(&self, persistent: &HashMap<usize, usize>) -> bool {
        // can only be computed if both variables are fixed
        if self.how_many_fixed(persistent) != 2 {
            return true;
        }

        // this unwrapping is safe, as we have already checked that both variables are fixed
        let x_i_value = *persistent.get(&self.x_i).unwrap();
        let x_j_value = *persistent.get(&self.x_j).unwrap();

        // check if we abide by the constraint
        match self.constr_type {
            ConstraintType::NoMoreThanOne => Self::no_more_than_one(x_i_value, x_j_value),
            ConstraintType::AtLeastOne => Self::at_least_one(x_i_value, x_j_value),
            ConstraintType::ExactlyOne => Self::exactly_one(x_i_value, x_j_value),
            ConstraintType::GreaterThan => Self::greater_than(x_i_value, x_j_value),
            ConstraintType::LessThan => Self::less_than(x_i_value, x_j_value),
            ConstraintType::Equal => Self::equal(x_i_value, x_j_value),
        }
    }

    /// Given a set of persistent variables, computes of an inference can be made and if so
    /// returns the index and value of the fixed variable
    pub fn make_inference(&self, persistent: &HashMap<usize, usize>) -> Option<(usize, usize)> {
        // count how many fixed variables we have
        let num_fixed = self.how_many_fixed(persistent);

        // if both are fixed or free, then we can't make any inferences
        if num_fixed == 2 || num_fixed == 0 {
            return None;
        }

        // give we have one fixed, then we can always make the standard form
        let (_, free, fixed_value) = self.get_standard_form(persistent).unwrap();

        // if we have only one fixed, then we can make an inference
        match self.constr_type {
            ConstraintType::NoMoreThanOne => Self::no_more_then_one_inference(free, fixed_value),
            ConstraintType::ExactlyOne => Self::exactly_one_inference(free, fixed_value),
            ConstraintType::AtLeastOne => Self::at_least_one_inference(free, fixed_value),
            ConstraintType::GreaterThan => self.greater_than_inference(persistent),
            ConstraintType::LessThan => self.less_than_inference(persistent),
            ConstraintType::Equal => Self::equal_inference(free, fixed_value),
        }
    }

    pub const fn no_more_than_one(x_i_val: usize, x_j_val: usize) -> bool {
        x_i_val + x_j_val <= 1
    }

    pub const fn at_least_one(x_i_val: usize, x_j_val: usize) -> bool {
        x_i_val + x_j_val >= 1
    }

    pub const fn exactly_one(x_i_val: usize, x_j_val: usize) -> bool {
        x_i_val + x_j_val == 1
    }

    pub const fn greater_than(x_i_val: usize, x_j_val: usize) -> bool {
        x_i_val >= x_j_val
    }

    pub const fn less_than(x_i_val: usize, x_j_val: usize) -> bool {
        x_i_val <= x_j_val
    }

    pub const fn equal(x_i_val: usize, x_j_val: usize) -> bool {
        x_i_val == x_j_val
    }

    pub fn how_many_fixed(&self, persistent: &HashMap<usize, usize>) -> usize {
        let mut count = 0;
        if Self::is_fixed(persistent, self.x_i) {
            count += 1;
        }
        if Self::is_fixed(persistent, self.x_j) {
            count += 1;
        }
        count
    }

    pub fn is_fixed(persistent: &HashMap<usize, usize>, index: usize) -> bool {
        persistent.contains_key(&index)
    }

    pub fn get_standard_form(
        &self,
        persistent: &HashMap<usize, usize>,
    ) -> Option<(usize, usize, usize)> {
        if self.how_many_fixed(persistent) != 1 {
            return None;
        }

        #[allow(clippy::option_if_let_else)]
        match persistent.get(&self.x_i) {
            Some(x_i_value) => Some((self.x_i, self.x_j, *x_i_value)),
            None => Some((self.x_j, self.x_i, *persistent.get(&self.x_j).unwrap())),
        }
    }

    /// Given a constraint of the type x_i + x_j <= 1, computes if we can make an inference on
    /// either x_i or x_j, given some fixed variables. If so, returns the index and value of the
    /// fixed value, otherwise returns None
    #[allow(clippy::unnecessary_wraps)]
    pub const fn no_more_then_one_inference(
        free_var: usize,
        fixed_value: usize,
    ) -> Option<(usize, usize)> {
        // if the fixed value is 1, then the free variable must be 0 else nothing can be said
        match fixed_value == 1 {
            true => Some((free_var, 0)),
            false => None,
        }
    }

    #[allow(clippy::unnecessary_wraps)]
    pub const fn exactly_one_inference(
        free_var: usize,
        fixed_value: usize,
    ) -> Option<(usize, usize)> {
        // if the fixed value is 1, then the free variable must be 0
        match fixed_value == 1 {
            true => Some((free_var, 0)),
            false => Some((free_var, 1)),
        }
    }

    pub const fn at_least_one_inference(
        free_var: usize,
        fixed_value: usize,
    ) -> Option<(usize, usize)> {
        // if the fixed value is 0, then the free variable must be 1
        match fixed_value {
            0 => Some((free_var, 1)),
            _ => None,
        }
    }

    pub fn greater_than_inference(
        &self,
        persistent: &HashMap<usize, usize>,
    ) -> Option<(usize, usize)> {
        // examines the constraint x_i >= x_j, to see if we can make a logical implication

        if let Some(x_i_value) = persistent.get(&self.x_i) {
            // if x_i is 0, then x_j must be 0
            if *x_i_value == 0 {
                return Some((self.x_j, 0));
            }
        }

        if let Some(x_j_value) = persistent.get(&self.x_j) {
            // if x_j is 1, then x_i must be 1
            if *x_j_value == 1 {
                return Some((self.x_i, 1));
            }
        }

        // if we can't make any inferences, then return None
        None
    }

    /// Given a constraint of the type x_i <= x_j, solves if we can make an inference on it
    pub fn less_than_inference(
        &self,
        persistent: &HashMap<usize, usize>,
    ) -> Option<(usize, usize)> {
        // examines the constraint x_i <= x_j, to see if we can make a logical implication

        if let Some(x_i_value) = persistent.get(&self.x_i) {
            // if x_i is 1, then x_j must be 1
            if *x_i_value == 1 {
                return Some((self.x_j, 1));
            }
        }

        if let Some(x_j_value) = persistent.get(&self.x_j) {
            // if x_j is 0, then x_i must be 0
            if *x_j_value == 0 {
                return Some((self.x_i, 0));
            }
        }

        // if we can't make any inferences, then return None
        None
    }

    /// Given a constraint of the type x_i = x_j, solves if we can make an inference on it
    ///
    /// In this case, we always can, as we can always set the free variable to the fixed value
    #[allow(clippy::unnecessary_wraps)]
    pub const fn equal_inference(free_var: usize, fixed_value: usize) -> Option<(usize, usize)> {
        Some((free_var, fixed_value))
    }
}

#[cfg(test)]
mod tests {
    use crate::constraint::{Constraint, ConstraintType};
    use std::collections::HashMap;

    #[test]
    fn test_constraints_10() {
        let mut persistent: HashMap<usize, usize> = HashMap::new();
        persistent.insert(0, 1);
        persistent.insert(1, 0);

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
        persistent.insert(0, 1);
        persistent.insert(1, 1);

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
        persistent.insert(0, 0);
        persistent.insert(1, 1);

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
        persistent.insert(0, 0);
        persistent.insert(1, 0);

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

    #[test]
    fn test_inference_0X() {
        let mut persistent = HashMap::new();
        persistent.insert(0, 0);

        let c_at_least = Constraint::new(0, 1, ConstraintType::AtLeastOne);
        let c_no_more = Constraint::new(0, 1, ConstraintType::NoMoreThanOne);
        let c_exactly = Constraint::new(0, 1, ConstraintType::ExactlyOne);
        let c_greater = Constraint::new(0, 1, ConstraintType::GreaterThan);
        let c_less = Constraint::new(0, 1, ConstraintType::LessThan);
        let c_equal = Constraint::new(0, 1, ConstraintType::Equal);

        assert_eq!(c_at_least.make_inference(&persistent), Some((1, 1)));
        assert_eq!(c_no_more.make_inference(&persistent), None);
        assert_eq!(c_exactly.make_inference(&persistent), Some((1, 1)));
        assert_eq!(c_greater.make_inference(&persistent), Some((1, 0)));
        assert_eq!(c_less.make_inference(&persistent), None);
        assert_eq!(c_equal.make_inference(&persistent), Some((1, 0)));
    }

    #[test]
    fn test_inference_1X() {
        let mut persistent = HashMap::new();
        persistent.insert(0, 1);

        let c_at_least = Constraint::new(0, 1, ConstraintType::AtLeastOne);
        let c_no_more = Constraint::new(0, 1, ConstraintType::NoMoreThanOne);
        let c_exactly = Constraint::new(0, 1, ConstraintType::ExactlyOne);
        let c_greater = Constraint::new(0, 1, ConstraintType::GreaterThan);
        let c_less = Constraint::new(0, 1, ConstraintType::LessThan);
        let c_equal = Constraint::new(0, 1, ConstraintType::Equal);

        assert_eq!(c_at_least.make_inference(&persistent), None);
        assert_eq!(c_no_more.make_inference(&persistent), Some((1, 0)));
        assert_eq!(c_exactly.make_inference(&persistent), Some((1, 0)));
        assert_eq!(c_greater.make_inference(&persistent), None);
        assert_eq!(c_less.make_inference(&persistent), Some((1, 1)));
        assert_eq!(c_equal.make_inference(&persistent), Some((1, 1)));
    }

    #[test]
    fn test_inference_X0() {
        let mut persistent = HashMap::new();
        persistent.insert(1, 0);

        let c_at_least = Constraint::new(0, 1, ConstraintType::AtLeastOne);
        let c_no_more = Constraint::new(0, 1, ConstraintType::NoMoreThanOne);
        let c_exactly = Constraint::new(0, 1, ConstraintType::ExactlyOne);
        let c_greater = Constraint::new(0, 1, ConstraintType::GreaterThan);
        let c_less = Constraint::new(0, 1, ConstraintType::LessThan);
        let c_equal = Constraint::new(0, 1, ConstraintType::Equal);

        assert_eq!(c_at_least.make_inference(&persistent), Some((0, 1)));
        assert_eq!(c_no_more.make_inference(&persistent), None);
        assert_eq!(c_exactly.make_inference(&persistent), Some((0, 1)));
        assert_eq!(c_greater.make_inference(&persistent), None);
        assert_eq!(c_less.make_inference(&persistent), Some((0, 0)));
        assert_eq!(c_equal.make_inference(&persistent), Some((0, 0)));
    }

    #[test]
    fn test_inference_X1() {
        let mut persistent = HashMap::new();
        persistent.insert(1, 1);

        let c_at_least = Constraint::new(0, 1, ConstraintType::AtLeastOne);
        let c_no_more = Constraint::new(0, 1, ConstraintType::NoMoreThanOne);
        let c_exactly = Constraint::new(0, 1, ConstraintType::ExactlyOne);
        let c_greater = Constraint::new(0, 1, ConstraintType::GreaterThan);
        let c_less = Constraint::new(0, 1, ConstraintType::LessThan);
        let c_equal = Constraint::new(0, 1, ConstraintType::Equal);

        assert_eq!(c_at_least.make_inference(&persistent), None);
        assert_eq!(c_no_more.make_inference(&persistent), Some((0, 0)));
        assert_eq!(c_exactly.make_inference(&persistent), Some((0, 0)));
        assert_eq!(c_greater.make_inference(&persistent), Some((0, 1)));
        assert_eq!(c_less.make_inference(&persistent), None);
        assert_eq!(c_equal.make_inference(&persistent), Some((0, 1)));
    }
}
