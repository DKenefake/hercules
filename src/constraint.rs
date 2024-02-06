use std::collections::HashMap;

pub enum ConstraintType {
    AtLeastOne,
    ExactlyOne,
    NoMoreThanOne,
    GreaterThan,
    LessThan,
}

pub struct Constraint {
    x_i: usize,
    x_j: usize,
    constraint_type: ConstraintType,
}

impl Constraint {
    /// Creates a new constraint with the given indices and type
    pub fn new(x_i: usize, x_j: usize, constraint_type: ConstraintType) -> Constraint {
        Constraint {
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
        return match self.constraint_type {
            ConstraintType::NoMoreThanOne => self.no_more_than_one(persistent),
            ConstraintType::AtLeastOne => self.at_least_one(persistent),
            ConstraintType::ExactlyOne => self.exactly_one(persistent),
            ConstraintType::GreaterThan => self.greater_than(persistent),
            ConstraintType::LessThan => self.less_than(persistent),
        };
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
            ConstraintType::GreaterThan => None,
            ConstraintType::LessThan => None,
        }
    }

    pub fn no_more_than_one(&self, persistent: &HashMap<usize, f64>) -> bool {
        // If we have explicitly checked both keys to be fixed so this is safe
        let x_i_val = *persistent.get(&self.x_i).unwrap();
        let x_j_val = *persistent.get(&self.x_j).unwrap();

        // We can't have both
        return !(x_i_val == 1.0 && x_j_val == 1.0);
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

    pub fn how_many_fixed(&self, persistent: &HashMap<usize, f64>) -> usize {
        let mut count = 0;
        if persistent.contains_key(&self.x_i) {
            count += 1;
        }
        if persistent.contains_key(&self.x_j) {
            count += 1;
        }
        count
    }

    pub fn is_fixed(&self, persistent: &HashMap<usize, f64>, index: usize) -> bool {
        return persistent.contains_key(&index);
    }

    pub fn get_standard_form(
        &self,
        persistent: &HashMap<usize, f64>,
    ) -> Option<(usize, usize, f64)> {
        if self.how_many_fixed(persistent) != 1 {
            return None;
        }

        return match persistent.get(&self.x_i) {
            Some(x_i_value) => Some((self.x_i, self.x_j, *x_i_value)),
            None => Some((self.x_j, self.x_i, *persistent.get(&self.x_j).unwrap())),
        };
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
}
