pub trait BaseVariable {
    fn to_f64(&self) -> f64;
    fn to_usize(&self) -> usize;
}

impl BaseVariable for usize {
    fn to_f64(&self) -> f64 {
        *self as f64
    }

    fn to_usize(&self) -> usize {
        *self
    }
}
impl BaseVariable for f64 {
    fn to_f64(&self) -> f64 {
        *self
    }

    fn to_usize(&self) -> usize {
        *self as usize
    }
}
