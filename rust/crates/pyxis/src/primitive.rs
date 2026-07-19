use core::fmt::{Debug, Display, LowerExp};

use num_traits::{ConstOne, Float, FloatConst, NumAssign};

pub trait GeoFloat: Float + FloatConst + ConstOne + NumAssign + Debug + LowerExp + Display {
    const TWO: Self;
}

macro_rules! num {
    ($value:expr) => {
        <T>::from($value).unwrap()
    };
}
pub(crate) use num;
impl GeoFloat for f32 {
    const TWO: Self = 2.0;
}

impl GeoFloat for f64 {
    const TWO: Self = 2.0;
}
