use std::fmt::{Debug, Display};

use num_traits::{ConstOne, Float, FloatConst, FromPrimitive};
pub trait GeoFloat: Float + FromPrimitive + FloatConst + ConstOne + Debug + Display {
    const TWO: Self;
}

impl GeoFloat for f32 {
    const TWO: Self = 2.0;
}

impl GeoFloat for f64 {
    const TWO: Self = 2.0;
}
