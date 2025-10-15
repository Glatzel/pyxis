use core::fmt::{Debug, Display, LowerExp};

use num_traits::{ConstOne, Float, FloatConst, NumAssign};

pub trait GeoFloat:
    Float
    + FloatConst
    + ConstOne
    + NumAssign
    + Debug
    + Display
    + LowerExp
    + GeoCast<f32>
    + GeoCast<f64>
    + GeoCast<Self>
    + PartialEq
{
    const TWO: Self;
}
pub trait GeoCast<Src>: Sized {
    fn from(src: Src) -> Self;
}

impl GeoCast<f32> for f32 {
    #[inline(always)]
    fn from(v: f32) -> Self { v }
}
impl GeoCast<f64> for f64 {
    #[inline(always)]
    fn from(v: f64) -> Self { v }
}

impl GeoCast<f32> for f64 {
    #[inline(always)]
    fn from(v: f32) -> Self { v as f64 }
}
impl GeoCast<f64> for f32 {
    #[inline(always)]
    fn from(v: f64) -> Self { v as f32 }
}

macro_rules! num {
    ($value:expr) => {
        <T as $crate::GeoCast<_>>::from($value)
    };
}
pub(crate) use num;
impl GeoFloat for f32 {
    const TWO: Self = 2.0;
}

impl GeoFloat for f64 {
    const TWO: Self = 2.0;
}
