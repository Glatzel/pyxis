use std::fmt::{Debug, Display, LowerExp};
use std::sync::LazyLock;

use num_traits::{ConstOne, Float, FloatConst, FromPrimitive};

use crate::Ellipsoid;
pub trait GeoFloat:
    Float + FromPrimitive + FloatConst + ConstOne + Debug + Display + LowerExp
{
    const TWO: Self;
}
#[macro_export]
macro_rules! num {
    ($value:expr) => {
        T::from($value).unwrap()
    };
}

impl GeoFloat for f32 {
    const TWO: Self = 2.0;
}

impl GeoFloat for f64 {
    const TWO: Self = 2.0;
}
pub trait ConstEllipsoid<T>
where
    T: GeoFloat,
{
    fn grs1980() -> LazyLock<Ellipsoid<T>>;
    fn krasovsky1940() -> LazyLock<Ellipsoid<T>>;
    fn wgs84() -> LazyLock<Ellipsoid<T>>;
}

impl ConstEllipsoid<f32> for f32 {
    fn grs1980() -> LazyLock<Ellipsoid<f32>> {
        LazyLock::new(|| Ellipsoid::from_semi_major_and_invf(6378137.0, 298.257222101))
    }
    fn krasovsky1940() -> LazyLock<Ellipsoid<f32>> {
        LazyLock::new(|| Ellipsoid::from_semi_major_and_invf(6378245.0, 298.3))
    }
    fn wgs84() -> LazyLock<Ellipsoid<f32>> {
        LazyLock::new(|| Ellipsoid::from_semi_major_and_invf(6378137.0, 298.257223563))
    }
}
impl ConstEllipsoid<f64> for f64 {
    fn grs1980() -> LazyLock<Ellipsoid<f64>> {
        LazyLock::new(|| Ellipsoid::from_semi_major_and_invf(6378137.0, 298.257222101))
    }
    fn krasovsky1940() -> LazyLock<Ellipsoid<f64>> {
        let e: LazyLock<Ellipsoid<f64>> =
            LazyLock::new(|| Ellipsoid::from_semi_major_and_invf(6378245.0, 298.3));
        e
    }
    fn wgs84() -> LazyLock<Ellipsoid<f64>> {
        LazyLock::new(|| Ellipsoid::from_semi_major_and_invf(6378137.0, 298.257223563))
    }
}
