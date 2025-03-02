use std::fmt::{Debug, Display, LowerExp};
use std::sync::LazyLock;

use num_traits::{ConstOne, Float, FloatConst, FromPrimitive};

use crate::Ellipsoid;
pub trait GeoFloat:
    Float + FromPrimitive + FloatConst + ConstOne + Debug + Display + LowerExp
{
    const TWO: Self;
}
#[macro_use]
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
    fn cgcs2000() -> &'static Ellipsoid<T>;
    fn iag1975() -> &'static Ellipsoid<T>;
    fn grs1980() -> &'static Ellipsoid<T>;
    fn krasovsky1940() -> &'static Ellipsoid<T>;
    fn wgs84() -> &'static Ellipsoid<T>;
}
//f32
static GRS1980_F32: LazyLock<Ellipsoid<f32>> =
    LazyLock::new(|| Ellipsoid::from_semi_major_and_invf(6378137.0, 298.257_23));
static IAG1975_F32: LazyLock<Ellipsoid<f32>> =
    LazyLock::new(|| Ellipsoid::from_semi_major_and_invf(6378140.0, 298.257));
static KRASOVSKY1940_F32: LazyLock<Ellipsoid<f32>> =
    LazyLock::new(|| Ellipsoid::from_semi_major_and_invf(6378245.0, 298.3));
static WGS84_F32: LazyLock<Ellipsoid<f32>> =
    LazyLock::new(|| Ellipsoid::from_semi_major_and_invf(6378137.0, 298.257_23));
impl ConstEllipsoid<f32> for f32 {
    fn cgcs2000() -> &'static Ellipsoid<f32> {
        &GRS1980_F32
    }
    fn iag1975() -> &'static Ellipsoid<f32> {
        &IAG1975_F32
    }
    fn grs1980() -> &'static Ellipsoid<f32> {
        &GRS1980_F32
    }
    fn krasovsky1940() -> &'static Ellipsoid<f32> {
        &KRASOVSKY1940_F32
    }
    fn wgs84() -> &'static Ellipsoid<f32> {
        &WGS84_F32
    }
}
//f64
static GRS1980_F64: LazyLock<Ellipsoid<f64>> =
    LazyLock::new(|| Ellipsoid::from_semi_major_and_invf(6378137.0, 298.257222101));
static IAG1975_F64: LazyLock<Ellipsoid<f64>> =
    LazyLock::new(|| Ellipsoid::from_semi_major_and_invf(6378140.0, 298.257));
static KRASOVSKY1940_F64: LazyLock<Ellipsoid<f64>> =
    LazyLock::new(|| Ellipsoid::from_semi_major_and_invf(6378245.0, 298.3));
static WGS84_F64: LazyLock<Ellipsoid<f64>> =
    LazyLock::new(|| Ellipsoid::from_semi_major_and_invf(6378137.0, 298.257223563));
impl ConstEllipsoid<f64> for f64 {
    fn cgcs2000() -> &'static Ellipsoid<f64> {
        &GRS1980_F64
    }
    fn iag1975() -> &'static Ellipsoid<f64> {
        &IAG1975_F64
    }
    fn grs1980() -> &'static Ellipsoid<f64> {
        &GRS1980_F64
    }
    fn krasovsky1940() -> &'static Ellipsoid<f64> {
        &KRASOVSKY1940_F64
    }
    fn wgs84() -> &'static Ellipsoid<f64> {
        &WGS84_F64
    }
}
