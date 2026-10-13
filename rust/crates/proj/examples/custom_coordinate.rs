use std::ptr::null_mut;

use float_cmp::assert_approx_eq;
use proj::{Area, ICoord, ICoords};
#[derive(Clone)]
struct MyCoord {
    a: f64,
    b: f64,
}
impl ICoord for MyCoord {
    fn x(&mut self) -> *mut f64 { &mut self.a }
    fn y(&mut self) -> *mut f64 { &mut self.b }
    fn z(&mut self) -> *mut f64 { null_mut::<f64>() }
    fn t(&mut self) -> *mut f64 { null_mut::<f64>() }
}
#[derive(Debug)]
struct MyCoordS {
    x: Vec<f64>,
    y: Vec<f64>,
}
impl ICoords for MyCoordS {
    fn x(&mut self) -> *mut f64 { self.x.as_mut_ptr() }
    fn y(&mut self) -> *mut f64 { self.y.as_mut_ptr() }
    fn z(&mut self) -> *mut f64 { null_mut::<f64>() }
    fn t(&mut self) -> *mut f64 { null_mut::<f64>() }
    fn x_len(&mut self) -> usize { self.x.len() }
    fn y_len(&mut self) -> usize { self.y.len() }
    fn z_len(&mut self) -> usize { 0 }
    fn t_len(&mut self) -> usize { 0 }
    fn x_step(&mut self) -> usize { size_of::<f64>() }
    fn y_step(&mut self) -> usize { size_of::<f64>() }
    fn z_step(&mut self) -> usize { 0 }
    fn t_step(&mut self) -> usize { 0 }
}
fn main() -> mischief::Result<()> {
    convert_scalar()?;
    convert_array()?;
    Ok(())
}
fn convert_scalar() -> mischief::Result<()> {
    let ctx = proj::Context::new()?;
    let pj = ctx
        .clone()
        .create_crs_to_crs("EPSG:4326", "EPSG:4496", &Area::default())?;

    let pj = ctx.normalize_for_visualization(&pj)?;
    let coord = MyCoord { a: 120.0, b: 30.0 };
    let coord = pj.convert(coord)?;
    assert_approx_eq!(f64, coord.a, 19955590.73888901);
    assert_approx_eq!(f64, coord.b, 3416780.562127255);
    Ok(())
}
fn convert_array() -> mischief::Result<()> {
    let ctx = proj::Context::new()?;
    let pj = ctx
        .clone()
        .create_crs_to_crs("EPSG:4326", "EPSG:4496", &Area::default())?;

    let pj = ctx.normalize_for_visualization(&pj)?;

    let coord = MyCoordS {
        x: vec![120.0, 50.0],
        y: vec![30.0, -80.0],
    };
    let coord = pj.convert_array(coord)?;
    assert_approx_eq!(f64, coord.x[0], 19955590.73888901);
    assert_approx_eq!(f64, coord.x[0], 3416780.562127255);
    Ok(())
}
#[test]
fn test_convert_scalar() -> mischief::Result<()> { convert_scalar() }
#[test]
fn test_convert_array() -> mischief::Result<()> { convert_array() }
