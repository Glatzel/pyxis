use std::ptr::null_mut;

use float_cmp::assert_approx_eq;
use proj::PjParams::CrsToCrs;
use proj::{Area, IPjCoord};
#[derive(Clone)]
struct MyCoord {
    a: f64,
    b: f64,
}
impl IPjCoord for MyCoord {
    fn x(&mut self) -> *mut f64 { &mut self.a }
    fn y(&mut self) -> *mut f64 { &mut self.b }
    fn z(&mut self) -> *mut f64 { null_mut::<f64>() }
    fn t(&mut self) -> *mut f64 { null_mut::<f64>() }
}

fn main() -> miette::Result<()> {
    convert_scalar()?;
    convert_array()?;
    Ok(())
}
fn convert_scalar() -> miette::Result<()> {
    let ctx = proj::Context::default();
    let pj = ctx.create_proj(CrsToCrs {
        source_crs: "EPSG:4326",
        target_crs: "EPSG:4496",
        area: &Area::default(),
    })?;

    let pj = ctx.normalize_for_visualization(&pj)?;
    let coord = MyCoord { a: 120.0, b: 30.0 };
    let coord = pj.convert(&coord)?;
    assert_approx_eq!(f64, coord.a, 19955590.73888901);
    assert_approx_eq!(f64, coord.b, 3416780.562127255);
    Ok(())
}
fn convert_array() -> miette::Result<()> {
    let ctx = proj::Context::default();
    let pj = ctx.create_proj(CrsToCrs {
        source_crs: "EPSG:4326",
        target_crs: "EPSG:4496",
        area: &Area::default(),
    })?;

    let pj = ctx.normalize_for_visualization(&pj)?;
    let mut coord = [MyCoord { a: 120.0, b: 30.0 }, MyCoord { a: 50.0, b: -80.0 }];
    pj.convert_array(&mut coord)?;
    assert_approx_eq!(f64, coord[0].a, 19955590.73888901);
    assert_approx_eq!(f64, coord[0].b, 3416780.562127255);
    Ok(())
}
