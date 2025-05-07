use std::ptr::null_mut;

use proj::PjParams::CrsToCrs;
use proj::{IPjCoord, PjArea};
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
    let ctx = proj::PjContext::default();
    let pj = ctx.create_proj(CrsToCrs {
        source_crs: "EPSG:4326",
        target_crs: "EPSG:4496",
        area: &PjArea::default(),
    })?;

    let pj = ctx.normalize_for_visualization(&pj)?;
    let coord = MyCoord { a: 120.0, b: 30.0 };
    let coord = pj.project(false, &coord)?;
    assert_eq!(coord.a, 19955590.73888901);
    assert_eq!(coord.b, 19955590.73888901);
    Ok(())
}
