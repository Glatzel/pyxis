use proj::PjArea;
use proj::PjParams::CrsToCrs;

fn main() -> miette::Result<()> {
    cvt_2d()?;
    cvt_3d()?;
    cvt_3d_array()?;
    Ok(())
}
fn cvt_2d() -> miette::Result<()> {
    let ctx = proj::PjContext::default();
    let pj = ctx.create_proj(CrsToCrs {
        source_crs: "EPSG:4326",
        target_crs: "EPSG:4496",
        area: &PjArea::default(),
    })?;

    let pj = ctx.normalize_for_visualization(&pj)?;
    let coord = [120.0, 30.0];
    let coord = pj.convert(&coord)?;
    assert_eq!(coord, [19955590.73888901, 3416780.562127255]);
    Ok(())
}
fn cvt_3d() -> miette::Result<()> {
    let ctx = proj::PjContext::default();
    let pj = ctx.create_proj(CrsToCrs {
        source_crs: "EPSG:4326",
        target_crs: "EPSG:4978",
        area: &PjArea::default(),
    })?;
    let pj = ctx.normalize_for_visualization(&pj)?;

    let coord = [120.0, 30.0, 10.0];
    let coord = pj.convert(&coord)?;
    assert_eq!(
        coord,
        [-2764132.649773435, 4787618.188267582, 3170378.735383637]
    );
    Ok(())
}
fn cvt_3d_array() -> miette::Result<()> {
    let ctx = proj::PjContext::default();
    let pj = ctx.create_proj(CrsToCrs {
        source_crs: "EPSG:4326",
        target_crs: "EPSG:4978",
        area: &PjArea::default(),
    })?;
    let pj = ctx.normalize_for_visualization(&pj)?;

    let mut coord = [[120.0, 30.0, 10.0], [50.0, -80.0, 0.0]];
    pj.convert_array(&mut coord)?;
    assert_eq!(
        coord,
        [
            [-2764132.649773435, 4787618.188267582, 3170378.735383637],
            [714243.0112756203, 851201.6746730272, -6259542.96102869]
        ]
    );
    Ok(())
}
