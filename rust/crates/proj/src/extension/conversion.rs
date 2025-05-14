use crate::IPjCoord;
use crate::PjDirection::{Fwd, Inv};

impl crate::Pj<'_> {
    pub fn project<T>(&self, inv: bool, coord: &T) -> miette::Result<T>
    where
        T: IPjCoord,
    {
        let direction = if inv { Inv } else { Fwd };
        let mut coord = coord.clone();
        let x = coord.x();
        let y = coord.y();
        let z = coord.z();
        let t = coord.t();
        match (x.is_null(), y.is_null(), z.is_null(), t.is_null()) {
            //2d
            (false, false, true, true) => unsafe {
                self.trans_generic(direction, x, 1, 1, y, 1, 1, z, 0, 0, t, 0, 0)
            },
            //3d
            (false, false, false, true) => unsafe {
                self.trans_generic(direction, x, 1, 1, y, 1, 1, z, 1, 1, t, 0, 0)
            },
            //4d
            (false, false, false, false) => unsafe {
                self.trans_generic(direction, x, 1, 1, y, 1, 1, z, 1, 1, t, 1, 1)
            },
            (x, y, z, t) => {
                miette::bail!(format!(
                    "Input data is not correct.x.is_null: {x},t.is_null: {y},z.is_null: {z},t.is_null: {t}"
                ))
            }
        }?;

        Ok(coord)
    }
    pub fn convert<T>(&self, coord: &T) -> miette::Result<T>
    where
        T: IPjCoord,
    {
        let mut coord = coord.clone();
        let x = coord.x();
        let y = coord.y();
        let z = coord.z();
        let t = coord.t();
        match (x.is_null(), y.is_null(), z.is_null(), t.is_null()) {
            //2d
            (false, false, true, true) => unsafe {
                self.trans_generic(Fwd, x, 1, 1, y, 1, 1, z, 0, 0, t, 0, 0)
            },
            //3d
            (false, false, false, true) => unsafe {
                self.trans_generic(Fwd, x, 1, 1, y, 1, 1, z, 1, 1, t, 0, 0)
            },
            //4d
            (false, false, false, false) => unsafe {
                self.trans_generic(Fwd, x, 1, 1, y, 1, 1, z, 1, 1, t, 1, 1)
            },
            (x, y, z, t) => {
                miette::bail!(format!(
                    "Input data is not correct.x.is_null: {x},t.is_null: {y},z.is_null: {z},t.is_null: {t}"
                ))
            }
        }?;

        Ok(coord)
    }
}

impl crate::Pj<'_> {
    pub fn project_array<T>(&self, inv: bool, coord: &mut [T]) -> miette::Result<&Self>
    where
        T: IPjCoord,
    {
        let direction = if inv { Inv } else { Fwd };
        let length = coord.len();
        let size = size_of::<T>();
        let x = coord[0].x();
        let y = coord[0].y();
        let z = coord[0].z();
        let t = coord[0].t();

        match (x.is_null(), y.is_null(), z.is_null(), t.is_null()) {
            //2d
            (false, false, true, true) => unsafe {
                self.trans_generic(
                    direction, x, size, length, y, size, length, z, 0, 0, t, 0, 0,
                )
            },
            //3d
            (false, false, false, true) => unsafe {
                self.trans_generic(
                    direction, x, size, length, y, size, length, z, size, length, t, 0, 0,
                )
            },
            //4d
            (false, false, false, false) => unsafe {
                self.trans_generic(
                    direction, x, size, length, y, size, length, z, size, length, t, size, length,
                )
            },
            (x, y, z, t) => {
                miette::bail!(format!(
                    "Input data is not correct.x.is_null: {x},t.is_null: {y},z.is_null: {z},t.is_null: {t}"
                ))
            }
        }?;
        Ok(self)
    }
    pub fn convert_array<T>(&self, coord: &mut [T]) -> miette::Result<&Self>
    where
        T: IPjCoord,
    {
        let length = coord.len();
        let size = size_of::<T>();
        let x = coord[0].x();
        let y = coord[0].y();
        let z = coord[0].z();
        let t = coord[0].t();

        match (x.is_null(), y.is_null(), z.is_null(), t.is_null()) {
            //2d
            (false, false, true, true) => unsafe {
                self.trans_generic(Fwd, x, size, length, y, size, length, z, 0, 0, t, 0, 0)
            },

            //3d
            (false, false, false, true) => unsafe {
                self.trans_generic(
                    Fwd, x, size, length, y, size, length, z, size, length, t, 0, 0,
                )
            },
            //4d
            (false, false, false, false) => unsafe {
                self.trans_generic(
                    Fwd, x, size, length, y, size, length, z, size, length, t, size, length,
                )
            },
            (x, y, z, t) => {
                miette::bail!(format!(
                    "Input data is not correct.x.is_null: {x},t.is_null: {y},z.is_null: {z},t.is_null: {t}"
                ))
            }
        }?;

        Ok(self)
    }
}
#[cfg(test)]
mod test {
    use float_cmp::assert_approx_eq;
    #[test]
    fn test_project_2d() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_crs_to_crs("EPSG:4326", "EPSG:4496", &crate::PjArea::default())?;
        // array
        {
            let pj = ctx.normalize_for_visualization(&pj)?;
            let coord = [120.0, 30.0];
            let coord = pj.project(false, &coord)?;
            assert_approx_eq!(f64, coord[0], 19955590.73888901);
            assert_approx_eq!(f64, coord[1], 3416780.562127255);
        }
        // tuple
        {
            let pj = ctx.normalize_for_visualization(&pj)?;
            let coord = (120.0, 30.0);
            let coord = pj.project(false, &coord)?;
            assert_approx_eq!(f64, coord.0, 19955590.73888901);
            assert_approx_eq!(f64, coord.1, 3416780.562127255);
        }
        Ok(())
    }
    #[test]
    fn test_project_3d() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_crs_to_crs("EPSG:4326", "EPSG:4978", &crate::PjArea::default())?;
        let pj = ctx.normalize_for_visualization(&pj)?;
        // array
        {
            let coord = [120.0, 30.0, 10.0];
            let coord = pj.project(false, &coord)?;
            assert_approx_eq!(f64, coord[0], -2764132.649773435);
            assert_approx_eq!(f64, coord[1], 4787618.188267582);
            assert_approx_eq!(f64, coord[2], 3170378.735383637);
        }
        // tuple
        {
            let coord = (120.0, 30.0, 10.0);
            let coord = pj.project(false, &coord)?;
            assert_approx_eq!(f64, coord.0, -2764132.649773435);
            assert_approx_eq!(f64, coord.1, 4787618.188267582);
            assert_approx_eq!(f64, coord.2, 3170378.735383637);
        }
        Ok(())
    }
    #[test]
    fn test_convert_2d() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_crs_to_crs("EPSG:4326", "EPSG:4496", &crate::PjArea::default())?;
        // array
        {
            let pj = ctx.normalize_for_visualization(&pj)?;
            let coord = [120.0, 30.0];
            let coord = pj.convert(&coord)?;
            assert_approx_eq!(f64, coord[0], 19955590.73888901);
            assert_approx_eq!(f64, coord[1], 3416780.562127255);
        }
        // tuple
        {
            let pj = ctx.normalize_for_visualization(&pj)?;
            let coord = (120.0, 30.0);
            let coord = pj.convert(&coord)?;
            assert_approx_eq!(f64, coord.0, 19955590.73888901);
            assert_approx_eq!(f64, coord.1, 3416780.562127255);
        }
        Ok(())
    }
    #[test]
    fn test_convert_3d() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_crs_to_crs("EPSG:4326", "EPSG:4978", &crate::PjArea::default())?;
        let pj = ctx.normalize_for_visualization(&pj)?;
        // array
        {
            let coord = [120.0, 30.0, 10.0];
            let coord = pj.convert(&coord)?;
            assert_approx_eq!(f64, coord[0], -2764132.649773435);
            assert_approx_eq!(f64, coord[1], 4787618.188267582);
            assert_approx_eq!(f64, coord[2], 3170378.735383637);
        }
        // tuple
        {
            let coord = (120.0, 30.0, 10.0);
            let coord = pj.convert(&coord)?;
            assert_approx_eq!(f64, coord.0, -2764132.649773435);
            assert_approx_eq!(f64, coord.1, 4787618.188267582);
            assert_approx_eq!(f64, coord.2, 3170378.735383637);
        }
        Ok(())
    }
    #[test]
    fn test_project_array_2d() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_crs_to_crs("EPSG:4326", "EPSG:4496", &crate::PjArea::default())?;
        let pj = ctx.normalize_for_visualization(&pj)?;
        let mut coord = [[120.0, 30.0], [50.0, -80.0]];

        pj.project_array(false, coord.as_mut_slice())?;
        assert_approx_eq!(f64, coord[0][0], 19955590.73888901);
        assert_approx_eq!(f64, coord[0][1], 3416780.562127255);
        assert_approx_eq!(f64, coord[1][0], 17583572.872089125);
        assert_approx_eq!(f64, coord[1][1], -9356989.97994042);
        Ok(())
    }
    #[test]
    fn test_project_array_3d() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_crs_to_crs("EPSG:4326", "EPSG:4978", &crate::PjArea::default())?;
        let pj = ctx.normalize_for_visualization(&pj)?;
        let mut coord = [[120.0, 30.0, 10.0], [50.0, -80.0, 0.0]];

        pj.project_array(false, coord.as_mut_slice())?;
        assert_approx_eq!(f64, coord[0][0], -2764132.649773435);
        assert_approx_eq!(f64, coord[0][1], 4787618.188267582);
        assert_approx_eq!(f64, coord[0][2], 3170378.735383637);
        assert_approx_eq!(f64, coord[1][0], 714243.0112756203);
        assert_approx_eq!(f64, coord[1][1], 851201.6746730272);
        assert_approx_eq!(f64, coord[1][2], -6259542.96102869);

        Ok(())
    }
    #[test]
    fn test_convert_array_2d() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_crs_to_crs("EPSG:4326", "EPSG:4496", &crate::PjArea::default())?;
        let pj = ctx.normalize_for_visualization(&pj)?;
        let mut coord = [[120.0, 30.0], [50.0, -80.0]];

        pj.convert_array(coord.as_mut_slice())?;
        assert_approx_eq!(f64, coord[0][0], 19955590.73888901);
        assert_approx_eq!(f64, coord[0][1], 3416780.562127255);
        assert_approx_eq!(f64, coord[1][0], 17583572.872089125);
        assert_approx_eq!(f64, coord[1][1], -9356989.97994042);
        Ok(())
    }
    #[test]
    fn test_convert_array_3d() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_crs_to_crs("EPSG:4326", "EPSG:4978", &crate::PjArea::default())?;
        let pj = ctx.normalize_for_visualization(&pj)?;
        let mut coord = [[120.0, 30.0, 10.0], [50.0, -80.0, 0.0]];

        pj.convert_array(coord.as_mut_slice())?;
        assert_approx_eq!(f64, coord[0][0], -2764132.649773435);
        assert_approx_eq!(f64, coord[0][1], 4787618.188267582);
        assert_approx_eq!(f64, coord[0][2], 3170378.735383637);
        assert_approx_eq!(f64, coord[1][0], 714243.0112756203);
        assert_approx_eq!(f64, coord[1][1], 851201.6746730272);
        assert_approx_eq!(f64, coord[1][2], -6259542.96102869);
        Ok(())
    }
}
