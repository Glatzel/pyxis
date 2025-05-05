use super::IPjCoord;
use crate::PjDirection::{PjFwd, PjInv};

impl crate::Pj {
    pub fn project<T>(&self, inv: bool, coord: &T) -> miette::Result<T>
    where
        T: IPjCoord,
    {
        let direction = if inv { PjInv } else { PjFwd };
        let mut coord = coord.clone();
        let x = coord.pj_x();
        let y = coord.pj_y();
        let z = coord.pj_z();
        let t = coord.pj_t();
        match (x.is_null(), y.is_null(), z.is_null(), t.is_null()) {
            //2d
            (false, false, true, true) => {
                unsafe { self.trans_generic(direction, x, 1, 1, y, 1, 1, z, 0, 0, t, 0, 0) }
            }
            //3d
            (false, false, false, true) => {
                unsafe { self.trans_generic(direction, x, 1, 1, y, 1, 1, z, 1, 1, t, 0, 0) }
            }
            //4d
            (false, false, false, false) => {
                unsafe { self.trans_generic(direction, x, 1, 1, y, 1, 1, z, 1, 1, t, 1, 1) }
            }
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
        let x = coord.pj_x();
        let y = coord.pj_y();
        let z = coord.pj_z();
        let t = coord.pj_t();
        match (x.is_null(), y.is_null(), z.is_null(), t.is_null()) {
            //2d
            (false, false, true, true) => {
                unsafe { self.trans_generic(PjFwd, x, 1, 1, y, 1, 1, z, 0, 0, t, 0, 0) }
            }
            //3d
            (false, false, false, true) => {
                unsafe { self.trans_generic(PjFwd, x, 1, 1, y, 1, 1, z, 1, 1, t, 0, 0) }
            }
            //4d
            (false, false, false, false) => {
                unsafe { self.trans_generic(PjFwd, x, 1, 1, y, 1, 1, z, 1, 1, t, 1, 1) }
            }
            (x, y, z, t) => {
                miette::bail!(format!(
                    "Input data is not correct.x.is_null: {x},t.is_null: {y},z.is_null: {z},t.is_null: {t}"
                ))
            }
        }?;

        Ok(coord)
    }
}

impl crate::Pj {
    pub fn project_array<T>(&self, inv: bool, coord: &mut [T]) -> miette::Result<&Self>
    where
        T: IPjCoord,
    {
        let direction = if inv { PjInv } else { PjFwd };
        let length = coord.len();
        let size = size_of::<T>();
        let x = coord[0].pj_x();
        let y = coord[0].pj_y();
        let z = coord[0].pj_z();
        let t = coord[0].pj_t();

        match (x.is_null(), y.is_null(), z.is_null(), t.is_null()) {
            //2d
            (false, false, true, true) => unsafe { self.trans_generic(
                direction, x, size, length, y, size, length, z, 0, 0, t, 0, 0,
            ) },
            //3d
            (false, false, false, true) => unsafe { self.trans_generic(
                direction, x, size, length, y, size, length, z, size, length, t, 0, 0,
            ) },
            //4d
            (false, false, false, false) => unsafe { self.trans_generic(
                direction, x, size, length, y, size, length, z, size, length, t, size, length,
            ) },
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
        println!("{length},{size}");
        let x = coord[0].pj_x();
        let y = coord[0].pj_y();
        let z = coord[0].pj_z();
        let t = coord[0].pj_t();

        match (x.is_null(), y.is_null(), z.is_null(), t.is_null()) {
            //2d
            (false, false, true, true) => {
                unsafe { self.trans_generic(PjFwd, x, size, length, y, size, length, z, 0, 0, t, 0, 0) }
            }

            //3d
            (false, false, false, true) => unsafe { self.trans_generic(
                PjFwd, x, size, length, y, size, length, z, size, length, t, 0, 0,
            ) },
            //4d
            (false, false, false, false) => unsafe { self.trans_generic(
                PjFwd, x, size, length, y, size, length, z, size, length, t, size, length,
            ) },
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
    #[test]
    fn test_project_2d() -> miette::Result<()> {
        let ctx = crate::PjContext::default();
        let pj = ctx.create_crs_to_crs("EPSG:4326", "EPSG:4496", &crate::PjArea::default())?;
        // array
        {
            let pj = ctx.normalize_for_visualization(&pj)?;
            let coord = [120.0, 30.0];
            let coord = pj.project(false, &coord)?;
            assert_eq!(coord, [19955590.73888901, 3416780.562127255]);
        }
        // tuple
        {
            let pj = ctx.normalize_for_visualization(&pj)?;
            let coord = (120.0, 30.0);
            let coord = pj.project(false, &coord)?;
            assert_eq!(coord, (19955590.73888901, 3416780.562127255));
        }
        Ok(())
    }
    #[test]
    fn test_project_3d() -> miette::Result<()> {
        let ctx = crate::PjContext::default();
        let pj = ctx.create_crs_to_crs("EPSG:4326", "EPSG:4978", &crate::PjArea::default())?;
        let pj = ctx.normalize_for_visualization(&pj)?;
        // array
        {
            let coord = [120.0, 30.0, 10.0];
            let coord = pj.project(false, &coord)?;
            assert_eq!(
                coord,
                [-2764132.649773435, 4787618.188267582, 3170378.735383637]
            );
        }
        // tuple
        {
            let coord = (120.0, 30.0, 10.0);
            let coord = pj.project(false, &coord)?;
            assert_eq!(
                coord,
                (-2764132.649773435, 4787618.188267582, 3170378.735383637)
            );
        }
        Ok(())
    }
    #[test]
    fn test_convert_2d() -> miette::Result<()> {
        let ctx = crate::PjContext::default();
        let pj = ctx.create_crs_to_crs("EPSG:4326", "EPSG:4496", &crate::PjArea::default())?;
        // array
        {
            let pj = ctx.normalize_for_visualization(&pj)?;
            let coord = [120.0, 30.0];
            let coord = pj.convert(&coord)?;
            assert_eq!(coord, [19955590.73888901, 3416780.562127255]);
        }
        // tuple
        {
            let pj = ctx.normalize_for_visualization(&pj)?;
            let coord = (120.0, 30.0);

            let coord = pj.convert(&coord)?;
            assert_eq!(coord, (19955590.73888901, 3416780.562127255));
        }
        Ok(())
    }
    #[test]
    fn test_convert_3d() -> miette::Result<()> {
        let ctx = crate::PjContext::default();
        let pj = ctx.create_crs_to_crs("EPSG:4326", "EPSG:4978", &crate::PjArea::default())?;
        let pj = ctx.normalize_for_visualization(&pj)?;
        // array
        {
            let coord = [120.0, 30.0, 10.0];
            let coord = pj.convert(&coord)?;
            assert_eq!(
                coord,
                [-2764132.649773435, 4787618.188267582, 3170378.735383637]
            );
        }
        // tuple
        {
            let coord = (120.0, 30.0, 10.0);
            let coord = pj.convert(&coord)?;
            assert_eq!(
                coord,
                (-2764132.649773435, 4787618.188267582, 3170378.735383637)
            );
        }
        Ok(())
    }
    #[test]
    fn test_project_array_2d() -> miette::Result<()> {
        let ctx = crate::PjContext::default();
        let pj = ctx.create_crs_to_crs("EPSG:4326", "EPSG:4496", &crate::PjArea::default())?;
        let pj = ctx.normalize_for_visualization(&pj)?;
        let mut coord = [[120.0, 30.0], [50.0, -80.0]];

        pj.project_array(false, coord.as_mut_slice())?;
        assert_eq!(
            coord,
            [
                [19955590.73888901, 3416780.562127255],
                [17583572.872089125, -9356989.97994042]
            ]
        );
        Ok(())
    }
    #[test]
    fn test_project_array_3d() -> miette::Result<()> {
        let ctx = crate::PjContext::default();
        let pj = ctx.create_crs_to_crs("EPSG:4326", "EPSG:4978", &crate::PjArea::default())?;
        let pj = ctx.normalize_for_visualization(&pj)?;
        let mut coord = [[120.0, 30.0, 10.0], [50.0, -80.0, 0.0]];

        pj.project_array(false, coord.as_mut_slice())?;
        assert_eq!(
            coord,
            [
                [-2764132.649773435, 4787618.188267582, 3170378.735383637],
                [714243.0112756203, 851201.6746730272, -6259542.96102869]
            ]
        );
        Ok(())
    }
    #[test]
    fn test_convert_array_2d() -> miette::Result<()> {
        let ctx = crate::PjContext::default();
        let pj = ctx.create_crs_to_crs("EPSG:4326", "EPSG:4496", &crate::PjArea::default())?;
        let pj = ctx.normalize_for_visualization(&pj)?;
        let mut coord = [[120.0, 30.0], [50.0, -80.0]];

        pj.convert_array(coord.as_mut_slice())?;
        assert_eq!(
            coord,
            [
                [19955590.73888901, 3416780.562127255],
                [17583572.872089125, -9356989.97994042]
            ]
        );
        Ok(())
    }
    #[test]
    fn test_convert_array_3d() -> miette::Result<()> {
        let ctx = crate::PjContext::default();
        let pj = ctx.create_crs_to_crs("EPSG:4326", "EPSG:4978", &crate::PjArea::default())?;
        let pj = ctx.normalize_for_visualization(&pj)?;
        let mut coord = [[120.0, 30.0, 10.0], [50.0, -80.0, 0.0]];

        pj.convert_array(coord.as_mut_slice())?;
        assert_eq!(
            coord,
            [
                [-2764132.649773435, 4787618.188267582, 3170378.735383637],
                [714243.0112756203, 851201.6746730272, -6259542.96102869]
            ]
        );
        Ok(())
    }
}
