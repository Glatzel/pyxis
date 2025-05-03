use core::f64;
use std::ptr::null_mut;

use crate::PjDirection::PjFwd;

pub trait IPjCoord {
    fn components(&self) -> usize;
    fn pj_x(&self) -> f64;
    fn pj_y(&self) -> f64;
    fn pj_z(&self) -> f64;
    fn pj_t(&self) -> f64;
    fn from_pj_coord(x: f64, y: f64, z: f64, t: f64) -> Self;
}
impl IPjCoord for (f64, f64) {
    fn components(&self) -> usize {
        2
    }
    fn pj_x(&self) -> f64 {
        self.0
    }

    fn pj_y(&self) -> f64 {
        self.1
    }

    fn pj_z(&self) -> f64 {
        f64::default()
    }

    fn pj_t(&self) -> f64 {
        f64::default()
    }
    fn from_pj_coord(x: f64, y: f64, _z: f64, _t: f64) -> Self {
        (x, y)
    }
}
impl IPjCoord for [f64; 2] {
    fn components(&self) -> usize {
        2
    }
    fn pj_x(&self) -> f64 {
        self[0]
    }

    fn pj_y(&self) -> f64 {
        self[1]
    }

    fn pj_z(&self) -> f64 {
        f64::default()
    }

    fn pj_t(&self) -> f64 {
        f64::default()
    }

    fn from_pj_coord(x: f64, y: f64, _z: f64, _t: f64) -> Self {
        [x, y]
    }
}
impl IPjCoord for (f64, f64, f64) {
    fn components(&self) -> usize {
        3
    }
    fn pj_x(&self) -> f64 {
        self.0
    }

    fn pj_y(&self) -> f64 {
        self.1
    }

    fn pj_z(&self) -> f64 {
        self.2
    }

    fn pj_t(&self) -> f64 {
        f64::default()
    }

    fn from_pj_coord(x: f64, y: f64, z: f64, _t: f64) -> Self {
        (x, y, z)
    }
}
impl IPjCoord for [f64; 3] {
    fn components(&self) -> usize {
        3
    }
    fn pj_x(&self) -> f64 {
        self[0]
    }

    fn pj_y(&self) -> f64 {
        self[1]
    }

    fn pj_z(&self) -> f64 {
        self[2]
    }

    fn pj_t(&self) -> f64 {
        f64::default()
    }

    fn from_pj_coord(x: f64, y: f64, z: f64, _t: f64) -> Self {
        [x, y, z]
    }
}
impl IPjCoord for (f64, f64, f64, f64) {
    fn components(&self) -> usize {
        4
    }
    fn pj_x(&self) -> f64 {
        self.0
    }

    fn pj_y(&self) -> f64 {
        self.1
    }

    fn pj_z(&self) -> f64 {
        self.2
    }

    fn pj_t(&self) -> f64 {
        self.3
    }

    fn from_pj_coord(x: f64, y: f64, z: f64, t: f64) -> Self {
        (x, y, z, t)
    }
}
impl IPjCoord for [f64; 4] {
    fn components(&self) -> usize {
        4
    }
    fn pj_x(&self) -> f64 {
        self[0]
    }

    fn pj_y(&self) -> f64 {
        self[1]
    }

    fn pj_z(&self) -> f64 {
        self[2]
    }

    fn pj_t(&self) -> f64 {
        self[3]
    }

    fn from_pj_coord(x: f64, y: f64, z: f64, t: f64) -> Self {
        [x, y, z, t]
    }
}
impl crate::Pj {
    pub fn project<T>(&self, inv: bool, coord: T) -> miette::Result<T>
    where
        T: IPjCoord,
    {
        let direction = if inv {
            crate::PjDirection::PjInv
        } else {
            PjFwd
        };
        let mut x = coord.pj_x();
        let mut y = coord.pj_y();
        let mut z = coord.pj_z();
        let mut t = coord.pj_t();
        match coord.components() {
            2 => self.trans_generic(
                direction,
                &mut x,
                1,
                1,
                &mut y,
                1,
                1,
                null_mut::<f64>(),
                0,
                0,
                null_mut::<f64>(),
                0,
                0,
            ),
            3 => self.trans_generic(
                direction,
                &mut x,
                1,
                1,
                &mut y,
                1,
                1,
                &mut z,
                1,
                1,
                null_mut::<f64>(),
                0,
                0,
            ),
            4 => self.trans_generic(
                direction, &mut x, 1, 1, &mut y, 1, 1, &mut z, 1, 1, &mut t, 1, 1,
            ),
            other => miette::bail!(format!("Unknow compenent count: {}", other)),
        }?;

        Ok(T::from_pj_coord(x, y, z, t))
    }
    pub fn convert<T>(&self, coord: T) -> miette::Result<T>
    where
        T: IPjCoord,
    {
        let mut x = coord.pj_x();
        let mut y = coord.pj_y();
        let mut z = coord.pj_z();
        let mut t = coord.pj_t();
        match coord.components() {
            2 => self.trans_generic(
                PjFwd,
                &mut x,
                1,
                1,
                &mut y,
                1,
                1,
                null_mut::<f64>(),
                0,
                0,
                &mut t,
                1,
                1,
            ),
            3 => self.trans_generic(
                PjFwd,
                &mut x,
                1,
                1,
                &mut y,
                1,
                1,
                &mut z,
                1,
                1,
                null_mut::<f64>(),
                0,
                0,
            ),
            4 => self.trans_generic(
                PjFwd, &mut x, 1, 1, &mut y, 1, 1, &mut z, 1, 1, &mut t, 1, 1,
            ),
            other => miette::bail!(format!("Unknow compenent count: {}", other)),
        }?;

        Ok(T::from_pj_coord(x, y, z, t))
    }
}
pub trait IPjCoordArray {
    fn length(&mut self) -> usize;
    fn size(&mut self) -> usize;
    fn pj_x(&mut self) -> *mut f64;
    fn pj_y(&mut self) -> *mut f64;
    fn pj_z(&mut self) -> *mut f64;
    fn pj_t(&mut self) -> *mut f64;
}
impl IPjCoordArray for &mut [[f64; 2]] {
    fn length(&mut self) -> usize {
        self.len()
    }
    fn size(&mut self) -> usize {
        std::mem::size_of::<[f64; 2]>()
    }
    fn pj_x(&mut self) -> *mut f64 {
        &mut self[0][0] as *mut f64
    }

    fn pj_y(&mut self) -> *mut f64 {
        &mut self[0][1] as *mut f64
    }

    fn pj_z(&mut self) -> *mut f64 {
        null_mut::<f64>()
    }

    fn pj_t(&mut self) -> *mut f64 {
        null_mut::<f64>()
    }
}
impl IPjCoordArray for &mut [[f64; 3]] {
    fn length(&mut self) -> usize {
        self.len()
    }
    fn size(&mut self) -> usize {
        std::mem::size_of::<[f64; 3]>()
    }
    fn pj_x(&mut self) -> *mut f64 {
        &mut self[0][0] as *mut f64
    }

    fn pj_y(&mut self) -> *mut f64 {
        &mut self[0][1] as *mut f64
    }

    fn pj_z(&mut self) -> *mut f64 {
        &mut self[0][2] as *mut f64
    }

    fn pj_t(&mut self) -> *mut f64 {
        null_mut::<f64>()
    }
}
impl IPjCoordArray for &mut [[f64; 4]] {
    fn length(&mut self) -> usize {
        self.len()
    }
    fn size(&mut self) -> usize {
        std::mem::size_of::<[f64; 4]>()
    }
    fn pj_x(&mut self) -> *mut f64 {
        &mut self[0][0] as *mut f64
    }

    fn pj_y(&mut self) -> *mut f64 {
        &mut self[0][1] as *mut f64
    }

    fn pj_z(&mut self) -> *mut f64 {
        &mut self[0][2] as *mut f64
    }

    fn pj_t(&mut self) -> *mut f64 {
        &mut self[0][3] as *mut f64
    }
}
impl crate::Pj {
    pub fn project_array<T>(&self, inv: bool, coord: &mut T) -> miette::Result<&Self>
    where
        T: IPjCoordArray,
    {
        let direction = if inv {
            crate::PjDirection::PjInv
        } else {
            PjFwd
        };
        let length = coord.length();
        let size = coord.size();
        let x = coord.pj_x();
        let y = coord.pj_y();
        let z = coord.pj_z();
        let t = coord.pj_t();

        match (x.is_null(), y.is_null(), z.is_null(), t.is_null()) {
            //2d
            (false, false, true, true) => self.trans_generic(
                direction, x, size, length, y, size, length, z, 0, 0, t, 0, 0,
            ),
            //3d
            (false, false, false, true) => self.trans_generic(
                direction, x, size, length, y, size, length, z, size, length, t, 0, 0,
            ),
            //4d
            (false, false, false, false) => self.trans_generic(
                direction, x, size, length, y, size, length, z, size, length, t, size, length,
            ),
            (x, y, z, t) => {
                miette::bail!(format!(
                    "Input data is not correct.x.is_null: {x},t.is_null: {y},z.is_null: {z},t.is_null: {t}"
                ))
            }
        }?;
        Ok(self)
    }
    pub fn convert_array<T>(&self, coord: &mut T) -> miette::Result<&Self>
    where
        T: IPjCoordArray,
    {
        let length = coord.length();
        let size = coord.size();
        println!("{length},{size}");
        let x = coord.pj_x();
        let y = coord.pj_y();
        let z = coord.pj_z();
        let t = coord.pj_t();

        match (x.is_null(), y.is_null(), z.is_null(), t.is_null()) {
            //2d
            (false, false, true, true) => {
                self.trans_generic(PjFwd, x, size, length, y, size, length, z, 0, 0, t, 0, 0)
            }

            //3d
            (false, false, false, true) => self.trans_generic(
                PjFwd, x, size, length, y, size, length, z, size, length, t, 0, 0,
            ),
            //4d
            (false, false, false, false) => self.trans_generic(
                PjFwd, x, size, length, y, size, length, z, size, length, t, size, length,
            ),
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
            let coord = pj.project(false, coord)?;
            assert_eq!(coord, [19955590.73888901, 3416780.562127255]);
        }
        // tuple
        {
            let pj = ctx.normalize_for_visualization(&pj)?;
            let coord = (120.0, 30.0);
            let coord = pj.project(false, coord)?;
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
            let coord = pj.project(false, coord)?;
            assert_eq!(
                coord,
                [-2764132.649773435, 4787618.188267582, 3170378.735383637]
            );
        }
        // tuple
        {
            let coord = (120.0, 30.0, 10.0);
            let coord = pj.project(false, coord)?;
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
            let coord = pj.convert(coord)?;
            assert_eq!(coord, [19955590.73888901, 3416780.562127255]);
        }
        // tuple
        {
            let pj = ctx.normalize_for_visualization(&pj)?;
            let coord = (120.0, 30.0);

            let coord = pj.convert(coord)?;
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
            let coord = pj.convert(coord)?;
            assert_eq!(
                coord,
                [-2764132.649773435, 4787618.188267582, 3170378.735383637]
            );
        }
        // tuple
        {
            let coord = (120.0, 30.0, 10.0);
            let coord = pj.convert(coord)?;
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

        pj.project_array(false, &mut coord.as_mut_slice())?;
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

        pj.project_array(false, &mut coord.as_mut_slice())?;
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

        pj.convert_array(&mut coord.as_mut_slice())?;
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

        pj.convert_array(&mut coord.as_mut_slice())?;
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
