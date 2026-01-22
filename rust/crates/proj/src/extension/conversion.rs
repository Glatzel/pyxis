use crate::Direction::{Fwd, Inv};
use crate::ICoord;
use crate::data_types::{ProjError, ProjErrorCode};

impl crate::Proj {
    /// Projects a single coordinate using the specified direction (forward or
    /// inverse).
    ///
    /// # Arguments
    ///
    /// * `inv`: If true, performs the inverse transformation; otherwise,
    ///   forward.
    /// * `coord`: The coordinate to transform.
    ///
    /// # Returns
    ///
    /// The transformed coordinate.
    pub fn project<T>(&self, inv: bool, coord: &T) -> Result<T, ProjError>
    where
        T: ICoord,
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
                self.trans_generic(direction, x, 1, 1, y, 1, 1, z, 0, 0, t, 0, 0)?
            },
            //3d
            (false, false, false, true) => unsafe {
                self.trans_generic(direction, x, 1, 1, y, 1, 1, z, 1, 1, t, 0, 0)?
            },
            //4d
            (false, false, false, false) => unsafe {
                self.trans_generic(direction, x, 1, 1, y, 1, 1, z, 1, 1, t, 1, 1)?
            },
            (x, y, z, t) => {
                return Err(ProjError::ProjError {
                    code: ProjErrorCode::Other,
                    message: format!(
                        "Input data is not correct.x.is_null: {x},t.is_null: {y},z.is_null: {z},t.is_null: {t}"
                    ),
                });
            }
        };
        Ok(coord)
    }
    /// Projects a single coordinate using the forward direction.
    ///
    /// # Arguments
    ///
    /// * `coord`: The coordinate to transform.
    ///
    /// # Returns
    ///
    /// The transformed coordinate.
    pub fn convert<T>(&self, coord: &T) -> Result<T, ProjError>
    where
        T: ICoord,
    {
        let mut coord = coord.clone();
        let x = coord.x();
        let y = coord.y();
        let z = coord.z();
        let t = coord.t();
        match (x.is_null(), y.is_null(), z.is_null(), t.is_null()) {
            //2d
            (false, false, true, true) => unsafe {
                self.trans_generic(Fwd, x, 1, 1, y, 1, 1, z, 0, 0, t, 0, 0)?
            },
            //3d
            (false, false, false, true) => unsafe {
                self.trans_generic(Fwd, x, 1, 1, y, 1, 1, z, 1, 1, t, 0, 0)?
            },
            //4d
            (false, false, false, false) => unsafe {
                self.trans_generic(Fwd, x, 1, 1, y, 1, 1, z, 1, 1, t, 1, 1)?
            },
            (x, y, z, t) => {
                return Err(ProjError::ProjError {
                    code: ProjErrorCode::Other,
                    message: format!(
                        "Input data is not correct.x.is_null: {x},t.is_null: {y},z.is_null: {z},t.is_null: {t}"
                    ),
                });
            }
        };

        Ok(coord)
    }
}

impl crate::Proj {
    /// Projects an array of coordinates using the specified direction (forward
    /// or inverse).
    ///
    /// # Arguments
    ///
    /// * `inv`: If true, performs the inverse transformation; otherwise,
    ///   forward.
    /// * `coord` - The mutable slice of coordinates to transform in-place.
    ///
    /// # Returns
    ///
    /// A reference to self for chaining.
    pub fn project_array<T>(&self, inv: bool, coord: &mut [T]) -> Result<&Self, ProjError>
    where
        T: ICoord,
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
                )?
            },
            //3d
            (false, false, false, true) => unsafe {
                self.trans_generic(
                    direction, x, size, length, y, size, length, z, size, length, t, 0, 0,
                )?
            },
            //4d
            (false, false, false, false) => unsafe {
                self.trans_generic(
                    direction, x, size, length, y, size, length, z, size, length, t, size, length,
                )?
            },
            (x, y, z, t) => {
                return Err(ProjError::ProjError {
                    code: ProjErrorCode::Other,
                    message: format!(
                        "Input data is not correct.x.is_null: {x},t.is_null: {y},z.is_null: {z},t.is_null: {t}"
                    ),
                });
            }
        };
        Ok(self)
    }
    /// Projects an array of coordinates using the forward direction.
    ///
    /// # Arguments
    ///
    /// * `coord`:The mutable slice of coordinates to transform in-place.
    ///
    /// # Returns
    ///
    /// A reference to self for chaining.
    pub fn convert_array<T>(&self, coord: &mut [T]) -> Result<&Self, ProjError>
    where
        T: ICoord,
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
                self.trans_generic(Fwd, x, size, length, y, size, length, z, 0, 0, t, 0, 0)?
            },

            //3d
            (false, false, false, true) => unsafe {
                self.trans_generic(
                    Fwd, x, size, length, y, size, length, z, size, length, t, 0, 0,
                )?
            },
            //4d
            (false, false, false, false) => unsafe {
                self.trans_generic(
                    Fwd, x, size, length, y, size, length, z, size, length, t, size, length,
                )?
            },
            (x, y, z, t) => {
                return Err(ProjError::ProjError {
                    code: crate::data_types::ProjErrorCode::Other,
                    message: format!(
                        "Input data is not correct.x.is_null: {x},t.is_null: {y},z.is_null: {z},t.is_null: {t}"
                    ),
                });
            }
        };

        Ok(self)
    }
}
#[cfg(test)]
mod test {
    use float_cmp::assert_approx_eq;
    #[test]
    fn test_project_2d() -> mischief::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_crs_to_crs("EPSG:4326", "EPSG:4496", &crate::Area::default())?;
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
    fn test_project_3d() -> mischief::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_crs_to_crs("EPSG:4326", "EPSG:4978", &crate::Area::default())?;
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
    fn test_project_4d() -> mischief::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_crs_to_crs("EPSG:8774", "EPSG:7789", &crate::Area::default())?;
        // array
        {
            let coord = [3879000.0, 1160000.0, 5000000.0, 2024.0];
            let coord = pj.project(false, &coord)?;
            println!("{coord:?}");
            assert_approx_eq!(f64, coord[0], 1935504.4269929447);
            assert_approx_eq!(f64, coord[1], -5521772.777432425);
            assert_approx_eq!(f64, coord[2], 5296676.095833973);
            assert_approx_eq!(f64, coord[3], 2024.0);
        }
        // tuple
        {
            let coord = (3879000.0, 1160000.0, 5000000.0, 2024.0);
            let coord = pj.project(false, &coord)?;
            println!("{coord:?}");
            assert_approx_eq!(f64, coord.0, 1935504.4269929447);
            assert_approx_eq!(f64, coord.1, -5521772.777432425);
            assert_approx_eq!(f64, coord.2, 5296676.095833973);
            assert_approx_eq!(f64, coord.3, 2024.0);
        }
        Ok(())
    }
    #[test]
    fn test_convert_2d() -> mischief::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_crs_to_crs("EPSG:4326", "EPSG:4496", &crate::Area::default())?;
        // array
        {
            let pj = ctx.normalize_for_visualization(&pj)?;
            let coord = [120.0, 30.0];
            let coord = pj.convert(&coord)?;
            println!("{coord:?}");
            assert_approx_eq!(f64, coord[0], 19955590.73888901);
            assert_approx_eq!(f64, coord[1], 3416780.562127255);
        }
        // tuple
        {
            let pj = ctx.normalize_for_visualization(&pj)?;
            let coord = (120.0, 30.0);
            let coord = pj.convert(&coord)?;
            println!("{coord:?}");
            assert_approx_eq!(f64, coord.0, 19955590.73888901);
            assert_approx_eq!(f64, coord.1, 3416780.562127255);
        }
        Ok(())
    }
    #[test]
    fn test_convert_3d() -> mischief::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_crs_to_crs("EPSG:4326", "EPSG:4978", &crate::Area::default())?;
        let pj = ctx.normalize_for_visualization(&pj)?;
        // array
        {
            let coord = [120.0, 30.0, 10.0];
            let coord = pj.convert(&coord)?;
            println!("{coord:?}");
            assert_approx_eq!(f64, coord[0], -2764132.649773435);
            assert_approx_eq!(f64, coord[1], 4787618.188267582);
            assert_approx_eq!(f64, coord[2], 3170378.735383637);
        }
        // tuple
        {
            let coord = (120.0, 30.0, 10.0);
            let coord = pj.convert(&coord)?;
            println!("{coord:?}");
            assert_approx_eq!(f64, coord.0, -2764132.649773435);
            assert_approx_eq!(f64, coord.1, 4787618.188267582);
            assert_approx_eq!(f64, coord.2, 3170378.735383637);
        }
        Ok(())
    }
    #[test]
    fn test_convert_4d() -> mischief::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_crs_to_crs("EPSG:8774", "EPSG:7789", &crate::Area::default())?;
        // array
        {
            let coord = [3879000.0, 1160000.0, 5000000.0, 2024.0];
            let coord = pj.convert(&coord)?;
            println!("{coord:?}");
            assert_approx_eq!(f64, coord[0], 1935504.4269929447);
            assert_approx_eq!(f64, coord[1], -5521772.777432425);
            assert_approx_eq!(f64, coord[2], 5296676.095833973);
            assert_approx_eq!(f64, coord[3], 2024.0);
        }
        // tuple
        {
            let coord = (3879000.0, 1160000.0, 5000000.0, 2024.0);
            let coord = pj.convert(&coord)?;
            println!("{coord:?}");
            assert_approx_eq!(f64, coord.0, 1935504.4269929447);
            assert_approx_eq!(f64, coord.1, -5521772.777432425);
            assert_approx_eq!(f64, coord.2, 5296676.095833973);
            assert_approx_eq!(f64, coord.3, 2024.0);
        }
        Ok(())
    }
    #[test]
    fn test_project_array_2d() -> mischief::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_crs_to_crs("EPSG:4326", "EPSG:4496", &crate::Area::default())?;
        let pj = ctx.normalize_for_visualization(&pj)?;
        let mut coord = [[120.0, 30.0], [50.0, -80.0]];

        pj.project_array(false, coord.as_mut_slice())?;
        println!("{coord:?}");
        assert_approx_eq!(f64, coord[0][0], 19955590.73888901);
        assert_approx_eq!(f64, coord[0][1], 3416780.562127255);
        assert_approx_eq!(f64, coord[1][0], 17583572.872089125);
        assert_approx_eq!(f64, coord[1][1], -9356989.97994042);
        Ok(())
    }
    #[test]
    fn test_project_array_3d() -> mischief::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_crs_to_crs("EPSG:4326", "EPSG:4978", &crate::Area::default())?;
        let pj = ctx.normalize_for_visualization(&pj)?;
        let mut coord = [[120.0, 30.0, 10.0], [50.0, -80.0, 0.0]];

        pj.project_array(false, coord.as_mut_slice())?;
        println!("{coord:?}");
        assert_approx_eq!(f64, coord[0][0], -2764132.649773435);
        assert_approx_eq!(f64, coord[0][1], 4787618.188267582);
        assert_approx_eq!(f64, coord[0][2], 3170378.735383637);
        assert_approx_eq!(f64, coord[1][0], 714243.0112756203);
        assert_approx_eq!(f64, coord[1][1], 851201.6746730272);
        assert_approx_eq!(f64, coord[1][2], -6259542.96102869);

        Ok(())
    }
    #[test]
    fn test_project_array_4d() -> mischief::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_crs_to_crs("EPSG:8774", "EPSG:7789", &crate::Area::default())?;
        let mut coord = [
            [3879000.0, 1160000.0, 5000000.0, 2024.0],
            [3879000.0, 1160000.0, 5000000.0, 2024.0],
        ];

        pj.project_array(false, coord.as_mut_slice())?;
        println!("{coord:?}");
        assert_approx_eq!(f64, coord[0][0], 1935504.4269929447);
        assert_approx_eq!(f64, coord[0][1], -5521772.777432425);
        assert_approx_eq!(f64, coord[0][2], 5296676.095833973);

        Ok(())
    }
    #[test]
    fn test_convert_array_2d() -> mischief::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_crs_to_crs("EPSG:4326", "EPSG:4496", &crate::Area::default())?;
        let mut coord = [[120.0, 30.0], [50.0, -80.0]];
        let pj = ctx.normalize_for_visualization(&pj)?;
        pj.convert_array(coord.as_mut_slice())?;
        println!("{coord:?}");
        assert_approx_eq!(f64, coord[0][0], 19955590.73888901);
        assert_approx_eq!(f64, coord[0][1], 3416780.562127255);
        assert_approx_eq!(f64, coord[1][0], 17583572.872089125);
        assert_approx_eq!(f64, coord[1][1], -9356989.97994042);
        Ok(())
    }
    #[test]
    fn test_convert_array_3d() -> mischief::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_crs_to_crs("EPSG:4326", "EPSG:4978", &crate::Area::default())?;
        let pj = ctx.normalize_for_visualization(&pj)?;
        let mut coord = [[120.0, 30.0, 10.0], [50.0, -80.0, 0.0]];

        pj.convert_array(coord.as_mut_slice())?;
        println!("{coord:?}");
        assert_approx_eq!(f64, coord[0][0], -2764132.649773435);
        assert_approx_eq!(f64, coord[0][1], 4787618.188267582);
        assert_approx_eq!(f64, coord[0][2], 3170378.735383637);
        assert_approx_eq!(f64, coord[1][0], 714243.0112756203);
        assert_approx_eq!(f64, coord[1][1], 851201.6746730272);
        assert_approx_eq!(f64, coord[1][2], -6259542.96102869);
        Ok(())
    }
    #[test]
    fn test_convert_array_4d() -> mischief::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_crs_to_crs("EPSG:8774", "EPSG:7789", &crate::Area::default())?;
        let mut coord = [
            [3879000.0, 1160000.0, 5000000.0, 2024.0],
            [3879000.0, 1160000.0, 5000000.0, 2024.0],
        ];

        pj.convert_array(coord.as_mut_slice())?;
        println!("{coord:?}");
        assert_approx_eq!(f64, coord[0][0], 1935504.4269929447);
        assert_approx_eq!(f64, coord[0][1], -5521772.777432425);
        assert_approx_eq!(f64, coord[0][2], 5296676.095833973);

        Ok(())
    }
}
