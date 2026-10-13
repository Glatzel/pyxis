use crate::Direction::{Fwd, Inv};
use crate::data_types::ProjError;
use crate::{ICoord, ICoords};

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
    pub fn project<T>(&self, inv: bool, mut coord: T) -> Result<T, ProjError>
    where
        T: ICoord,
    {
        let direction = if inv { Inv } else { Fwd };
        self.trans_generic(
            direction,
            coord.x(),
            1,
            1,
            coord.y(),
            1,
            1,
            coord.z(),
            1,
            1,
            coord.t(),
            1,
            1,
        )?;
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
    pub fn convert<T>(&self, coord: T) -> Result<T, ProjError>
    where
        T: ICoord,
    {
        self.project(false, coord)
    }

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
    pub fn project_array<T>(&self, inv: bool, mut coord: T) -> Result<T, ProjError>
    where
        T: ICoords,
    {
        let direction = if inv { Inv } else { Fwd };

        self.trans_generic(
            direction,
            coord.x(),
            coord.x_step(),
            coord.x_len(),
            coord.y(),
            coord.y_step(),
            coord.y_len(),
            coord.z(),
            coord.z_step(),
            coord.z_len(),
            coord.t(),
            coord.t_step(),
            coord.t_len(),
        )?;

        Ok(coord)
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
    pub fn convert_array<T>(&self, coord: T) -> Result<T, ProjError>
    where
        T: ICoords,
    {
        self.project_array(false, coord)
    }
}
#[cfg(test)]
mod test {
    use std::vec;

    use float_cmp::assert_approx_eq;

    use crate::ICoords;
    #[derive(Debug)]
    struct TestCoord {
        x: Vec<f64>,
        y: Vec<f64>,
        z: Vec<f64>,
        t: Vec<f64>,
    }
    impl ICoords for TestCoord {
        fn x(&mut self) -> *mut f64 { self.x.as_mut_ptr() }
        fn y(&mut self) -> *mut f64 { self.y.as_mut_ptr() }
        fn z(&mut self) -> *mut f64 { self.z.as_mut_ptr() }
        fn t(&mut self) -> *mut f64 { self.t.as_mut_ptr() }
        fn x_len(&mut self) -> usize { self.x.len() }
        fn y_len(&mut self) -> usize { self.y.len() }
        fn z_len(&mut self) -> usize { self.z.len() }
        fn t_len(&mut self) -> usize { self.t.len() }
        fn x_step(&mut self) -> usize { size_of::<f64>() }
        fn y_step(&mut self) -> usize { size_of::<f64>() }
        fn z_step(&mut self) -> usize { size_of::<f64>() }
        fn t_step(&mut self) -> usize { size_of::<f64>() }
    }

    #[test]
    fn test_convert_2d() -> mischief::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_crs_to_crs("EPSG:4326", "EPSG:4496", &crate::Area::default())?;
        // array
        {
            let pj = ctx.normalize_for_visualization(&pj)?;
            let coord = [120.0, 30.0];
            let coord = pj.convert(coord)?;
            println!("{coord:?}");
            assert_approx_eq!(f64, coord[0], 19955590.73888901);
            assert_approx_eq!(f64, coord[1], 3416780.562127255);
        }
        // tuple
        {
            let pj = ctx.normalize_for_visualization(&pj)?;
            let coord = (120.0, 30.0);
            let coord = pj.convert(coord)?;
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
            let coord = pj.convert(coord)?;
            println!("{coord:?}");
            assert_approx_eq!(f64, coord[0], -2764132.649773435);
            assert_approx_eq!(f64, coord[1], 4787618.188267582);
            assert_approx_eq!(f64, coord[2], 3170378.735383637);
        }
        // tuple
        {
            let coord = (120.0, 30.0, 10.0);
            let coord = pj.convert(coord)?;
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
            let coord = pj.convert(coord)?;
            println!("{coord:?}");
            assert_approx_eq!(f64, coord[0], 1935504.4269929447);
            assert_approx_eq!(f64, coord[1], -5521772.777432425);
            assert_approx_eq!(f64, coord[2], 5296676.095833973);
            assert_approx_eq!(f64, coord[3], 2024.0);
        }
        // tuple
        {
            let coord = (3879000.0, 1160000.0, 5000000.0, 2024.0);
            let coord = pj.convert(coord)?;
            println!("{coord:?}");
            assert_approx_eq!(f64, coord.0, 1935504.4269929447);
            assert_approx_eq!(f64, coord.1, -5521772.777432425);
            assert_approx_eq!(f64, coord.2, 5296676.095833973);
            assert_approx_eq!(f64, coord.3, 2024.0);
        }
        Ok(())
    }

    #[test]
    fn test_convert_array_2d() -> mischief::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_crs_to_crs("EPSG:4326", "EPSG:4496", &crate::Area::default())?;
        let coord = TestCoord {
            x: vec![120.0, 50.0],
            y: vec![30.0, -80.0],
            z: vec![],
            t: vec![],
        };
        let pj = ctx.normalize_for_visualization(&pj)?;
        let coord = pj.convert_array(coord)?;
        println!("{coord:?}");
        assert_approx_eq!(f64, coord.x[0], 19955590.73888901);
        assert_approx_eq!(f64, coord.y[0], 3416780.562127255);
        assert_approx_eq!(f64, coord.x[1], 17583572.872089125);
        assert_approx_eq!(f64, coord.y[1], -9356989.97994042);
        Ok(())
    }
    #[test]
    fn test_convert_array_3d() -> mischief::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_crs_to_crs("EPSG:4326", "EPSG:4978", &crate::Area::default())?;
        let pj = ctx.normalize_for_visualization(&pj)?;
        let coord = TestCoord {
            x: vec![120.0, 50.0],
            y: vec![30.0, -80.0],
            z: vec![10.0, 0.0],
            t: vec![],
        };

        let coord = pj.convert_array(coord)?;
        println!("{coord:?}");
        assert_approx_eq!(f64, coord.x[0], -2764132.649773435);
        assert_approx_eq!(f64, coord.y[0], 4787618.188267582);
        assert_approx_eq!(f64, coord.z[0], 3170378.735383637);
        assert_approx_eq!(f64, coord.x[1], 714243.0112756203);
        assert_approx_eq!(f64, coord.y[1], 851201.6746730272);
        assert_approx_eq!(f64, coord.z[1], -6259542.96102869);
        Ok(())
    }
    #[test]
    fn test_convert_array_4d() -> mischief::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_crs_to_crs("EPSG:8774", "EPSG:7789", &crate::Area::default())?;
        let coord = TestCoord {
            x: vec![3879000.0, 3879000.0],
            y: vec![1160000.0, 1160000.0],
            z: vec![5000000.0, 5000000.0],
            t: vec![2024.0, 2025.0],
        };

        let coord = pj.convert_array(coord)?;
        println!("{coord:?}");
        assert_approx_eq!(f64, coord.x[0], 1935504.4269929447);
        assert_approx_eq!(f64, coord.y[0], -5521772.777432425);
        assert_approx_eq!(f64, coord.z[0], 5296676.095833973);
        assert_approx_eq!(f64, coord.x[1], 1935504.4055178885);
        assert_approx_eq!(f64, coord.y[1], -5521772.779423727);
        assert_approx_eq!(f64, coord.z[1], 5296676.102978117);
        Ok(())
    }
}
