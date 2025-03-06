use cust::prelude::*;

use crate::PyxisCudaContext;
const PTX: &str = include_str!("./crypto_cuda.ptx");

impl PyxisCudaContext {
    pub fn bd09_to_gcj02_cuda<'a>(&self, lon: &mut DeviceBuffer<f64>, lat: &mut DeviceBuffer<f64>) {
        assert_eq!(lon.len(), lat.len());
        let length: usize = lon.len();
        let module = Module::from_ptx(PTX, &[]).unwrap();
        let func = module.get_function("bd09_to_gcj02_cuda").unwrap();
        let stream = self.stream();
        let (grid_size, block_size) = self.get_grid_block(&func, length);

        unsafe {
            launch!(
                func<<<grid_size, block_size, 0, stream>>>(
                    lon.as_device_ptr(),
                    lat.as_device_ptr()
                )
            )
            .unwrap();
        }
        stream.synchronize().unwrap();
    }
    pub fn gcj02_to_bd09_cuda<'a>(&self, lon: &mut DeviceBuffer<f64>, lat: &mut DeviceBuffer<f64>) {
        assert_eq!(lon.len(), lat.len());
        let length: usize = lon.len();
        let module = Module::from_ptx(PTX, &[]).unwrap();
        let func = module.get_function("gcj02_to_bd09_cuda").unwrap();
        let stream = self.stream();
        let (grid_size, block_size) = self.get_grid_block(&func, length);

        unsafe {
            launch!(
                func<<<grid_size, block_size, 0, stream>>>(
                    lon.as_device_ptr(),
                    lat.as_device_ptr()
                )
            )
            .unwrap();
        }
        stream.synchronize().unwrap();
    }
    pub fn gcj02_to_wgs84_cuda<'a>(
        &self,
        lon: &mut DeviceBuffer<f64>,
        lat: &mut DeviceBuffer<f64>,
    ) {
        assert_eq!(lon.len(), lat.len());
        let length: usize = lon.len();
        let module = Module::from_ptx(PTX, &[]).unwrap();
        let func = module.get_function("gcj02_to_wgs84_cuda").unwrap();
        let stream = self.stream();
        let (grid_size, block_size) = self.get_grid_block(&func, length);

        unsafe {
            launch!(
                func<<<grid_size, block_size, 0, stream>>>(
                    lon.as_device_ptr(),
                    lat.as_device_ptr()
                )
            )
            .unwrap();
        }
        stream.synchronize().unwrap();
    }
    pub fn wgs84_to_gcj02_cuda<'a>(
        &self,
        lon: &mut DeviceBuffer<f64>,
        lat: &mut DeviceBuffer<f64>,
    ) {
        assert_eq!(lon.len(), lat.len());
        let length: usize = lon.len();
        let module = Module::from_ptx(PTX, &[]).unwrap();
        let func = module.get_function("wgs84_to_gcj02_cuda").unwrap();
        let stream = self.stream();
        let (grid_size, block_size) = self.get_grid_block(&func, length);

        unsafe {
            launch!(
                func<<<grid_size, block_size, 0, stream>>>(
                    lon.as_device_ptr(),
                    lat.as_device_ptr()
                )
            )
            .unwrap();
        }
        stream.synchronize().unwrap();
    }
    pub fn wgs84_to_bd09_cuda<'a>(&self, lon: &mut DeviceBuffer<f64>, lat: &mut DeviceBuffer<f64>) {
        assert_eq!(lon.len(), lat.len());
        let length: usize = lon.len();
        let module = Module::from_ptx(PTX, &[]).unwrap();
        let func = module.get_function("wgs84_to_bd09_cuda").unwrap();
        let stream = self.stream();
        let (grid_size, block_size) = self.get_grid_block(&func, length);

        unsafe {
            launch!(
                func<<<grid_size, block_size, 0, stream>>>(
                    lon.as_device_ptr(),
                    lat.as_device_ptr()
                )
            )
            .unwrap();
        }
        stream.synchronize().unwrap();
    }
    pub fn bd09_to_wgs84_cuda<'a>(&self, lon: &mut DeviceBuffer<f64>, lat: &mut DeviceBuffer<f64>) {
        assert_eq!(lon.len(), lat.len());
        let length: usize = lon.len();
        let module = Module::from_ptx(PTX, &[]).unwrap();
        let func = module.get_function("bd09_to_wgs84_cuda").unwrap();
        let stream = self.stream();
        let (grid_size, block_size) = self.get_grid_block(&func, length);

        unsafe {
            launch!(
                func<<<grid_size, block_size, 0, stream>>>(
                    lon.as_device_ptr(),
                    lat.as_device_ptr()
                )
            )
            .unwrap();
        }
        stream.synchronize().unwrap();
    }
}

#[cfg(test)]
mod test {
    use cust::memory::CopyDestination;
    use float_cmp::assert_approx_eq;
    #[test]
    fn test_bd09_to_gcj02_cuda() {
        let mut lon: Vec<f64> = vec![pyxis::crypto::BD09_LON, pyxis::crypto::BD09_LON];
        let mut lat: Vec<f64> = vec![pyxis::crypto::BD09_LAT, pyxis::crypto::BD09_LAT];
        let expect_gcj = pyxis::crypto::bd09_to_gcj02(lon[0], lat[0]);
        let ctx = crate::PyxisCudaContext::new();
        let mut dlon = ctx.from_slice(&lon);
        let mut dlat = ctx.from_slice(&lat);
        ctx.bd09_to_gcj02_cuda(&mut dlon, &mut dlat);
        dlon.copy_to(&mut lon).unwrap();
        dlat.copy_to(&mut lat).unwrap();

        println!("yc:{:?}", lon);
        println!("yc:{:?}", lat);

        assert_approx_eq!(f64, lon[0], expect_gcj.0, epsilon = 1e-14);
        assert_approx_eq!(f64, lat[0], expect_gcj.1, epsilon = 1e-14);
    }
    #[test]
    fn test_gcj02_to_bd09_cuda() {
        let mut lon: Vec<f64> = vec![pyxis::crypto::GCJ02_LON, pyxis::crypto::GCJ02_LON];
        let mut lat: Vec<f64> = vec![pyxis::crypto::GCJ02_LAT, pyxis::crypto::GCJ02_LAT];

        let ctx = crate::PyxisCudaContext::new();
        let mut dlon = ctx.from_slice(&lon);
        let mut dlat = ctx.from_slice(&lat);
        ctx.gcj02_to_bd09_cuda(&mut dlon, &mut dlat);
        dlon.copy_to(&mut lon).unwrap();
        dlat.copy_to(&mut lat).unwrap();

        println!("yc:{:?}", lon);
        println!("yc:{:?}", lat);
        assert_approx_eq!(f64, lon[0], pyxis::crypto::BD09_LON, epsilon = 1e-14);
        assert_approx_eq!(f64, lat[0], pyxis::crypto::BD09_LAT, epsilon = 1e-14);
    }
    #[test]
    fn test_gcj02_to_wgs84_cuda() {
        let mut lon: Vec<f64> = vec![pyxis::crypto::GCJ02_LON, pyxis::crypto::GCJ02_LON];
        let mut lat: Vec<f64> = vec![pyxis::crypto::GCJ02_LAT, pyxis::crypto::GCJ02_LAT];
        let expect_wgs = pyxis::crypto::gcj02_to_wgs84(lon[0], lat[0]);
        let ctx = crate::PyxisCudaContext::new();
        let mut dlon = ctx.from_slice(&lon);
        let mut dlat = ctx.from_slice(&lat);
        ctx.gcj02_to_wgs84_cuda(&mut dlon, &mut dlat);
        dlon.copy_to(&mut lon).unwrap();
        dlat.copy_to(&mut lat).unwrap();

        println!("yc:{:?}", lon);
        println!("yc:{:?}", lat);

        assert_approx_eq!(f64, lon[0], expect_wgs.0, epsilon = 1e-17);
        assert_approx_eq!(f64, lat[0], expect_wgs.1, epsilon = 1e-17);
    }
    #[test]
    fn test_wgs84_to_gcj02_cuda() {
        let mut lon: Vec<f64> = vec![pyxis::crypto::WGS84_LON, pyxis::crypto::WGS84_LON];
        let mut lat: Vec<f64> = vec![pyxis::crypto::WGS84_LAT, pyxis::crypto::WGS84_LAT];
        let ctx = crate::PyxisCudaContext::new();
        let mut dlon = ctx.from_slice(&lon);
        let mut dlat = ctx.from_slice(&lat);
        ctx.wgs84_to_gcj02_cuda(&mut dlon, &mut dlat);
        dlon.copy_to(&mut lon).unwrap();
        dlat.copy_to(&mut lat).unwrap();

        println!("lon:{:?}", lon);
        println!("lat:{:?}", lat);
        assert_approx_eq!(f64, lon[0], pyxis::crypto::GCJ02_LON, epsilon = 1e-17);
        assert_approx_eq!(f64, lat[0], pyxis::crypto::GCJ02_LAT, epsilon = 1e-17);
    }
    #[test]
    fn test_wgs84_to_bd09_cuda() {
        let mut lon: Vec<f64> = vec![pyxis::crypto::WGS84_LON, pyxis::crypto::WGS84_LON];
        let mut lat: Vec<f64> = vec![pyxis::crypto::WGS84_LAT, pyxis::crypto::WGS84_LAT];
        let ctx = crate::PyxisCudaContext::new();
        let mut dlon = ctx.from_slice(&lon);
        let mut dlat = ctx.from_slice(&lat);
        ctx.wgs84_to_bd09_cuda(&mut dlon, &mut dlat);
        dlon.copy_to(&mut lon).unwrap();
        dlat.copy_to(&mut lat).unwrap();

        println!("lon:{:?}", lon);
        println!("lat:{:?}", lat);
        assert_approx_eq!(f64, lon[0], pyxis::crypto::BD09_LON, epsilon = 1e-14);
        assert_approx_eq!(f64, lat[0], pyxis::crypto::BD09_LAT, epsilon = 1e-14);
    }
    #[test]
    fn test_bd09_to_wgs84_cuda() {
        let mut lon: Vec<f64> = vec![pyxis::crypto::BD09_LON, pyxis::crypto::BD09_LON];
        let mut lat: Vec<f64> = vec![pyxis::crypto::BD09_LAT, pyxis::crypto::BD09_LAT];
        let expect_wgs = pyxis::crypto::bd09_to_wgs84(lon[0], lat[0]);
        let ctx = crate::PyxisCudaContext::new();
        let mut dlon = ctx.from_slice(&lon);
        let mut dlat = ctx.from_slice(&lat);
        ctx.bd09_to_wgs84_cuda(&mut dlon, &mut dlat);
        dlon.copy_to(&mut lon).unwrap();
        dlat.copy_to(&mut lat).unwrap();

        println!("yc:{:?}", lon);
        println!("yc:{:?}", lat);

        assert_approx_eq!(f64, lon[0], expect_wgs.0, epsilon = 1e-14);
        assert_approx_eq!(f64, lat[0], expect_wgs.1, epsilon = 1e-14);
    }
}
