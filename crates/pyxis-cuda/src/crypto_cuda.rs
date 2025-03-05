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
}

#[cfg(test)]
mod test {
    use cust::memory::CopyDestination;

    #[test]
    fn test_bd09_to_gcj02_cuda() {
        let mut lon: Vec<f64> = vec![pyxis::crypto::BD09_LON, pyxis::crypto::BD09_LON];
        let mut lat: Vec<f64> = vec![pyxis::crypto::BD09_LAT, pyxis::crypto::BD09_LAT];
        let ctx = crate::PyxisCudaContext::new();
        let mut dlon = ctx.from_slice(&lon);
        let mut dlat = ctx.from_slice(&lat);
        ctx.bd09_to_gcj02_cuda(&mut dlon, &mut dlat);
        dlon.copy_to(&mut lon).unwrap();
        dlat.copy_to(&mut lat).unwrap();

        println!("yc:{:?}", lon);
        println!("yc:{:?}", lat);
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
    }
}
