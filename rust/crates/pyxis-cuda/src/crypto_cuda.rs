use cust::memory::DeviceCopy;
use cust::prelude::*;
use pyxis::GeoFloat;
use pyxis::crypto::{CryptoSpace, CryptoThresholdMode};

use crate::{PyxisCudaContext, PyxisPtx};

const PTX_STR: &str = include_str!("./crypto_cuda.ptx");
const PTX: PyxisPtx = crate::PyxisPtx {
    name: "datum_compense_cuda",
    content: PTX_STR,
    size: PTX_STR.len(),
};
impl PyxisCudaContext {
    pub fn bd09_to_gcj02_cuda<T: 'static + DeviceCopy + GeoFloat>(
        &self,
        lon: &mut DeviceBuffer<T>,
        lat: &mut DeviceBuffer<T>,
    ) {
        assert_eq!(lon.len(), lat.len());
        let length: usize = lon.len();
        let module = self.get_module(&PTX);
        let func = self.get_function::<T>(&module, "bd09_to_gcj02_cuda");
        let stream = &self.stream;
        let (grid_size, block_size) = self.get_grid_block(&func, length);

        unsafe {
            launch!(
                func<<<grid_size, block_size, 0, stream>>>(length,
                    lon.as_device_ptr(),
                    lat.as_device_ptr()
                )
            )
            .unwrap();
        }
        stream.synchronize().unwrap();
    }
    pub fn gcj02_to_bd09_cuda<T: 'static + DeviceCopy + GeoFloat>(
        &self,
        lon: &mut DeviceBuffer<T>,
        lat: &mut DeviceBuffer<T>,
    ) {
        assert_eq!(lon.len(), lat.len());
        let length: usize = lon.len();
        let module = self.get_module(&PTX);
        let func = self.get_function::<T>(&module, "gcj02_to_bd09_cuda");
        let stream = &self.stream;
        let (grid_size, block_size) = self.get_grid_block(&func, length);

        unsafe {
            launch!(
                func<<<grid_size, block_size, 0, stream>>>(length,
                    lon.as_device_ptr(),
                    lat.as_device_ptr()
                )
            )
            .unwrap();
        }
        stream.synchronize().unwrap();
    }
    pub fn gcj02_to_wgs84_cuda<T: 'static + DeviceCopy + GeoFloat>(
        &self,
        lon: &mut DeviceBuffer<T>,
        lat: &mut DeviceBuffer<T>,
    ) {
        assert_eq!(lon.len(), lat.len());
        let length: usize = lon.len();
        let module = self.get_module(&PTX);
        let func = self.get_function::<T>(&module, "gcj02_to_wgs84_cuda");
        let stream = &self.stream;
        let (grid_size, block_size) = self.get_grid_block(&func, length);

        unsafe {
            launch!(
                func<<<grid_size, block_size, 0, stream>>>(length,
                    lon.as_device_ptr(),
                    lat.as_device_ptr()
                )
            )
            .unwrap();
        }
        stream.synchronize().unwrap();
    }
    pub fn wgs84_to_gcj02_cuda<T: 'static + DeviceCopy + GeoFloat>(
        &self,
        lon: &mut DeviceBuffer<T>,
        lat: &mut DeviceBuffer<T>,
    ) {
        assert_eq!(lon.len(), lat.len());
        let length: usize = lon.len();
        let module = self.get_module(&PTX);
        let func = self.get_function::<T>(&module, "wgs84_to_gcj02_cuda");
        let stream = &self.stream;
        let (grid_size, block_size) = self.get_grid_block(&func, length);

        unsafe {
            launch!(
                func<<<grid_size, block_size, 0, stream>>>(length,
                    lon.as_device_ptr(),
                    lat.as_device_ptr()
                )
            )
            .unwrap();
        }
        stream.synchronize().unwrap();
    }
    pub fn wgs84_to_bd09_cuda<T: 'static + DeviceCopy + GeoFloat>(
        &self,
        lon: &mut DeviceBuffer<T>,
        lat: &mut DeviceBuffer<T>,
    ) {
        assert_eq!(lon.len(), lat.len());
        let length: usize = lon.len();
        let module = self.get_module(&PTX);
        let func = self.get_function::<T>(&module, "wgs84_to_bd09_cuda");
        let stream = &self.stream;
        let (grid_size, block_size) = self.get_grid_block(&func, length);

        unsafe {
            launch!(
                func<<<grid_size, block_size, 0, stream>>>(length,
                    lon.as_device_ptr(),
                    lat.as_device_ptr()
                )
            )
            .unwrap();
        }
        stream.synchronize().unwrap();
    }
    pub fn bd09_to_wgs84_cuda<T: 'static + DeviceCopy + GeoFloat>(
        &self,
        lon: &mut DeviceBuffer<T>,
        lat: &mut DeviceBuffer<T>,
    ) {
        assert_eq!(lon.len(), lat.len());
        let length: usize = lon.len();
        let module = self.get_module(&PTX);
        let func = self.get_function::<T>(&module, "bd09_to_wgs84_cuda");
        let stream = &self.stream;
        let (grid_size, block_size) = self.get_grid_block(&func, length);

        unsafe {
            launch!(
                func<<<grid_size, block_size, 0, stream>>>(length,
                    lon.as_device_ptr(),
                    lat.as_device_ptr()
                )
            )
            .unwrap();
        }
        stream.synchronize().unwrap();
    }
    pub fn crypto_exact_cuda<T: 'static + DeviceCopy + GeoFloat>(
        &self,
        lon: &mut DeviceBuffer<T>,
        lat: &mut DeviceBuffer<T>,
        from: CryptoSpace,
        to: CryptoSpace,
        threshold: T,
        threshold_mode: CryptoThresholdMode,
        max_iter: usize,
    ) {
        assert_eq!(lon.len(), lat.len());
        let length: usize = lon.len();
        let module = self.get_module(&PTX);
        let func = match (from, to) {
            (CryptoSpace::GCJ02, CryptoSpace::WGS84) => {
                self.get_function::<T>(&module, "gcj02_to_wgs84_exact_cuda")
            }
            (CryptoSpace::BD09, CryptoSpace::WGS84) => {
                self.get_function::<T>(&module, "bd09_to_wgs84_exact_cuda")
            }
            (CryptoSpace::BD09, CryptoSpace::GCJ02) => {
                self.get_function::<T>(&module, "bd09_to_gcj02_exact_cuda")
            }
            _ => panic!("Unsupported "),
        };
        let distance_mode = match threshold_mode {
            CryptoThresholdMode::Distance => true,
            CryptoThresholdMode::LonLat => false,
        };
        let stream = &self.stream;
        let (grid_size, block_size) = self.get_grid_block(&func, length);
        unsafe {
            launch!(
                func<<<grid_size, block_size, 0, stream>>>(length,
                    lon.as_device_ptr(),
                    lat.as_device_ptr(),
                    threshold,
                    distance_mode,
                    max_iter
                )
            )
            .unwrap();
        }
        stream.synchronize().unwrap();
    }
}

#[cfg(all(feature = "log", test))]
mod test {
    use cust::memory::CopyDestination;
    use float_cmp::assert_approx_eq;
    use pyxis::crypto::*;
    use rand::Rng;

    #[test]
    fn test_bd09_to_gcj02_cuda() {
        let mut lon: Vec<f64> = vec![BD09_LON, BD09_LON];
        let mut lat: Vec<f64> = vec![BD09_LAT, BD09_LAT];
        let expect_gcj = pyxis::crypto::bd09_to_gcj02(BD09_LON, BD09_LAT);
        let ctx = &crate::CONTEXT;
        let mut dlon = ctx.device_buffer_from_slice(&lon);
        let mut dlat = ctx.device_buffer_from_slice(&lat);
        ctx.bd09_to_gcj02_cuda(&mut dlon, &mut dlat);
        dlon.copy_to(&mut lon).unwrap();
        dlat.copy_to(&mut lat).unwrap();

        println!("lon:{:?}", lon);
        println!("lat:{:?}", lat);

        assert_approx_eq!(f64, lon[0], expect_gcj.0, epsilon = 1e-14);
        assert_approx_eq!(f64, lat[0], expect_gcj.1, epsilon = 1e-14);
    }
    #[test]
    fn test_gcj02_to_bd09_cuda() {
        let ctx = &crate::CONTEXT;
        let mut lon: Vec<f64> = vec![GCJ02_LON, GCJ02_LON];
        let mut lat: Vec<f64> = vec![GCJ02_LAT, GCJ02_LAT];
        let mut dlon = ctx.device_buffer_from_slice(&lon);
        let mut dlat = ctx.device_buffer_from_slice(&lat);
        ctx.gcj02_to_bd09_cuda(&mut dlon, &mut dlat);
        dlon.copy_to(&mut lon).unwrap();
        dlat.copy_to(&mut lat).unwrap();
        println!("gcj02:{GCJ02_LON},{GCJ02_LAT}");
        println!("yc:{:?}", lon);
        println!("yc:{:?}", lat);
        println!("bd09:{BD09_LON},{BD09_LAT}");
        assert_approx_eq!(f64, lon[0], BD09_LON, epsilon = 1e-17);
        assert_approx_eq!(f64, lat[0], BD09_LAT, epsilon = 1e-17);
    }
    #[test]
    fn test_gcj02_to_wgs84_cuda() {
        let mut lon: Vec<f64> = vec![GCJ02_LON, GCJ02_LON];
        let mut lat: Vec<f64> = vec![GCJ02_LAT, GCJ02_LAT];
        let expect_wgs = pyxis::crypto::gcj02_to_wgs84(lon[0], lat[0]);
        let ctx = &crate::CONTEXT;
        let mut dlon = ctx.device_buffer_from_slice(&lon);
        let mut dlat = ctx.device_buffer_from_slice(&lat);
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
        let mut lon: Vec<f64> = vec![WGS84_LON, WGS84_LON];
        let mut lat: Vec<f64> = vec![WGS84_LAT, WGS84_LAT];
        let ctx = &crate::CONTEXT;
        let mut dlon = ctx.device_buffer_from_slice(&lon);
        let mut dlat = ctx.device_buffer_from_slice(&lat);
        ctx.wgs84_to_gcj02_cuda(&mut dlon, &mut dlat);
        dlon.copy_to(&mut lon).unwrap();
        dlat.copy_to(&mut lat).unwrap();
        println!("wgs84:{WGS84_LON},{WGS84_LAT}");
        println!("lon:{:?}", lon);
        println!("lat:{:?}", lat);
        println!("gcj02:{GCJ02_LON},{GCJ02_LAT}");
        assert_approx_eq!(f64, lon[0], GCJ02_LON, epsilon = 1e-17);
        assert_approx_eq!(f64, lat[0], GCJ02_LAT, epsilon = 1e-17);
    }
    #[test]
    fn test_wgs84_to_bd09_cuda() {
        let mut lon: Vec<f64> = vec![WGS84_LON, WGS84_LON];
        let mut lat: Vec<f64> = vec![WGS84_LAT, WGS84_LAT];
        let ctx = &crate::CONTEXT;
        let mut dlon = ctx.device_buffer_from_slice(&lon);
        let mut dlat = ctx.device_buffer_from_slice(&lat);
        ctx.wgs84_to_bd09_cuda(&mut dlon, &mut dlat);
        dlon.copy_to(&mut lon).unwrap();
        dlat.copy_to(&mut lat).unwrap();

        println!("lon:{:?}", lon);
        println!("lat:{:?}", lat);
        assert_approx_eq!(f64, lon[0], BD09_LON, epsilon = 1e-15);
        assert_approx_eq!(f64, lat[0], BD09_LAT, epsilon = 1e-15);
    }
    #[test]
    fn test_bd09_to_wgs84_cuda() {
        let mut lon: Vec<f64> = vec![BD09_LON, BD09_LON];
        let mut lat: Vec<f64> = vec![BD09_LAT, BD09_LAT];
        let expect_wgs = pyxis::crypto::bd09_to_wgs84(lon[0], lat[0]);
        let ctx = &crate::CONTEXT;
        let mut dlon = ctx.device_buffer_from_slice(&lon);
        let mut dlat = ctx.device_buffer_from_slice(&lat);
        ctx.bd09_to_wgs84_cuda(&mut dlon, &mut dlat);
        dlon.copy_to(&mut lon).unwrap();
        dlat.copy_to(&mut lat).unwrap();

        println!("yc:{:?}", lon);
        println!("yc:{:?}", lat);

        assert_approx_eq!(f64, lon[0], expect_wgs.0, epsilon = 1e-14);
        assert_approx_eq!(f64, lat[0], expect_wgs.1, epsilon = 1e-14);
    }
    #[test]
    fn test_bd09_to_wgs84_exact_cuda() {
        let mut lon: Vec<f64> = vec![BD09_LON, BD09_LON];
        let mut lat: Vec<f64> = vec![BD09_LAT, BD09_LAT];
        let ctx = &crate::CONTEXT;
        let mut dlon = ctx.device_buffer_from_slice(&lon);
        let mut dlat = ctx.device_buffer_from_slice(&lat);
        ctx.crypto_exact_cuda(
            &mut dlon,
            &mut dlat,
            CryptoSpace::BD09,
            CryptoSpace::WGS84,
            1e-20,
            CryptoThresholdMode::LonLat,
            100,
        );
        dlon.copy_to(&mut lon).unwrap();
        dlat.copy_to(&mut lat).unwrap();

        println!("bd09: {BD09_LON},{BD09_LAT}");
        println!("lon:{:?}", lon);
        println!("lat:{:?}", lat);
        println!("wgs84: {WGS84_LON},{WGS84_LAT}");
        assert_approx_eq!(f64, lon[0], WGS84_LON, epsilon = 1e-13);
        assert_approx_eq!(f64, lat[0], WGS84_LAT, epsilon = 1e-13);
    }
    #[test]
    fn test_bd09_to_gcj02_exact_cuda() {
        let mut lon: Vec<f64> = vec![BD09_LON, BD09_LON];
        let mut lat: Vec<f64> = vec![BD09_LAT, BD09_LAT];
        let ctx = &crate::CONTEXT;
        let mut dlon = ctx.device_buffer_from_slice(&lon);
        let mut dlat = ctx.device_buffer_from_slice(&lat);
        ctx.crypto_exact_cuda(
            &mut dlon,
            &mut dlat,
            CryptoSpace::BD09,
            CryptoSpace::GCJ02,
            1e-20,
            CryptoThresholdMode::LonLat,
            100,
        );
        dlon.copy_to(&mut lon).unwrap();
        dlat.copy_to(&mut lat).unwrap();

        println!("bd09: {BD09_LON},{BD09_LAT}");
        println!("lon:{:?}", lon);
        println!("lat:{:?}", lat);
        println!("gcj02: {GCJ02_LON},{GCJ02_LAT}");
        assert_approx_eq!(f64, lon[0], GCJ02_LON, epsilon = 1e-13);
        assert_approx_eq!(f64, lat[0], GCJ02_LAT, epsilon = 1e-13);
    }
    #[test]
    fn test_gcj02_to_wgs84_exact_cuda() {
        let mut lon: Vec<f64> = vec![GCJ02_LON, GCJ02_LON];
        let mut lat: Vec<f64> = vec![GCJ02_LAT, GCJ02_LAT];
        let ctx = &crate::CONTEXT;
        let mut dlon = ctx.device_buffer_from_slice(&lon);
        let mut dlat = ctx.device_buffer_from_slice(&lat);
        ctx.crypto_exact_cuda(
            &mut dlon,
            &mut dlat,
            CryptoSpace::GCJ02,
            CryptoSpace::WGS84,
            1e-20,
            CryptoThresholdMode::LonLat,
            100,
        );
        dlon.copy_to(&mut lon).unwrap();
        dlat.copy_to(&mut lat).unwrap();

        println!("bd09: {BD09_LON},{BD09_LAT}");
        println!("lon:{:?}", lon);
        println!("lat:{:?}", lat);
        println!("wgs84: {WGS84_LON},{WGS84_LAT}");
        assert_approx_eq!(f64, lon[0], WGS84_LON, epsilon = 1e-13);
        assert_approx_eq!(f64, lat[0], WGS84_LAT, epsilon = 1e-13);
    }
    #[test]
    fn test_exact_cuda() {
        let is_ci = std::env::var("CI").is_ok();
        let mut rng = rand::rng();
        let count = if is_ci { 10 } else { 150000 };
        let wgs_lon = (0..count)
            .map(|_| rng.random_range(72.004..137.8347))
            .collect::<Vec<f64>>();
        let wgs_lat = (0..count)
            .map(|_| rng.random_range(0.8293..55.8271))
            .collect::<Vec<f64>>();
        let gcj_lon = wgs_lon
            .iter()
            .zip(wgs_lat.iter())
            .map(|(lon, lat)| wgs84_to_gcj02(*lon, *lat).0)
            .collect::<Vec<f64>>();
        let gcj_lat = wgs_lon
            .iter()
            .zip(wgs_lat.iter())
            .map(|(lon, lat)| wgs84_to_gcj02(*lon, *lat).1)
            .collect::<Vec<f64>>();
        let bd_lon = wgs_lon
            .iter()
            .zip(wgs_lat.iter())
            .map(|(lon, lat)| wgs84_to_bd09(*lon, *lat).0)
            .collect::<Vec<f64>>();
        let bd_lat = wgs_lon
            .iter()
            .zip(wgs_lat.iter())
            .map(|(lon, lat)| wgs84_to_bd09(*lon, *lat).1)
            .collect::<Vec<f64>>();
        {
            let mut max_dist: f64 = 0.0;
            let mut max_lonlat: f64 = 0.0;
            let mut all_dist = 0.0;
            let mut all_lonlat = 0.0;
            let ctx = &crate::CONTEXT;
            let mut lon = gcj_lon.clone();
            let mut lat = gcj_lat.clone();
            let mut dlon = ctx.device_buffer_from_slice(&lon);
            let mut dlat = ctx.device_buffer_from_slice(&lat);
            ctx.crypto_exact_cuda(
                &mut dlon,
                &mut dlat,
                CryptoSpace::GCJ02,
                CryptoSpace::WGS84,
                1e-20,
                CryptoThresholdMode::LonLat,
                100,
            );
            dlon.copy_to(&mut lon).unwrap();
            dlat.copy_to(&mut lat).unwrap();
            lon.iter()
                .zip(lat.iter())
                .zip(wgs_lon.iter())
                .zip(wgs_lat.iter())
                .for_each(|(((x, y), a), b)| {
                    max_dist = max_dist.max(haversine_distance(*x, *y, *a, *b).abs());
                    max_lonlat = max_lonlat.max((x - a).abs()).max((y - b).abs());
                    all_dist += haversine_distance(*x, *y, *a, *b).abs();
                    all_lonlat += (x - a).abs() + (y - b).abs();
                });
            println!("gcj02 to wgs84 exact cuda");
            println!("average distance: {:.2e}", all_dist / count as f64);
            println!("max distance: {:.2e}", max_dist);
            println!("average lonlat: {:.2e}", all_lonlat / count as f64 / 2.0);
            println!("max lonlat: {:.2e}", max_lonlat);
        }
        {
            let mut max_dist: f64 = 0.0;
            let mut max_lonlat: f64 = 0.0;
            let mut all_dist = 0.0;
            let mut all_lonlat = 0.0;
            let ctx = &crate::CONTEXT;
            let mut lon = bd_lon.clone();
            let mut lat = bd_lat.clone();
            let mut dlon = ctx.device_buffer_from_slice(&lon);
            let mut dlat = ctx.device_buffer_from_slice(&lat);
            ctx.crypto_exact_cuda(
                &mut dlon,
                &mut dlat,
                CryptoSpace::BD09,
                CryptoSpace::WGS84,
                1e-20,
                CryptoThresholdMode::LonLat,
                100,
            );
            dlon.copy_to(&mut lon).unwrap();
            dlat.copy_to(&mut lat).unwrap();
            lon.iter()
                .zip(lat.iter())
                .zip(wgs_lon.iter())
                .zip(wgs_lat.iter())
                .for_each(|(((x, y), a), b)| {
                    max_dist = max_dist.max(haversine_distance(*x, *y, *a, *b).abs());
                    max_lonlat = max_lonlat.max((x - a).abs()).max((y - b).abs());
                    all_dist += haversine_distance(*x, *y, *a, *b).abs();
                    all_lonlat += (x - a).abs() + (y - b).abs();
                });
            println!("bd09 to wgs84 exact cuda");
            println!("average distance: {:.2e}", all_dist / count as f64);
            println!("max distance: {:.2e}", max_dist);
            println!("average lonlat: {:.2e}", all_lonlat / count as f64 / 2.0);
            println!("max lonlat: {:.2e}", max_lonlat);
        }
        {
            let mut max_dist: f64 = 0.0;
            let mut max_lonlat: f64 = 0.0;
            let mut all_dist = 0.0;
            let mut all_lonlat = 0.0;
            let ctx = &crate::CONTEXT;
            let mut lon = bd_lon.clone();
            let mut lat = bd_lat.clone();
            let mut dlon = ctx.device_buffer_from_slice(&lon);
            let mut dlat = ctx.device_buffer_from_slice(&lat);
            ctx.crypto_exact_cuda(
                &mut dlon,
                &mut dlat,
                CryptoSpace::BD09,
                CryptoSpace::GCJ02,
                1e-20,
                CryptoThresholdMode::LonLat,
                100,
            );
            dlon.copy_to(&mut lon).unwrap();
            dlat.copy_to(&mut lat).unwrap();
            lon.iter()
                .zip(lat.iter())
                .zip(gcj_lon.iter())
                .zip(gcj_lat.iter())
                .for_each(|(((x, y), a), b)| {
                    max_dist = max_dist.max(haversine_distance(*x, *y, *a, *b).abs());
                    max_lonlat = max_lonlat.max((x - a).abs()).max((y - b).abs());
                    all_dist += haversine_distance(*x, *y, *a, *b).abs();
                    all_lonlat += (x - a).abs() + (y - b).abs();
                });
            println!("bd09 to gcj02 exact cuda");
            println!("average distance: {:.2e}", all_dist / count as f64);
            println!("max distance: {:.2e}", max_dist);
            println!("average lonlat: {:.2e}", all_lonlat / count as f64 / 2.0);
            println!("max lonlat: {:.2e}", max_lonlat);
        }
    }
}
