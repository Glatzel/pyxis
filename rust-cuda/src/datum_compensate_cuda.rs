use cust::memory::DeviceCopy;
use cust::prelude::*;
use pyxis::GeoFloat;

use crate::context::PyxisCudaContext;

const PTX_STR: &str = include_str!("./datum_compensate_cuda.ptx");
const PTX: crate::context::PyxisPtx = crate::context::PyxisPtx {
    name: "datum_compensate_cuda",
    content: PTX_STR,
    size: PTX_STR.len(),
};
impl PyxisCudaContext {
    pub fn datum_compensate_cuda<T: 'static + DeviceCopy + GeoFloat>(
        &self,
        xc: &mut DeviceBuffer<T>,
        yc: &mut DeviceBuffer<T>,
        hb: T,
        radius: T,
        x0: T,
        y0: T,
    ) {
        assert_eq!(xc.len(), yc.len());
        let length: usize = xc.len();
        let module = self.get_module(&PTX);
        // let func = module.get_function("datum_compensate_cuda_double").unwrap();
        let func = self.get_function::<T>(&module, "datum_compensate_cuda");
        let stream = &self.stream;
        let (grid_size, block_size) = self.get_grid_block(length);

        let ratio = hb / radius;
        let factor = ratio / (T::ONE + ratio);

        unsafe {
            launch!(
                func<<<grid_size, block_size, 0, stream>>>(length,
                    xc.as_device_ptr(),
                    yc.as_device_ptr(),
                   factor,
                   x0,
                   y0,
                    xc.as_device_ptr(),
                    yc.as_device_ptr(),
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
    fn test_datum_compensate_cuda() {
        let mut xc: Vec<f64> = vec![469704.6693, 469704.6693];
        let mut yc: Vec<f64> = vec![2821940.796, 2821940.796];
        let ctx = &crate::CONTEXT;
        let mut dxc = ctx.device_buffer_from_slice(&xc);
        let mut dyc = ctx.device_buffer_from_slice(&yc);
        ctx.datum_compensate_cuda(&mut dxc, &mut dyc, 400.0, 6_378_137.0, 500_000.0, 0.0);
        dxc.copy_to(&mut xc).unwrap();
        dyc.copy_to(&mut yc).unwrap();

        println!("yc:{:?}", xc);
        println!("yc:{:?}", yc);
    }
}
