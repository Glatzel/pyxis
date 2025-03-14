use cust::{memory::DeviceCopy, prelude::*};
use pyxis::{GeoFloat, IDatumCompenseParms};

use crate::context::PyxisCudaContext;

const PTX_STR: &str = include_str!("./datum_compense_cuda.ptx");
const PTX: crate::context::PyxisPtx = crate::context::PyxisPtx {
    name: "datum_compense_cuda",
    content: PTX_STR,
    size: PTX_STR.len(),
};
impl PyxisCudaContext {
    pub fn datum_compense_cuda<T: 'static + DeviceCopy + GeoFloat>(
        &self,
        xc: &mut DeviceBuffer<T>,
        yc: &mut DeviceBuffer<T>,
        parms: &impl IDatumCompenseParms<T>,
    ) {
        assert_eq!(xc.len(), yc.len());
        let length: usize = xc.len();
        let module = self.get_module(&PTX);
        // let func = module.get_function("datum_compense_cuda_double").unwrap();
        let func = self.get_function::<T>(&module, "datum_compense_cuda");
        let stream = &self.stream;
        let (grid_size, block_size) = self.get_grid_block(&func, length);

        unsafe {
            launch!(
                func<<<grid_size, block_size, 0, stream>>>(
                    xc.as_device_ptr(),
                    yc.as_device_ptr(),
                    parms.factor(),
                    parms.x0(),
                    parms.y0(),
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
    fn test_datum_compense_cuda() {
        let mut xc: Vec<f64> = vec![469704.6693, 469704.6693];
        let mut yc: Vec<f64> = vec![2821940.796, 2821940.796];
        let ctx = &crate::CONTEXT;
        let mut dxc = ctx.device_buffer_from_slice(&xc);
        let mut dyc = ctx.device_buffer_from_slice(&yc);
        let parms = pyxis::DatumCompenseParms::new(400.0, 6_378_137.0, 500_000.0, 0.0);
        ctx.datum_compense_cuda(&mut dxc, &mut dyc, &parms);
        dxc.copy_to(&mut xc).unwrap();
        dyc.copy_to(&mut yc).unwrap();

        println!("yc:{:?}", xc);
        println!("yc:{:?}", yc);
    }
}
