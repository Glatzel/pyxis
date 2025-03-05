use cust::prelude::*;
use pyxis::IDatumCompenseParms;

use crate::PyxisCudaContext;
const PTX: &str = include_str!("./datum_compense.ptx");

impl PyxisCudaContext {
    pub fn datum_compense_cuda<'a>(
        &self,
        xc: &mut DeviceBuffer<f64>,
        yc: &mut DeviceBuffer<f64>,
        parms: &impl IDatumCompenseParms<f64>,
    ) {
        let length: usize = xc.len();
        let module = Module::from_ptx(PTX, &[]).unwrap();
        let func = module.get_function("datum_compense").unwrap();
        let stream = self.stream();
        let (_, block_size) = func.suggested_launch_configuration(0, 0.into()).unwrap();
        let grid_size = (length as u32).div_ceil(block_size);
        
        #[cfg(feature = "log")]
        {
            tracing::debug!(
                "using {} blocks and {} threads per block",
                grid_size,
                block_size
            );
        }

        unsafe {
            launch!(
                func<<<grid_size, block_size, 0, stream>>>(
                    xc.as_device_ptr(),
                    yc.as_device_ptr(),parms.factor(),parms.x0(),parms.y0()
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
        let ctx = crate::PyxisCudaContext::new();
        let mut dxc = ctx.from_slice(&xc);
        let mut dyc = ctx.from_slice(&yc);
        let parms = pyxis::DatumCompenseParms::new(400.0, 6_378_137.0, 500_000.0, 0.0);
        ctx.datum_compense_cuda(&mut dxc, &mut dyc, &parms);
        dxc.copy_to(&mut xc).unwrap();
        dyc.copy_to(&mut yc).unwrap();

        println!("yc:{:?}", xc);
        println!("yc:{:?}", yc);
    }
}
