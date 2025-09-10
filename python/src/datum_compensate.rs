use numpy::{PyArrayDyn, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use pyo3::{Python, pyfunction};
use rayon::prelude::*;
#[pyfunction]
pub fn py_datum_compensate(
    py: Python<'_>,
    xc_py: Py<PyAny>,
    yc_py: Py<PyAny>,
    hb: f64,
    r: f64,
    x0: f64,
    y0: f64,
) -> Result<pyo3::Bound<'_, PyTuple>, PyErr> {
    let processor = pyxis::DatumCompensate::new(hb, r, x0, y0);
    if let (Ok(xc_ref), Ok(yc_ref)) = (
        xc_py.downcast_bound::<PyArrayDyn<f64>>(py),
        yc_py.downcast_bound::<PyArrayDyn<f64>>(py),
    ) {
        let xc_array = unsafe { xc_ref.as_slice_mut().unwrap() };
        let yc_array = unsafe { yc_ref.as_slice_mut().unwrap() };
        xc_array
            .par_iter_mut()
            .zip(yc_array.par_iter_mut())
            .for_each(|(x, y)| {
                (*x, *y) = processor.transform(*x, *y);
            });
        (xc_ref, yc_ref).into_pyobject(py)
    } else if let (Ok(xc), Ok(yc)) = (xc_py.extract::<f64>(py), yc_py.extract::<f64>(py)) {
        processor.transform(xc, yc).into_pyobject(py)
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Input must be a float or a 1D numpy.ndarray of floats.",
        ))
    }
}
