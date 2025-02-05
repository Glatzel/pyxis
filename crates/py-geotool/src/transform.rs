use numpy::{PyArrayDyn, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use pyo3::{pyfunction, PyObject, Python};
use rayon::prelude::*;
#[pyfunction]
pub fn py_datum_compense(
    py: Python,
    xc_py: PyObject,
    yc_py: PyObject,
    hb: f64,
    r: f64,
    x0: f64,
    y0: f64,
) -> Result<pyo3::Bound<'_, PyTuple>, PyErr> {
    if let (Ok(xc), Ok(yc)) = (xc_py.extract::<f64>(py), yc_py.extract::<f64>(py)) {
        geotool_algorithm::datum_compense(xc, yc, hb, r, x0, y0).into_pyobject(py)
    } else if let (Ok(xc_ref), Ok(yc_ref)) = (
        xc_py.downcast_bound::<PyArrayDyn<f64>>(py),
        yc_py.downcast_bound::<PyArrayDyn<f64>>(py),
    ) {
        let xc_array = unsafe { xc_ref.as_slice_mut().unwrap() };
        let yc_array = unsafe { yc_ref.as_slice_mut().unwrap() };
        xc_array
            .par_iter_mut()
            .zip(yc_array.par_iter_mut())
            .for_each(|(x, y)| {
                (*x, *y) = geotool_algorithm::datum_compense(*x, *y, hb, r, x0, y0);
            });
        (xc_ref, yc_ref).into_pyobject(py)
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Input must be a float or a 1D numpy.ndarray of floats.",
        ))
    }
}
#[pyfunction]
pub fn py_lbh2xyz(
    py: Python,
    lon_py: PyObject,
    lat_py: PyObject,
    height_py: PyObject,
    semi_major_axis: f64,
    inverse_flattening: f64,
) -> Result<pyo3::Bound<'_, PyTuple>, PyErr> {
    if let (Ok(lon), Ok(lat), Ok(height)) = (
        lon_py.extract::<f64>(py),
        lat_py.extract::<f64>(py),
        height_py.extract::<f64>(py),
    ) {
        geotool_algorithm::lbh2xyz(lon, lat, height, semi_major_axis, inverse_flattening)
            .into_pyobject(py)
    } else if let (Ok(lon_ref), Ok(lat_ref), Ok(height_ref)) = (
        lon_py.downcast_bound::<PyArrayDyn<f64>>(py),
        lat_py.downcast_bound::<PyArrayDyn<f64>>(py),
        height_py.downcast_bound::<PyArrayDyn<f64>>(py),
    ) {
        let lon_array = unsafe { lon_ref.as_slice_mut().unwrap() };
        let lat_array = unsafe { lat_ref.as_slice_mut().unwrap() };
        let height_array = unsafe { height_ref.as_slice_mut().unwrap() };

        lon_array
            .par_iter_mut()
            .zip(lat_array.par_iter_mut())
            .zip(height_array.par_iter_mut())
            .for_each(|((lon, lat), height)| {
                (*lon, *lat, *height) = geotool_algorithm::lbh2xyz(
                    *lon,
                    *lat,
                    *height,
                    semi_major_axis,
                    inverse_flattening,
                );
            });
        (lon_ref, lat_ref, height_ref).into_pyobject(py)
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Input must be a float or a 1D numpy.ndarray of floats.",
        ))
    }
}
#[pyfunction]
pub fn py_xyz2lbh(
    py: Python,
    x_py: PyObject,
    y_py: PyObject,
    z_py: PyObject,
    semi_major_axis: f64,
    inverse_flattening: f64,
    tolerance: f64,
    max_iterations: u32,
) -> Result<pyo3::Bound<'_, PyTuple>, PyErr> {
    if let (Ok(x), Ok(y), Ok(z)) = (
        x_py.extract::<f64>(py),
        y_py.extract::<f64>(py),
        z_py.extract::<f64>(py),
    ) {
        geotool_algorithm::xyz2lbh(
            x,
            y,
            z,
            semi_major_axis,
            inverse_flattening,
            Some(tolerance),
            Some(max_iterations),
        )
        .into_pyobject(py)
    } else if let (Ok(x_ref), Ok(y_ref), Ok(z_ref)) = (
        x_py.downcast_bound::<PyArrayDyn<f64>>(py),
        y_py.downcast_bound::<PyArrayDyn<f64>>(py),
        z_py.downcast_bound::<PyArrayDyn<f64>>(py),
    ) {
        let x_array = unsafe { x_ref.as_slice_mut().unwrap() };
        let y_array = unsafe { y_ref.as_slice_mut().unwrap() };
        let z_array = unsafe { z_ref.as_slice_mut().unwrap() };

        x_array
            .par_iter_mut()
            .zip(y_array.par_iter_mut())
            .zip(z_array.par_iter_mut())
            .for_each(|((l, b), h)| {
                (*l, *b, *h) = geotool_algorithm::xyz2lbh(
                    *l,
                    *b,
                    *h,
                    semi_major_axis,
                    inverse_flattening,
                    Some(tolerance),
                    Some(max_iterations),
                );
            });
        (x_ref, y_ref, z_ref).into_pyobject(py)
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Input must be a float or a 1D numpy.ndarray of floats.",
        ))
    }
}
