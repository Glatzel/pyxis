use numpy::{PyArrayDyn, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use pyo3::{PyObject, Python, pyfunction};
use rayon::prelude::*;

#[pyfunction]
pub fn py_bd09_to_gcj02(
    py: Python,
    lon_py: PyObject,
    lat_py: PyObject,
) -> Result<pyo3::Bound<'_, PyTuple>, PyErr> {
    if let (Ok(lon_ref), Ok(lat_ref)) = (
        lon_py.downcast_bound::<PyArrayDyn<f64>>(py),
        lat_py.downcast_bound::<PyArrayDyn<f64>>(py),
    ) {
        let lon_array = unsafe { lon_ref.as_slice_mut().unwrap() };
        let lat_array = unsafe { lat_ref.as_slice_mut().unwrap() };
        lon_array
            .par_iter_mut()
            .zip(lat_array.par_iter_mut())
            .for_each(|(x, y)| {
                (*x, *y) = geotool_algorithm::bd09_to_gcj02(*x, *y);
            });
        (lon_ref, lat_ref).into_pyobject(py)
    } else if let (Ok(lon), Ok(lat)) = (lon_py.extract::<f64>(py), lat_py.extract::<f64>(py)) {
        geotool_algorithm::bd09_to_gcj02(lon, lat).into_pyobject(py)
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Input must be a float or a 1D numpy.ndarray of floats.",
        ))
    }
}
#[pyfunction]
pub fn py_bd09_to_wgs84(
    py: Python,
    lon_py: PyObject,
    lat_py: PyObject,
) -> Result<pyo3::Bound<'_, PyTuple>, PyErr> {
    if let (Ok(lon_ref), Ok(lat_ref)) = (
        lon_py.downcast_bound::<PyArrayDyn<f64>>(py),
        lat_py.downcast_bound::<PyArrayDyn<f64>>(py),
    ) {
        let lon_array = unsafe { lon_ref.as_slice_mut().unwrap() };
        let lat_array = unsafe { lat_ref.as_slice_mut().unwrap() };
        lon_array
            .par_iter_mut()
            .zip(lat_array.par_iter_mut())
            .for_each(|(x, y)| {
                (*x, *y) = geotool_algorithm::bd09_to_wgs84(*x, *y);
            });
        (lon_ref, lat_ref).into_pyobject(py)
    } else if let (Ok(lon), Ok(lat)) = (lon_py.extract::<f64>(py), lat_py.extract::<f64>(py)) {
        geotool_algorithm::bd09_to_wgs84(lon, lat).into_pyobject(py)
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Input must be a float or a 1D numpy.ndarray of floats.",
        ))
    }
}
#[pyfunction]
pub fn py_gcj02_to_bd09(
    py: Python,
    lon_py: PyObject,
    lat_py: PyObject,
) -> Result<pyo3::Bound<'_, PyTuple>, PyErr> {
    if let (Ok(lon_ref), Ok(lat_ref)) = (
        lon_py.downcast_bound::<PyArrayDyn<f64>>(py),
        lat_py.downcast_bound::<PyArrayDyn<f64>>(py),
    ) {
        let lon_array = unsafe { lon_ref.as_slice_mut().unwrap() };
        let lat_array = unsafe { lat_ref.as_slice_mut().unwrap() };
        lon_array
            .par_iter_mut()
            .zip(lat_array.par_iter_mut())
            .for_each(|(x, y)| {
                (*x, *y) = geotool_algorithm::gcj02_to_bd09(*x, *y);
            });
        (lon_ref, lat_ref).into_pyobject(py)
    } else if let (Ok(lon), Ok(lat)) = (lon_py.extract::<f64>(py), lat_py.extract::<f64>(py)) {
        geotool_algorithm::gcj02_to_bd09(lon, lat).into_pyobject(py)
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Input must be a float or a 1D numpy.ndarray of floats.",
        ))
    }
}
#[pyfunction]
pub fn py_gcj02_to_wgs84(
    py: Python,
    lon_py: PyObject,
    lat_py: PyObject,
) -> Result<pyo3::Bound<'_, PyTuple>, PyErr> {
    if let (Ok(lon_ref), Ok(lat_ref)) = (
        lon_py.downcast_bound::<PyArrayDyn<f64>>(py),
        lat_py.downcast_bound::<PyArrayDyn<f64>>(py),
    ) {
        let lon_array = unsafe { lon_ref.as_slice_mut().unwrap() };
        let lat_array = unsafe { lat_ref.as_slice_mut().unwrap() };
        lon_array
            .par_iter_mut()
            .zip(lat_array.par_iter_mut())
            .for_each(|(x, y)| {
                (*x, *y) = geotool_algorithm::gcj02_to_wgs84(*x, *y);
            });
        (lon_ref, lat_ref).into_pyobject(py)
    } else if let (Ok(lon), Ok(lat)) = (lon_py.extract::<f64>(py), lat_py.extract::<f64>(py)) {
        geotool_algorithm::gcj02_to_wgs84(lon, lat).into_pyobject(py)
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Input must be a float or a 1D numpy.ndarray of floats.",
        ))
    }
}
#[pyfunction]
pub fn py_wgs84_to_bd09(
    py: Python,
    lon_py: PyObject,
    lat_py: PyObject,
) -> Result<pyo3::Bound<'_, PyTuple>, PyErr> {
    if let (Ok(lon_ref), Ok(lat_ref)) = (
        lon_py.downcast_bound::<PyArrayDyn<f64>>(py),
        lat_py.downcast_bound::<PyArrayDyn<f64>>(py),
    ) {
        let lon_array = unsafe { lon_ref.as_slice_mut().unwrap() };
        let lat_array = unsafe { lat_ref.as_slice_mut().unwrap() };
        lon_array
            .par_iter_mut()
            .zip(lat_array.par_iter_mut())
            .for_each(|(x, y)| {
                (*x, *y) = geotool_algorithm::wgs84_to_bd09(*x, *y);
            });
        (lon_ref, lat_ref).into_pyobject(py)
    } else if let (Ok(lon), Ok(lat)) = (lon_py.extract::<f64>(py), lat_py.extract::<f64>(py)) {
        geotool_algorithm::wgs84_to_bd09(lon, lat).into_pyobject(py)
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Input must be a float or a 1D numpy.ndarray of floats.",
        ))
    }
}
#[pyfunction]
pub fn py_wgs84_to_gcj02(
    py: Python,
    lon_py: PyObject,
    lat_py: PyObject,
) -> Result<pyo3::Bound<'_, PyTuple>, PyErr> {
    if let (Ok(lon_ref), Ok(lat_ref)) = (
        lon_py.downcast_bound::<PyArrayDyn<f64>>(py),
        lat_py.downcast_bound::<PyArrayDyn<f64>>(py),
    ) {
        let lon_array = unsafe { lon_ref.as_slice_mut().unwrap() };
        let lat_array = unsafe { lat_ref.as_slice_mut().unwrap() };
        lon_array
            .par_iter_mut()
            .zip(lat_array.par_iter_mut())
            .for_each(|(x, y)| {
                (*x, *y) = geotool_algorithm::wgs84_to_gcj02(*x, *y);
            });
        (lon_ref, lat_ref).into_pyobject(py)
    } else if let (Ok(lon), Ok(lat)) = (lon_py.extract::<f64>(py), lat_py.extract::<f64>(py)) {
        geotool_algorithm::wgs84_to_gcj02(lon, lat).into_pyobject(py)
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Input must be a float or a 1D numpy.ndarray of floats.",
        ))
    }
}
