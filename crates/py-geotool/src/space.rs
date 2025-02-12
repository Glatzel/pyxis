use numpy::{PyArrayDyn, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use pyo3::{pyfunction, PyObject, Python};
use rayon::prelude::*;
#[pyfunction]
pub fn py_cartesian_to_cylindrical(
    py: Python,
    x_py: PyObject,
    y_py: PyObject,
    z_py: PyObject,
) -> Result<pyo3::Bound<'_, PyTuple>, PyErr> {
    if let (Ok(x_ref), Ok(y_ref), Ok(z_ref)) = (
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
            .for_each(|((x, y), z)| {
                (*x, *y, *z) = geotool_algorithm::cartesian_to_cylindrical(*x, *y, *z);
            });
        (x_ref, y_ref, z_ref).into_pyobject(py) // r,u,z
    } else if let (Ok(x), Ok(y), Ok(z)) = (
        x_py.extract::<f64>(py),
        y_py.extract::<f64>(py),
        z_py.extract::<f64>(py),
    ) {
        geotool_algorithm::cartesian_to_cylindrical(x, y, z).into_pyobject(py)
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Input must be a float or a 1D numpy.ndarray of floats.",
        ))
    }
}
#[pyfunction]
pub fn py_cartesian_to_spherical(
    py: Python,
    x_py: PyObject,
    y_py: PyObject,
    z_py: PyObject,
) -> Result<pyo3::Bound<'_, PyTuple>, PyErr> {
    if let (Ok(x_ref), Ok(y_ref), Ok(z_ref)) = (
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
            .for_each(|((x, y), z)| {
                (*x, *y, *z) = geotool_algorithm::cartesian_to_spherical(*x, *y, *z);
            });
        (x_ref, y_ref, z_ref).into_pyobject(py) // u,v,r
    } else if let (Ok(x), Ok(y), Ok(z)) = (
        x_py.extract::<f64>(py),
        y_py.extract::<f64>(py),
        z_py.extract::<f64>(py),
    ) {
        geotool_algorithm::cartesian_to_spherical(x, y, z).into_pyobject(py)
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Input must be a float or a 1D numpy.ndarray of floats.",
        ))
    }
}
#[pyfunction]
pub fn py_cylindrical_to_cartesian(
    py: Python,
    r_py: PyObject,
    u_py: PyObject,
    z_py: PyObject,
) -> Result<pyo3::Bound<'_, PyTuple>, PyErr> {
    if let (Ok(r_ref), Ok(u_ref), Ok(z_ref)) = (
        r_py.downcast_bound::<PyArrayDyn<f64>>(py),
        u_py.downcast_bound::<PyArrayDyn<f64>>(py),
        z_py.downcast_bound::<PyArrayDyn<f64>>(py),
    ) {
        let r_array = unsafe { r_ref.as_slice_mut().unwrap() };
        let u_array = unsafe { u_ref.as_slice_mut().unwrap() };
        let z_array = unsafe { z_ref.as_slice_mut().unwrap() };

        r_array
            .par_iter_mut()
            .zip(u_array.par_iter_mut())
            .zip(z_array.par_iter_mut())
            .for_each(|((r, u), z)| {
                (*r, *u, *z) = geotool_algorithm::cylindrical_to_cartesian(*r, *u, *z);
            });
        (r_ref, u_ref, z_ref).into_pyobject(py) // x,y,z
    } else if let (Ok(r), Ok(u), Ok(z)) = (
        r_py.extract::<f64>(py),
        u_py.extract::<f64>(py),
        z_py.extract::<f64>(py),
    ) {
        geotool_algorithm::cylindrical_to_cartesian(r, u, z).into_pyobject(py)
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Input must be a float or a 1D numpy.ndarray of floats.",
        ))
    }
}

#[pyfunction]
pub fn py_cylindrical_to_spherical(
    py: Python,
    r_py: PyObject,
    u_py: PyObject,
    z_py: PyObject,
) -> Result<pyo3::Bound<'_, PyTuple>, PyErr> {
    if let (Ok(r_ref), Ok(u_ref), Ok(z_ref)) = (
        r_py.downcast_bound::<PyArrayDyn<f64>>(py),
        u_py.downcast_bound::<PyArrayDyn<f64>>(py),
        z_py.downcast_bound::<PyArrayDyn<f64>>(py),
    ) {
        let r_array = unsafe { r_ref.as_slice_mut().unwrap() };
        let u_array = unsafe { u_ref.as_slice_mut().unwrap() };
        let z_array = unsafe { z_ref.as_slice_mut().unwrap() };

        r_array
            .par_iter_mut()
            .zip(u_array.par_iter_mut())
            .zip(z_array.par_iter_mut())
            .for_each(|((r, u), z)| {
                (*r, *u, *z) = geotool_algorithm::cylindrical_to_spherical(*r, *u, *z);
            });
        (r_ref, u_ref, z_ref).into_pyobject(py) // u,v,r
    } else if let (Ok(r), Ok(u), Ok(z)) = (
        r_py.extract::<f64>(py),
        u_py.extract::<f64>(py),
        z_py.extract::<f64>(py),
    ) {
        geotool_algorithm::cylindrical_to_spherical(r, u, z).into_pyobject(py)
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Input must be a float or a 1D numpy.ndarray of floats.",
        ))
    }
}
#[pyfunction]
pub fn py_spherical_to_cartesian(
    py: Python,
    u_py: PyObject,
    v_py: PyObject,
    r_py: PyObject,
) -> Result<pyo3::Bound<'_, PyTuple>, PyErr> {
    if let (Ok(u_ref), Ok(v_ref), Ok(r_ref)) = (
        u_py.downcast_bound::<PyArrayDyn<f64>>(py),
        v_py.downcast_bound::<PyArrayDyn<f64>>(py),
        r_py.downcast_bound::<PyArrayDyn<f64>>(py),
    ) {
        let u_array = unsafe { u_ref.as_slice_mut().unwrap() };
        let v_array = unsafe { v_ref.as_slice_mut().unwrap() };
        let r_array = unsafe { r_ref.as_slice_mut().unwrap() };

        u_array
            .par_iter_mut()
            .zip(v_array.par_iter_mut())
            .zip(r_array.par_iter_mut())
            .for_each(|((u, v), z)| {
                (*u, *v, *z) = geotool_algorithm::spherical_to_cartesian(*u, *v, *z);
            });
        (u_ref, v_ref, r_ref).into_pyobject(py) // x,y,z
    } else if let (Ok(u), Ok(v), Ok(r)) = (
        u_py.extract::<f64>(py),
        v_py.extract::<f64>(py),
        r_py.extract::<f64>(py),
    ) {
        geotool_algorithm::spherical_to_cartesian(u, v, r).into_pyobject(py)
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Input must be a float or a 1D numpy.ndarray of floats.",
        ))
    }
}
#[pyfunction]
pub fn py_spherical_to_cylindrical(
    py: Python,
    u_py: PyObject,
    v_py: PyObject,
    r_py: PyObject,
) -> Result<pyo3::Bound<'_, PyTuple>, PyErr> {
    if let (Ok(u_ref), Ok(v_ref), Ok(r_ref)) = (
        u_py.downcast_bound::<PyArrayDyn<f64>>(py),
        v_py.downcast_bound::<PyArrayDyn<f64>>(py),
        r_py.downcast_bound::<PyArrayDyn<f64>>(py),
    ) {
        let u_array = unsafe { u_ref.as_slice_mut().unwrap() };
        let v_array = unsafe { v_ref.as_slice_mut().unwrap() };
        let r_array = unsafe { r_ref.as_slice_mut().unwrap() };

        u_array
            .par_iter_mut()
            .zip(v_array.par_iter_mut())
            .zip(r_array.par_iter_mut())
            .for_each(|((u, v), z)| {
                (*u, *v, *z) = geotool_algorithm::spherical_to_cylindrical(*u, *v, *z);
            });
        (u_ref, v_ref, r_ref).into_pyobject(py) // x,y,z
    } else if let (Ok(u), Ok(v), Ok(r)) = (
        u_py.extract::<f64>(py),
        v_py.extract::<f64>(py),
        r_py.extract::<f64>(py),
    ) {
        geotool_algorithm::spherical_to_cylindrical(u, v, r).into_pyobject(py)
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Input must be a float or a 1D numpy.ndarray of floats.",
        ))
    }
}
