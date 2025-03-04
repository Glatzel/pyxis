use numpy::{PyArrayDyn, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use pyo3::{PyObject, Python, pyfunction};
use pyxis_algorithm::*;
use rayon::prelude::*;
fn get_space_fn(from: &str, to: &str) -> miette::Result<impl Fn(f64, f64, f64) -> (f64, f64, f64)> {
    match (from.to_lowercase().as_str(), to.to_lowercase().as_str()) {
        ("cartesian", "cylindrical") => {
            Ok(cartesian_to_cylindrical as fn(f64, f64, f64) -> (f64, f64, f64))
        }
        ("cartesian", "spherical") => {
            Ok(cartesian_to_spherical as fn(f64, f64, f64) -> (f64, f64, f64))
        }
        ("cylindrical", "cartesian") => {
            Ok(cylindrical_to_cartesian as fn(f64, f64, f64) -> (f64, f64, f64))
        }
        ("cylindrical", "spherical") => {
            Ok(cylindrical_to_spherical as fn(f64, f64, f64) -> (f64, f64, f64))
        }
        ("spherical", "cartesian") => {
            Ok(spherical_to_cartesian as fn(f64, f64, f64) -> (f64, f64, f64))
        }
        ("spherical", "cylindrical") => {
            Ok(spherical_to_cylindrical as fn(f64, f64, f64) -> (f64, f64, f64))
        }

        _ => miette::bail!("unknow from to"),
    }
}
#[pyfunction]
pub fn py_space(
    py: Python,
    x_py: PyObject,
    y_py: PyObject,
    z_py: PyObject,
    from: String,
    to: String,
) -> Result<pyo3::Bound<'_, PyTuple>, PyErr> {
    let space_fn = get_space_fn(&from, &to).unwrap();
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
                (*x, *y, *z) = space_fn(*x, *y, *z);
            });
        (x_ref, y_ref, z_ref).into_pyobject(py) // r,u,z
    } else if let (Ok(x), Ok(y), Ok(z)) = (
        x_py.extract::<f64>(py),
        y_py.extract::<f64>(py),
        z_py.extract::<f64>(py),
    ) {
        space_fn(x, y, z).into_pyobject(py)
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Input must be a float or a 1D numpy.ndarray of floats.",
        ))
    }
}
