use numpy::{PyArrayDyn, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use pyo3::{PyObject, Python, pyfunction};
use pyxis::crypto::*;
use rayon::prelude::*;
fn get_crypto_fn(
    from: &str,
    to: &str,
    exact: bool,
) -> Result<impl Fn(f64, f64) -> (f64, f64), PyErr> {
    match (
        from.to_lowercase().as_str(),
        to.to_lowercase().as_str(),
        exact,
    ) {
        ("bd09", "gcj02", false) => Ok(bd09_to_gcj02 as fn(f64, f64) -> (f64, f64)),
        ("bd09", "gcj02", true) => Ok(|src_lon, src_lat| {
            crypto_exact(
                src_lon,
                src_lat,
                &bd09_to_gcj02,
                &gcj02_to_bd09,
                1e-17,
                CryptoThresholdMode::LonLat,
                100,
            )
        }),
        ("bd09", "wgs84", false) => Ok(bd09_to_wgs84 as fn(f64, f64) -> (f64, f64)),
        ("bd09", "wgs84", true) => Ok(|src_lon, src_lat| {
            crypto_exact(
                src_lon,
                src_lat,
                &bd09_to_wgs84,
                &wgs84_to_bd09,
                1e-17,
                CryptoThresholdMode::LonLat,
                100,
            )
        }),
        ("gcj02", "bd09", _) => Ok(gcj02_to_bd09 as fn(f64, f64) -> (f64, f64)),
        ("gcj02", "wgs84", false) => Ok(gcj02_to_wgs84 as fn(f64, f64) -> (f64, f64)),
        ("gcj02", "wgs84", true) => Ok(|src_lon, src_lat| {
            crypto_exact(
                src_lon,
                src_lat,
                &gcj02_to_wgs84,
                &wgs84_to_gcj02,
                1e-17,
                CryptoThresholdMode::LonLat,
                100,
            )
        }),
        ("wgs84", "bd09", _) => Ok(wgs84_to_bd09 as fn(f64, f64) -> (f64, f64)),
        ("wgs84", "gcj02", _) => Ok(wgs84_to_gcj02 as fn(f64, f64) -> (f64, f64)),
        _ => Err(pyo3::exceptions::PyTypeError::new_err(
            "Unsupported input: from: {from}, to: {to}",
        )),
    }
}
#[pyfunction]
pub fn py_crypto(
    py: Python<'_>,
    lon_py: PyObject,
    lat_py: PyObject,
    from: String,
    to: String,
    exact: bool,
) -> Result<pyo3::Bound<'_, PyTuple>, PyErr> {
    let crypto_fn = get_crypto_fn(&from, &to, exact).unwrap();
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
                (*x, *y) = crypto_fn(*x, *y);
            });
        (lon_ref, lat_ref).into_pyobject(py)
    } else if let (Ok(lon), Ok(lat)) = (lon_py.extract::<f64>(py), lat_py.extract::<f64>(py)) {
        crypto_fn(lon, lat).into_pyobject(py)
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Input must be a float or a 1D numpy.ndarray of floats.",
        ))
    }
}
