mod crypto;
mod space;
mod transform;
use pyo3::prelude::*;

#[pymodule]
fn py_geotool(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // crypto
    m.add_wrapped(wrap_pyfunction!(crypto::py_bd09_to_gcj02))?;
    m.add_wrapped(wrap_pyfunction!(crypto::py_bd09_to_wgs84))?;
    m.add_wrapped(wrap_pyfunction!(crypto::py_gcj02_to_bd09))?;
    m.add_wrapped(wrap_pyfunction!(crypto::py_gcj02_to_wgs84))?;
    m.add_wrapped(wrap_pyfunction!(crypto::py_wgs84_to_bd09))?;
    m.add_wrapped(wrap_pyfunction!(crypto::py_wgs84_to_gcj02))?;

    // geometry_coordinate
    m.add_wrapped(wrap_pyfunction!(space::py_cartesian_to_cylindrical))?;
    m.add_wrapped(wrap_pyfunction!(space::py_cartesian_to_spherical))?;
    m.add_wrapped(wrap_pyfunction!(space::py_cylindrical_to_cartesian))?;
    m.add_wrapped(wrap_pyfunction!(space::py_cylindrical_to_spherical))?;
    m.add_wrapped(wrap_pyfunction!(space::py_spherical_to_cartesian))?;
    m.add_wrapped(wrap_pyfunction!(space::py_spherical_to_cylindrical))?;

    // transform
    m.add_wrapped(wrap_pyfunction!(transform::py_datum_compense))?;
    m.add_wrapped(wrap_pyfunction!(transform::py_lbh2xyz))?;
    m.add_wrapped(wrap_pyfunction!(transform::py_xyz2lbh))?;

    Ok(())
}
