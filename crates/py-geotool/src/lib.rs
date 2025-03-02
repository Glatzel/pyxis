mod crypto;
mod misc;
mod space;
use pyo3::prelude::*;

#[pymodule]
fn py_geotool(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // crypto
    m.add_wrapped(wrap_pyfunction!(crypto::py_crypto))?;

    // geometry_coordinate
    m.add_wrapped(wrap_pyfunction!(space::py_cartesian_to_cylindrical))?;
    m.add_wrapped(wrap_pyfunction!(space::py_cartesian_to_spherical))?;
    m.add_wrapped(wrap_pyfunction!(space::py_cylindrical_to_cartesian))?;
    m.add_wrapped(wrap_pyfunction!(space::py_cylindrical_to_spherical))?;
    m.add_wrapped(wrap_pyfunction!(space::py_spherical_to_cartesian))?;
    m.add_wrapped(wrap_pyfunction!(space::py_spherical_to_cylindrical))?;

    // transform
    m.add_wrapped(wrap_pyfunction!(misc::py_datum_compense))?;
    m.add_wrapped(wrap_pyfunction!(misc::py_lbh2xyz))?;
    m.add_wrapped(wrap_pyfunction!(misc::py_xyz2lbh))?;

    Ok(())
}
