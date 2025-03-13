mod crypto;
mod datum_compense;
mod gauss_projection;
mod space;
use pyo3::prelude::*;

#[pymodule]
fn pyxis_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // crypto
    m.add_wrapped(wrap_pyfunction!(crypto::py_crypto))?;

    // space
    m.add_wrapped(wrap_pyfunction!(space::py_space))?;

    m.add_wrapped(wrap_pyfunction!(datum_compense::py_datum_compense))?;

    m.add_wrapped(wrap_pyfunction!(gauss_projection::py_lbh2xyz))?;
    m.add_wrapped(wrap_pyfunction!(gauss_projection::py_xyz2lbh))?;

    Ok(())
}
