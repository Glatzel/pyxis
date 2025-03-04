mod crypto;
mod misc;
mod space;
use pyo3::prelude::*;

#[pymodule]
fn py_pyxis(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // crypto
    m.add_wrapped(wrap_pyfunction!(crypto::py_crypto))?;

    // space
    m.add_wrapped(wrap_pyfunction!(space::py_space))?;

    // transform
    m.add_wrapped(wrap_pyfunction!(misc::py_datum_compense))?;
    m.add_wrapped(wrap_pyfunction!(misc::py_lbh2xyz))?;
    m.add_wrapped(wrap_pyfunction!(misc::py_xyz2lbh))?;

    Ok(())
}
