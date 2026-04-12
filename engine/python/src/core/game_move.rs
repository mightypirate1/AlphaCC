use pyo3::prelude::*;
use pyo3_stub_gen_derive::{gen_stub_pyclass, gen_stub_pymethods};

use alpha_cc_core::cc::HexCoord;

use super::hexcoord::PyHexCoord;

#[gen_stub_pyclass]
#[pyclass(name = "Move", module = "alpha_cc_engine", from_py_object)]
#[derive(Clone)]
pub struct PyMove(pub alpha_cc_core::Move<HexCoord>);

impl From<alpha_cc_core::Move<HexCoord>> for PyMove {
    fn from(m: alpha_cc_core::Move<HexCoord>) -> Self { PyMove(m) }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyMove {
    #[getter]
    #[allow(clippy::wrong_self_convention)]
    fn from_coord(&self) -> PyHexCoord {
        PyHexCoord(self.0.from_coord)
    }

    #[getter]
    fn to_coord(&self) -> PyHexCoord {
        PyHexCoord(self.0.to_coord)
    }

    fn __repr__(&self) -> String {
        let fc = self.0.from_coord;
        let tc = self.0.to_coord;
        format!("Move[{} -> {}]", fc.repr(), tc.repr())
    }
}
