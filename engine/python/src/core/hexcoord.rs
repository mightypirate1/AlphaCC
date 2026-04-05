use pyo3::prelude::*;
use pyo3_stub_gen_derive::{gen_stub_pyclass, gen_stub_pymethods};

#[gen_stub_pyclass]
#[pyclass(name = "HexCoord", module = "alpha_cc_engine", from_py_object)]
#[derive(Clone)]
pub struct PyHexCoord(pub alpha_cc_core::HexCoord);

impl From<alpha_cc_core::HexCoord> for PyHexCoord {
    fn from(h: alpha_cc_core::HexCoord) -> Self { PyHexCoord(h) }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyHexCoord {
    fn get_all_neighbours(&self, distance: usize) -> Vec<PyHexCoord> {
        self.0.get_all_neighbours(distance)
            .into_iter().map(PyHexCoord::from).collect()
    }

    #[getter]
    fn get_x(&self) -> u8 {
        self.0.x
    }

    #[getter]
    fn get_y(&self) -> u8 {
        self.0.y
    }

    fn flip(&self) -> PyHexCoord {
        PyHexCoord(self.0.flip())
    }

    fn repr(&self) -> String {
        self.0.repr()
    }

    fn __repr__(&self) -> String {
        self.0.repr()
    }
}
