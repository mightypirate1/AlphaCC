use pyo3::prelude::*;
use pyo3::types::{PyTuple, PyBytes};
use pyo3_stub_gen_derive::{gen_stub_pyclass, gen_stub_pymethods};

use super::board_info::PyBoardInfo;
use super::game_move::PyMove;

#[gen_stub_pyclass]
#[pyclass(name = "Board", module = "alpha_cc_engine", from_py_object)]
#[derive(Clone)]
pub struct PyBoard(pub alpha_cc_core::Board);

impl From<alpha_cc_core::Board> for PyBoard {
    fn from(b: alpha_cc_core::Board) -> Self { PyBoard(b) }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyBoard {
    #[new]
    #[pyo3(signature = (*py_args))]
    fn new(py_args: &Bound<'_, PyTuple>) -> Self {
        match py_args.len() {
            1 => {
                if let Ok(size) = py_args.get_item(0).unwrap().extract::<usize>() {
                    return PyBoard(alpha_cc_core::Board::create(size))
                }
                panic!("expected a single int as input");
            },
            0 => {
                PyBoard(alpha_cc_core::Board::create(9))
            },
            _ => { unreachable!() }
        }
    }

    #[getter]
    fn info(&self) -> PyBoardInfo {
        PyBoardInfo::from(self.0.get_info())
    }

    fn reset(&self) -> PyBoard {
        PyBoard(self.0.reset())
    }

    fn get_moves(&self) -> Vec<PyMove> {
        self.0.get_moves().into_iter().map(PyMove::from).collect()
    }

    fn get_next_states(&self) -> Vec<PyBoard> {
        self.0.get_next_states().into_iter().map(PyBoard::from).collect()
    }

    fn get_matrix(&self) -> alpha_cc_core::BoardMatrix {
        self.0.get_matrix()
    }

    fn apply(&self, mv: &PyMove) -> PyBoard {
        PyBoard(self.0.apply(&mv.0))
    }

    fn get_unflipped_matrix(&self) -> alpha_cc_core::BoardMatrix {
        self.0.get_unflipped_matrix()
    }

    fn render(&self) {
        self.0.render()
    }

    fn __setstate__(&mut self, py: Python, state: Py<PyAny>) -> PyResult<()> {
        let py_bytes = state.extract::<Bound<'_, PyBytes>>(py)?;
        let bytes = py_bytes.as_bytes();
        self.0 = alpha_cc_core::Board::deserialize_rs(bytes);
        Ok(())
    }

    fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        Ok(PyBytes::new(py, &self.0.serialize_rs()))
    }

    fn __hash__(&self) -> u64 {
        self.0.compute_hash()
    }

    fn __eq__(&self, other: &PyBoard) -> bool {
        self.0 == other.0
    }
}
