use pyo3::prelude::*;
use pyo3::types::{PyTuple, PyBytes};
use pyo3_stub_gen_derive::{gen_stub_pyclass, gen_stub_pymethods};

use alpha_cc_core::Board;
use alpha_cc_core::cc::CCBoard;
use alpha_cc_nn::BoardEncoding;

use super::board_info::PyBoardInfo;
use super::game_move::PyMove;

#[gen_stub_pyclass]
#[pyclass(name = "Board", module = "alpha_cc_engine", from_py_object)]
#[derive(Clone)]
pub struct PyBoard(pub alpha_cc_core::cc::CCBoard);

impl From<alpha_cc_core::cc::CCBoard> for PyBoard {
    fn from(b: alpha_cc_core::cc::CCBoard) -> Self { PyBoard(b) }
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
                    return PyBoard(alpha_cc_core::cc::CCBoard::create(size))
                }
                panic!("expected a single int as input");
            },
            0 => {
                PyBoard(alpha_cc_core::cc::CCBoard::create(9))
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

    fn get_matrix(&self) -> alpha_cc_core::cc::CCBoardMatrix {
        self.0.get_matrix()
    }

    fn apply(&self, mv: &PyMove) -> PyBoard {
        PyBoard(self.0.apply(&mv.0))
    }

    fn get_unflipped_matrix(&self) -> alpha_cc_core::cc::CCBoardMatrix {
        self.0.get_unflipped_matrix()
    }

    fn render(&self) {
        self.0.render()
    }

    // ── BoardEncoding methods ──

    /// State tensor as a flat f32 array of shape (state_channels * s * s).
    /// Reshape to (state_channels, s, s) on the Python side.
    fn state_tensor<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<f32>> {
        let (s, _) = self.0.get_sizes();
        let s = s as usize;
        let mut buf = vec![0.0f32; CCBoard::STATE_CHANNELS * s * s];
        self.0.encode_state(&mut buf);
        numpy::IntoPyArray::into_pyarray(numpy::ndarray::Array1::from_vec(buf), py)
    }

    /// Flat policy indices for all legal moves: shape (n_moves,).
    /// Each entry is a flat index into a (policy_size,) tensor.
    fn policy_indices<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<u32>> {
        let (s, _) = self.0.get_sizes();
        let s = s as usize;
        let moves = self.0.legal_moves();
        let mut move_buf = vec![0u8; CCBoard::MOVE_BYTES];
        let indices: Vec<u32> = moves.iter().map(|mv| {
            CCBoard::encode_move(mv, &mut move_buf);
            CCBoard::move_to_policy_index(&move_buf, s) as u32
        }).collect();
        numpy::IntoPyArray::into_pyarray(numpy::ndarray::Array1::from_vec(indices), py)
    }

    /// Scatter a per-move value vector into a flat policy tensor of shape (policy_size,).
    /// `values` must have length == number of legal moves.
    fn scatter_policy<'py>(
        &self,
        py: Python<'py>,
        values: numpy::PyReadonlyArray1<'py, f32>,
    ) -> Bound<'py, numpy::PyArray1<f32>> {
        let (s, _) = self.0.get_sizes();
        let s = s as usize;
        let moves = self.0.legal_moves();
        let values_slice = values.as_slice().expect("values not contiguous");
        assert_eq!(values_slice.len(), moves.len(), "values length must match number of legal moves");

        let policy_size = CCBoard::policy_size(s);
        let mut policy = vec![0.0f32; policy_size];
        let mut move_buf = vec![0u8; CCBoard::MOVE_BYTES];
        for (mv, &val) in moves.iter().zip(values_slice) {
            CCBoard::encode_move(mv, &mut move_buf);
            let idx = CCBoard::move_to_policy_index(&move_buf, s);
            policy[idx] = val;
        }
        numpy::IntoPyArray::into_pyarray(numpy::ndarray::Array1::from_vec(policy), py)
    }

    /// Policy mask of shape (policy_size,) with 1 at legal move positions, 0 elsewhere.
    /// Use `.astype(bool)` or `.view(np.bool_)` in Python if a bool mask is needed.
    fn policy_mask<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<u8>> {
        let (s, _) = self.0.get_sizes();
        let s = s as usize;
        let moves = self.0.legal_moves();
        let policy_size = CCBoard::policy_size(s);
        let mut mask = vec![0u8; policy_size];
        let mut move_buf = vec![0u8; CCBoard::MOVE_BYTES];
        for mv in &moves {
            CCBoard::encode_move(mv, &mut move_buf);
            let idx = CCBoard::move_to_policy_index(&move_buf, s);
            mask[idx] = 1;
        }
        numpy::IntoPyArray::into_pyarray(numpy::ndarray::Array1::from_vec(mask), py)
    }

    fn __setstate__(&mut self, py: Python, state: Py<PyAny>) -> PyResult<()> {
        let py_bytes = state.extract::<Bound<'_, PyBytes>>(py)?;
        let bytes = py_bytes.as_bytes();
        self.0 = alpha_cc_core::cc::CCBoard::deserialize_rs(bytes);
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
