use pyo3::prelude::*;
use pyo3_stub_gen_derive::{gen_stub_pyclass, gen_stub_pymethods};

use alpha_cc_core::moves::find_all_moves;

use crate::core::{PyBoard, PyMove};

#[gen_stub_pyclass]
#[pyclass(name = "MCTSNode", module = "alpha_cc_engine")]
pub struct PyMCTSNode(pub alpha_cc_mcts::MCTSNode);

impl From<alpha_cc_mcts::MCTSNode> for PyMCTSNode {
    fn from(n: alpha_cc_mcts::MCTSNode) -> Self { PyMCTSNode(n) }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyMCTSNode {
    #[getter(n)]
    fn get_n_py(&self) -> Vec<u32> {
        (0..self.0.num_actions()).map(|a| self.0.get_n(a)).collect()
    }

    #[getter(q)]
    fn get_q_py(&self) -> Vec<f32> {
        (0..self.0.num_actions()).map(|a| self.0.get_q(a)).collect()
    }

    #[getter(pi)]
    fn get_pi_py(&self) -> Vec<f32> {
        alpha_cc_nn::NNQuantizedPi::dequantize_vec(&self.0.pi)
    }

    #[getter(v)]
    fn get_v_py(&self) -> f32 {
        self.0.get_v()
    }

    #[getter(blended_wdl)]
    fn get_blended_wdl_py(&self) -> [f32; 3] {
        let wdl = self.0.blended_wdl();
        [wdl.win, wdl.draw, wdl.loss]
    }

    /// Get moves for a board position. Not a getter — requires the board as argument.
    fn get_moves(&self, board: &PyBoard) -> Vec<PyMove> {
        find_all_moves(&board.0).into_iter().map(PyMove::from).collect()
    }
}
