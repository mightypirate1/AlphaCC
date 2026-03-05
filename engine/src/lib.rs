pub mod cc;

use pyo3::prelude::*;
use crate::cc::{Board, BoardInfo, HexCoord, Move};
use crate::cc::{create_move_mask, create_move_index_map};
use crate::cc::rollouts::{MCTS, MCTSNode, FetchStats};
use crate::cc::pred_db::{NNPred, PredDBChannel, InferenceBatch, preds_from_logits, enqueue_responses, build_inference_request, fetch_and_build_tensor};

/// A Python module implemented in Rust.
#[pymodule]
#[pyo3(name = "alpha_cc_engine")]
fn alpha_cc(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<Board>()?;
    m.add_class::<BoardInfo>()?;
    m.add_class::<HexCoord>()?;
    m.add_class::<Move>()?;
    m.add_function(wrap_pyfunction!(create_move_mask, m)?)?;
    m.add_function(wrap_pyfunction!(create_move_index_map, m)?)?;
    m.add_class::<MCTS>()?;
    m.add_class::<MCTSNode>()?;
    m.add_class::<PredDBChannel>()?;
    m.add_class::<NNPred>()?;
    m.add_class::<InferenceBatch>()?;
    m.add_function(wrap_pyfunction!(preds_from_logits, m)?)?;
    m.add_function(wrap_pyfunction!(enqueue_responses, m)?)?;
    m.add_function(wrap_pyfunction!(build_inference_request, m)?)?;
    m.add_function(wrap_pyfunction!(fetch_and_build_tensor, m)?)?;
    m.add_class::<FetchStats>()?;
    Ok(())
}
