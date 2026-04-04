#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use pyo3::prelude::*;

use alpha_cc_core::{Board, BoardInfo, HexCoord, Move};
use alpha_cc_core::{create_move_mask, create_move_index_map};
use alpha_cc_nn::{NNPred, FetchStats, preds_from_logits, build_inference_request};
use alpha_cc_mcts::{MCTS, MCTSNode};

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
    m.add_class::<NNPred>()?;
    m.add_function(wrap_pyfunction!(preds_from_logits, m)?)?;
    m.add_function(wrap_pyfunction!(build_inference_request, m)?)?;
    m.add_class::<FetchStats>()?;
    Ok(())
}
