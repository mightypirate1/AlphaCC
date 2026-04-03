pub mod cc;
pub mod db;
pub mod nn;

#[cfg(feature = "tui")]
pub mod tui;

#[cfg(feature = "extension-module")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[cfg(feature = "extension-module")]
use pyo3::prelude::*;
#[cfg(feature = "extension-module")]
use crate::cc::{Board, BoardInfo, HexCoord, Move};
#[cfg(feature = "extension-module")]
use crate::cc::{create_move_mask, create_move_index_map};
#[cfg(feature = "extension-module")]
use crate::cc::rollouts::{MCTS, MCTSNode, FetchStats};
#[cfg(feature = "extension-module")]
use crate::cc::predictions::{NNPred, preds_from_logits, build_inference_request};

/// A Python module implemented in Rust.
#[cfg(feature = "extension-module")]
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
