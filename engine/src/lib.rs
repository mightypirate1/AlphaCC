pub mod cc;

use pyo3::prelude::*;
use crate::cc::{Board, BoardInfo, HexCoord, Move};
use crate::cc::{create_move_mask, create_move_index_map};
use crate::cc::rollouts::{MCTS, MCTSNode};
use crate::cc::pred_db::{NNPred, PredDBChannel};

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
    Ok(())
}
