#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use pyo3::prelude::*;
use pyo3_stub_gen::define_stub_info_gatherer;

pub mod core;
pub mod nn;
pub mod mcts;

use core::*;
use nn::*;
use mcts::*;

#[pymodule]
#[pyo3(name = "alpha_cc_engine")]
fn alpha_cc(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyBoard>()?;
    m.add_class::<PyBoardInfo>()?;
    m.add_class::<PyHexCoord>()?;
    m.add_class::<PyMove>()?;
    m.add_class::<PyMCTS>()?;
    m.add_class::<PyMCTSNode>()?;
    m.add_class::<PyRolloutResult>()?;
    m.add_class::<PyNNPred>()?;
    m.add_class::<PyFetchStats>()?;
    m.add_function(wrap_pyfunction!(create_move_mask, m)?)?;
    m.add_function(wrap_pyfunction!(create_move_index_map, m)?)?;
    m.add_function(wrap_pyfunction!(preds_from_logits, m)?)?;
    m.add_function(wrap_pyfunction!(build_inference_request, m)?)?;
    Ok(())
}

define_stub_info_gatherer!(stub_info);
