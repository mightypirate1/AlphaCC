extern crate pyo3;
pub mod cc;

use pyo3::prelude::*;
use crate::cc::{Board, BoardInfo, HexCoord, Move};
use crate::cc::{create_move_mask, create_move_index_map};


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
    Ok(())
}
