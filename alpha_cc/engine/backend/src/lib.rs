extern crate pyo3;
mod board;
mod moves;
mod hexcoordinate;
use pyo3::prelude::*;
use crate::board::Board;
use crate::board::BoardInfo;
use crate::hexcoordinate::HexCoordinate;
use crate::moves::Move;


/// A Python module implemented in Rust.
#[pymodule]
#[pyo3(name = "alpha_cc_engine")]
fn alpha_cc(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<Board>()?;
    m.add_class::<BoardInfo>()?;
    m.add_class::<HexCoordinate>()?;
    m.add_class::<Move>()?;
    Ok(())
}
