extern crate pyo3;
mod board;
mod moves;
mod hexcoordinate;
use pyo3::prelude::*;
use crate::board::Board;
use crate::board::BoardInfo;
use crate::hexcoordinate::HexCoordinate;
/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
#[pyo3(name = "alpha_cc_engine")]
fn alpha_cc(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Board>()?;
    m.add_class::<BoardInfo>()?;
    m.add_class::<HexCoordinate>()?;
    Ok(())
}
