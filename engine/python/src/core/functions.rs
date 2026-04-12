use std::collections::HashMap;

use alpha_cc_core::cc::HexCoord;
use pyo3::prelude::*;
use pyo3_stub_gen_derive::gen_stub_pyfunction;

use alpha_cc_core::cc::MAX_SIZE;

use super::hexcoord::PyHexCoord;
use super::game_move::PyMove;

#[gen_stub_pyfunction]
#[pyfunction]
pub fn create_move_mask(moves: Vec<PyMove>) -> [[[[bool; MAX_SIZE]; MAX_SIZE]; MAX_SIZE]; MAX_SIZE] {
    let inner_moves: Vec<alpha_cc_core::Move<HexCoord>> = moves.into_iter().map(|m| m.0).collect();
    alpha_cc_core::cc::moves::create_move_mask(inner_moves)
}

#[gen_stub_pyfunction]
#[pyfunction]
pub fn create_move_index_map(moves: Vec<PyMove>) -> HashMap<usize, (PyHexCoord, PyHexCoord)> {
    let inner_moves: Vec<alpha_cc_core::Move<HexCoord>> = moves.into_iter().map(|m| m.0).collect();
    let inner_map = alpha_cc_core::cc::moves::create_move_index_map(inner_moves);
    inner_map.into_iter()
        .map(|(k, (a, b))| (k, (PyHexCoord(a), PyHexCoord(b))))
        .collect()
}
