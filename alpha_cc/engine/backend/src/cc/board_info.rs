extern crate pyo3;
use pyo3::prelude::*;


#[pyclass]
pub struct BoardInfo {
    #[pyo3(get)]
    pub current_player: i8,
    #[pyo3(get)]
    pub winner: i8,
    #[pyo3(get)]
    pub size: usize,
    #[pyo3(get)]
    pub duration: u16,
    #[pyo3(get)]
    pub game_over: bool,
    #[pyo3(get)]
    pub reward: i8,
}