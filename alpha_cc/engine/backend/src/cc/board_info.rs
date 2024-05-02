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


#[pymethods]
impl BoardInfo {
    pub fn __repr__(&self) -> String {
        format!(
            "BoardInfo[\n  game_over: {}\n  current_player: {}\n  winner: {}\n  reward: {}\n  duration: {}\n  size: {}\n]",
            self.game_over,
            self.current_player,
            self.winner,
            self.reward,
            self.duration,
            self.size,
        )
    }
}