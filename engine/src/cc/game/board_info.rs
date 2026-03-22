#[cfg(feature = "extension-module")]
extern crate pyo3;

use crate::cc::dtypes;

#[cfg_attr(feature = "extension-module", pyo3::prelude::pyclass(module="alpha_cc_engine", get_all))]
pub struct BoardInfo {
    pub current_player: i8,
    pub winner: i8,
    pub size: dtypes::BoardSize,
    pub duration: dtypes::GameDuration,
    pub game_over: bool,
    pub reward: f32,
}


#[cfg(feature = "extension-module")]
#[pyo3::prelude::pymethods]
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