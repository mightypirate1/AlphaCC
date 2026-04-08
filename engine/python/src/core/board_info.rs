use pyo3::prelude::*;
use pyo3_stub_gen_derive::{gen_stub_pyclass, gen_stub_pymethods};

#[gen_stub_pyclass]
#[pyclass(name = "BoardInfo", module = "alpha_cc_engine", get_all)]
pub struct PyBoardInfo {
    pub current_player: i8,
    pub winner: i8,
    pub size: u8,
    pub duration: u16,
    pub game_over: bool,
    pub reward: f32,
    pub wdl: (f32, f32, f32),
}

impl From<alpha_cc_core::BoardInfo> for PyBoardInfo {
    fn from(bi: alpha_cc_core::BoardInfo) -> Self {
        PyBoardInfo {
            current_player: bi.current_player,
            winner: bi.winner,
            size: bi.size,
            duration: bi.duration,
            game_over: bi.game_over,
            reward: bi.wdl.to_value(),
            wdl: (bi.wdl.win, bi.wdl.draw, bi.wdl.loss),
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyBoardInfo {
    fn __repr__(&self) -> String {
        format!(
            "BoardInfo[\n  game_over: {}\n  current_player: {}\n  winner: {}\n  wdl: ({:.3}, {:.3}, {:.3})\n  reward: {:.3}\n  duration: {}\n  size: {}\n]",
            self.game_over,
            self.current_player,
            self.winner,
            self.wdl.0, self.wdl.1, self.wdl.2,
            self.reward,
            self.duration,
            self.size,
        )
    }
}
