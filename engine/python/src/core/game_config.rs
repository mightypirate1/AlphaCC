use pyo3::prelude::*;
use pyo3_stub_gen_derive::{gen_stub_pyclass, gen_stub_pymethods};

use alpha_cc_nn::Game;

#[gen_stub_pyclass]
#[pyclass(name = "GameConfig", module = "alpha_cc_engine", from_py_object)]
#[derive(Clone)]
pub struct PyGameConfig(pub alpha_cc_nn::GameConfig);

#[gen_stub_pymethods]
#[pymethods]
impl PyGameConfig {
    /// Create a GameConfig from a game string like "cc:9", "cc:5", etc.
    #[new]
    fn new(game: &str) -> Self {
        Self(Game::parse(game).config())
    }

    #[getter]
    fn name(&self) -> &str {
        self.0.name
    }

    #[getter]
    fn board_size(&self) -> usize {
        self.0.board_size
    }

    #[getter]
    fn state_channels(&self) -> usize {
        self.0.state_channels
    }

    #[getter]
    fn policy_size(&self) -> usize {
        self.0.policy_size
    }

    #[getter]
    fn policy_shape(&self) -> Vec<usize> {
        self.0.policy_shape.clone()
    }

    #[getter]
    fn move_bytes(&self) -> usize {
        self.0.move_bytes
    }

    fn pad_item_len(&self) -> usize {
        self.0.pad_item_len()
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
}
