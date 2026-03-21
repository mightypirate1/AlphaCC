mod backend;
pub mod encoder;
pub mod inference;
pub mod decoder;

use pyo3::prelude::*;

pub type PyTensor = Py<PyAny>;
pub type PyModel = Py<PyAny>;

pub use backend::PyTorchBackend;
