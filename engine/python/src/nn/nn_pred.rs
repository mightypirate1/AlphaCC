use pyo3::prelude::*;
use pyo3_stub_gen_derive::{gen_stub_pyclass, gen_stub_pymethods};

#[gen_stub_pyclass]
#[pyclass(name = "NNPred", module = "alpha_cc_engine", from_py_object)]
#[derive(Clone)]
pub struct PyNNPred(pub alpha_cc_nn::NNPred);

impl From<alpha_cc_nn::NNPred> for PyNNPred {
    fn from(p: alpha_cc_nn::NNPred) -> Self { PyNNPred(p) }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyNNPred {
    #[new]
    fn new(pi: Vec<f32>, value: f32) -> Self {
        PyNNPred(alpha_cc_nn::NNPred::new(pi, value))
    }

    #[getter]
    fn get_pi(&self) -> Vec<f32> {
        self.0.pi()
    }

    #[getter]
    fn get_value(&self) -> f32 {
        self.0.value()
    }

    fn __repr__(&self) -> String {
        format!("NNPred[val={}, pi={:?}]", self.0.value(), self.0.pi())
    }
}
