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
    fn new(pi_logits: Vec<f32>, wdl_logits: [f32; 3]) -> Self {
        PyNNPred(alpha_cc_nn::NNPred::new(&pi_logits, wdl_logits))
    }

    #[getter]
    fn get_pi(&self) -> Vec<f32> {
        self.0.pi()
    }

    #[getter]
    fn get_pi_logits(&self) -> Vec<f32> {
        self.0.pi_logits()
    }

    #[getter]
    fn get_wdl(&self) -> Vec<f32> {
        self.0.wdl()
    }

    #[getter]
    fn get_wdl_logits(&self) -> [f32; 3] {
        self.0.wdl_logits()
    }

    #[getter]
    fn get_value(&self) -> f32 {
        self.0.expected_value()
    }

    fn __repr__(&self) -> String {
        let wdl = self.0.wdl();
        format!("NNPred[wdl=({:.3},{:.3},{:.3}), pi={:?}]", wdl[0], wdl[1], wdl[2], self.0.pi())
    }
}
