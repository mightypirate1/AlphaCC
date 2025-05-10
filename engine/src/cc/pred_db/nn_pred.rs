use bincode::{self, config::standard};
extern crate pyo3;
use pyo3::prelude::*;


#[pyclass(module="alpha_cc_engine")]
#[derive(Clone, bincode::Encode, bincode::Decode)]
pub struct NNPred {
    pub pi: Vec<f32>,
    pub value: f32,
}

impl NNPred {
    pub fn serialize(&self) -> Vec<u8> {
        bincode::encode_to_vec(self, standard())
            .unwrap_or_else(|e| {
                panic!("Failed to serialize: {:?}", e);
            })
    }

    pub fn deserialize(data: &[u8]) -> NNPred {
        bincode::decode_from_slice(data, standard())
            .unwrap_or_else(|e| {
                panic!("Failed to deserialize: {:?}", e);
            })
            .0
    }
}

#[pymethods]
impl NNPred {
    #[new]
    fn new(pi: Vec<f32>, value: f32) -> Self {
        NNPred { pi, value }
    }

    #[getter]
    fn get_pi(&self) -> Vec<f32> {
        self.pi.clone()
    }

    #[getter]
    fn get_value(&self) -> f32 {
        self.value
    }
}
