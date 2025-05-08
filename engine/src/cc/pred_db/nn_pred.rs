use bincode::{self, config::standard};
extern crate pyo3;
use pyo3::prelude::*;


#[pyclass(module="alpha_cc_engine")]
#[derive(Clone)]
pub struct NNPred {
    pub pi: Vec<f32>,
    pub value: f32,
}

impl NNPred {
    pub fn serialize(&self) -> Vec<u8> {
        let data = (self.pi.clone(), self.value);
        match bincode::encode_to_vec(&data, standard()) {
            Ok(encoded) => encoded,
            Err(e) => {
                println!("error: {:?}", e);
                panic!("failed to serialize")
            }
        }
    }

    pub fn deserialize(data: &[u8]) -> NNPred {
        let decoded: (Vec<f32>, f32) = match bincode::decode_from_slice(data, standard()) {
            Ok((decoded, _)) => decoded,
            Err(e) => {
                eprintln!("Deserialization error: {:?}", e);
                panic!("Failed to deserialize");
            }
        };
        NNPred::new(decoded.0, decoded.1)
    }
}

#[pymethods]
impl NNPred {
    #[new]
    fn new(pi: Vec<f32>, value: f32) -> Self {
        NNPred { pi, value }
    }
}
