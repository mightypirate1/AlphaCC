use bincode;
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
        match bincode::serialize(&data) {
            Ok(encoded) => encoded,
            Err(e) => {
                println!("error: {:?}", e);
                panic!("failed to serialize")
            }
        }
    }

    pub fn deserialize(data: Vec<u8>) -> NNPred {
        let decoded: (Vec<f32>, f32) = bincode::deserialize(&data).unwrap();
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
