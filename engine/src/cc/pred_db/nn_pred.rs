use bincode::{self, config::standard};
extern crate pyo3;
use pyo3::prelude::*;

use crate::cc::dtypes::{self, NNQuantizedPi, NNQuantizedValue};

#[pyclass(module="alpha_cc_engine")]
#[derive(Clone, bincode::Encode, bincode::BorrowDecode)]
pub struct NNPred {
    quant_pi: Vec<dtypes::NNQuantizedPi>,
    quant_value: dtypes::NNQuantizedValue,
}

impl NNPred {
        pub fn pi(&self) -> Vec<f32> {
        self.quant_pi.iter().map(|q| q.dequantize()).collect()
    }

    pub fn value(&self) -> f32 {
        self.quant_value.dequantize()
    }

    pub fn serialize(&self) -> Vec<u8> {
        bincode::encode_to_vec(self, standard())
            .unwrap_or_else(|e| {
                panic!("Failed to serialize: {:?}", e);
            })
    }

    pub fn deserialize(data: &[u8]) -> NNPred {
        bincode::borrow_decode_from_slice(data, standard())
            .unwrap_or_else(|e| {
                panic!("Failed to deserialize: {:?}", e);
            })
            .0
    }
}

#[pymethods]
impl NNPred {
    #[new]
    pub fn new(pi: Vec<f32>, value: f32) -> Self {
        NNPred {
            quant_pi: NNQuantizedPi::quantize_vec(&pi),
            quant_value: NNQuantizedValue::quantize(value),
        }
    }

    #[getter]
    fn get_pi(&self) -> Vec<f32> {
        self.quant_pi.iter().map(|qp|qp.dequantize()).collect()
    }

    #[getter]
    fn get_value(&self) -> f32 {
        self.quant_value.dequantize()
    }

    fn __repr__(&self) -> String {
        format!("NNPred[val={}, pi={:?}]", self.get_value(), self.get_pi())
    }
}
