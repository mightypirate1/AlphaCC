#[cfg(feature = "extension-module")]
extern crate pyo3;

use crate::nn_dtypes::{NNQuantizedPi, NNQuantizedValue};

#[cfg_attr(feature = "extension-module", pyo3::prelude::pyclass(module="alpha_cc_engine", from_py_object))]
#[derive(Clone, bitcode::Encode, bitcode::Decode)]
pub struct NNPred {
    quant_pi: Vec<NNQuantizedPi>,
    quant_value: NNQuantizedValue,
}

impl NNPred {
    pub fn new(pi: Vec<f32>, value: f32) -> Self {
        NNPred {
            quant_pi: NNQuantizedPi::quantize_vec(&pi),
            quant_value: NNQuantizedValue::quantize(value),
        }
    }

    pub fn pi(&self) -> Vec<f32> {
        self.quant_pi.iter().map(|q| q.dequantize()).collect()
    }

    pub fn value(&self) -> f32 {
        self.quant_value.dequantize()
    }

    pub fn serialize(&self) -> Vec<u8> {
        bitcode::encode(self)
    }

    pub fn deserialize(data: &[u8]) -> NNPred {
        bitcode::decode(data)
            .unwrap_or_else(|e| {
                panic!("Failed to deserialize: {:?}", e);
            })
    }
}

#[cfg(feature = "extension-module")]
#[pyo3::prelude::pymethods]
impl NNPred {
    #[new]
    fn py_new(pi: Vec<f32>, value: f32) -> Self {
        NNPred::new(pi, value)
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
