use crate::nn_dtypes::{NNQuantizedPi, NNQuantizedValue};

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

