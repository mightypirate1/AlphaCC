use crate::nn_dtypes::{NNQuantizedPi, NNQuantizedWDL};

#[derive(Clone, bitcode::Encode, bitcode::Decode)]
pub struct NNPred {
    quant_pi: Vec<NNQuantizedPi>,
    quant_wdl: NNQuantizedWDL,
}

impl NNPred {
    pub fn new(pi: Vec<f32>, wdl: [f32; 3]) -> Self {
        NNPred {
            quant_pi: NNQuantizedPi::quantize_vec(&pi),
            quant_wdl: NNQuantizedWDL::quantize(wdl),
        }
    }

    pub fn pi(&self) -> Vec<f32> {
        self.quant_pi.iter().map(|q| q.dequantize()).collect()
    }

    pub fn wdl(&self) -> [f32; 3] {
        self.quant_wdl.dequantize()
    }

    pub fn quant_wdl(&self) -> NNQuantizedWDL {
        self.quant_wdl
    }

    /// Expected value = P(win) - P(loss), computed efficiently in quantized space.
    pub fn expected_value(&self) -> f32 {
        self.quant_wdl.expected_value()
    }

    /// Backward-compatible alias for expected_value().
    pub fn value(&self) -> f32 {
        self.expected_value()
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
