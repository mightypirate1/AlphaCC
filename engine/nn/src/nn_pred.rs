use crate::inference_utils::softmax;

#[derive(Clone, bitcode::Encode, bitcode::Decode)]
pub struct NNPred {
    pi_logits: Vec<f32>,
    wdl_logits: [f32; 3],
    expected_value: f32,
}

impl NNPred {
    pub fn new(pi_logits: &[f32], wdl_logits: [f32; 3]) -> Self {
        let wdl = softmax(&wdl_logits);
        NNPred {
            pi_logits: pi_logits.into(),
            wdl_logits,
            expected_value: wdl[0] - wdl[2],
        }
    }

    pub fn pi(&self) -> Vec<f32> {
        softmax(&self.pi_logits)
    }

    pub fn wdl(&self) -> Vec<f32> {
        softmax(&self.wdl_logits)
    }

    pub fn pi_logits(&self) -> Vec<f32> {
        self.pi_logits.clone()
    }

    pub fn wdl_logits(&self) -> [f32; 3] {
        self.wdl_logits
    }

    pub fn expected_value(&self) -> f32 {
        self.expected_value
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
