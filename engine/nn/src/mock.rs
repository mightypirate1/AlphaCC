use alpha_cc_core::Board;
use crate::nn_pred::NNPred;
use crate::prediction_source::PredictionSource;

/// A deterministic prediction source for testing.
///
/// Returns a uniform policy over legal moves and a fixed WDL.
/// Optionally biases the first legal move to have higher probability.
pub struct MockPredictor {
    wdl: [f32; 3],
    /// If > 0, the first move gets this fraction of the total probability.
    first_move_bias: f32,
}

impl MockPredictor {
    /// Uniform policy, fixed expected value. WDL derived as [w, 0, l] where w-l = value.
    pub fn uniform(value: f32) -> Self {
        let v = value.clamp(-1.0, 1.0);
        let wdl = [(1.0 + v) / 2.0, 0.0, (1.0 - v) / 2.0];
        Self { wdl, first_move_bias: 0.0 }
    }

    /// Biased policy: first move gets `bias` fraction, rest is uniform.
    pub fn biased(value: f32, first_move_bias: f32) -> Self {
        let v = value.clamp(-1.0, 1.0);
        let wdl = [(1.0 + v) / 2.0, 0.0, (1.0 - v) / 2.0];
        Self { wdl, first_move_bias }
    }

    /// Explicit WDL values.
    pub fn with_wdl(wdl: [f32; 3], first_move_bias: f32) -> Self {
        Self { wdl, first_move_bias }
    }
}

impl<B: Board> PredictionSource<B> for MockPredictor {
    fn predict(&self, board: &B, _model_id: u32) -> NNPred {
        let n = board.legal_moves().len();
        let pi = if n == 0 {
            vec![]
        } else if self.first_move_bias > 0.0 {
            let mut pi = vec![(1.0 - self.first_move_bias) / (n - 1).max(1) as f32; n];
            pi[0] = self.first_move_bias;
            pi
        } else {
            vec![1.0 / n as f32; n]
        };
        let pi_logits: Vec<f32> = pi.iter().map(|pia| pia.ln()).collect();
        let wdl_logits = self.wdl.map(|w| w.ln());
        NNPred::new(&pi_logits, wdl_logits)
    }
}
