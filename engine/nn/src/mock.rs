use alpha_cc_core::Board;
use alpha_cc_core::moves::find_all_moves;
use crate::nn_pred::NNPred;
use crate::prediction_source::PredictionSource;

/// A deterministic prediction source for testing.
///
/// Returns a uniform policy over legal moves and a fixed value.
/// Optionally biases the first legal move to have higher probability.
pub struct MockPredictor {
    value: f32,
    /// If > 0, the first move gets this fraction of the total probability.
    first_move_bias: f32,
}

impl MockPredictor {
    /// Uniform policy, fixed value.
    pub fn uniform(value: f32) -> Self {
        Self { value, first_move_bias: 0.0 }
    }

    /// Biased policy: first move gets `bias` fraction, rest is uniform.
    pub fn biased(value: f32, first_move_bias: f32) -> Self {
        Self { value, first_move_bias }
    }
}

impl PredictionSource for MockPredictor {
    fn predict(&self, board: &Board, _model_id: u32) -> NNPred {
        let n = find_all_moves(board).len();
        let pi = if n == 0 {
            vec![]
        } else if self.first_move_bias > 0.0 {
            let mut pi = vec![(1.0 - self.first_move_bias) / (n - 1).max(1) as f32; n];
            pi[0] = self.first_move_bias;
            pi
        } else {
            vec![1.0 / n as f32; n]
        };
        NNPred::new(pi, self.value)
    }
}
