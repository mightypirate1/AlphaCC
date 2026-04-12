use alpha_cc_core::Board;
use crate::nn_pred::NNPred;

/// Trait for anything that can provide neural network predictions for a board position.
/// Implemented by NNRemote (gRPC client) in nn-service, or by test mocks.
pub trait PredictionSource<B: Board>: Send + Sync {
    fn predict(&self, board: &B, model_id: u32) -> NNPred;
}

/// Stats about NN fetch latency. Used by Python bindings.
pub struct FetchStats {
    pub total_fetch_time_us: u64,
    pub total_fetches: u32,
}
