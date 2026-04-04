use tokio::sync::oneshot;
use alpha_cc_nn::proto::{PredictRequest, PredictResponse};


pub struct PendingPrediction {
    pub request: PredictRequest,
    pub reply: oneshot::Sender<PredictResponse>,
}

impl PendingPrediction {
    pub fn new(request: PredictRequest, reply: oneshot::Sender<PredictResponse>) -> Self {
        Self { request, reply }
    }
}

pub type StateBytes = Vec<u8>;
pub type MovesBytes = Vec<u8>;

/// A reply destination: the request_id to echo back, plus the oneshot to send on.
pub type ReplyHandle = (u64, oneshot::Sender<PredictResponse>);

/// Data traveling through the pipeline stages.
/// Carries the reply handles and per-request move bytes alongside the stage payload.
/// The moves side-channel bypasses user closures and arrives at the responder.
pub struct PipelineItem<T> {
    pub model_id: u32,
    pub replies: Vec<ReplyHandle>,
    pub moves: Vec<MovesBytes>,
    pub payload: T,
}
