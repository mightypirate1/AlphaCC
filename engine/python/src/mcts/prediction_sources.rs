use alpha_cc_core::cc::CCBoard;
use alpha_cc_nn::PredictionSource;

/// Unifies the two NN prediction backends the Python binding can use: a mock predictor
/// for dummy/testing runs, and a real gRPC client for the nn-service.
pub enum PredictionSources {
    Dummy(alpha_cc_nn::mock::MockPredictor),
    Real(alpha_cc_nn_service::NNRemote<CCBoard>),
}

impl PredictionSources {
    pub fn dummy() -> Self {
        Self::Dummy(alpha_cc_nn::mock::MockPredictor::uniform(0.0))
    }

    pub fn real(addr: &str) -> Self {
        let nn = alpha_cc_nn_service::NNRemote::connect(addr);
        Self::Real(nn)
    }
}

impl PredictionSource<CCBoard> for PredictionSources {
    fn predict(&self, board: &CCBoard, model_id: u32) -> alpha_cc_nn::NNPred {
        match self {
            Self::Dummy(dummy) => dummy.predict(board, model_id),
            Self::Real(nn_remote) => nn_remote.predict(board, model_id),
        }
    }
}

/// Build `n` prediction sources — real gRPC clients to `addr`, or mock predictors
/// if `dummy` is set.
pub fn build_services(addr: &str, n: usize, dummy: bool) -> Vec<PredictionSources> {
    let n = n.max(1);
    if dummy {
        (0..n).map(|_| PredictionSources::dummy()).collect()
    } else {
        (0..n).map(|_| PredictionSources::real(addr)).collect()
    }
}
