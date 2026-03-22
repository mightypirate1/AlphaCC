use std::sync::Mutex;
use std::time::Duration;

use crate::cc::game::board::Board;
use crate::cc::predictions::inference_utils::softmax;
use crate::cc::predictions::NNPred;
use crate::nn::client::PredictClient;
use crate::nn::io;

const MAX_RETRIES: usize = 5;

/// A synchronous, resilient prediction service.
///
/// Wraps the async `PredictClient` with a blocking interface, automatic
/// timeout with exponential backoff, and transparent reconnection. MCTS
/// threads call `predict()` without knowing about gRPC, tokio, encoding,
/// or connection management.
pub struct NNRemote {
    addr: String,
    rt: tokio::runtime::Runtime,
    client: Mutex<PredictClient>,
    base_timeout: Duration,
}

impl NNRemote {
    pub fn connect(addr: &str, timeout: Duration) -> Self {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("failed to create tokio runtime");
        let client = rt
            .block_on(PredictClient::connect(addr))
            .unwrap_or_else(|e| panic!("failed to connect to nn-service at {addr}: {e}"));
        Self {
            addr: addr.to_string(),
            rt,
            client: Mutex::new(client),
            base_timeout: timeout,
        }
    }

    /// Predict policy + value for a board position.
    ///
    /// Encodes the board, sends it to the nn-service, decodes the response,
    /// and applies softmax. On timeout or disconnect, reconnects with
    /// exponential backoff and retries.
    pub fn predict(&self, board: &Board, model_id: u32) -> Result<NNPred, PredictionError> {
        let (state_tensor, moves) = io::encode_request(board);
        for attempt in 0..MAX_RETRIES {
            let timeout = self.base_timeout * 2u32.pow(attempt as u32);
            match self.try_predict(state_tensor.clone(), moves.clone(), model_id, timeout) {
                Ok(pred) => return Ok(pred),
                Err(e) => {
                    eprintln!(
                        "[NNRemote] attempt {}/{MAX_RETRIES}: {e} (timeout={timeout:?}), reconnecting to {}...",
                        attempt + 1, self.addr,
                    );
                    if let Err(re) = self.reconnect() {
                        eprintln!("[NNRemote] reconnect failed: {re}");
                    }
                }
            }
        }
        Err(PredictionError::Transport(
            format!("all {MAX_RETRIES} attempts failed for {}", self.addr),
        ))
    }

    /// Drop the current connection and establish a new one.
    ///
    /// DNS resolves fresh on each call, so this may connect to a different
    /// nn-service replica. Called automatically on prediction failure, and
    /// can be called explicitly to rebalance (e.g. between games).
    pub fn reconnect(&self) -> Result<(), PredictionError> {
        let new_client = self
            .rt
            .block_on(PredictClient::connect(&self.addr))
            .map_err(|e| PredictionError::Transport(e.to_string()))?;
        *self.client.lock().unwrap() = new_client;
        Ok(())
    }

    fn try_predict(
        &self,
        state_tensor: Vec<u8>,
        moves: Vec<u8>,
        model_id: u32,
        timeout: Duration,
    ) -> Result<NNPred, PredictionError> {
        let client = self.client.lock().unwrap().clone();
        let fut = client.predict(state_tensor, moves, model_id);
        let resp = self
            .rt
            .block_on(async { tokio::time::timeout(timeout, fut).await })
            .map_err(|_| PredictionError::Timeout)?
            .map_err(|e| PredictionError::Transport(e.to_string()))?;
        let (logits, value) = io::decode_response(&resp);
        let pi = softmax(&logits);
        Ok(NNPred::new(pi, value))
    }
}

#[derive(Debug)]
pub enum PredictionError {
    Timeout,
    Transport(String),
}

impl std::fmt::Display for PredictionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PredictionError::Timeout => write!(f, "prediction timed out"),
            PredictionError::Transport(msg) => write!(f, "transport error: {msg}"),
        }
    }
}
