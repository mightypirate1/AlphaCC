use std::sync::Mutex;

use rand::RngExt;

use crate::cc::game::board::Board;
use crate::cc::predictions::inference_utils::softmax;
use crate::cc::predictions::NNPred;
use crate::nn::client::PredictClient;
use crate::nn::io;

const WARN_THRESHOLD: usize = 5;

/// A synchronous prediction client.
///
/// Wraps the async `PredictClient` with a blocking interface. Retries
/// indefinitely on transport errors (e.g. nn-service closing the stream
/// during reload), reconnecting via DNS re-resolve on each attempt.
/// Warns if more than 5 retries are needed.
pub struct NNRemote {
    addr: String,
    rt: tokio::runtime::Runtime,
    client: Mutex<PredictClient>,
}

impl NNRemote {
    pub fn connect(addr: &str) -> Self {
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
        }
    }

    /// Predict policy + value for a board position.
    ///
    /// Encodes the board, sends it to the nn-service, decodes the response,
    /// and applies softmax. Retries indefinitely on error, reconnecting
    /// each time.
    pub fn predict(&self, board: &Board, model_id: u32) -> NNPred {
        let (state_tensor, moves) = io::encode_request(board);
        let mut attempt = 0usize;
        loop {
            let client = self.client.lock().unwrap().clone();
            let fut = client.predict(state_tensor.clone(), moves.clone(), model_id);
            let start = std::time::Instant::now();
            match self.rt.block_on(fut) {
                Ok(resp) => {
                    let elapsed = start.elapsed();
                    if elapsed.as_millis() > 50 {
                        eprintln!("[NNRemote] slow predict: {}ms", elapsed.as_millis());
                    }
                    if attempt >= WARN_THRESHOLD {
                        eprintln!("[NNRemote] recovered after {attempt} retries (model_id={model_id})");
                    }
                    let (logits, value) = io::decode_response(&resp);
                    let pi = softmax(&logits);
                    return NNPred::new(pi, value);
                }
                Err(e) => {
                    attempt += 1;
                    if attempt == WARN_THRESHOLD {
                        eprintln!(
                            "[NNRemote] warning: {attempt} consecutive failures (model_id={model_id}): {e}, still retrying {}...",
                            self.addr,
                        );
                    }
                    let jitter = rand::rng().random_range(10..=100);
                    std::thread::sleep(std::time::Duration::from_millis(jitter));
                    let _ = self.reconnect();
                }
            }
        }
    }

    /// Drop the current connection and establish a new one.
    ///
    /// DNS resolves fresh on each call, so this may connect to a different
    /// nn-service replica. Called automatically on prediction failure, and
    /// can be called explicitly to rebalance (e.g. between games).
    pub fn reconnect(&self) -> Result<(), String> {
        let new_client = self
            .rt
            .block_on(PredictClient::connect(&self.addr))
            .map_err(|e| e.to_string())?;
        *self.client.lock().unwrap() = new_client;
        Ok(())
    }
}
