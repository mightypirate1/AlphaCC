//! Prediction client that wraps a gRPC bidirectional stream.
//!
//! # Usage
//!
//! ```rust,no_run
//! use alpha_cc_nn_service::client::PredictClient;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let client = PredictClient::connect("http://[::1]:50051").await?;
//!
//! // Simple request/response — the stream is managed for you.
//! let response = client.predict(vec![1, 2, 3], vec![], 0).await?;
//! println!("value: {}", response.value);
//! # Ok(())
//! # }
//! ```
//!
//! # How it works
//!
//! On [`connect`](PredictClient::connect), the client:
//! 1. Opens a gRPC channel to the server.
//! 2. Starts a bidirectional stream (`Predict` RPC).
//! 3. Spawns a background task that reads responses from the stream and routes
//!    them to the correct caller via `request_id` → `oneshot` lookup.
//!
//! Each call to [`predict`](PredictClient::predict):
//! 1. Assigns a unique `request_id`.
//! 2. Sends the request on the outbound stream.
//! 3. Awaits the response on a oneshot channel.
//!
//! This means multiple tasks/threads can share one `PredictClient` and call
//! `predict` concurrently — they'll all multiplex over the same stream.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use futures::StreamExt;
use tokio::sync::{mpsc, oneshot, Mutex};
use tokio_stream::wrappers::ReceiverStream;

use crate::proto::prediction_service_client::PredictionServiceClient;
use crate::proto::{PredictRequest, PredictResponse};

/// Error type for prediction calls.
#[derive(Debug)]
pub enum PredictError {
    /// The gRPC connection or stream failed.
    Transport(String),
    /// The background reader task has exited (server disconnected).
    Disconnected,
}

impl std::fmt::Display for PredictError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PredictError::Transport(msg) => write!(f, "transport error: {msg}"),
            PredictError::Disconnected => write!(f, "prediction stream disconnected"),
        }
    }
}

impl std::error::Error for PredictError {}

/// A client handle for sending predictions to the server.
///
/// Clone-friendly: all clones share the same underlying stream.
#[derive(Clone)]
pub struct PredictClient {
    /// Sends requests into the outbound stream.
    req_tx: mpsc::Sender<PredictRequest>,

    /// Shared map of pending request_id → oneshot sender.
    /// The background reader task resolves these.
    pending: Arc<Mutex<HashMap<u64, oneshot::Sender<PredictResponse>>>>,

    /// Monotonic request ID counter.
    next_id: Arc<AtomicU64>,
}

impl PredictClient {
    /// Connect to the prediction server and open a bidirectional stream.
    ///
    /// This spawns a background tokio task to read responses. The task exits
    /// when the stream closes or the last `PredictClient` clone is dropped.
    pub async fn connect(addr: &str) -> Result<Self, PredictError> {
        let endpoint = tonic::transport::Endpoint::from_shared(addr.to_string())
            .map_err(|e| PredictError::Transport(e.to_string()))?
            .connect_timeout(std::time::Duration::from_secs(2));
        let mut grpc_client = PredictionServiceClient::connect(endpoint)
            .await
            .map_err(|e| PredictError::Transport(e.to_string()))?;

        // Channel for the outbound side of the bidi stream.
        let (req_tx, req_rx) = mpsc::channel::<PredictRequest>(64);

        // Open the bidi stream.
        let response_stream = grpc_client
            .predict(ReceiverStream::new(req_rx))
            .await
            .map_err(|e| PredictError::Transport(e.to_string()))?
            .into_inner();

        let pending: Arc<Mutex<HashMap<u64, oneshot::Sender<PredictResponse>>>> =
            Arc::new(Mutex::new(HashMap::new()));

        // Spawn background reader: reads responses and resolves pending oneshots.
        let pending_clone = pending.clone();
        tokio::spawn(async move {
            let mut stream = response_stream;
            while let Some(Ok(resp)) = stream.next().await {
                let mut map = pending_clone.lock().await;
                if let Some(tx) = map.remove(&resp.request_id) {
                    let _ = tx.send(resp);
                }
            }
            // Stream ended — cancel all pending requests.
            let mut map = pending_clone.lock().await;
            map.clear();
        });

        Ok(Self {
            req_tx,
            pending,
            next_id: Arc::new(AtomicU64::new(0)),
        })
    }

    /// Send a prediction request and wait for the response.
    ///
    /// `state` is the raw bytes of your input tensor (board state, features,
    /// etc). The server will batch this with other workers' requests for
    /// efficient inference.
    ///
    /// This method is safe to call concurrently from multiple tasks.
    pub async fn predict(&self, state_tensor: Vec<u8>, moves: Vec<u8>, model_id: u32) -> Result<PredictResponse, PredictError> {
        let request_id = self.next_id.fetch_add(1, Ordering::Relaxed);

        // Set up the reply channel BEFORE sending, to avoid races.
        let (reply_tx, reply_rx) = oneshot::channel();
        {
            let mut map = self.pending.lock().await;
            map.insert(request_id, reply_tx);
        }

        // Send the request on the outbound stream.
        let req = PredictRequest { request_id, state_tensor, moves, model_id };
        self.req_tx
            .send(req)
            .await
            .map_err(|_| PredictError::Disconnected)?;

        // Wait for the batcher to process it and send us the result.
        reply_rx.await.map_err(|_| PredictError::Disconnected)
    }
}
