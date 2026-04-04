use std::sync::Arc;

use tokio::sync::mpsc;

use crate::backends::Backend;
use crate::server::types::{PipelineItem, StateBytes};


/// Converts raw state bytes into an encoded batch for inference.
/// The moves side-channel passes through untouched.
pub async fn run_encoder<B: Backend>(
    backend: Arc<B>,
    mut encoder_rx: mpsc::Receiver<PipelineItem<Vec<StateBytes>>>,
    inference_tx: mpsc::Sender<PipelineItem<B::Encoded>>,
) {
    while let Some(item) = encoder_rx.recv().await {
        let backend = backend.clone();
        let out = tokio::task::spawn_blocking(move || {
            let encoded = backend.encode(item.payload);
            PipelineItem {
                model_id: item.model_id,
                replies: item.replies,
                moves: item.moves,
                payload: encoded,
            }
        }).await.unwrap();
        if inference_tx.send(out).await.is_err() {
            return; // Inference stage gone.
        }
    }
}
