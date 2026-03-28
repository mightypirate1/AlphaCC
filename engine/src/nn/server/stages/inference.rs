use std::sync::Arc;
use tokio::sync::mpsc;

use crate::nn::backends::Backend;
use crate::nn::server::types::PipelineItem;


/// Runs the inference function on each batch and forwards the results
/// to the decoder stage. The moves side-channel passes through untouched.
pub async fn run_inference<B: Backend>(
    backend: Arc<B>,
    mut inference_rx: mpsc::Receiver<PipelineItem<B::Encoded>>,
    decoder_tx: mpsc::Sender<PipelineItem<B::Inferred>>,
) {
    while let Some(item) = inference_rx.recv().await {
        let backend = backend.clone();
        let out = tokio::task::spawn_blocking(move || {
            let output = backend.inference(item.model_id, item.payload);
            PipelineItem {
                model_id: item.model_id,
                replies: item.replies,
                moves: item.moves,
                payload: output,
            }
        }).await.unwrap();
        if decoder_tx.send(out).await.is_err() {
            return;  // Decoder is down!
        };
    }
}
