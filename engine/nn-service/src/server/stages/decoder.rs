use std::sync::Arc;

use tokio::sync::mpsc;

use crate::backends::Backend;
use crate::server::types::PipelineItem;


/// Decodes inference output into per-request `(pi_logits_bytes, value)`
/// pairs. Runs the backend's decode on the whole batch, then forwards
/// the decoded items to the responder stage for per-item move masking and dispatch.
pub async fn run_decoder<B: Backend>(
    backend: Arc<B>,
    mut decoder_rx: mpsc::Receiver<PipelineItem<B::Inferred>>,
    responder_tx: mpsc::Sender<PipelineItem<Vec<crate::backends::DecodedPrediction>>>,
) {
    while let Some(item) = decoder_rx.recv().await {
        let backend = backend.clone();
        let out = tokio::task::spawn_blocking(move || {
            let decoded = backend.decode(item.payload);
            PipelineItem {
                model_id: item.model_id,
                replies: item.replies,
                moves: item.moves,
                payload: decoded,
            }
        }).await.unwrap();
        if responder_tx.send(out).await.is_err() {
            return; // Responder is down.
        }
    }
}
