use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use tokio::sync::mpsc;

use crate::nn::backends::Backend;
use crate::nn::proto::PredictResponse;
use crate::nn::server::types::PipelineItem;


/// Shared counters for throughput reporting.
struct Stats {
    predictions: AtomicU64,
    batches: AtomicU64,
}

/// Spawns a background task that prints throughput stats every `interval`.
fn spawn_stats_printer(stats: Arc<Stats>, interval: Duration) {
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(interval);
        ticker.tick().await; // skip first immediate tick
        loop {
            ticker.tick().await;
            let preds = stats.predictions.swap(0, Ordering::Relaxed);
            let batches = stats.batches.swap(0, Ordering::Relaxed);
            let secs = interval.as_secs_f64();
            eprintln!(
                "[nn-service] {:.0} preds/s  ({} preds in {} batches, avg batch={:.1})",
                preds as f64 / secs,
                preds,
                batches,
                if batches > 0 { preds as f64 / batches as f64 } else { 0.0 },
            );
        }
    });
}


/// Dispatches responses to workers one at a time. For each item in the
/// decoded batch, calls the backend's respond method (e.g. to apply
/// move masking), builds the `PredictResponse`, and sends it on the
/// oneshot channel immediately — so workers resume as soon as their
/// individual response is ready, not after the whole batch is processed.
pub async fn run_responder<B: Backend>(
    backend: Arc<B>,
    mut responder_rx: mpsc::Receiver<PipelineItem<Vec<(Vec<u8>, f32)>>>,
) {
    let stats = Arc::new(Stats {
        predictions: AtomicU64::new(0),
        batches: AtomicU64::new(0),
    });
    spawn_stats_printer(stats.clone(), Duration::from_secs(10));

    while let Some(item) = responder_rx.recv().await {
        let batch_size = item.replies.len() as u64;
        for (((request_id, reply_tx), (pi_bytes, value)), move_bytes) in
            item.replies.into_iter().zip(item.payload).zip(item.moves)
        {
            let (pi_logits, value) = backend.respond(pi_bytes, value, move_bytes);
            let response = PredictResponse { request_id, pi_logits, value };
            let _ = reply_tx.send(response);
        }
        stats.predictions.fetch_add(batch_size, Ordering::Relaxed);
        stats.batches.fetch_add(1, Ordering::Relaxed);
    }
}
