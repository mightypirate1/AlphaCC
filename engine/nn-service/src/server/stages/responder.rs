use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use tokio::sync::mpsc;

use crate::backends::Backend;
use crate::proto::PredictResponse;
use crate::server::types::PipelineItem;


/// Shared counters for throughput reporting.
struct Stats {
    predictions: AtomicU64,
    batches: AtomicU64,
}

/// Spawns a background task that prints throughput stats every `interval`.
/// Only prints when at least one prediction was served in the window.
/// Label shows which models are currently loaded for this pipeline's model_ids.
fn spawn_stats_printer<B: Backend>(stats: Arc<Stats>, interval: Duration, backend: Arc<B>, model_ids: Vec<u32>, current_wait_us: Arc<AtomicU64>) {
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(interval);
        ticker.tick().await; // skip first immediate tick
        loop {
            ticker.tick().await;
            let preds = stats.predictions.swap(0, Ordering::Relaxed);
            let batches = stats.batches.swap(0, Ordering::Relaxed);
            if batches == 0 {
                continue;
            }
            let loaded: Vec<String> = model_ids.iter()
                .filter_map(|&id| {
                    let guard = backend.model_store().load(id as usize);
                    guard.as_ref().as_ref().map(|vm| format!("{}:v{}", id, vm.version))
                })
                .collect();
            let label = if loaded.is_empty() { "none".to_string() } else { loaded.join(",") };
            let wait_us = current_wait_us.load(Ordering::Relaxed);
            let wait_ms = wait_us as f64 / 1000.0;
            let secs = interval.as_secs_f64();
            log::info!(
                "[nn-service:{label}] {:.0} preds/s  ({} preds in {} batches, avg batch={:.1}, wait={:.1}ms)",
                preds as f64 / secs,
                preds,
                batches,
                preds as f64 / batches as f64,
                wait_ms,
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
    model_ids: Vec<u32>,
    current_wait_us: Arc<AtomicU64>,
) {
    let stats = Arc::new(Stats {
        predictions: AtomicU64::new(0),
        batches: AtomicU64::new(0),
    });
    spawn_stats_printer(stats.clone(), Duration::from_secs(10), backend.clone(), model_ids, current_wait_us);

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
