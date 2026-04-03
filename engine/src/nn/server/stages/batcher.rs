use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use tokio::sync::mpsc;
use crate::nn::server::config::BatcherConfig;
use crate::nn::server::types::{PipelineItem, PendingPrediction, ReplyHandle, StateBytes, MovesBytes};


struct Batch {
    replies: Vec<ReplyHandle>,
    moves: Vec<MovesBytes>,
    states: Vec<StateBytes>,
}

impl Batch {
    fn new() -> Self {
        Self {
            replies: Vec::new(),
            moves: Vec::new(),
            states: Vec::new(),
        }
    }

    fn push(&mut self, item: PendingPrediction) {
        self.replies.push((item.request.request_id, item.reply));
        self.moves.push(item.request.moves);
        self.states.push(item.request.state_tensor);
    }

    fn len(&self) -> usize {
        self.states.len()
    }

    fn take(&mut self) -> (Vec<ReplyHandle>, Vec<MovesBytes>, Vec<StateBytes>) {
        (
            std::mem::take(&mut self.replies),
            std::mem::take(&mut self.moves),
            std::mem::take(&mut self.states),
        )
    }
}


/// Collects requests, waits for the adaptive deadline, flushes one batch,
/// adjusts wait, repeats. Zero-pads to `max_batch_size` if `pad_to_max` is set.
pub async fn run_batcher(
    config: BatcherConfig,
    mut rx: mpsc::Receiver<PendingPrediction>,
    prepared_tx: mpsc::Sender<PipelineItem<Vec<StateBytes>>>,
    current_wait_us: Arc<AtomicU64>,
) {
    assert!(config.half_life > 0.0, "half_life must be > 0");
    let mut current_wait = config.min_wait;
    current_wait_us.store(current_wait.as_micros() as u64, Ordering::Relaxed);

    loop {
        // Time the full cycle (including wait for first sample) for alpha computation.
        let cycle_start = tokio::time::Instant::now();

        // Block until at least one request arrives.
        let first = match rx.recv().await {
            Some(item) => item,
            None => return,
        };
        let mut batch = Batch::new();
        let model_id = first.request.model_id;
        batch.push(first);

        // Accumulate until deadline or batch is full.
        // elapsed (for fill rate estimate) starts AFTER the first sample.
        let batch_start = tokio::time::Instant::now();
        let deadline = batch_start + current_wait;
        while batch.len() < config.max_batch_size {
            match rx.try_recv() {
                Ok(item) => batch.push(item),
                Err(_) => {
                    let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
                    if remaining.is_zero() {
                        break;
                    }
                    match tokio::time::timeout(remaining, rx.recv()).await {
                        Ok(Some(item)) => batch.push(item),
                        Ok(None) => return,
                        Err(_) => break,
                    }
                }
            }
        }

        // Flush.
        let batch_len = batch.len();
        let elapsed = batch_start.elapsed();
        let cycle_duration = cycle_start.elapsed().as_secs_f64(); // capture before send — excludes pipeline backpressure
        let (replies, moves, mut states) = batch.take();
        if config.pad_to_max {
            states.resize_with(config.max_batch_size, || vec![0u8; config.pad_item_len]);
        }
        let item = PipelineItem { model_id, replies, moves, payload: states };
        if prepared_tx.send(item).await.is_err() {
            return;
        }

        // Adaptive wait adjustment.
        // Fill rate estimate: elapsed covers samples 2..N, so N-1 intervals.
        let effective_count = (batch_len - 1).max(1) as f64;
        let inv_fill_rate = (config.max_batch_size as f64 - 1.0) / effective_count;
        let estimated_optimal_wait = elapsed.mul_f64(inv_fill_rate);
        let alpha = (1.0 - (0.5_f64).powf(cycle_duration / config.half_life)).clamp(0.0, 1.0);

        current_wait = alpha_blend_duration(current_wait, estimated_optimal_wait, alpha);
        current_wait = current_wait.max(config.min_wait).min(config.max_wait);
        current_wait_us.store(current_wait.as_micros() as u64, Ordering::Relaxed);
    }
}

fn alpha_blend_duration(current: Duration, estimated: Duration, alpha: f64) -> Duration {
    current.mul_f64(1.0 - alpha) + estimated.mul_f64(alpha)
}
