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


/// Collects requests, waits for the adaptive deadline, flushes one batch per
/// model_id, adjusts wait, repeats. Zero-pads to `fixed_batch_size` if set.
pub async fn run_batcher(
    config: BatcherConfig,
    mut rx: mpsc::Receiver<PendingPrediction>,
    prepared_tx: mpsc::Sender<PipelineItem<Vec<StateBytes>>>,
    current_wait_us: Arc<AtomicU64>,
) {
    let mut current_wait = config.max_wait;
    let scale_down = 1.0 - config.adaptive_rate;
    let scale_up = 1.0 + config.adaptive_rate * config.wait_upward_drift;

    loop {
        // Block until at least one request arrives.
        let first = match rx.recv().await {
            Some(item) => item,
            None => return,
        };
        let mut batch = Batch::new();
        let model_id = first.request.model_id;
        batch.push(first);

        // Accumulate until deadline or batch is full.
        let deadline = tokio::time::Instant::now() + current_wait;
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
        let (replies, moves, mut states) = batch.take();
        if let Some(fixed) = config.fixed_batch_size {
            states.resize_with(fixed, || vec![0u8; config.pad_item_len]);
        }
        let item = PipelineItem { model_id, replies, moves, payload: states };
        if prepared_tx.send(item).await.is_err() {
            return;
        }

        // Adaptive wait: drift up, push down when full.
        if config.adaptive_rate > 0.0 {
            if batch_len < config.max_batch_size {
                current_wait = duration_mul(current_wait, scale_up);
            } else {
                current_wait = duration_mul(current_wait, scale_down);
            }
            current_wait = current_wait.max(config.min_wait).min(config.max_wait);
            current_wait_us.store(current_wait.as_micros() as u64, Ordering::Relaxed);
        }
    }
}

fn duration_mul(d: Duration, factor: f64) -> Duration {
    Duration::from_secs_f64(d.as_secs_f64() * factor)
}
