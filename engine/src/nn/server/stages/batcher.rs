use std::collections::HashMap;

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


/// Drains the shared mpsc channel from all gRPC streams. Collects requests
/// into per-model batches using recv (blocking) + try_recv (greedy drain),
/// then forwards each batch to the encoder stage.
///
/// Requests are grouped by `model_id` so each model gets its own batch.
/// A batch is flushed immediately if it hits `max_batch_size`, or when
/// the deadline expires all non-empty batches are flushed.
pub async fn run_batcher(
    config: BatcherConfig,
    mut rx: mpsc::Receiver<PendingPrediction>,
    prepared_tx: mpsc::Sender<PipelineItem<Vec<StateBytes>>>,
) {
    let mut buckets: HashMap<u32, Batch> = HashMap::new();

    loop {
        buckets.clear();

        // Block until at least one request arrives.
        let first = match rx.recv().await {
            Some(item) => item,
            None => return, // All senders dropped — server shutting down.
        };
        let first_model_id = first.request.model_id;
        buckets.entry(first_model_id).or_insert_with(Batch::new).push(first);

        // Greedily drain everything available, then fall back to a short
        // timeout if no bucket is full yet. The try_recv loop is the
        // fast path under heavy load; the timeout catches the gap between
        // inference waves where workers are briefly idle before resubmitting.
        let deadline = tokio::time::Instant::now() + config.max_wait;
        loop {
            // Check if any bucket hit max_batch_size — flush it immediately.
            let full_ids: Vec<u32> = buckets.iter()
                .filter(|(_, b)| b.len() >= config.max_batch_size)
                .map(|(&id, _)| id)
                .collect();
            for model_id in full_ids {
                if let Some(mut batch) = buckets.remove(&model_id) {
                    let (replies, moves, states) = batch.take();
                    let item = PipelineItem { model_id, replies, moves, payload: states };
                    if prepared_tx.send(item).await.is_err() {
                        return;
                    }
                }
            }

            // Try to grab more requests.
            match rx.try_recv() {
                Ok(item) => {
                    let model_id = item.request.model_id;
                    buckets.entry(model_id).or_insert_with(Batch::new).push(item);
                }
                Err(_) => {
                    // Channel empty — wait briefly for more to arrive.
                    let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
                    if remaining.is_zero() {
                        break; // Deadline expired — flush all non-empty buckets.
                    }
                    match tokio::time::timeout(remaining, rx.recv()).await {
                        Ok(Some(item)) => {
                            let model_id = item.request.model_id;
                            buckets.entry(model_id).or_insert_with(Batch::new).push(item);
                        }
                        Ok(None) => return, // Channel closed.
                        Err(_) => break,    // Timeout — flush all non-empty buckets.
                    }
                }
            }
        }

        // Flush all remaining non-empty buckets.
        for (model_id, mut batch) in buckets.drain() {
            let (replies, moves, states) = batch.take();
            let item = PipelineItem { model_id, replies, moves, payload: states };
            if prepared_tx.send(item).await.is_err() {
                return;
            }
        }
    }
}
