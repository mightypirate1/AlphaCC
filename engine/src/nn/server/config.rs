use std::time::Duration;

#[derive(Clone, Debug)]
pub struct ServerConfig {
    pub port: u16,
    pub game_size: usize,
    pub pipelines: Vec<PipelineChannelConfig>,
}

/// Configuration for a single pipeline channel (e.g., primary training or secondary tournament).
#[derive(Clone, Debug)]
pub struct PipelineChannelConfig {
    pub batcher: BatcherConfig,
    pub pipeline: PipelineConfig,
    /// Which model_id values this pipeline serves.
    pub model_ids: Vec<u32>,
    /// Batch size qualifier for the Redis weight key (e.g., Some(170) → `weights-0042-b170`).
    pub weight_batch_size: Option<usize>,
}

#[derive(Clone, Debug)]
pub struct PipelineConfig {
    /// Buffer size for channels feeding INTO inference (batcher→encoder, encoder→inference).
    /// Low values (1-2) create backpressure that forces larger batches → better GPU utilization.
    pub intake_buffer: usize,
    /// Buffer size for channels AFTER inference (inference→decoder, decoder→responder).
    /// Higher values prevent the GPU from stalling on slow downstream consumers.
    pub outtake_buffer: usize,
}

#[derive(Clone, Debug)]
pub struct BatcherConfig {
    /// Maximum number of requests in a single batch.
    pub max_batch_size: usize,

    /// Minimum wait time before flushing (floor for adaptive adjustment).
    pub min_wait: Duration,

    /// Maximum wait time before flushing (ceiling for adaptive adjustment).
    pub max_wait: Duration,

    /// Buffer size for the internal channel that all streams feed into.
    pub channel_buffer: usize,

    /// Rate for exponential wait adjustment (e.g., 0.05 = 5% per cycle).
    /// 0.0 disables adaptation (static wait at max_wait).
    pub adaptive_rate: f64,

    /// Zero-pad all batches to `max_batch_size`.
    pub pad_to_max: bool,

    /// Byte length of a single zero-pad item (2 * game_size^2 * sizeof(f32)).
    pub pad_item_len: usize,
}
