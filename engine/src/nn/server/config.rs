use std::time::Duration;

#[derive(Clone, Debug)]
pub struct ServerConfig {
    pub port: u16,
    pub game_size: usize,
    pub batcher: BatcherConfig,
    pub pipeline: PipelineConfig,
}

#[derive(Clone, Debug)]
pub struct PipelineConfig {
    /// Buffer size for channels feeding INTO inference (batcher→encoder, encoder→inference).
    /// Low values (1) create backpressure that forces larger batches → better GPU utilization.
    pub intake_buffer: usize,
    /// Buffer size for channels AFTER inference (inference→decoder, decoder→responder).
    /// Higher values prevent the GPU from stalling on slow downstream consumers.
    pub outtake_buffer: usize,
}

#[derive(Clone, Debug)]
pub struct BatcherConfig {
    /// Maximum number of requests in a single batch.
    ///
    /// ML-SPECIFIC: Set this to your model's optimal batch size. Larger batches
    /// = better GPU utilization, but more latency for early arrivals.
    pub max_batch_size: usize,

    /// Maximum time to wait for a full batch before sending an undersized one.
    ///
    /// This is your primary latency knob. A request arriving just after a batch
    /// fires will wait at most this long. Typical values: 1–10ms.
    pub max_wait: Duration,

    /// Buffer size for the internal channel that all streams feed into.
    /// Should be >= max concurrent in-flight requests across all workers.
    pub channel_buffer: usize,
}
