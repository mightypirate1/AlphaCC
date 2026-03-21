use std::time::Duration;

#[derive(Clone, Debug)]
pub struct ServerConfig {
    pub port: u16,
    pub game_size: usize,
    pub device: tch::Device,
    pub batcher: BatcherConfig,
    pub pipeline: PipelineConfig,
}

#[derive(Clone, Debug)]
pub struct PipelineConfig {
    pub buffer_size: usize,
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

pub fn detect_device() -> tch::Device {
    if tch::Cuda::is_available() {
        tch::Device::Cuda(0)
    } else {
        tch::Device::Cpu
    }
}
