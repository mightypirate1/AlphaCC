use std::io::Error;
use std::time::Instant;

use alpha_cc_core::Board;
use crate::nn_pred::NNPred;
use crate::client::PredictClient;
use crate::io;

use crate::inference_utils::softmax;

#[derive(Default)]
pub struct FetchStatsAccumulator {
    total_fetch_time_us: u64,
    total_fetches: u32,
}

impl FetchStatsAccumulator {
    fn record_fetch(&mut self, elapsed: std::time::Duration) {
        self.total_fetch_time_us += elapsed.as_micros() as u64;
        self.total_fetches += 1;
    }

    pub fn snapshot_and_reset(&mut self) -> FetchStats {
        let stats = FetchStats {
            total_fetch_time_us: self.total_fetch_time_us,
            total_fetches: self.total_fetches,
        };
        *self = FetchStatsAccumulator::default();
        stats
    }
}

#[cfg_attr(feature = "extension-module", pyo3::prelude::pyclass(module="alpha_cc_engine", get_all))]
pub struct FetchStats {
    pub total_fetch_time_us: u64,
    pub total_fetches: u32,
}

pub struct PredictionClient {
    rt: tokio::runtime::Runtime,
    client: PredictClient,
    model_id: u32,
    stats: FetchStatsAccumulator,
}

impl PredictionClient {
    pub fn new(addr: &str, model_id: u32) -> Self {
        let rt = tokio::runtime::Runtime::new()
            .expect("failed to create tokio runtime for PredictionClient");
        let client = rt.block_on(PredictClient::connect(addr))
            .unwrap_or_else(|e| panic!("failed to connect to nn-service at {addr}: {e}"));
        Self {
            rt,
            client,
            model_id,
            stats: FetchStatsAccumulator::default(),
        }
    }

    pub fn fetch_pred(&mut self, board: &Board) -> Result<NNPred, Error> {
        let start = Instant::now();

        let (state_tensor, moves) = io::encode_request(board);
        let resp = self.rt
            .block_on(self.client.predict(state_tensor, moves, self.model_id))
            .map_err(|e| Error::other(format!("prediction failed: {e}")))?;
        let (logits, value) = io::decode_response(&resp);
        let pi = softmax(&logits);

        self.stats.record_fetch(start.elapsed());
        Ok(NNPred::new(pi, value))
    }

    pub fn get_fetch_stats(&mut self) -> FetchStats {
        self.stats.snapshot_and_reset()
    }
}
