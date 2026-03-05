use std::io::Error;
use std::time::Instant;

use pyo3::prelude::*;

use crate::cc::pred_db::PredDBChannel;
use crate::cc::pred_db::NNPred;
use crate::cc::game::board::Board;

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

#[pyclass(module="alpha_cc_engine")]
pub struct FetchStats {
    #[pyo3(get)]
    pub total_fetch_time_us: u64,
    #[pyo3(get)]
    pub total_fetches: u32,
}

pub struct NNRemote {
    pred_db: PredDBChannel,
    stats: FetchStatsAccumulator,
}

impl NNRemote {
    pub fn new(pred_db: PredDBChannel) -> Self {
        NNRemote {
            pred_db,
            stats: FetchStatsAccumulator::default(),
        }
    }

    pub fn get_fetch_stats(&mut self) -> FetchStats {
        self.stats.snapshot_and_reset()
    }

    pub fn fetch_pred(&mut self, board: &Board) -> Result<NNPred, Error> {
        let start = Instant::now();
        self.pred_db.add_to_pred_queue(board);
        let nn_pred = self.pred_db.recv_pred()?;
        self.stats.record_fetch(start.elapsed());
        Ok(nn_pred)
    }
}
