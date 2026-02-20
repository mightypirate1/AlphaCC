use core::time::Duration;
use std::io::{Error, ErrorKind};
use std::time::Instant;

use pyo3::prelude::*;

use crate::cc::pred_db::PredDBChannel;
use crate::cc::pred_db::NNPred;
use crate::cc::game::board::Board;

const INITIAL_PATIENCE: Duration = Duration::from_millis(5);
const REPOST_THRESHOLD: Duration = Duration::from_millis(100);
const WARNING_THRESHOLD: Duration = Duration::from_millis(1000);
const FAIL_THRESHOLD: Duration = Duration::from_millis(10000);
const BACKOFF_SCALING: f32 = 2.0;
const PATIENCE_SCALE_UP: f32 = 1.02;
const PATIENCE_SCALE_DOWN: f32 = 0.99;

const MAX_ATTEMPTS: usize = 16;

#[derive(Default)]
pub struct FetchStatsAccumulator {
    resolved_at_attempt: [u32; MAX_ATTEMPTS],
    attempt_total_wait_us: [u64; MAX_ATTEMPTS],
    timeouts: u32,
    total_gets: u32,
    total_misses: u32,
    total_fetch_time_us: u64,
    total_fetches: u32,
}

impl FetchStatsAccumulator {

    fn record_get_hit(&mut self) {
        self.total_gets += 1;
    }

    fn record_get_miss(&mut self) {
        self.total_gets += 1;
        self.total_misses += 1;
    }

    fn record_resolved(&mut self, attempt: usize, elapsed: Duration) {
        let attempt = attempt.min(MAX_ATTEMPTS - 1);
        self.resolved_at_attempt[attempt] += 1;
        self.attempt_total_wait_us[attempt] += elapsed.as_micros() as u64;
        self.total_fetch_time_us += elapsed.as_micros() as u64;
        self.total_fetches += 1;
    }

    fn record_timeout(&mut self, elapsed: Duration) {
        self.timeouts += 1;
        self.total_fetch_time_us += elapsed.as_micros() as u64;
        self.total_fetches += 1;
    }

    pub fn snapshot_and_reset(&mut self, patience: Duration) -> FetchStats {
        let stats = FetchStats {
            resolved_at_attempt: self.resolved_at_attempt.to_vec(),
            attempt_total_wait_us: self.attempt_total_wait_us.to_vec(),
            timeouts: self.timeouts,
            total_gets: self.total_gets,
            total_misses: self.total_misses,
            total_fetch_time_us: self.total_fetch_time_us,
            total_fetches: self.total_fetches,
            current_patience_us: patience.as_micros() as u64,
        };
        *self = FetchStatsAccumulator::default();
        stats
    }
}

#[pyclass(module="alpha_cc_engine")]
pub struct FetchStats {
    #[pyo3(get)]
    pub resolved_at_attempt: Vec<u32>,
    #[pyo3(get)]
    pub attempt_total_wait_us: Vec<u64>,
    #[pyo3(get)]
    pub timeouts: u32,
    #[pyo3(get)]
    pub total_gets: u32,
    #[pyo3(get)]
    pub total_misses: u32,
    #[pyo3(get)]
    pub total_fetch_time_us: u64,
    #[pyo3(get)]
    pub total_fetches: u32,
    #[pyo3(get)]
    pub current_patience_us: u64,
}

pub struct NNRemote {
    pred_db: PredDBChannel,
    patience: Duration,
    stats: FetchStatsAccumulator,
}

impl NNRemote {
    pub fn new(pred_db: PredDBChannel) -> Self {
        NNRemote {
            pred_db,
            patience: INITIAL_PATIENCE,
            stats: FetchStatsAccumulator::default(),
        }
    }

    pub fn get_fetch_stats(&mut self) -> FetchStats {
        self.stats.snapshot_and_reset(self.patience)
    }

    fn check_pred(&mut self, board: &Board) -> Option<NNPred> {
        let result = self.pred_db.get_pred(board);
        if result.is_some() {
            self.stats.record_get_hit();
        } else {
            self.stats.record_get_miss();
        }
        result
    }

    pub fn fetch_pred(&mut self, board: &Board) -> Result<NNPred, Error> {
        let start = Instant::now();

        // attempt 0: cache hit
        if let Some(nn_pred) = self.check_pred(board) {
            self.stats.record_resolved(0, start.elapsed());
            return Ok(nn_pred);
        }

        self.pred_db.add_to_pred_queue(board);
        std::thread::sleep(self.patience);

        // attempt 1: after first patience sleep
        if let Some(nn_pred) = self.check_pred(board) {
            self.patience = self.patience.mul_f32(PATIENCE_SCALE_DOWN);
            self.stats.record_resolved(1, start.elapsed());
            return Ok(nn_pred);
        }

        self.patience = self.patience.mul_f32(PATIENCE_SCALE_UP);

        // backoff loop: attempts 2, 3, 4, ...
        let mut total_wait = self.patience;
        let mut delay = self.patience;
        let mut attempt: usize = 2;
        while total_wait < FAIL_THRESHOLD {
            std::thread::sleep(delay);
            total_wait += delay;

            if total_wait >= REPOST_THRESHOLD {
                self.pred_db.add_to_pred_queue(board);
            }
            if total_wait >= WARNING_THRESHOLD {
                println!("service[channel: {}] slow or unavailable: retrying...",
                    self.pred_db.get_channel(),
                );
            }

            if let Some(nn_pred) = self.check_pred(board) {
                self.stats.record_resolved(attempt, start.elapsed());
                return Ok(nn_pred);
            }

            delay = delay.mul_f32(BACKOFF_SCALING);
            attempt += 1;
        }

        println!(
            "service[channel: {}] not responding in {} ms",
            self.pred_db.get_channel(),
            FAIL_THRESHOLD.as_millis(),
        );
        self.stats.record_timeout(start.elapsed());
        Err(Error::new(
            ErrorKind::TimedOut,
            "NN service not responding",
        ))
    }
}
