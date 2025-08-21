use core::time::Duration;
use std::io::{Error, ErrorKind};

use crate::cc::pred_db::PredDBChannel;
use crate::cc::pred_db::NNPred;
use crate::cc::board::Board;

const INITIAL_PATIENCE: Duration = Duration::from_millis(5);
const REPOST_THRESHOLD: Duration = Duration::from_millis(100);
const WARNING_THRESHOLD: Duration = Duration::from_millis(1000);
const FAIL_THRESHOLD: Duration = Duration::from_millis(10000);
const BACKOFF_SCALING: f32 = 2.0;
const PATIENCE_SCALE_UP: f32 = 1.02;
const PATIENCE_SCALE_DOWN: f32 = 0.99;

pub struct NNRemote {
    pred_db: PredDBChannel,
    patience: Duration,
}

impl NNRemote {
    pub fn new(pred_db: PredDBChannel) -> Self {
        NNRemote { pred_db, patience: INITIAL_PATIENCE }
    }

    pub fn fetch_pred(&mut self, board: &Board) -> Result<NNPred, Error> {
        // assumes there is a service running that will eventually
        // provide the prediction
        match self.pred_db.get_pred(board) {
            Some(nn_pred) => Ok(nn_pred),
            None => {
                /* resons for retry:
                 * - weight reloads in the nn service needs to flush old preds,
                 *   so there is a chance that a prediction was produced, but
                 *   flushed before we could get it.
                 * - low key impossible, but the nn service could respond faster
                 *   than it takes for us to set up the subscription.
                 * - the nn service has crashed.
                 * - a tournament started, and the nn service has not yet fired
                 *   up the network on the tournament channels.
                 */
                self.pred_db.add_to_pred_queue(board);
                std::thread::sleep(self.patience);

                // if we have the prediction after the first wait, we adjust down the patience
                if let Some(nn_pred) = self.pred_db.get_pred(board) {
                    self.patience = self.patience.mul_f32(PATIENCE_SCALE_DOWN);
                    return Ok(nn_pred);
                }
                // if we don't have the prediction, we adjust up the patience...
                self.patience = self.patience.mul_f32(PATIENCE_SCALE_UP);

                // ...and proceed to retry with exponential backoff
                let mut total_wait = self.patience;
                let mut delay = self.patience;
                while total_wait < FAIL_THRESHOLD {
                    // wait and count up
                    std::thread::sleep(delay);
                    total_wait += delay;

                    // warn/repost as needed
                    if total_wait >= REPOST_THRESHOLD {
                        self.pred_db.add_to_pred_queue(board);
                    }
                    if total_wait >= WARNING_THRESHOLD {
                        println!("service[channel: {}] slow or unavailable: retrying...",
                            self.pred_db.get_channel(),
                        );
                    }

                    // check if we have the prediction now
                    if let Some(nn_pred) = self.pred_db.get_pred(board) {
                        return Ok(nn_pred);
                    }
                    delay = delay.mul_f32(BACKOFF_SCALING);
                }
                println!(
                    "service[channel: {}] not responding in {} ms",
                    self.pred_db.get_channel(),
                    FAIL_THRESHOLD.as_millis(),
                );
                Err(Error::new(
                    ErrorKind::TimedOut,
                    "NN service not responding",
                ))
            }
        }
    }
}
