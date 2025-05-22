use core::time::Duration;
use std::io::{Error, ErrorKind};
use crate::cc::pred_db::PredDBChannel;
use crate::cc::pred_db::NNPred;
use crate::cc::board::Board;

const ATTEMPT_PATIENCES: [Duration; 5] = [
    Duration::from_millis(10),
    Duration::from_millis(100),
    Duration::from_millis(1000),
    Duration::from_millis(10000),
    Duration::from_millis(10000),
];

pub struct NNRemote {
    pred_db: PredDBChannel,
}

impl NNRemote {
    pub fn new(pred_db: PredDBChannel) -> Self {
        NNRemote { pred_db }
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
                for patience in ATTEMPT_PATIENCES {
                    self.pred_db.add_to_pred_queue(board);
                    if let Some(nn_pred) = self.pred_db.await_pred(board, Some(patience)) {
                        return Ok(nn_pred);
                    }
                    if patience > Duration::from_millis(1000) {
                        println!("service[channel: {}] slow or unavailable: retrying...",
                            self.pred_db.get_channel(),
                        );
                    }
                }
                println!(
                    "service[channel: {}] not responding in {} attemps",
                    self.pred_db.get_channel(),
                    ATTEMPT_PATIENCES.len(),
                );
                Err(Error::new(
                    ErrorKind::TimedOut,
                    "NN service not responding",
                ))
            }
        }
    }
}
