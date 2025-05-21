use core::time::Duration;
use crate::cc::pred_db::PredDBChannel;
use crate::cc::pred_db::NNPred;
use crate::cc::board::Board;

const PATIENCE: Duration = Duration::from_millis(1000);
const NUM_ATTEMPTS: usize = 3;

pub struct NNRemote {
    pred_db: PredDBChannel,
}

impl NNRemote {
    pub fn new(pred_db: PredDBChannel) -> Self {
        NNRemote { pred_db }
    }

    pub fn fetch_pred(&mut self, board: &Board) -> NNPred {
        // assumes there is a service running that will eventually
        // provide the prediction
        match self.pred_db.get_pred(board) {
            Some(nn_pred) => nn_pred,
            None => {
                /* resons for retry:
                 * - weight reloads in the nn service needs to flush old preds,
                 *   so there is a chance that a prediction was produced, but
                 *   flushed before we could get it.
                 * - low key impossible, but the nn service could respond faster
                 *   than it takes for us to set up the subscription.
                 * - the nn service has crashed.
                 */
                for _ in 0..NUM_ATTEMPTS {
                    self.pred_db.add_to_pred_queue(board);
                    if let Some(nn_pred) = self.pred_db.await_pred(board, Some(PATIENCE)) {
                        return nn_pred;
                    }
                }
                panic!(
                    "service[channel: {}] not responding in {} attemps",
                    self.pred_db.get_channel(),
                    NUM_ATTEMPTS,
                );
            }
        }
    }
}
