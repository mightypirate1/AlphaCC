use std::{thread, time::Duration};
use crate::cc::pred_db::PredDBChannel;
use crate::cc::pred_db::NNPred;
use crate::cc::board::Board;


const PATIENCE: u64 = 10000;
const DELAY_NS: u64 = 100;

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
        let mut patience = PATIENCE;
        match self.pred_db.get_pred(board) {
            Some(nn_pred) => nn_pred,
            None => {
                self.pred_db.add_to_pred_queue(board);
                loop {
                    if self.pred_db.has_pred(board) {
                        return self.pred_db.get_pred(board).unwrap();
                    }
                    thread::sleep(Duration::from_nanos(DELAY_NS));
                    patience -= 1;
                    if patience == 0 {
                        panic!("service not responding in {}ms", PATIENCE * DELAY_NS / 1_000);
                    }
                }
            }
        }
    }   
}
