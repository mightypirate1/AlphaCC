use std::{thread, time::Duration};
use crate::cc::pred_db::PredDBChannel;
use crate::cc::pred_db::NNPred;
use crate::cc::board::Board;


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
        if self.pred_db.has_pred(board) {
            return self.pred_db.get_pred(board).unwrap();
        }
        self.pred_db.add_to_pred_queue(board);
        loop {
            if self.pred_db.has_pred(board) {
                return self.pred_db.get_pred(board).unwrap();
            }
            thread::sleep(Duration::from_nanos(100));
        }
    }   
}
