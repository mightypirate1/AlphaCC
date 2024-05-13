use std::{thread, time::Duration};
use crate::cc::rollouts::pred_db::PredDB;
use crate::cc::rollouts::nn_pred::NNPred;
use crate::cc::board::Board;


pub struct NNRemote {
    pred_db: PredDB,
}

impl NNRemote {
    pub fn new(pred_db: PredDB) -> Self {
        NNRemote { pred_db }
    }

    pub fn fetch_pred(&mut self, board: &Board) -> NNPred {
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
