use pyo3::prelude::*;
use redis::{Client, Commands, Connection, ConnectionLike, RedisResult, RedisError};

use crate::cc::Board;
use crate::cc::rollouts::nn_pred::NNPred;


const PRED_QUEUE: &str = "pred-queue";

#[pyclass(module="alpha_cc_engine")]
pub struct PredDB {
    queue_conn: Connection,
    pred_conn: Connection,
}


impl PredDB {
    pub fn new(url: &str) -> Self {
        let queue_conn = PredDB::connect(url, 1);        
        let pred_conn = PredDB::connect(url, 2);        

        PredDB { queue_conn, pred_conn }
    }

    pub fn add_to_pred_queue(&mut self, board: &Board) {
        let value = board.serialize_rs();
        let result: RedisResult<()> = self.queue_conn.rpush(PRED_QUEUE, value);
        match result {
            Ok(_) => {},
            Err(e) => {
                println!("error: {:?}", e);
            },
        };
    }

    pub fn has_pred(&mut self, board: &Board) -> bool {
        let key = board.compute_hash();
        self.pred_conn.exists(key).unwrap()
    }

    pub fn set_pred(&mut self, board: &Board, nn_pred: &NNPred) {
        let key = board.compute_hash();
        let encoded = nn_pred.serialize();
        let result: RedisResult<()> = self.pred_conn.set(key, encoded);
        match result {
            Ok(_) => {},
            Err(e) => {
                println!("set error: {:?}", e);
            },
        };
    }

    pub fn get_pred(&mut self, board: &Board) -> Option<NNPred> {
        let key = board.compute_hash();
        match self.pred_conn.get(key) {
            Ok(encoded) => {
                Some(NNPred::deserialize(encoded))
            },
            Err(e) => {
                println!("get error: {:?}", e);
                None
            },
        }
    }

    pub fn ping(& mut self) -> bool {
        self.pred_conn.check_connection() && self.queue_conn.check_connection()
    }

    fn pop_bytes_from_queue(&mut self) -> Option<Vec<u8>> {
        let encoded_board: Vec<u8>;
        match self.queue_conn.lpop(PRED_QUEUE, None) {
            Ok(encoded) => {
                encoded_board = encoded;
                if !encoded_board.is_empty() {
                    return Some(encoded_board)
                }
                None
            },
            Err(e) => {
                println!("error: {:?}", e);
                None
            },
        }
    }
    fn bpop_bytes_from_queue(&mut self) -> Option<Vec<u8>> {
        let encoded_board: Vec<u8>;
        let queue_and_result: Result<(String, Vec<u8>), RedisError> = self.queue_conn.blpop(
            PRED_QUEUE, 0.0
        );
        match queue_and_result {
            Ok((_, encoded)) => {
                encoded_board = encoded;
                if !encoded_board.is_empty() {
                    return Some(encoded_board)
                }
                None
            },
            Err(e) => {
                println!("error: {:?}", e);
                None
            },
        }
    }

    fn decode_bytes_as_board(encoded_board: Vec<u8>) -> Option<Board> {
        if !encoded_board.is_empty() {
            return Some(Board::deserialize_rs(encoded_board))
        }
        None
    }

    fn pop_board_from_queue(&mut self) -> Option<Board> {
        match self.pop_bytes_from_queue() {
            Some(encoded_board) => {
                PredDB::decode_bytes_as_board(encoded_board)
            },
            None => None,
        }
    }
    fn bpop_board_from_queue(&mut self) -> Option<Board> {
        match self.bpop_bytes_from_queue() {
            Some(encoded_board) => {
                PredDB::decode_bytes_as_board(encoded_board)
            },
            None => None,
        }
    }

    fn connect(url: &str, db: usize) -> Connection {
        let client = Client::open(format!("redis://{url}/{db}", url=url, db=db))
            .expect("Invalid connection URL");
        client.get_connection()
            .expect("failed to connect to Redis")
    }
}

#[pymethods]
impl PredDB {
    #[new]
    fn new_py(url: &str) -> Self {
        PredDB::new(url)
    }

    pub fn fetch_all(&mut self) -> Vec<Board> {
        let mut boards: Vec<Board>= Vec::new();
        if let Some(board) = self.bpop_board_from_queue() {
            boards.push(board);
        }
        while let Some(board) = self.pop_board_from_queue() {
            boards.push(board);
        }
        boards
    }

    pub fn post_pred(&mut self, board: &Board, nn_pred: &NNPred) {
        self.set_pred(board, nn_pred);
    }

    pub fn flush_preds(&mut self) {
        redis::cmd("FLUSHDB").execute(&mut self.pred_conn);
    }
}
