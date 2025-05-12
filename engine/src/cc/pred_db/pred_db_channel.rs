use std::num::NonZero;

use pyo3::prelude::*;
use redis::{Client, Commands, Connection, ConnectionLike, RedisResult};

use crate::cc::Board;
use crate::cc::pred_db::nn_pred::NNPred;

const PRED_QUEUE: &str = "queue";

#[pyclass(module="alpha_cc_engine")]
pub struct PredDBChannel {
    conn: Connection,
    channel: usize,
}


impl PredDBChannel {
    pub fn new(url: &str, channel: usize) -> Self {
        let conn = PredDBChannel::connect(url, channel);       
        PredDBChannel { 
            conn,
            channel,
         }
    }


    fn connect(url: &str, db: usize) -> Connection {
        let client = Client::open(format!("redis://{url}/{db}", url=url, db=db))
            .expect("Invalid connection URL");
        client.get_connection()
            .expect("failed to connect to Redis")
    }

    pub fn ping(& mut self) -> bool {
        self.conn.check_connection() && self.conn.check_connection()
    }

    pub fn add_to_pred_queue(&mut self, board: &Board) {
        let value = board.serialize_rs();
        let result: RedisResult<()> = self.conn.rpush(PRED_QUEUE, value);
        match result {
            Ok(_) => {},
            Err(e) => {
                println!("error: {:?}", e);
            },
        };
    }

    pub fn has_pred(&mut self, board: &Board) -> bool {
        let key = board.compute_hash();
        self.conn.exists(key).unwrap()
    }

    pub fn set_pred(&mut self, board: &Board, nn_pred: &NNPred) {
        let key = board.compute_hash();
        let encoded = nn_pred.serialize();
        let result: RedisResult<()> = self.conn.set(key, encoded);
        match result {
            Ok(_) => {},
            Err(e) => {
                println!("set error: {:?}", e);
            },
        };
    }

    pub fn set_preds(&mut self, boards: &[Board], nn_preds: &[NNPred]) {
        let items = boards.iter()
            .zip(nn_preds.iter())
            .map(|(board, nn_pred)| {
                let field = board.compute_hash();
                let encoded = nn_pred.serialize();
                (field, encoded)
            })
            .collect::<Vec<(u64, Vec<u8>)>>();
        let result: RedisResult<()> = self.conn.mset(&items);
        match result {
            Ok(_) => {},
            Err(e) => {
                println!("set error: {:?}", e);
            },
        };
    }

    pub fn get_pred(&mut self, board: &Board) -> Option<NNPred> {
        let key: u64 = board.compute_hash();
        match self.conn.get::<_, Vec<u8>>(key) {
            Ok(encoded) => {
                if encoded.is_empty() {
                    // TODO: check if this actually happens
                    return None;
                }
                Some(NNPred::deserialize(&encoded))
            },
            Err(e) => {
                println!("get error: {:?}", e);
                None
            },
        }
    }

    fn queue_len(&mut self) -> usize {
        match self.conn.llen(PRED_QUEUE) {
            Ok(len) => len,
            Err(e) => {
                println!("Error fetching queue length: {:?}", e);
                0
            },
        }
    }

    fn pop_all_boards_from_queue(&mut self) -> Vec<Board> {
        let queue_len = self.queue_len();
        
        if queue_len == 0 {
            return Vec::new();
        }

        let mut boards: Vec<Board> = Vec::with_capacity(queue_len);
        let encoded_boards = 
            self.conn.lpop::<_, Vec<Vec<u8>>>(PRED_QUEUE, NonZero::new(queue_len));
        match encoded_boards {
            Ok(encoded_boards) => {
                for encoded_board in encoded_boards {
                    boards.push(Board::deserialize_rs(&encoded_board));
                }
            },
            Err(e) => {
                panic!("Error fetching boards from queue: {:?}", e);
            },
        }
        
        boards
    }
}

#[pymethods]
impl PredDBChannel {
    #[new]
    fn new_py(url: &str, channel: usize) -> Self {
        PredDBChannel::new(url, channel)
    }

    #[getter]
    fn get_channel(&self) -> usize {
        self.channel
    }

    pub fn fetch_all(&mut self) -> Vec<Board> {
        self.pop_all_boards_from_queue()
    }

    pub fn request_pred(&mut self, board: &Board) -> Option<NNPred> {
        self.get_pred(board)
    }

    pub fn dbg(&mut self, board: &Board) {
        let result: RedisResult<()> = self.conn.set("HEJ", 1);
        match result {
            Ok(_) => {},
            Err(e) => {
                println!("set error: {:?}", e);
            },
        };
        self.add_to_pred_queue(board);
    }

    pub fn post_pred(&mut self, board: &Board, nn_pred: &NNPred) {
        self.set_pred(board, nn_pred);
    }

    pub fn post_preds(&mut self, boards: Vec<Board>, nn_preds: Vec<NNPred>) {
        self.set_preds(&boards, &nn_preds);
    }

    pub fn flush_preds(&mut self) {
        redis::cmd("FLUSHDB").exec(&mut self.conn).unwrap();
    }
}
