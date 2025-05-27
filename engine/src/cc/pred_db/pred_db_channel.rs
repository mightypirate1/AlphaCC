use std::num::NonZero;

use pyo3::prelude::*;
use redis::{Commands, ConnectionLike, RedisResult};

use crate::cc::Board;
use crate::cc::pred_db::nn_pred::NNPred;

const PRED_QUEUE: &str = "queue";
const KEYDB_DB: usize = 2;


#[pyclass(module="alpha_cc_engine")]
pub struct PredDBChannel {
    keydb_conn: redis::Connection,
    memcached_client: memcache::Client,
    channel: usize,
}


impl PredDBChannel {
    pub fn new(keydb_host: &str, memcached_host: &str, channel: usize) -> Self {
        let (keydb_conn, memcached_client) = PredDBChannel::connect(keydb_host, memcached_host, KEYDB_DB);
        PredDBChannel { 
            keydb_conn,
            memcached_client,
            channel,
        }
    }

    fn connect(keydb_host: &str, memcached_host: &str, db: usize) -> (redis::Connection, memcache::Client) {
        let keydb_url = format!("redis://{host}/{db}", host=keydb_host, db=db);
        let memcached_url = format!("{host}:{port}", host=memcached_host, port=11211);
        let client = redis::Client::open(keydb_url)
            .expect("Invalid connection URL");
        let keydb_conn = client.get_connection()
            .expect("failed to connect to Redis");
        let memcached_client = memcache::Client::connect(memcached_url)
            .expect("Failed to connect to Memcached");
        (keydb_conn, memcached_client)
    }

    pub fn add_to_pred_queue(&mut self, board: &Board) {
        let value = board.serialize_rs();
        let result: RedisResult<()> = self.keydb_conn.rpush(PRED_QUEUE, value);
        match result {
            Ok(_) => {},
            Err(e) => {
                println!("error: {:?}", e);
            },
        };
    }

    fn pop_boards_from_queue(&mut self, count: usize) -> Vec<Board> {
        let mut boards: Vec<Board> = Vec::with_capacity(count);
        let encoded_boards = 
            self.keydb_conn.lpop::<_, Vec<Vec<u8>>>(PRED_QUEUE, NonZero::new(count));
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

    pub fn set_pred(&mut self, board: &Board, nn_pred: &NNPred) {
        let key = self.pred_key(board);
        self.memcached_client.set(&key, nn_pred.serialize().as_slice(), 0)
            .expect("Failed to set prediction in Memcached");
    }

    pub fn set_preds(&mut self, boards: &[Board], nn_preds: &[NNPred]) {
        for (board, nn_pred) in boards.iter().zip(nn_preds.iter()) {
            self.set_pred(board, nn_pred);
        }
    }

    pub fn get_pred(&mut self, board: &Board) -> Option<NNPred> {
        let key = self.pred_key(board);
        if let Ok(value) = self.memcached_client.get::<Vec<u8>>(&key) {
            if let Some(encoded) = value {
                return Some(NNPred::deserialize(&encoded));
            }
        } else {
            panic!("Failed to get prediction from Memcached for key: {}", key);
        }
        None
    }

    fn pred_key(&self, board: &Board) -> String {
        format!("{}:{}", self.channel, board.compute_hash())
    }
}

#[pymethods]
impl PredDBChannel {
    #[new]
    pub fn new_py(keydb_host: &str, memcached_host: &str, channel: usize) -> Self {
        PredDBChannel::new(keydb_host, memcached_host, channel)
    }

    #[getter]
    pub fn get_channel(&self) -> usize {
        self.channel
    }

    pub fn has_pred(&mut self, board: &Board) -> bool {
        self.get_pred(board).is_some()
    }

    pub fn ping(& mut self) -> bool {
        self.keydb_conn.check_connection() && self.memcached_client.set("ping", "pong", 0).is_ok()
    }

    pub fn fetch_requests(&mut self, count: usize) -> Vec<Board> {
        self.pop_boards_from_queue(count)
    }

    pub fn request_pred(&mut self, board: &Board) {
        self.add_to_pred_queue(board)
    }

    pub fn post_pred(&mut self, board: &Board, nn_pred: &NNPred) {
        self.set_pred(board, nn_pred);
    }

    pub fn post_preds(&mut self, boards: Vec<Board>, nn_preds: Vec<NNPred>) {
        self.set_preds(&boards, &nn_preds);
    }

    pub fn flush_preds(&mut self) {
        self.memcached_client.flush()
            .expect("Failed to flush Memcached");
    }
}
