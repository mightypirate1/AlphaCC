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
                Some(NNPred::deserialize(&encoded))
            },
            Err(e) => {
                println!("get error: {:?}", e);
                None
            },
        }
    }

    pub fn ping(& mut self) -> bool {
        self.conn.check_connection() && self.conn.check_connection()
    }

    fn pop_all_boards_from_queue(&mut self) -> Vec<Board> {
        let encoded_boards: Vec<Vec<u8>> = match self.conn.lrange(PRED_QUEUE, 0, -1) {
            Ok(boards) => boards,
            Err(e) => {
                println!("Error fetching all from queue: {:?}", e);
                return Vec::new();
            },
        };
        // NOTE: if this crashes or misbehaves - there was a filtering here in .is_empty() that looked redundant...
        encoded_boards.into_iter()
            .map(|encoded_board| Board::deserialize_rs(&encoded_board))
            .collect()
    }

    fn connect(url: &str, db: usize) -> Connection {
        let client = Client::open(format!("redis://{url}/{db}", url=url, db=db))
            .expect("Invalid connection URL");
        client.get_connection()
            .expect("failed to connect to Redis")
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
