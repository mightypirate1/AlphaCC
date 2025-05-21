use std::num::NonZero;
use std::time::Duration;

use pyo3::prelude::*;
use redis::{Client, Commands, Connection, ConnectionLike, RedisResult};

use crate::cc::Board;
use crate::cc::pred_db::nn_pred::NNPred;

const PRED_QUEUE: &str = "queue";
const CHANNEL_DB_OFFSET: usize = 2;


#[pyclass(module="alpha_cc_engine")]
pub struct PredDBChannel {
    conn: Connection,
    channel: usize,
}


impl PredDBChannel {
    pub fn new(url: &str, channel: usize) -> Self {
        let conn = PredDBChannel::connect(url, channel + CHANNEL_DB_OFFSET);
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

    pub fn set_pred(&mut self, board: &Board, nn_pred: &NNPred) {
        let key = board.compute_hash();
        let encoded = nn_pred.serialize();
        let set_result: RedisResult<()> = self.conn.set(key, &encoded);
        match set_result {
            Ok(_) => {
                let notification_key = PredDBChannel::notify_key(&board.compute_hash());
                let notify_result: RedisResult<()> = self.conn.publish(notification_key, &encoded);
                match notify_result {
                    Ok(_) => {},
                    Err(e) => {
                        println!("notify error: {:?}", e);
                    },
                };
            },
            Err(e) => {
                println!("set error: {:?}", e);
            },
        };
    }

    pub fn set_preds(&mut self, boards: &[Board], nn_preds: &[NNPred]) {
        let items = boards.iter()
            .zip(nn_preds.iter())
            .map(|(board, nn_pred)| {
                let pred_key = board.compute_hash();
                let encoded_pred = nn_pred.serialize();
                (pred_key, encoded_pred)
            })
            .collect::<Vec<(u64, Vec<u8>)>>();
        let set_result: RedisResult<()> = self.conn.mset(&items);
        match set_result {
            Ok(_) => {
                let mut pipe = redis::pipe();
                items
                    .iter()
                    .for_each(|(pred_key, encoded_pred)| {
                        let notification_key = PredDBChannel::notify_key(pred_key);
                        pipe.publish(notification_key, encoded_pred).ignore();
                    });
                let notify_result: RedisResult<()> = pipe.query::<()>(&mut self.conn);
                match notify_result {
                    Ok(_) => {},
                    Err(e) => {
                        println!("notify error: {:?}", e);
                    },
                };
            },
            Err(e) => {
                println!("set error: {:?}", e);
            },
        };
    }

    pub fn get_pred(&mut self, board: &Board) -> Option<NNPred> {
        let key: u64 = board.compute_hash();
        let encoded = self.conn.get::<_, Vec<u8>>(key).unwrap();
        if !encoded.is_empty() {
            Some(NNPred::deserialize(&encoded))
        } else {
            None
        }
    }

    pub fn await_pred(&mut self, board: &Board, timeout: Option<Duration>) -> Option<NNPred> {
        fn await_notification(
            conn: &mut Connection,
            notification_key: &str,
            timeout: Option<Duration>,
        ) -> RedisResult<NNPred> {
            let mut pubsub = conn.as_pubsub();
            pubsub.set_read_timeout(timeout)?;
            pubsub.subscribe(notification_key)?;
            let message = pubsub.get_message()?;
            let encoded_pred: Vec<u8> = message.get_payload()?;
            let nn_pred = NNPred::deserialize(&encoded_pred);
            match pubsub.unsubscribe(notification_key) {
                Ok(_) => {},
                Err(e) => {
                    println!("unsubscribe error: {:?}", e);
                },
            };
            Ok(nn_pred)
        }

        let pred_opt = self.get_pred(board);
        if pred_opt.is_some() {
            return pred_opt;
        }
        
        let notification_key = PredDBChannel::notify_key(&board.compute_hash());
        let result = await_notification(
            &mut self.conn,
            &notification_key,
            timeout,
        );
        match result {
            Ok(nn_pred) => {
                Some(nn_pred)
            },
            Err(e) => {
                // TODO: decide whether this is actually a problem
                println!("pubsub error: {:?}", e);
                None
            },
        }
    }

    fn notify_key(key: &u64) -> String {
        format!("notify:{}", key)
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
    pub fn get_channel(&self) -> usize {
        self.channel
    }

    pub fn has_pred(&mut self, board: &Board) -> bool {
        let key = board.compute_hash();
        self.conn.exists(key).unwrap()
    }

    pub fn ping(& mut self) -> bool {
        self.conn.check_connection() && self.conn.check_connection()
    }

    pub fn fetch_all_requests(&mut self) -> Vec<Board> {
        self.pop_all_boards_from_queue()
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

    pub fn fetch_pred(&mut self, board: &Board, timeout_ms: Option<u64>) -> Option<NNPred> {
        if timeout_ms.is_some() {
            self.await_pred(board, timeout_ms.map(Duration::from_millis))
        } else {
            self.get_pred(board)
        }
    }

    pub fn flush_preds(&mut self) {
        redis::cmd("FLUSHDB").exec(&mut self.conn).unwrap();
    }
}
