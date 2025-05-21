use std::num::NonZero;
use std::time::Duration;

use pyo3::prelude::*;
use redis::{Client, Commands, Connection, ConnectionLike, RedisResult};

use crate::cc::Board;
use crate::cc::pred_db::nn_pred::NNPred;

const PRED_QUEUE: &str = "queue";
const CHANNEL_DB_OFFSET: usize = 2;
const PUBSUB_BACKOFF_MS: u64 = 10;
const PUBSUB_RETRY_COUNT: u64 = 5;


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
        self.set_preds(&[board.clone()], &[nn_pred.clone()]);
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
        if let Ok(encoded) = self.conn.get::<_, Vec<u8>>(key) {
            if !encoded.is_empty() {
                return Some(NNPred::deserialize(&encoded));
            }
        }
        None
    }

    pub fn await_pred(&mut self, board: &Board, timeout: Option<Duration>) -> Option<NNPred> {
        fn await_notification_with_retry(
            conn: &mut Connection,
            notification_key: &str,
            timeout: Option<Duration>,
        ) -> RedisResult<NNPred> {
            let mut pubsub = conn.as_pubsub();
            pubsub.set_read_timeout(timeout)?;

            let mut retry_count: u64 = 0;
            loop {
                match pubsub.subscribe(notification_key) {
                    Ok(_) => break,
                    Err(e) => {
                        if retry_count >= PUBSUB_RETRY_COUNT - 1 {
                            return Err(e);
                        }
                        retry_count += 1;
                        std::thread::sleep(Duration::from_millis(PUBSUB_BACKOFF_MS * retry_count));
                    },
                }
            }
            let message = pubsub.get_message();
            match pubsub.unsubscribe(notification_key) {
                Ok(_) => {},
                Err(e) => {
                    println!("unsubscribe error: {:?}", e);
                }
            };
            let encoded_pred: Vec<u8> = message?.get_payload()?;
            let nn_pred = NNPred::deserialize(&encoded_pred);
            Ok(nn_pred)
        }

        let notification_key = PredDBChannel::notify_key(&board.compute_hash());
        let result = await_notification_with_retry(
            &mut self.conn,
            &notification_key,
            timeout,
        );

        /*
         * Under high load, the keydb sometimes does not respond
         * as quickly as we would wish. Thus we convert all errors
         * to `None`, which semantically means that the prediction
         * was not found.
         * 
         * TODO: see what can be done to fix that, and decide
         * what to do about this case. For now, I simply pretend
         * there was no prediction, and the caller will have to
         * handle it.
         */
        result.ok()
    }

    fn notify_key(key: &u64) -> String {
        format!("notify:{}", key)
    }

    fn pop_boards_from_queue(&mut self, count: usize) -> Vec<Board> {
        let mut boards: Vec<Board> = Vec::with_capacity(count);
        let encoded_boards = 
            self.conn.lpop::<_, Vec<Vec<u8>>>(PRED_QUEUE, NonZero::new(count));
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

    pub fn fetch_pred(&mut self, board: &Board, timeout_ms: Option<u64>) -> Option<NNPred> {
        if let Some(timeout) = timeout_ms {
            self.get_pred(board).or_else(|| {
                self.await_pred(board, Some(Duration::from_millis(timeout)))
            })
        } else {
            self.get_pred(board)
        }
    }

    pub fn flush_preds(&mut self) {
        redis::cmd("FLUSHDB").arg("ASYNC").exec(&mut self.conn).unwrap_or_else(|e| {
            println!("Error flushing predictions: {:?}", e);
        });
    }
}
