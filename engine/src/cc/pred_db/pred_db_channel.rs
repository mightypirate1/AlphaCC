use std::num::NonZero;
use std::time::Duration;

use pyo3::prelude::*;
use redis::{Client, Commands, Connection, ConnectionLike, RedisResult};

use crate::cc::board::{BoardHash, EncBoard};
use crate::cc::Board;
use crate::cc::pred_db::nn_pred::{EncPred, NNPred};

const PRED_QUEUE: &str = "queue";
const CHANNEL_DB_OFFSET: usize = 2;

const PUBSUB_BACKOFF_MS: u64 = 10;
const PUBSUB_RETRY_COUNT: u64 = 5;

const QUEUE_CONN_IDX: usize = 0;


#[pyclass(module="alpha_cc_engine")]
pub struct PredDBChannel {
    conns: Vec<Connection>,
    channel: usize,
    n_shards: u64,
    shard_urls: Vec<String>,  // for reconnects
}


impl PredDBChannel {
    pub fn new(shard_urls: Vec<String>, channel: usize) -> Self {
        if shard_urls.len() < 2 {
            panic!("provide one queue URL, and at least one shard URL");
        }
        let conns = shard_urls.iter()
            .map(|url| {
                PredDBChannel::connect(&url, channel)
            })
            .collect::<Vec<Connection>>();
        let n_shards = conns.len() as u64 - 1;
        PredDBChannel { 
            conns,
            channel,
            n_shards,
            shard_urls,
        }
    }

    fn connect(url: &str, channel: usize) -> Connection {
        let db = channel + CHANNEL_DB_OFFSET;
        let client = Client::open(format!("redis://{url}/{db}", url=url, db=db))
            .expect("Invalid connection URL");
        client.get_connection()
            .expect("failed to connect to Redis")
    }

    fn reconnect_shard_conn(&mut self, shard_idx: usize) {
        let conn_idx = self.conn_idx_for_shard_idx(shard_idx);
        if conn_idx >= self.conns.len() {
            panic!("Invalid connection index: {}", conn_idx);
        }
        println!("Reconnecting connection shard: {}", conn_idx);
        let url = &self.shard_urls[conn_idx];
        self.conns[conn_idx] = PredDBChannel::connect(url, self.channel);
    }

    pub fn add_to_pred_queue(&mut self, board: &Board) {
        let value = board.serialize_rs();
        let result: RedisResult<()> = self.queue_conn().rpush(PRED_QUEUE, value);
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
        // group into items, split by shards
        let mut shards_items: Vec<Vec<(BoardHash, EncPred)>> = (0..self.n_shards as usize)
            .map(|_| Vec::new())
            .collect();
        boards.iter()
            .zip(nn_preds.iter())
            .for_each(|(board, nn_pred)| {
                let key = board.compute_hash();
                let encoded_pred = nn_pred.serialize();
                let item = (key, encoded_pred);
                let shard_items = shards_items.get_mut(
                    self.shard_idx_for_key(key)
                ).unwrap();
                shard_items.push(item);
            });

        // for each shard:
        for (shard_idx, shard_items) in shards_items.iter().enumerate() {
            if shard_items.is_empty() {
                continue;
            }
            // publish preds
            let set_result: RedisResult<()> = self.conn_by_shard_idx(shard_idx).mset(shard_items);
            // publish notifications
            match set_result {
                Ok(_) => {
                    let mut pipe = redis::pipe();
                    shard_items.iter()
                        .for_each(|(key, encoded_pred)| {
                            let notification_key = PredDBChannel::notify_key(key);
                            pipe.publish(notification_key, encoded_pred).ignore();
                        });
                    let mut conn = self.conn_by_shard_idx(shard_idx);
                        match pipe.query::<()>(&mut conn) {
                            Ok(_) => {},
                            Err(e) => {
                                println!("notify error: {:?}", e);
                            }
                        };
                },
                Err(e) => {
                    println!("set error: {:?}", e);
                    self.reconnect_shard_conn(shard_idx);
                },
            };
        }
    }

    pub fn get_pred(&mut self, board: &Board) -> Option<NNPred> {
        let key: u64 = board.compute_hash();
        if let Ok(encoded) = self.conn(key).get::<_, EncPred>(key) {
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
            let encoded_pred: EncPred = message?.get_payload()?;
            let nn_pred = NNPred::deserialize(&encoded_pred);
            Ok(nn_pred)
        }

        let key = board.compute_hash();
        let notification_key = PredDBChannel::notify_key(&key);
        let result = await_notification_with_retry(
            self.conn(key),
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
            self.queue_conn().lpop::<_, Vec<EncBoard>>(PRED_QUEUE, NonZero::new(count));
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

    fn queue_conn(&mut self) -> &mut Connection {
        &mut self.conns[QUEUE_CONN_IDX]
    }

    fn conn(&mut self, key: u64) -> &mut Connection {
        let conn_idx = self.conn_idx_for_key(key);
        self.conns.get_mut(conn_idx).unwrap_or_else(|| {
            panic!("Invalid connection index for key: {} which maps to conn_idx: {}", key, conn_idx);
        })
    }

    fn conn_by_shard_idx(&mut self, shard_idx: usize) -> &mut Connection {
        // one for the queue, and one for each shard
        self.conns.get_mut(1 + shard_idx).unwrap_or_else(|| {
            panic!("Invalid shard index: {}", shard_idx)
        })
    }

    fn shard_idx_for_key(&self, key: u64) -> usize {
        (key % self.n_shards) as usize
    }

    fn conn_idx_for_shard_idx(&self, shard_idx: usize) -> usize {
        // one for the queue, and one for each shard
        1 + shard_idx
    }

    fn conn_idx_for_key(&self, key: u64) -> usize {
        // one for the queue, and one for each shard
        (1 + self.shard_idx_for_key(key)) as usize
    }
}

#[pymethods]
impl PredDBChannel {
    #[new]
    pub fn new_py(shard_urls: Vec<String>, channel: usize) -> Self {
        PredDBChannel::new(shard_urls, channel)
    }

    #[getter]
    pub fn get_channel(&self) -> usize {
        self.channel
    }

    pub fn has_pred(&mut self, board: &Board) -> bool {
        let key = board.compute_hash();
        self.conn(key).exists(key).unwrap()
    }

    pub fn ping(& mut self) -> bool {
        self.conns.iter_mut()
            .all(|conn| conn.check_connection())
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
        for conn in self.conns.iter_mut() {
            redis::cmd("FLUSHDB").arg("ASYNC").exec(conn).unwrap_or_else(|e| {
                println!("Error flushing predictions: {:?}", e);
            });
        }
    }
}
