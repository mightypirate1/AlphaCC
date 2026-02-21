use std::cell::RefCell;
use pyo3::prelude::*;

use crate::cc::pred_db::memcached_binary::MemcachedBinaryClient;
use crate::cc::pred_db::nn_pred::NNPred;
use crate::cc::game::board::Board;
use crate::cc::game::moves::find_all_moves;

const ZMQ_HWM: i32 = 10000;

lazy_static::lazy_static! {
    static ref ZMQ_CONTEXT: zmq::Context = zmq::Context::new();
    static ref SHARED_PULL_SOCKETS: [std::sync::Mutex<Option<zmq::Socket>>; 3] = [
        std::sync::Mutex::new(None),
        std::sync::Mutex::new(None),
        std::sync::Mutex::new(None),
    ];
}

thread_local! {
    static THREAD_SOCKETS_PUSH: [RefCell<Option<zmq::Socket>>; 3] = const {[
        RefCell::new(None),
        RefCell::new(None),
        RefCell::new(None),
    ]};
}

#[pyclass(module="alpha_cc_engine")]
pub struct PredDBChannel {
    zmq_url: String,
    memcached_client: MemcachedBinaryClient,
    channel: usize,
}


impl PredDBChannel {
    pub fn new(zmq_url: &str, memcached_host: &str, channel: usize) -> Self {
        let memcached_client = MemcachedBinaryClient::new(memcached_host, 11211);

        PredDBChannel {
            zmq_url: zmq_url.to_string(),
            memcached_client,
            channel,
        }
    }

    pub fn add_to_pred_queue(&self, board: &Board) {
        let value = board.serialize_rs();
        self.with_push_socket(|socket| {
            if let Err(e) = socket.send(&value, 0) {
                println!("ZMQ error: {:?}", e);
            }
        });
    }

    pub fn pop_boards_from_queue(&self, count: usize) -> Vec<Board> {
        let mut boards = Vec::with_capacity(count);
        self.with_pull_socket(|socket| {
            for _ in 0..count {
                match socket.recv_bytes(zmq::DONTWAIT) {
                    Ok(data) => {
                        boards.push(Board::deserialize_rs(&data));
                    },
                    Err(zmq::Error::EAGAIN) => {
                        break;
                    },
                    Err(e) => {
                        println!("ZMQ receive error: {:?}", e);
                        break;
                    }
                }
            }
        });
        boards
    }

    pub fn set_pred(&mut self, board: &Board, nn_pred: &NNPred) {
        let key = self.pred_key(board);
        self.memcached_client.set(key.as_bytes(), &nn_pred.serialize());
    }

    pub fn set_preds(&mut self, boards: &[Board], nn_preds: &[NNPred]) {
        let keys: Vec<String> = boards.iter().map(|b| self.pred_key(b)).collect();
        let values: Vec<Vec<u8>> = nn_preds.iter().map(|p| p.serialize()).collect();
        let kvs: Vec<(&[u8], &[u8])> = keys.iter().zip(values.iter())
            .map(|(k, v)| (k.as_bytes(), v.as_slice()))
            .collect();
        self.memcached_client.set_multi(&kvs);
    }

    pub fn get_pred(&mut self, board: &Board) -> Option<NNPred> {
        let key = self.pred_key(board);
        self.memcached_client
            .get(key.as_bytes())
            .map(|data: Vec<u8>| NNPred::deserialize(&data))
    }

    fn with_push_socket<F, R>(&self, f: F) -> R where F: FnOnce(&zmq::Socket) -> R {
        THREAD_SOCKETS_PUSH.with(|sockets_array| {
            let socket_cell = &sockets_array[self.channel];
            let mut socket_opt = socket_cell.borrow_mut();

            if socket_opt.is_none() {
                let socket = ZMQ_CONTEXT.socket(zmq::PUSH).expect("Failed to create socket");
                socket.set_sndhwm(ZMQ_HWM).expect("Failed to set HWM");
                let port = 5555 + self.channel;
                socket.connect(&format!("tcp://{}:{}", self.zmq_url, port))
                    .unwrap_or_else(|_| panic!("Failed to connect to ZMQ server on port {}", port));
                *socket_opt = Some(socket);
            }

            f(socket_opt.as_ref().unwrap())
        })
    }

    fn with_pull_socket<F, R>(&self, f: F) -> R where F: FnOnce(&zmq::Socket) -> R {
        let mut socket_guard = SHARED_PULL_SOCKETS[self.channel].lock().unwrap();

        if socket_guard.is_none() {
            let socket = ZMQ_CONTEXT.socket(zmq::PULL).expect("Failed to create socket");
            socket.set_rcvhwm(ZMQ_HWM).expect("Failed to set HWM");
            socket.set_linger(0).expect("Failed to set LINGER");
            let port = 5555 + self.channel;
            socket.bind(&format!("tcp://0.0.0.0:{}", port))
                .unwrap_or_else(|_| panic!("Failed to bind ZMQ socket on port {}", port));

            *socket_guard = Some(socket);
        }

        f(socket_guard.as_ref().unwrap())
    }

    fn pred_key(&self, board: &Board) -> String {
        format!("{}:{}", self.channel, board.compute_hash())
    }
}

#[pymethods]
impl PredDBChannel {
    #[new]
    pub fn new_py(zmq_url: &str, memcached_host: &str, channel: usize) -> Self {
        PredDBChannel::new(zmq_url, memcached_host, channel)
    }

    #[getter]
    pub fn get_channel(&self) -> usize {
        self.channel
    }

    pub fn has_pred(&mut self, board: &Board) -> bool {
        self.get_pred(board).is_some()
    }

    pub fn ping(&mut self) -> bool {
        self.memcached_client.set(b"ping", b"pong");
        true
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
        self.memcached_client.flush_all();
    }
}

#[pyfunction]
pub fn preds_from_logits<'py>(
    logits_flat: numpy::PyReadonlyArray1<'py, f32>,
    values_flat: numpy::PyReadonlyArray1<'py, f32>,
    boards: Vec<Board>,
    board_size: usize,
) -> PyResult<Vec<NNPred>> {
    let logits = logits_flat.as_slice()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("logits not contiguous: {e}")))?;
    let values = values_flat.as_slice()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("values not contiguous: {e}")))?;
    let s = board_size;
    let stride = s * s * s * s;
    let mut preds = Vec::with_capacity(boards.len());

    for (i, board) in boards.iter().enumerate() {
        let logits_slice = &logits[i * stride..(i + 1) * stride];
        let moves = find_all_moves(board);

        let move_logits: Vec<f32> = moves.iter().map(|m| {
            let fx = m.from_coord.x as usize;
            let fy = m.from_coord.y as usize;
            let tx = m.to_coord.x as usize;
            let ty = m.to_coord.y as usize;
            logits_slice[fx * s * s * s + fy * s * s + tx * s + ty]
        }).collect();

        let pi = softmax(&move_logits);
        let value = values[i];
        preds.push(NNPred::new(pi, value));
    }

    Ok(preds)
}

#[pyfunction]
pub fn post_preds_from_logits<'py>(
    pred_db: &mut PredDBChannel,
    logits_flat: numpy::PyReadonlyArray1<'py, f32>,
    values_flat: numpy::PyReadonlyArray1<'py, f32>,
    boards: Vec<Board>,
    board_size: usize,
) -> PyResult<()> {
    let preds = preds_from_logits(logits_flat, values_flat, boards.clone(), board_size)?;
    pred_db.set_preds(&boards, &preds);
    Ok(())
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}
