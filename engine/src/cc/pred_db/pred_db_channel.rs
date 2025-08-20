use std::cell::RefCell;
use pyo3::prelude::*;

use crate::cc::Board;
use crate::cc::pred_db::nn_pred::NNPred;

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
    memcached_client: memcache::Client,
    channel: usize,
}


impl PredDBChannel {
    pub fn new(zmq_url: &str, memcached_host: &str, channel: usize) -> Self {
        let memcached_url = format!("memcache://{host}:{port}?binary=true&tcp_nodelay=true", host=memcached_host, port=11211);
        let memcached_client = memcache::Client::connect(memcached_url)
            .expect("Failed to connect to memcached");
        
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

    pub fn ping(& mut self) -> bool {
        self.memcached_client.set("ping", "pong", 0).is_ok()
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
