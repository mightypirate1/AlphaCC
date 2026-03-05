use std::cell::RefCell;

use pyo3::prelude::*;

use crate::cc::pred_db::nn_pred::NNPred;
use crate::cc::game::board::Board;
use crate::cc::game::moves::find_all_moves;

const ZMQ_HWM: i32 = 10000;

lazy_static::lazy_static! {
    static ref ZMQ_CONTEXT: zmq::Context = zmq::Context::new();
    static ref SHARED_ROUTER_SOCKETS: [std::sync::Mutex<Option<zmq::Socket>>; 3] = [
        std::sync::Mutex::new(None),
        std::sync::Mutex::new(None),
        std::sync::Mutex::new(None),
    ];
    static ref RESPONSE_QUEUES: [std::sync::Mutex<Vec<PendingResponse>>; 3] = [
        std::sync::Mutex::new(Vec::new()),
        std::sync::Mutex::new(Vec::new()),
        std::sync::Mutex::new(Vec::new()),
    ];
}

thread_local! {
    static THREAD_SOCKETS_DEALER: [RefCell<Option<zmq::Socket>>; 3] = const {[
        RefCell::new(None),
        RefCell::new(None),
        RefCell::new(None),
    ]};
}

/// Wire format for a single prediction request sent from worker to NN service.
/// Contains everything the NN service needs — no Board deserialization required.
#[derive(bitcode::Encode, bitcode::Decode)]
struct InferenceRequest {
    tensor_data: Vec<f32>,              // 2*s*s elements: one-hot encoded board state
    move_coords: Vec<(u8, u8, u8, u8)>, // (fx, fy, tx, ty) per legal move
}

/// Wire format for a prediction response sent from NN service back to worker.
/// Contains masked logits (one per legal move); worker applies softmax.
#[derive(bitcode::Encode, bitcode::Decode)]
struct InferenceResponse {
    logits: Vec<f32>,
    value: f32,
}

/// A pending response waiting to be sent on the ROUTER socket.
struct PendingResponse {
    identity: Vec<u8>,
    data: Vec<u8>, // pre-encoded InferenceResponse (bitcode)
}

impl InferenceRequest {
    fn from_board(board: &Board) -> Self {
        let s = board.get_size() as usize;
        let matrix = board.get_matrix();
        let mut tensor_data = vec![0.0f32; 2 * s * s];
        for (x, row) in matrix.iter().enumerate().take(s) {
            for (y, &val) in row.iter().enumerate().take(s) {
                let idx = x * s + y;
                if val == 1 {
                    tensor_data[idx] = 1.0;
                } else if val == 2 {
                    tensor_data[s * s + idx] = 1.0;
                }
            }
        }

        let moves = find_all_moves(board);
        let move_coords: Vec<(u8, u8, u8, u8)> = moves.iter().map(|m| {
            (m.from_coord.x, m.from_coord.y,
             m.to_coord.x, m.to_coord.y)
        }).collect();

        InferenceRequest {
            tensor_data,
            move_coords,
        }
    }
}

/// Opaque batch handle passed from fetch to post — holds the data the NN service
/// needs to route responses back to the correct worker.
#[pyclass(module="alpha_cc_engine", from_py_object)]
#[derive(Clone)]
pub struct InferenceBatch {
    identities: Vec<Vec<u8>>,
    move_coords: Vec<Vec<(u8, u8, u8, u8)>>,
    board_size: usize,
    channel: usize,
}

#[pymethods]
impl InferenceBatch {
    fn __len__(&self) -> usize {
        self.identities.len()
    }

    fn slice(&self, start: usize, end: usize) -> InferenceBatch {
        let end = end.min(self.identities.len());
        InferenceBatch {
            identities: self.identities[start..end].to_vec(),
            move_coords: self.move_coords[start..end].to_vec(),
            board_size: self.board_size,
            channel: self.channel,
        }
    }

    fn extend(&mut self, other: &InferenceBatch) {
        self.identities.extend_from_slice(&other.identities);
        self.move_coords.extend(other.move_coords.iter().cloned());
    }
}

impl InferenceBatch {
    pub fn channel(&self) -> usize {
        self.channel
    }
}

// ---------------------------------------------------------------------------
// ROUTER socket helpers (NN service side)
// ---------------------------------------------------------------------------

fn with_router_socket<F, R>(channel: usize, f: F) -> R where F: FnOnce(&zmq::Socket) -> R {
    let mut socket_guard = SHARED_ROUTER_SOCKETS[channel].lock().unwrap();

    if socket_guard.is_none() {
        let socket = ZMQ_CONTEXT.socket(zmq::ROUTER).expect("Failed to create ROUTER socket");
        socket.set_rcvhwm(ZMQ_HWM).expect("Failed to set RCVHWM");
        socket.set_sndhwm(ZMQ_HWM).expect("Failed to set SNDHWM");
        socket.set_linger(0).expect("Failed to set LINGER");
        let port = 5555 + channel;
        socket.bind(&format!("tcp://0.0.0.0:{}", port))
            .unwrap_or_else(|_| panic!("Failed to bind ZMQ ROUTER socket on port {}", port));

        *socket_guard = Some(socket);
    }

    f(socket_guard.as_ref().unwrap())
}

/// Drain pending responses from the queue and send them on the ROUTER socket.
fn drain_response_queue(channel: usize) {
    let responses: Vec<PendingResponse> = {
        let mut queue = RESPONSE_QUEUES[channel].lock().unwrap();
        std::mem::take(&mut *queue)
    };
    if responses.is_empty() {
        return;
    }

    with_router_socket(channel, |socket| {
        for resp in responses {
            socket.send(&resp.identity, zmq::SNDMORE)
                .expect("ZMQ ROUTER send identity failed");
            socket.send(&resp.data, 0)
                .expect("ZMQ ROUTER send data failed");
        }
    });
}

/// Receive inference requests from workers via the ROUTER socket.
/// Drains whatever is immediately available, up to max_count.
fn recv_inference_requests(channel: usize, max_count: usize) -> (Vec<Vec<u8>>, Vec<InferenceRequest>) {
    with_router_socket(channel, |socket| {
        let mut identities: Vec<Vec<u8>> = Vec::with_capacity(max_count);
        let mut requests: Vec<InferenceRequest> = Vec::with_capacity(max_count);

        while requests.len() < max_count {
            match socket.recv_bytes(zmq::DONTWAIT) {
                Ok(identity) => {
                    let data = socket.recv_bytes(0)
                        .expect("ZMQ ROUTER recv data frame failed after receiving identity");
                    identities.push(identity);
                    requests.push(bitcode::decode(&data).unwrap());
                }
                Err(zmq::Error::EAGAIN) => break,
                Err(e) => panic!("ZMQ ROUTER recv failed: {:?}", e),
            }
        }

        (identities, requests)
    })
}

// ---------------------------------------------------------------------------
// PredDBChannel (worker side) — instance methods on per-thread DEALER sockets
// ---------------------------------------------------------------------------

#[pyclass(module="alpha_cc_engine")]
pub struct PredDBChannel {
    zmq_url: String,
    channel: usize,
}

impl PredDBChannel {
    pub fn new(zmq_url: &str, channel: usize) -> Self {
        PredDBChannel {
            zmq_url: zmq_url.to_string(),
            channel,
        }
    }

    pub fn add_to_pred_queue(&self, board: &Board) {
        let req = InferenceRequest::from_board(board);
        let encoded = bitcode::encode(&req);
        self.with_dealer_socket(|socket| {
            socket.send(&encoded, 0).expect("ZMQ DEALER send failed");
        });
    }

    pub fn recv_pred(&self) -> Result<NNPred, std::io::Error> {
        self.with_dealer_socket(|socket| {
            match socket.recv_bytes(0) {
                Ok(data) => {
                    let resp: InferenceResponse = bitcode::decode(&data)
                        .map_err(|e| std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            format!("decode error: {:?}", e),
                        ))?;
                    let pi = softmax(&resp.logits);
                    Ok(NNPred::new(pi, resp.value))
                }
                Err(e) => Err(std::io::Error::other(format!("ZMQ recv error: {:?}", e))),
            }
        })
    }

    fn with_dealer_socket<F, R>(&self, f: F) -> R where F: FnOnce(&zmq::Socket) -> R {
        THREAD_SOCKETS_DEALER.with(|sockets_array| {
            let socket_cell = &sockets_array[self.channel];
            let mut socket_opt = socket_cell.borrow_mut();

            if socket_opt.is_none() {
                let socket = ZMQ_CONTEXT.socket(zmq::DEALER).expect("Failed to create DEALER socket");
                socket.set_sndhwm(ZMQ_HWM).expect("Failed to set SNDHWM");
                socket.set_rcvhwm(ZMQ_HWM).expect("Failed to set RCVHWM");
                let port = 5555 + self.channel;
                socket.connect(&format!("tcp://{}:{}", self.zmq_url, port))
                    .unwrap_or_else(|_| panic!("Failed to connect DEALER to ZMQ server on port {}", port));
                *socket_opt = Some(socket);
            }

            f(socket_opt.as_ref().unwrap())
        })
    }
}

#[pymethods]
impl PredDBChannel {
    #[new]
    pub fn new_py(zmq_url: &str, channel: usize) -> Self {
        PredDBChannel::new(zmq_url, channel)
    }

    #[getter]
    pub fn get_channel(&self) -> usize {
        self.channel
    }
}

// ---------------------------------------------------------------------------
// NN Service free functions (operate on static ROUTER sockets / response queues)
// ---------------------------------------------------------------------------

/// Fetch inference requests from ZMQ and build a batch tensor in one fused call.
/// Drains the response queue first (sends pending responses to workers), then
/// receives new requests. This keeps all ROUTER socket ops in one thread.
#[pyfunction]
pub fn fetch_and_build_tensor<'py>(
    py: Python<'py>,
    channel: usize,
    max_count: usize,
    board_size: usize,
) -> Option<(InferenceBatch, Bound<'py, numpy::PyArray4<f32>>)> {
    // Phase 1: Release GIL — drain responses, ZMQ recv, bitcode decode
    let result = py.detach(|| {
        drain_response_queue(channel);

        let (identities, requests) = recv_inference_requests(channel, max_count);
        if requests.is_empty() {
            return None;
        }

        let s = board_size;
        let batch = requests.len();
        let mut tensor_data = vec![0.0f32; batch * 2 * s * s];
        let mut move_coords = Vec::with_capacity(batch);

        for (i, req) in requests.into_iter().enumerate() {
            let offset = i * 2 * s * s;
            tensor_data[offset..offset + 2 * s * s].copy_from_slice(&req.tensor_data);
            move_coords.push(req.move_coords);
        }

        Some((tensor_data, identities, move_coords, batch, s))
    });

    // Phase 2: GIL held — create numpy array
    let (tensor_data, identities, move_coords, batch, s) = result?;

    let arr = numpy::ndarray::Array4::<f32>::from_shape_vec(
        [batch, 2, s, s], tensor_data,
    ).unwrap();
    let numpy_tensor = numpy::IntoPyArray::into_pyarray(arr, py);

    let inference_batch = InferenceBatch {
        identities,
        move_coords,
        board_size: s,
        channel,
    };

    Some((inference_batch, numpy_tensor))
}

/// Pure computation: build responses from logits (masked, no softmax).
/// Workers apply softmax on their end.
fn build_responses_from_logits(
    logits: &[f32],
    values: &[f32],
    batch: &InferenceBatch,
) -> Vec<PendingResponse> {
    let s = batch.board_size;
    let stride = s * s * s * s;
    let mut responses = Vec::with_capacity(batch.identities.len());

    for (i, move_coords) in batch.move_coords.iter().enumerate() {
        let logits_slice = &logits[i * stride..(i + 1) * stride];

        let move_logits: Vec<f32> = move_coords.iter().map(|&(fx, fy, tx, ty)| {
            logits_slice[fx as usize * s*s*s + fy as usize * s*s + tx as usize * s + ty as usize]
        }).collect();

        let resp = InferenceResponse { logits: move_logits, value: values[i] };
        let data = bitcode::encode(&resp);

        responses.push(PendingResponse {
            identity: batch.identities[i].clone(),
            data,
        });
    }

    responses
}

/// Enqueue inference responses for sending back to workers.
/// Called from the post thread; responses are drained and sent by the prefetch thread.
#[pyfunction]
pub fn enqueue_responses<'py>(
    logits_flat: numpy::PyReadonlyArray1<'py, f32>,
    values_flat: numpy::PyReadonlyArray1<'py, f32>,
    batch: &InferenceBatch,
) -> PyResult<()> {
    let logits = logits_flat.as_slice()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("logits not contiguous: {e}")))?;
    let values = values_flat.as_slice()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("values not contiguous: {e}")))?;

    let responses = build_responses_from_logits(logits, values, batch);
    RESPONSE_QUEUES[batch.channel()].lock().unwrap().extend(responses);
    Ok(())
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

type MoveCoords = Vec<(u8, u8, u8, u8)>;

/// Expose InferenceRequest::from_board for testing.
/// Returns (tensor_data as numpy (2, s, s), move_coords).
#[pyfunction]
#[allow(clippy::type_complexity)]
pub fn build_inference_request<'py>(
    py: Python<'py>,
    board: &Board,
) -> (Bound<'py, numpy::PyArray3<f32>>, MoveCoords) {
    let req = InferenceRequest::from_board(board);
    let s = board.get_size() as usize;
    let arr = numpy::ndarray::Array3::<f32>::from_shape_vec([2, s, s], req.tensor_data).unwrap();
    let numpy_arr = numpy::IntoPyArray::into_pyarray(arr, py);
    (numpy_arr, req.move_coords)
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}
