#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::{PyTuple, PyBytes};
use pyo3_stub_gen::define_stub_info_gatherer;
use pyo3_stub_gen_derive::{gen_stub_pyclass, gen_stub_pymethods, gen_stub_pyfunction};

use alpha_cc_core::board::MAX_SIZE;
use alpha_cc_core::moves::find_all_moves;

// ──────────────────────────────────────────────────
// HexCoord
// ──────────────────────────────────────────────────

#[gen_stub_pyclass]
#[pyclass(name = "HexCoord", module = "alpha_cc_engine", from_py_object)]
#[derive(Clone)]
pub struct PyHexCoord(pub alpha_cc_core::HexCoord);

impl From<alpha_cc_core::HexCoord> for PyHexCoord {
    fn from(h: alpha_cc_core::HexCoord) -> Self { PyHexCoord(h) }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyHexCoord {
    fn get_all_neighbours(&self, distance: usize) -> Vec<PyHexCoord> {
        self.0.get_all_neighbours(distance)
            .into_iter().map(PyHexCoord::from).collect()
    }

    #[getter]
    fn get_x(&self) -> u8 {
        self.0.x
    }

    #[getter]
    fn get_y(&self) -> u8 {
        self.0.y
    }

    fn flip(&self) -> PyHexCoord {
        PyHexCoord(self.0.flip())
    }

    fn repr(&self) -> String {
        self.0.repr()
    }

    fn __repr__(&self) -> String {
        self.0.repr()
    }
}

// ──────────────────────────────────────────────────
// BoardInfo
// ──────────────────────────────────────────────────

#[gen_stub_pyclass]
#[pyclass(name = "BoardInfo", module = "alpha_cc_engine", get_all)]
pub struct PyBoardInfo {
    pub current_player: i8,
    pub winner: i8,
    pub size: u8,
    pub duration: u16,
    pub game_over: bool,
    pub reward: f32,
}

impl From<alpha_cc_core::BoardInfo> for PyBoardInfo {
    fn from(bi: alpha_cc_core::BoardInfo) -> Self {
        PyBoardInfo {
            current_player: bi.current_player,
            winner: bi.winner,
            size: bi.size,
            duration: bi.duration,
            game_over: bi.game_over,
            reward: bi.reward,
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyBoardInfo {
    fn __repr__(&self) -> String {
        format!(
            "BoardInfo[\n  game_over: {}\n  current_player: {}\n  winner: {}\n  reward: {}\n  duration: {}\n  size: {}\n]",
            self.game_over,
            self.current_player,
            self.winner,
            self.reward,
            self.duration,
            self.size,
        )
    }
}

// ──────────────────────────────────────────────────
// Move
// ──────────────────────────────────────────────────

#[gen_stub_pyclass]
#[pyclass(name = "Move", module = "alpha_cc_engine", from_py_object)]
#[derive(Clone)]
pub struct PyMove(pub alpha_cc_core::Move);

impl From<alpha_cc_core::Move> for PyMove {
    fn from(m: alpha_cc_core::Move) -> Self { PyMove(m) }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyMove {
    #[getter]
    fn from_coord(&self) -> PyHexCoord {
        PyHexCoord(self.0.from_coord)
    }

    #[getter]
    fn to_coord(&self) -> PyHexCoord {
        PyHexCoord(self.0.to_coord)
    }

    #[getter]
    fn path(&self) -> Vec<PyHexCoord> {
        self.0.path.iter().map(|c| PyHexCoord(*c)).collect()
    }

    fn __repr__(&self) -> String {
        let fc = self.0.from_coord;
        let tc = self.0.to_coord;
        format!("Move[{} -> {}]", fc.repr(), tc.repr())
    }
}

// ──────────────────────────────────────────────────
// Board
// ──────────────────────────────────────────────────

#[gen_stub_pyclass]
#[pyclass(name = "Board", module = "alpha_cc_engine", from_py_object)]
#[derive(Clone)]
pub struct PyBoard(pub alpha_cc_core::Board);

impl From<alpha_cc_core::Board> for PyBoard {
    fn from(b: alpha_cc_core::Board) -> Self { PyBoard(b) }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyBoard {
    #[new]
    #[pyo3(signature = (*py_args))]
    fn new(py_args: &Bound<'_, PyTuple>) -> Self {
        match py_args.len() {
            1 => {
                if let Ok(size) = py_args.get_item(0).unwrap().extract::<usize>() {
                    return PyBoard(alpha_cc_core::Board::create(size))
                }
                panic!("expected a single int as input");
            },
            0 => {
                PyBoard(alpha_cc_core::Board::create(9))
            },
            _ => { unreachable!() }
        }
    }

    #[getter]
    fn info(&self) -> PyBoardInfo {
        PyBoardInfo::from(self.0.get_info())
    }

    fn reset(&self) -> PyBoard {
        PyBoard(self.0.reset())
    }

    fn get_moves(&self) -> Vec<PyMove> {
        self.0.get_moves().into_iter().map(PyMove::from).collect()
    }

    fn get_next_states(&self) -> Vec<PyBoard> {
        self.0.get_next_states().into_iter().map(PyBoard::from).collect()
    }

    fn get_matrix(&self) -> alpha_cc_core::BoardMatrix {
        self.0.get_matrix()
    }

    fn apply(&self, mv: &PyMove) -> PyBoard {
        PyBoard(self.0.apply(&mv.0))
    }

    fn get_unflipped_matrix(&self) -> alpha_cc_core::BoardMatrix {
        self.0.get_unflipped_matrix()
    }

    fn render(&self) {
        self.0.render()
    }

    fn __setstate__(&mut self, py: Python, state: Py<PyAny>) -> PyResult<()> {
        let py_bytes = state.extract::<Bound<'_, PyBytes>>(py)?;
        let bytes = py_bytes.as_bytes();
        self.0 = alpha_cc_core::Board::deserialize_rs(bytes);
        Ok(())
    }

    fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        Ok(PyBytes::new(py, &self.0.serialize_rs()))
    }

    fn __hash__(&self) -> u64 {
        self.0.compute_hash()
    }

    fn __eq__(&self, other: &PyBoard) -> bool {
        self.0 == other.0
    }
}

// ──────────────────────────────────────────────────
// NNPred
// ──────────────────────────────────────────────────

#[gen_stub_pyclass]
#[pyclass(name = "NNPred", module = "alpha_cc_engine", from_py_object)]
#[derive(Clone)]
pub struct PyNNPred(pub alpha_cc_nn::NNPred);

impl From<alpha_cc_nn::NNPred> for PyNNPred {
    fn from(p: alpha_cc_nn::NNPred) -> Self { PyNNPred(p) }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyNNPred {
    #[new]
    fn new(pi: Vec<f32>, value: f32) -> Self {
        PyNNPred(alpha_cc_nn::NNPred::new(pi, value))
    }

    #[getter]
    fn get_pi(&self) -> Vec<f32> {
        self.0.pi()
    }

    #[getter]
    fn get_value(&self) -> f32 {
        self.0.value()
    }

    fn __repr__(&self) -> String {
        format!("NNPred[val={}, pi={:?}]", self.0.value(), self.0.pi())
    }
}

// ──────────────────────────────────────────────────
// FetchStats
// ──────────────────────────────────────────────────

#[gen_stub_pyclass]
#[pyclass(name = "FetchStats", module = "alpha_cc_engine", get_all)]
pub struct PyFetchStats {
    pub total_fetch_time_us: u64,
    pub total_fetches: u32,
}

impl From<alpha_cc_nn::FetchStats> for PyFetchStats {
    fn from(fs: alpha_cc_nn::FetchStats) -> Self {
        PyFetchStats {
            total_fetch_time_us: fs.total_fetch_time_us,
            total_fetches: fs.total_fetches,
        }
    }
}

// ──────────────────────────────────────────────────
// MCTSNode
// ──────────────────────────────────────────────────

#[gen_stub_pyclass]
#[pyclass(name = "MCTSNode", module = "alpha_cc_engine")]
pub struct PyMCTSNode(pub alpha_cc_mcts::MCTSNode);

impl From<alpha_cc_mcts::MCTSNode> for PyMCTSNode {
    fn from(n: alpha_cc_mcts::MCTSNode) -> Self { PyMCTSNode(n) }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyMCTSNode {
    #[getter(n)]
    fn get_n_py(&self) -> Vec<u32> {
        (0..self.0.num_actions()).map(|a| self.0.get_n(a)).collect()
    }

    #[getter(q)]
    fn get_q_py(&self) -> Vec<f32> {
        (0..self.0.num_actions()).map(|a| self.0.get_q(a)).collect()
    }

    #[getter(pi)]
    fn get_pi_py(&self) -> Vec<f32> {
        alpha_cc_nn::NNQuantizedPi::dequantize_vec(&self.0.pi)
    }

    #[getter(v)]
    fn get_v_py(&self) -> f32 {
        self.0.get_v()
    }

    /// Get moves for a board position. Not a getter — requires the board as argument.
    fn get_moves(&self, board: &PyBoard) -> Vec<PyMove> {
        find_all_moves(&board.0).into_iter().map(PyMove::from).collect()
    }
}

// ──────────────────────────────────────────────────
// MCTS
// ──────────────────────────────────────────────────

type MCTS = alpha_cc_mcts::MCTS<alpha_cc_nn_service::NNRemote>;

#[gen_stub_pyclass]
#[pyclass(name = "MCTS", module = "alpha_cc_engine")]
pub struct PyMCTS(pub MCTS);

#[gen_stub_pymethods]
#[pymethods]
impl PyMCTS {
    #[allow(clippy::too_many_arguments)]
    #[new]
    #[pyo3(signature = (nn_service_addr, channel, gamma, dirichlet_weight, dirichlet_leaf_weight, dirichlet_alpha, c_puct_init, c_puct_base, n_threads=1, pruning_tree=false, debug_prints=false))]
    fn new(
        nn_service_addr: String,
        channel: u32,
        gamma: f32,
        dirichlet_weight: f32,
        dirichlet_leaf_weight: f32,
        dirichlet_alpha: f32,
        c_puct_init: f32,
        c_puct_base: f32,
        n_threads: usize,
        pruning_tree: bool,
        debug_prints: bool,
    ) -> Self {
        let n = n_threads.max(1);
        let services: Vec<_> = (0..n)
            .map(|_| alpha_cc_nn_service::NNRemote::connect(&nn_service_addr))
            .collect();
        PyMCTS(MCTS::new(
            services,
            channel,
            alpha_cc_mcts::MCTSParams {
                gamma,
                dirichlet_weight,
                dirichlet_leaf_weight,
                dirichlet_alpha,
                c_puct_init,
                c_puct_base,
            },
            pruning_tree,
            debug_prints,
        ))
    }

    fn run(&self, board: &PyBoard, rollout_depth: usize) -> f32 {
        let (_, value) = self.0.run_rollouts_inner(&board.0, 1, rollout_depth, 1.0);
        value
    }

    #[pyo3(signature = (board, n_rollouts, rollout_depth, temperature=1.0))]
    fn run_rollouts<'py>(
        &self,
        py: Python<'py>,
        board: &PyBoard,
        n_rollouts: usize,
        rollout_depth: usize,
        temperature: f32,
    ) -> (Bound<'py, numpy::PyArray1<f32>>, f32) {
        let (pi, mean_value) = self.0.run_rollouts_inner(&board.0, n_rollouts, rollout_depth, temperature);
        let pi_arr = numpy::IntoPyArray::into_pyarray(
            numpy::ndarray::Array1::from_vec(pi), py,
        );
        (pi_arr, mean_value)
    }

    fn get_node(&self, board: &PyBoard) -> Option<PyMCTSNode> {
        self.0.get_node_snapshot(&board.0).map(PyMCTSNode::from)
    }

    fn get_nodes(&self) -> Vec<(PyBoard, PyMCTSNode)> {
        self.0.get_all_nodes()
            .into_iter()
            .map(|(board, node)| (PyBoard(board), PyMCTSNode(node)))
            .collect()
    }

    fn on_move_applied(&self, board: &PyBoard) {
        self.0.notify_move_applied(&board.0);
    }

    fn clear_nodes(&self) {
        self.0.clear_tree();
    }

    fn get_fetch_stats(&self) -> PyFetchStats {
        PyFetchStats {
            total_fetch_time_us: 0,
            total_fetches: 0,
        }
    }
}

// ──────────────────────────────────────────────────
// Standalone functions
// ──────────────────────────────────────────────────

#[gen_stub_pyfunction]
#[pyfunction]
fn create_move_mask(moves: Vec<PyMove>) -> [[[[bool; MAX_SIZE]; MAX_SIZE]; MAX_SIZE]; MAX_SIZE] {
    let inner_moves: Vec<alpha_cc_core::Move> = moves.into_iter().map(|m| m.0).collect();
    alpha_cc_core::create_move_mask(inner_moves)
}

#[gen_stub_pyfunction]
#[pyfunction]
fn create_move_index_map(moves: Vec<PyMove>) -> HashMap<usize, (PyHexCoord, PyHexCoord)> {
    let inner_moves: Vec<alpha_cc_core::Move> = moves.into_iter().map(|m| m.0).collect();
    let inner_map = alpha_cc_core::create_move_index_map(inner_moves);
    inner_map.into_iter()
        .map(|(k, (a, b))| (k, (PyHexCoord(a), PyHexCoord(b))))
        .collect()
}

#[gen_stub_pyfunction]
#[pyfunction]
fn preds_from_logits<'py>(
    logits_flat: numpy::PyReadonlyArray1<'py, f32>,
    values_flat: numpy::PyReadonlyArray1<'py, f32>,
    boards: Vec<PyBoard>,
    board_size: usize,
) -> PyResult<Vec<PyNNPred>> {
    let logits = logits_flat.as_slice()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("logits not contiguous: {e}")))?;
    let values = values_flat.as_slice()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("values not contiguous: {e}")))?;
    let s = board_size;
    let stride = s * s * s * s;
    let mut preds = Vec::with_capacity(boards.len());

    for (i, py_board) in boards.iter().enumerate() {
        let logits_slice = &logits[i * stride..(i + 1) * stride];
        let moves = find_all_moves(&py_board.0);

        let move_logits: Vec<f32> = moves.iter().map(|m| {
            let fx = m.from_coord.x as usize;
            let fy = m.from_coord.y as usize;
            let tx = m.to_coord.x as usize;
            let ty = m.to_coord.y as usize;
            logits_slice[fx * s * s * s + fy * s * s + tx * s + ty]
        }).collect();

        let pi = alpha_cc_nn::softmax(&move_logits);
        let value = values[i];
        preds.push(PyNNPred(alpha_cc_nn::NNPred::new(pi, value)));
    }

    Ok(preds)
}

type MoveCoords = Vec<(u8, u8, u8, u8)>;

#[gen_stub_pyfunction]
#[pyfunction]
#[allow(clippy::type_complexity)]
fn build_inference_request<'py>(
    py: Python<'py>,
    board: &PyBoard,
) -> (Bound<'py, numpy::PyArray3<f32>>, MoveCoords) {
    let s = board.0.get_size() as usize;
    let matrix = board.0.get_matrix();
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

    let moves = find_all_moves(&board.0);
    let move_coords: MoveCoords = moves.iter().map(|m| {
        (m.from_coord.x, m.from_coord.y, m.to_coord.x, m.to_coord.y)
    }).collect();

    let arr = numpy::ndarray::Array3::<f32>::from_shape_vec([2, s, s], tensor_data).unwrap();
    let numpy_arr = numpy::IntoPyArray::into_pyarray(arr, py);
    (numpy_arr, move_coords)
}


// ──────────────────────────────────────────────────
// Module definition
// ──────────────────────────────────────────────────

#[pymodule]
#[pyo3(name = "alpha_cc_engine")]
fn alpha_cc(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyBoard>()?;
    m.add_class::<PyBoardInfo>()?;
    m.add_class::<PyHexCoord>()?;
    m.add_class::<PyMove>()?;
    m.add_class::<PyMCTS>()?;
    m.add_class::<PyMCTSNode>()?;
    m.add_class::<PyNNPred>()?;
    m.add_class::<PyFetchStats>()?;
    m.add_function(wrap_pyfunction!(create_move_mask, m)?)?;
    m.add_function(wrap_pyfunction!(create_move_index_map, m)?)?;
    m.add_function(wrap_pyfunction!(preds_from_logits, m)?)?;
    m.add_function(wrap_pyfunction!(build_inference_request, m)?)?;
    Ok(())
}

define_stub_info_gatherer!(stub_info);
