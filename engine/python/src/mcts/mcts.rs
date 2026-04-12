use alpha_cc_core::cc::CCBoard;
use pyo3::prelude::*;
use pyo3_stub_gen_derive::{gen_stub_pyclass, gen_stub_pymethods};

use alpha_cc_nn::PredictionSource;
use crate::core::PyBoard;
use crate::nn::PyFetchStats;
use super::mcts_node::PyMCTSNode;

enum PredictionSources {
    Dummy(alpha_cc_nn::mock::MockPredictor),
    Real(alpha_cc_nn_service::NNRemote<CCBoard>),
}

impl PredictionSources {
    fn dummy() -> Self {
        Self::Dummy(alpha_cc_nn::mock::MockPredictor::uniform(0.0))
    }

    fn real(addr: &str) -> Self {
        let nn = alpha_cc_nn_service::NNRemote::connect(addr);
        Self::Real(nn)
    }
}

impl PredictionSource<CCBoard> for PredictionSources {
    fn predict(&self, board: &alpha_cc_core::cc::CCBoard, model_id: u32) -> alpha_cc_nn::NNPred {
        match self {
            Self::Dummy(dummy) => dummy.predict(board, model_id),
            Self::Real(nn_remote) => nn_remote.predict(board, model_id),
        }
    }
}

type MCTS = alpha_cc_mcts::MCTS<CCBoard, PredictionSources>;

#[gen_stub_pyclass]
#[pyclass(name = "RolloutResult", module = "alpha_cc_engine")]
pub struct PyRolloutResult(alpha_cc_mcts::RolloutResult);

#[gen_stub_pymethods]
#[pymethods]
impl PyRolloutResult {
    #[new]
    #[pyo3(signature = (pi, value, mcts_wdl, greedy_backup_wdl=None))]
    fn new(pi: Vec<f32>, value: f32, mcts_wdl: [f32; 3], greedy_backup_wdl: Option<[f32; 3]>) -> Self {
        let greedy_backup_wdl = greedy_backup_wdl.unwrap_or(mcts_wdl);
        Self(alpha_cc_mcts::RolloutResult {
            pi, value, mcts_wdl, greedy_backup_wdl,
            search_stats: alpha_cc_mcts::SearchStats::default(),
        })
    }

    #[getter]
    fn pi<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<f32>> {
        numpy::IntoPyArray::into_pyarray(
            numpy::ndarray::Array1::from_vec(self.0.pi.clone()), py,
        )
    }

    #[getter]
    fn value(&self) -> f32 {
        self.0.value
    }

    #[getter]
    fn mcts_wdl(&self) -> [f32; 3] {
        self.0.mcts_wdl
    }

    #[getter]
    fn greedy_backup_wdl(&self) -> [f32; 3] {
        self.0.greedy_backup_wdl
    }

    #[getter]
    fn prior_entropy(&self) -> f32 {
        self.0.search_stats.prior_entropy
    }

    #[getter]
    fn target_entropy(&self) -> f32 {
        self.0.search_stats.target_entropy
    }

    #[getter]
    fn logit_std(&self) -> f32 {
        self.0.search_stats.logit_std
    }

    #[getter]
    fn sigma_q_std(&self) -> f32 {
        self.0.search_stats.sigma_q_std
    }
}

#[gen_stub_pyclass]
#[pyclass(name = "MCTS", module = "alpha_cc_engine")]
pub struct PyMCTS(MCTS);

#[gen_stub_pymethods]
#[pymethods]
impl PyMCTS {
    #[allow(clippy::too_many_arguments)]
    #[new]
    #[pyo3(signature = (nn_service_addr, channel, gamma, c_visit=50.0, c_scale=1.0, all_at_least_once=false, base_count=16, floor_count=5, keep_frac=0.5, n_threads=1, pruning_tree=false, debug_prints=false, dummy_preds=false))]
    fn new(
        nn_service_addr: String,
        channel: u32,
        gamma: f32,
        c_visit: f32,
        c_scale: f32,
        all_at_least_once: bool,
        base_count: usize,
        floor_count: usize,
        keep_frac: f32,
        n_threads: usize,
        pruning_tree: bool,
        debug_prints: bool,
        dummy_preds: bool,
    ) -> Self {
        let n = n_threads.max(1);
        let services = if dummy_preds {
            (0..n).map(|_| PredictionSources::dummy()).collect()
        } else {
            (0..n).map(|_| PredictionSources::real(&nn_service_addr)).collect()
        };
        PyMCTS(MCTS::new(
            services,
            channel,
            alpha_cc_mcts::MCTSParams {
                gamma,
                c_visit,
                c_scale,
                gumbel: alpha_cc_mcts::GumbelParams {
                    all_at_least_once,
                    base_count,
                    floor_count,
                    keep_frac,
                },
            },
            pruning_tree,
            debug_prints,
        ))
    }

    fn run(&self, board: &PyBoard, rollout_depth: usize) -> f32 {
        self.0.run_rollout_threads(&board.0, 1, rollout_depth).value
    }

    #[pyo3(signature = (board, n_rollouts, rollout_depth))]
    fn run_rollouts(
        &self,
        board: &PyBoard,
        n_rollouts: usize,
        rollout_depth: usize,
    ) -> PyRolloutResult {
        PyRolloutResult(self.0.run_rollout_threads(&board.0, n_rollouts, rollout_depth))
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
