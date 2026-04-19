use std::collections::HashMap;

use alpha_cc_core::cc::CCBoard;
use pyo3::prelude::*;
use pyo3_stub_gen_derive::{gen_stub_pyclass, gen_stub_pymethods};

use alpha_cc_mcts::descent::{Descent, DirichletParams, PuctParams, SigmaParams};
use alpha_cc_mcts::scheduler::{FreeConfig, RootScheduler};
use alpha_cc_mcts::{RolloutResult as MctsRolloutResult, MCTS};
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

/// Type-erasing backend trait — lets PyMCTS hold any (Descent, Scheduler) combination
/// behind a single `Box<dyn PyMctsBackend>`, so pyo3 doesn't need to know which is in use.
trait PyMctsBackend: Send + Sync {
    fn run_rollouts(&self, board: &CCBoard, n_rollouts: usize, rollout_depth: usize) -> MctsRolloutResult;
    fn notify_move_applied(&self, board: &CCBoard);
    fn clear_tree(&self);
    fn get_node_snapshot(&self, board: &CCBoard) -> Option<alpha_cc_mcts::MCTSNode>;
    fn get_all_nodes(&self) -> HashMap<CCBoard, alpha_cc_mcts::MCTSNode>;
}

impl<D, S> PyMctsBackend for MCTS<CCBoard, PredictionSources, D, S>
where D: Descent + 'static, S: RootScheduler<CCBoard, PredictionSources, D> + 'static
{
    fn run_rollouts(&self, board: &CCBoard, n_rollouts: usize, rollout_depth: usize) -> MctsRolloutResult {
        self.run_rollout_threads(board, n_rollouts, rollout_depth)
    }
    fn notify_move_applied(&self, board: &CCBoard) { MCTS::notify_move_applied(self, board); }
    fn clear_tree(&self) { MCTS::clear_tree(self); }
    fn get_node_snapshot(&self, board: &CCBoard) -> Option<alpha_cc_mcts::MCTSNode> {
        MCTS::get_node_snapshot(self, board)
    }
    fn get_all_nodes(&self) -> HashMap<CCBoard, alpha_cc_mcts::MCTSNode> {
        MCTS::get_all_nodes(self)
    }
}

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

    #[getter]
    fn kl_prior_posterior(&self) -> f32 {
        self.0.search_stats.kl_prior_posterior
    }

    #[getter]
    fn kl_posterior_prior(&self) -> f32 {
        self.0.search_stats.kl_posterior_prior
    }
}

#[gen_stub_pyclass]
#[pyclass(name = "ImprovedHalvingParams", module = "alpha_cc_engine", from_py_object)]
#[derive(Clone)]
pub struct PyImprovedHalvingParams {
    #[pyo3(get, set)] pub c_visit: f32,
    #[pyo3(get, set)] pub c_scale: f32,
    #[pyo3(get, set)] pub all_at_least_once: bool,
    #[pyo3(get, set)] pub base_count: usize,
    #[pyo3(get, set)] pub floor_count: usize,
    #[pyo3(get, set)] pub keep_frac: f32,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyImprovedHalvingParams {
    #[new]
    #[pyo3(signature = (c_visit=50.0, c_scale=1.0, all_at_least_once=false, base_count=16, floor_count=5, keep_frac=0.5))]
    fn new(c_visit: f32, c_scale: f32, all_at_least_once: bool, base_count: usize, floor_count: usize, keep_frac: f32) -> Self {
        Self { c_visit, c_scale, all_at_least_once, base_count, floor_count, keep_frac }
    }
}

#[gen_stub_pyclass]
#[pyclass(name = "PuctFreeParams", module = "alpha_cc_engine", from_py_object)]
#[derive(Clone)]
pub struct PyPuctFreeParams {
    #[pyo3(get, set)] pub c_puct_init: f32,
    #[pyo3(get, set)] pub c_puct_base: f32,
    #[pyo3(get, set)] pub dirichlet_weight: f32,
    #[pyo3(get, set)] pub dirichlet_alpha: f32,
    #[pyo3(get, set)] pub temperature: f32,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyPuctFreeParams {
    #[new]
    #[pyo3(signature = (c_puct_init=2.0, c_puct_base=10000.0, dirichlet_weight=0.15, dirichlet_alpha=0.15, temperature=1.0))]
    fn new(c_puct_init: f32, c_puct_base: f32, dirichlet_weight: f32, dirichlet_alpha: f32, temperature: f32) -> Self {
        Self { c_puct_init, c_puct_base, dirichlet_weight, dirichlet_alpha, temperature }
    }
}

#[gen_stub_pyclass]
#[pyclass(name = "MCTS", module = "alpha_cc_engine")]
pub struct PyMCTS(Box<dyn PyMctsBackend>);

fn build_services(addr: &str, n: usize, dummy: bool) -> Vec<PredictionSources> {
    let n = n.max(1);
    if dummy {
        (0..n).map(|_| PredictionSources::dummy()).collect()
    } else {
        (0..n).map(|_| PredictionSources::real(addr)).collect()
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyMCTS {
    #[allow(clippy::too_many_arguments)]
    #[new]
    #[pyo3(signature = (nn_service_addr, channel, gamma, improved_halving=None, puct_free=None, n_threads=1, pruning_tree=false, debug_prints=false, dummy_preds=false))]
    fn new(
        nn_service_addr: String,
        channel: u32,
        gamma: f32,
        improved_halving: Option<PyImprovedHalvingParams>,
        puct_free: Option<PyPuctFreeParams>,
        n_threads: usize,
        pruning_tree: bool,
        debug_prints: bool,
        dummy_preds: bool,
    ) -> PyResult<Self> {
        let services = build_services(&nn_service_addr, n_threads, dummy_preds);
        match (improved_halving, puct_free) {
            (Some(p), None) => {
                let sigma = SigmaParams { c_visit: p.c_visit, c_scale: p.c_scale };
                let gumbel = alpha_cc_mcts::GumbelParams {
                    all_at_least_once: p.all_at_least_once,
                    base_count: p.base_count,
                    floor_count: p.floor_count,
                    keep_frac: p.keep_frac,
                };
                Ok(PyMCTS(Box::new(MCTS::new_improved_halving(
                    services, channel, gamma, sigma, gumbel, pruning_tree, debug_prints,
                ))))
            }
            (None, Some(p)) => {
                let dirichlet = if p.dirichlet_weight > 0.0 {
                    Some(DirichletParams { weight: p.dirichlet_weight, alpha: p.dirichlet_alpha })
                } else { None };
                let puct = PuctParams {
                    c_puct_init: p.c_puct_init,
                    c_puct_base: p.c_puct_base,
                    dirichlet,
                };
                let free = FreeConfig { temperature: p.temperature };
                Ok(PyMCTS(Box::new(MCTS::new_puct_free(
                    services, channel, gamma, puct, free, pruning_tree, debug_prints,
                ))))
            }
            (Some(_), Some(_)) => Err(pyo3::exceptions::PyValueError::new_err(
                "MCTS: specify exactly one of `improved_halving` or `puct_free`, not both",
            )),
            (None, None) => Err(pyo3::exceptions::PyValueError::new_err(
                "MCTS: specify exactly one of `improved_halving` or `puct_free`",
            )),
        }
    }

    fn run(&self, board: &PyBoard, rollout_depth: usize) -> f32 {
        self.0.run_rollouts(&board.0, 1, rollout_depth).value
    }

    #[pyo3(signature = (board, n_rollouts, rollout_depth))]
    fn run_rollouts(&self, board: &PyBoard, n_rollouts: usize, rollout_depth: usize) -> PyRolloutResult {
        PyRolloutResult(self.0.run_rollouts(&board.0, n_rollouts, rollout_depth))
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
