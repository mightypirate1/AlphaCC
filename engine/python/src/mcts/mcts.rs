use std::collections::HashMap;

use alpha_cc_core::cc::CCBoard;
use pyo3::prelude::*;
use pyo3_stub_gen_derive::{gen_stub_pyclass, gen_stub_pymethods};

use alpha_cc_mcts::{
    DirichletParams, FreeConfig, FreeScheduler, HalvingScheduler, PuctParams,
    RolloutResult as MctsRolloutResult, Scheduler, SigmaParams,
};

use crate::core::PyBoard;
use crate::nn::PyFetchStats;
use super::mcts_node::PyMCTSNode;
use super::params::{PyImprovedHalvingParams, PyPuctFreeParams};
use super::prediction_sources::{build_services, PredictionSources};
use super::rollout_result::PyRolloutResult;

/// Type-erasing backend trait — lets PyMCTS hold any concrete scheduler (and thus
/// any Descent) behind a single `Box<dyn PyMctsBackend>`, so pyo3 doesn't need to
/// know which combination is in use.
trait PyMctsBackend: Send + Sync {
    fn run_rollouts(&self, board: &CCBoard, n_rollouts: usize, rollout_depth: usize) -> MctsRolloutResult;
    fn notify_move_applied(&self, board: &CCBoard);
    fn clear_tree(&self);
    fn get_node_snapshot(&self, board: &CCBoard) -> Option<alpha_cc_mcts::MCTSNode>;
    fn get_all_nodes(&self) -> HashMap<CCBoard, alpha_cc_mcts::MCTSNode>;
}

macro_rules! impl_pymcts_backend {
    ($ty:ty) => {
        impl PyMctsBackend for $ty {
            fn run_rollouts(&self, board: &CCBoard, n_rollouts: usize, rollout_depth: usize) -> MctsRolloutResult {
                self.run(board, n_rollouts, rollout_depth)
            }
            fn notify_move_applied(&self, board: &CCBoard) { self.mcts().notify_move_applied(board); }
            fn clear_tree(&self) { self.mcts().clear_tree(); }
            fn get_node_snapshot(&self, board: &CCBoard) -> Option<alpha_cc_mcts::MCTSNode> {
                self.mcts().get_node_snapshot(board)
            }
            fn get_all_nodes(&self) -> HashMap<CCBoard, alpha_cc_mcts::MCTSNode> {
                self.mcts().get_all_nodes()
            }
        }
    };
}

impl_pymcts_backend!(HalvingScheduler<CCBoard, PredictionSources, alpha_cc_mcts::ImprovedPolicyDescent>);
impl_pymcts_backend!(FreeScheduler<CCBoard, PredictionSources, alpha_cc_mcts::PuctDescent>);

#[gen_stub_pyclass]
#[pyclass(name = "MCTS", module = "alpha_cc_engine")]
pub struct PyMCTS(Box<dyn PyMctsBackend>);

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
                Ok(PyMCTS(Box::new(HalvingScheduler::build_improved_halving(
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
                Ok(PyMCTS(Box::new(FreeScheduler::build_puct_free(
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
