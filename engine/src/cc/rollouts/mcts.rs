#[cfg(feature = "extension-module")]
extern crate pyo3;

use std::sync::Arc;
#[cfg(feature = "extension-module")]
use std::collections::HashMap;

#[cfg(feature = "extension-module")]
use pyo3::prelude::*;

use crate::cc::dtypes::NNQuantizedPi;
use crate::cc::game::board::Board;
use crate::cc::game::moves::find_all_moves;
#[cfg(feature = "extension-module")]
use crate::cc::predictions::FetchStats;
use crate::cc::rollouts::mcts_node::MCTSNode;
use crate::cc::rollouts::noise;
use crate::cc::rollouts::tree::Tree;
use crate::cc::predictions::nn_remote::NNRemote;

#[cfg_attr(feature = "extension-module", pyo3::prelude::pyclass(module="alpha_cc_engine"))]
pub struct MCTS {
    tree: Arc<Tree>,
    services: Vec<NNRemote>,
    model_id: u32,
    mcts_params: MCTSParams,
}

#[derive(Clone)]
pub struct MCTSParams {
    pub gamma: f32,
    pub dirichlet_weight: f32,
    pub dirichlet_leaf_weight: f32,
    pub dirichlet_alpha: f32,
    pub c_puct_init: f32,
    pub c_puct_base: f32,
}


impl MCTS {
    pub fn new(
        nn_service_addr: &str,
        model_id: u32,
        mcts_params: MCTSParams,
        n_threads: usize,
        pruning_tree: bool,
        debug_prints: bool,
    ) -> Self {
        let n = n_threads.max(1);
        let services = (0..n)
            .map(|_| NNRemote::connect(nn_service_addr))
            .collect();
        MCTS {
            tree: Arc::new(Tree::new(n, pruning_tree, debug_prints)),
            services,
            model_id,
            mcts_params,
        }
    }

    /// Perform a single rollout from `board` down the tree.
    fn rollout(
        tree: &Tree,
        service: &NNRemote,
        model_id: u32,
        board: &Board,
        remaining_depth: usize,
        params: &MCTSParams,
        root_noise: &Option<Vec<f32>>,
        thread_id: usize,
    ) -> f32 {
        let info = board.get_info();
        if info.game_over {
            return -info.reward;
        }

        if let Some(data) = tree.visit(board, thread_id) {
            if remaining_depth == 0 {
                return -data.get_v();
            }

            let a = find_best_action(&data, params.c_puct_init, params.c_puct_base, root_noise, params);
            let moves = find_all_moves(board);
            let s_prime = board.apply(&moves[a]);
            tree.record_action(thread_id, &s_prime, a);

            data.apply_virtual_loss(a);
            drop(data);

            let v = MCTS::rollout(tree, service, model_id, &s_prime, remaining_depth - 1, params, &None, thread_id);

            let data = tree.get_data(board)
                .expect("node data disappeared mid-rollout");
            data.resolve_virtual_loss(a, params.gamma * v);

            return -(params.gamma * v);
        }

        let node = MCTS::new_leaf_for(board, service, model_id, params);
        let v = node.v.dequantize();
        tree.insert(board, node);
        -v
    }

    fn new_leaf_for(board: &Board, service: &NNRemote, model_id: u32, params: &MCTSParams) -> MCTSNode {
        let nn_pred = service.predict(board, model_id);
        let v = nn_pred.value();
        let mut pi = nn_pred.pi();
        let num_actions = pi.len();

        if params.dirichlet_leaf_weight > 0.0 {
            noise::apply_dirichlet_noise(&mut pi, params);
        }

        MCTSNode::new(pi, v, num_actions)
    }

    pub fn notify_move_applied(&self, board: &Board) {
        if let Some(action) = self.tree.maybe_prune(board) {
            if self.tree.debug_prints() {
                let report = self.tree.memory_report();
                log::debug!("[mcts] pruned action={action}: {report}");
            }
        }
    }

    pub fn run_rollouts_inner(
        &self,
        board: &Board,
        n_rollouts: usize,
        rollout_depth: usize,
        temperature: f32,
    ) -> (Vec<f32>, f32) {
        let n_threads = self.services.len().min(n_rollouts);
        let rollouts_per_thread = n_rollouts / n_threads;
        let remainder = n_rollouts % n_threads;

        let tree = &self.tree;
        let params = &self.mcts_params;
        let model_id = self.model_id;

        let noise = if let Some(node) = tree.get_data(board) {
            let pi = NNQuantizedPi::dequantize_vec(&node.pi);
            noise::generate_dirichlet_noise(&pi, params).ok()
        } else {
            let node = MCTS::new_leaf_for(board, &self.services[0], model_id, params);
            let pi = NNQuantizedPi::dequantize_vec(&node.pi);
            tree.insert(board, node);
            noise::generate_dirichlet_noise(&pi, params).ok()
        };

        let value_sum: f64 = std::thread::scope(|s| {
            let handles: Vec<_> = self.services[..n_threads].iter().enumerate().map(|(i, svc)| {
                let board = board.clone();
                let extra = if i < remainder { 1 } else { 0 };
                let count = rollouts_per_thread + extra;
                let noise = noise.clone();
                s.spawn(move || {
                    let mut local_sum = 0.0f64;
                    for _ in 0..count {
                        tree.begin_rollout(i);
                        let v = MCTS::rollout(tree, svc, model_id, &board, rollout_depth, params, &noise, i);
                        local_sum += (-v) as f64;
                    }
                    local_sum
                })
            }).collect();

            handles.into_iter()
                .map(|h| h.join().unwrap())
                .sum()
        });

        // Merge thread paths into the tracking tree.
        tree.finalize_rollouts(board);

        let mean_value = (value_sum / n_rollouts as f64) as f32;

        // Build pi from root node visit counts
        let pi = if let Some(data) = self.tree.get_data(board) {
            let mut weights: Vec<f32> = (0..data.num_actions())
                .map(|a| data.get_n(a) as f32)
                .collect();
            if temperature != 1.0 {
                let inv_temp = 1.0 / temperature;
                for w in weights.iter_mut() {
                    *w = w.powf(inv_temp);
                }
            }
            let sum: f32 = weights.iter().sum();
            if sum > 0.0 {
                weights.iter().map(|&w| w / sum).collect()
            } else {
                let n = weights.len();
                vec![1.0 / n as f32; n]
            }
        } else {
            let moves = find_all_moves(board);
            let n = moves.len();
            vec![1.0 / n as f32; n]
        };

        (pi, mean_value)
    }
}


fn find_best_action(data: &MCTSNode, c_puct_init: f32, c_puct_base: f32, root_noise: &Option<Vec<f32>>, params: &MCTSParams) -> usize {
    let sum_n_f = data.total_visits() as f32;
    let c_puct = c_puct_init + ((sum_n_f + c_puct_base + 1.0) / c_puct_base).ln();

    let mut best_action = 0;
    let mut best_u = f32::MIN;
    let mut pi = NNQuantizedPi::dequantize_vec(&data.pi);
    if let Some(noise) = root_noise {
        noise::blend_with_noise(&mut pi, noise, params.dirichlet_weight);
    }
    for i in 0..data.num_actions() {
        let pi_a = pi[i];
        let n_a = data.get_n(i) as f32;
        let prior_weight = c_puct * sum_n_f.sqrt() / (1.0 + n_a);

        let u = data.get_q(i) + prior_weight * pi_a;
        if u > best_u {
            best_u = u;
            best_action = i;
        }
    }
    best_action
}


// ── Python interface ──

#[cfg(feature = "extension-module")]
#[pymethods]
impl MCTS {
    #[allow(clippy::too_many_arguments)]
    #[new]
    #[pyo3(signature = (nn_service_addr, channel, gamma, dirichlet_weight, dirichlet_leaf_weight, dirichlet_alpha, c_puct_init, c_puct_base, n_threads=1, pruning_tree=false, debug_prints=false))]
    fn create(
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
    ) -> MCTS {
        MCTS::new(
            &nn_service_addr,
            channel,
            MCTSParams {
                gamma,
                dirichlet_weight,
                dirichlet_leaf_weight,
                dirichlet_alpha,
                c_puct_init,
                c_puct_base,
            },
            n_threads,
            pruning_tree,
            debug_prints,
        )
    }

    pub fn run(&self, board: &Board, rollout_depth: usize) -> f32 {
        let (_, value) = self.run_rollouts_inner(board, 1, rollout_depth, 1.0);
        value
    }

    #[pyo3(signature = (board, n_rollouts, rollout_depth, temperature=1.0))]
    pub fn run_rollouts<'py>(
        &self,
        py: Python<'py>,
        board: &Board,
        n_rollouts: usize,
        rollout_depth: usize,
        temperature: f32,
    ) -> (Bound<'py, numpy::PyArray1<f32>>, f32) {
        let (pi, mean_value) = self.run_rollouts_inner(board, n_rollouts, rollout_depth, temperature);
        let pi_arr = numpy::IntoPyArray::into_pyarray(
            numpy::ndarray::Array1::from_vec(pi), py,
        );
        (pi_arr, mean_value)
    }

    pub fn get_node(&self, board: &Board) -> Option<MCTSNode> {
        self.tree.get_data(board).map(|data| data.snapshot())
    }

    pub fn get_nodes(&self) -> HashMap<Board, MCTSNode> {
        self.tree.iter_data()
            .map(|entry| (entry.key().clone(), entry.value().snapshot()))
            .collect()
    }

    pub fn on_move_applied(&self, board: &Board) {
        self.notify_move_applied(board);
    }

    pub fn clear_nodes(&self) {
        self.tree.clear();
    }

    pub fn get_fetch_stats(&self) -> FetchStats {
        FetchStats {
            total_fetch_time_us: 0,
            total_fetches: 0,
        }
    }
}
