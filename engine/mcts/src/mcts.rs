use std::sync::Arc;

use alpha_cc_nn::NNQuantizedPi;
use alpha_cc_nn::PredictionSource;
use alpha_cc_core::Board;
use alpha_cc_core::moves::find_all_moves;
use crate::mcts_node::MCTSNode;
use crate::noise;
use crate::tree::Tree;

pub struct MCTS<T: PredictionSource> {
    tree: Arc<Tree>,
    services: Vec<T>,
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


impl<T: PredictionSource> MCTS<T> {
    pub fn new(
        services: Vec<T>,
        model_id: u32,
        mcts_params: MCTSParams,
        pruning_tree: bool,
        debug_prints: bool,
    ) -> Self {
        let n = services.len().max(1);
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
        service: &T,
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

            let v = MCTS::<T>::rollout(tree, service, model_id, &s_prime, remaining_depth - 1, params, &None, thread_id);

            let data = tree.get_data(board)
                .expect("node data disappeared mid-rollout");
            data.resolve_virtual_loss(a, params.gamma * v);

            return -(params.gamma * v);
        }

        let node = MCTS::<T>::new_leaf_for(board, service, model_id, params);
        let v = node.v.dequantize();
        tree.insert(board, node);
        -v
    }

    fn new_leaf_for(board: &Board, service: &T, model_id: u32, params: &MCTSParams) -> MCTSNode {
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

    /// Get a snapshot of the MCTS node for a board position (if it exists in the tree).
    pub fn get_node_snapshot(&self, board: &Board) -> Option<MCTSNode> {
        self.tree.get_data(board).map(|data| data.snapshot())
    }

    /// Clear all tree nodes and tracking state.
    pub fn clear_tree(&self) {
        self.tree.clear();
    }

    /// Get snapshots of all nodes in the tree.
    pub fn get_all_nodes(&self) -> std::collections::HashMap<Board, MCTSNode> {
        self.tree.iter_data()
            .map(|entry| (entry.key().clone(), entry.value().snapshot()))
            .collect()
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
            let node = MCTS::<T>::new_leaf_for(board, &self.services[0], model_id, params);
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
                        let v = MCTS::<T>::rollout(tree, svc, model_id, &board, rollout_depth, params, &noise, i);
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
                let max_w = weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                if max_w > 0.0 {
                    for w in weights.iter_mut() {
                        // Divide by max before exponentiating to prevent overflow
                        *w = (*w / max_w).powf(inv_temp);
                    }
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


