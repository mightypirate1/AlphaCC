use std::sync::Arc;

use alpha_cc_core::Board;
use alpha_cc_nn::{softmax, PredictionSource};
use crate::mcts_node::MCTSNode;
use crate::outcome::Outcome;
use crate::noise;
use crate::tree::Tree;

pub struct RolloutResult {
    pub pi: Vec<f32>,
    pub value: f32,
    pub mcts_wdl: [f32; 3],
    pub greedy_backup_wdl: [f32; 3],
    pub search_stats: SearchStats,
}

#[derive(Clone, Default)]
pub struct SearchStats {
    pub prior_entropy: f32,
    pub target_entropy: f32,
    pub logit_std: f32,
    pub sigma_q_std: f32,
}

fn entropy(probs: &[f32]) -> f32 {
    -probs.iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| p * p.ln())
        .sum::<f32>()
}

fn std_dev(values: &[f32]) -> f32 {
    if values.is_empty() { return 0.0; }
    let n = values.len() as f32;
    let mean = values.iter().sum::<f32>() / n;
    let variance = values.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / n;
    variance.sqrt()
}

pub struct MCTS<B: Board, T: PredictionSource<B>> {
    tree: Arc<Tree<B>>,
    services: Vec<T>,
    model_id: u32,
    mcts_params: MCTSParams,
}

#[derive(Clone)]
pub struct MCTSParams {
    pub gamma: f32,
    pub c_visit: f32,
    pub c_scale: f32,
    pub gumbel: GumbelParams,
}

#[derive(Clone)]
pub struct GumbelParams {
    pub all_at_least_once: bool,
    pub base_count: usize,
    pub floor_count: usize,
    pub keep_frac: f32,
}

/// σ(q) = (c_visit + N_max) * c_scale * q
#[inline]
fn sigma(q: f32, n_max: u32, c_visit: f32, c_scale: f32) -> f32 {
    (c_visit + n_max as f32) * c_scale * q
}

impl<B: Board, T: PredictionSource<B>> MCTS<B, T> {
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
    /// Returns (value, outcome) from the caller's (parent's) perspective.
    fn rollout(
        &self,
        model_id: u32,
        board: &B,
        remaining_depth: usize,
        forced_action: Option<usize>,
        thread_id: usize,
    ) -> (f32, Option<Outcome>) {
        let info = board.get_info();
        if info.game_over {
            let outcome = Outcome::from_wdl(&info.wdl);
            return (-info.wdl.to_value(), Some(outcome.flip()));
        }

        if let Some(data) = self.tree.visit(board, thread_id) {
            if remaining_depth == 0 {
                return (-data.get_v(), None);
            }

            let a = forced_action.unwrap_or_else(|| self.find_best_action(&data));
            let moves = board.legal_moves();
            let s_prime = board.apply_move(&moves[a]);
            self.tree.record_action(thread_id, &s_prime, a);

            data.apply_virtual_loss(a);
            drop(data);

            let (v, outcome) = self.rollout(model_id, &s_prime, remaining_depth - 1, None, thread_id);

            let data = self.tree.get_data(board)
                .expect("node data disappeared mid-rollout");
            if let Some(o) = outcome {
                data.tick_outcome(o);
            }
            data.resolve_virtual_loss(a, self.mcts_params.gamma * v);

            return (-(self.mcts_params.gamma * v), outcome.map(|o| o.flip()));
        }

        let node = self.new_leaf_for(board, &self.services[thread_id], model_id);
        let v = node.v.dequantize();
        self.tree.insert(board, node);
        (-v, None)
    }

    fn new_leaf_for(&self, board: &B, service: &T, model_id: u32) -> MCTSNode {
        let nn_pred = service.predict(board, model_id);
        let v = nn_pred.expected_value();
        let pi_logits = nn_pred.pi_logits();
        let num_actions = pi_logits.len();
        let wdl = nn_pred.wdl();
        MCTSNode::new(pi_logits, v, [wdl[0], wdl[1], wdl[2]], num_actions)
    }

    pub fn notify_move_applied(&self, board: &B) {
        if let Some(action) = self.tree.maybe_prune(board) {
            if self.tree.debug_prints() {
                let report = self.tree.memory_report();
                log::debug!("[mcts] pruned action={action}: {report}");
            }
        }
    }

    /// Get a snapshot of the MCTS node for a board position (if it exists in the tree).
    pub fn get_node_snapshot(&self, board: &B) -> Option<MCTSNode> {
        self.tree.get_data(board).map(|data| data.snapshot())
    }

    /// Clear all tree nodes and tracking state.
    pub fn clear_tree(&self) {
        self.tree.clear();
    }

    /// Get snapshots of all nodes in the tree.
    pub fn get_all_nodes(&self) -> std::collections::HashMap<B, MCTSNode> {
        self.tree.iter_data()
            .map(|entry| (entry.key().clone(), entry.value().snapshot()))
            .collect()
    }

    pub fn run_rollout_threads(
        &self,
        board: &B,
        n_rollouts: usize,
        rollout_depth: usize,
    ) -> RolloutResult {
        let tree = &self.tree;
        let model_id = self.model_id;
        let gumbel = &self.mcts_params.gumbel;

        // Ensure root node exists.
        if tree.get_data(board).is_none() {
            let node = self.new_leaf_for(board, &self.services[0], model_id);
            tree.insert(board, node);
        }

        let root = tree.get_data(board).unwrap();
        let num_actions = root.num_actions();
        let logits = root.pi_logits.clone();
        drop(root);

        // Sample Gumbel noise and compute initial scores.
        let gumbels: Vec<f32> = (0..num_actions).map(|_| noise::gumbel()).collect();
        let gumbel_scores: Vec<f32> = (0..num_actions)
            .map(|a| gumbels[a] + logits[a])
            .collect();

        // Initial candidate set: all actions or top base_count.
        let mut candidates: Vec<usize> = (0..num_actions).collect();
        candidates.sort_by(|&a, &b| gumbel_scores[b].partial_cmp(&gumbel_scores[a]).unwrap());
        if !gumbel.all_at_least_once {
            candidates.truncate(gumbel.base_count.min(num_actions));
        }

        let mut sim_budget = n_rollouts;

        // Sequential halving loop.
        while sim_budget > 0 && !candidates.is_empty() {
            let sims_this_round = candidates.len().min(sim_budget);
            let actions_this_round: Vec<usize> = candidates[..sims_this_round].to_vec();

            // Run rollouts in parallel, one per candidate action (forced at root).
            let n_threads = self.services.len().min(sims_this_round);
            let rollouts_per_thread = sims_this_round / n_threads;
            let remainder = sims_this_round % n_threads;

            std::thread::scope(|s| {
                let handles: Vec<_> = (0..n_threads).map(|thread_id| {
                    let board = board.clone();
                    let start = thread_id * rollouts_per_thread + thread_id.min(remainder);
                    let count = rollouts_per_thread + if thread_id < remainder { 1 } else { 0 };
                    let actions = &actions_this_round[start..start + count];
                    s.spawn(move || {
                        for &action in actions {
                            self.tree.begin_rollout(thread_id);
                            self.rollout(model_id, &board, rollout_depth, Some(action), thread_id);
                        }
                    })
                }).collect();
                for h in handles {
                    h.join().unwrap();
                }
            });

            tree.finalize_rollouts(board);
            sim_budget -= sims_this_round;

            // Re-score and narrow candidates.
            if sim_budget > 0 {
                let root = tree.get_data(board).unwrap();
                let n_max = (0..num_actions).map(|a| root.get_n(a)).max().unwrap_or(0);
                let c_visit = self.mcts_params.c_visit;
                let c_scale = self.mcts_params.c_scale;

                candidates.sort_by(|&a, &b| {
                    let sa = gumbels[a] + sigma(root.completed_q(a), n_max, c_visit, c_scale);
                    let sb = gumbels[b] + sigma(root.completed_q(b), n_max, c_visit, c_scale);
                    sb.partial_cmp(&sa).unwrap()
                });
                drop(root);

                let next_size = (candidates.len() as f32 * gumbel.keep_frac) as usize;
                let next_size = next_size.max(gumbel.floor_count).min(candidates.len());
                candidates.truncate(next_size);
            }
        }

        // Compute improved policy target over ALL actions.
        let root = tree.get_data(board).unwrap();
        let n_max = (0..num_actions).map(|a| root.get_n(a)).max().unwrap_or(0);
        let c_visit = self.mcts_params.c_visit;
        let c_scale = self.mcts_params.c_scale;

        let sigma_q_values: Vec<f32> = (0..num_actions)
            .map(|a| sigma(root.completed_q(a), n_max, c_visit, c_scale))
            .collect();
        let improved_logits: Vec<f32> = (0..num_actions)
            .map(|a| logits[a] + sigma_q_values[a])
            .collect();
        let pi = softmax(&improved_logits);

        let prior_pi = softmax(&logits);
        let search_stats = SearchStats {
            prior_entropy: entropy(&prior_pi),
            target_entropy: entropy(&pi),
            logit_std: std_dev(&logits),
            sigma_q_std: std_dev(&sigma_q_values),
        };

        // Value: mean of backed-up Q across all visited actions, weighted by visit count.
        let total_n: u32 = (0..num_actions).map(|a| root.get_n(a)).sum();
        let value = if total_n > 0 {
            (0..num_actions)
                .map(|a| root.get_n(a) as f32 * root.get_q(a))
                .sum::<f32>() / total_n as f32
        } else {
            root.get_v()
        };

        let wdl = root.blended_wdl();
        let mcts_wdl = [wdl.win, wdl.draw, wdl.loss];
        drop(root);

        let greedy_backup_wdl = self.greedy_backup_wdl(board, rollout_depth);

        RolloutResult {
            pi,
            value,
            mcts_wdl,
            greedy_backup_wdl,
            search_stats,
        }
    }

    /// Walk the tree greedily from `board`, always picking the most-visited child.
    /// Returns the WDL of the deepest reachable node, from the root player's perspective.
    fn greedy_backup_wdl(&self, board: &B, max_depth: usize) -> [f32; 3] {
        let mut current = board.clone();
        let mut depth = 0;

        loop {
            let info = current.get_info();
            if info.game_over {
                let wdl = if depth % 2 == 0 { info.wdl } else { info.wdl.flip() };
                return [wdl.win, wdl.draw, wdl.loss];
            }

            let Some(data) = self.tree.get_data(&current) else {
                log::warn!("[mcts] greedy_backup_wdl: node missing from tree at depth {depth}");
                return [0.0, 1.0, 0.0];
            };

            if data.total_visits() == 0 || depth >= max_depth {
                let wdl = data.blended_wdl();
                let wdl = if depth % 2 == 0 { wdl } else { wdl.flip() };
                return [wdl.win, wdl.draw, wdl.loss];
            }

            let n_max = (0..data.num_actions()).map(|a| data.get_n(a)).max().unwrap_or(0);
            let c_scale = self.mcts_params.c_scale;
            let c_visit = self.mcts_params.c_visit;
            let best_action = (0..data.num_actions())
                .max_by(|&a, &b| {
                    let score_a = data.pi_logits[a] + sigma(data.get_q(a), n_max, c_visit, c_scale);
                    let score_b = data.pi_logits[b] + sigma(data.get_q(b), n_max, c_visit, c_scale);
                    score_a.partial_cmp(&score_b).unwrap()
                })
                .unwrap_or(0);

            let moves = current.legal_moves();
            current = current.apply_move(&moves[best_action]);
            drop(data);
            depth += 1;
        }
    }

    /// Select action using the improved policy proportional rule:
    /// argmax_a [ π'(a) - N(a) / (1 + Σ N(b)) ]
    /// where π' = softmax(logits + σ(completedQ))
    fn find_best_action(&self, data: &MCTSNode) -> usize {
        let n_max = (0..data.num_actions()).map(|a| data.get_n(a)).max().unwrap_or(0);
        let sum_n = data.total_visits() as f32;
        let denom = 1.0 + sum_n;
        let c_visit = self.mcts_params.c_visit;
        let c_scale = self.mcts_params.c_scale;

        let improved_logits: Vec<f32> = (0..data.num_actions())
            .map(|a| data.pi_logits[a] + sigma(data.completed_q(a), n_max, c_visit, c_scale))
            .collect();
        let pi_improved = softmax(&improved_logits);

        let mut best_action = 0;
        let mut best_score = f32::NEG_INFINITY;
        for a in 0..data.num_actions() {
            let score = pi_improved[a] - (data.get_n(a) as f32 / denom);
            if score > best_score {
                best_score = score;
                best_action = a;
            }
        }
        best_action
    }
}
