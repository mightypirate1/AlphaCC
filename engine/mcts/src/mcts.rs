use std::sync::Arc;

use alpha_cc_core::Board;
use alpha_cc_nn::PredictionSource;
use crate::mcts_node::MCTSNode;
use crate::outcome::Outcome;
use crate::search::descent::Descent;
use crate::stats::SearchStats;
use crate::tree::Tree;

pub struct RolloutResult {
    pub pi: Vec<f32>,
    pub value: f32,
    pub mcts_wdl: [f32; 3],
    pub greedy_backup_wdl: [f32; 3],
    pub search_stats: SearchStats,
}

/// Core search engine: owns the transposition tree, prediction services, and a
/// descent strategy. The top-level search flow (how to organise rollouts from the
/// root) lives in schedulers (see `crate::search::scheduler`) which own an `MCTS`
/// and drive it.
pub struct MCTS<B, T, D>
where B: Board, T: PredictionSource<B>, D: Descent
{
    tree: Arc<Tree<B>>,
    services: Vec<T>,
    model_id: u32,
    mcts_params: MCTSParams,
    descent: D,
}

#[derive(Clone)]
pub struct MCTSParams {
    pub gamma: f32,
}

impl<B, T, D> MCTS<B, T, D>
where B: Board, T: PredictionSource<B>, D: Descent
{
    pub fn new(
        services: Vec<T>,
        model_id: u32,
        mcts_params: MCTSParams,
        descent: D,
        pruning_tree: bool,
        debug_prints: bool,
    ) -> Self {
        let n = services.len().max(1);
        MCTS {
            tree: Arc::new(Tree::new(n, pruning_tree, debug_prints)),
            services,
            model_id,
            mcts_params,
            descent,
        }
    }

    // ── accessors used by schedulers ──

    pub fn tree(&self) -> &Tree<B> { &self.tree }
    pub fn services(&self) -> &[T] { &self.services }
    pub fn model_id(&self) -> u32 { self.model_id }
    pub fn descent(&self) -> &D { &self.descent }
    pub fn gamma(&self) -> f32 { self.mcts_params.gamma }

    // ── rollout plumbing ──

    /// Ensure the root node is present in the tree (inserts a fresh NN leaf if not).
    pub fn ensure_root(&self, board: &B) {
        if self.tree.get_data(board).is_none() {
            let node = self.new_leaf_for(board, &self.services[0], self.model_id);
            self.tree.insert(board, node);
        }
    }

    pub fn begin_rollout(&self, thread_id: usize) {
        self.tree.begin_rollout(thread_id);
    }

    pub fn finalize_rollouts(&self, board: &B) {
        self.tree.finalize_rollouts(board);
    }

    /// Perform a single rollout from `board` down the tree.
    /// Returns (value, outcome) from the caller's (parent's) perspective.
    /// `root_state` is Some only at the outermost call; inner recursions pass None.
    pub fn rollout(
        &self,
        board: &B,
        remaining_depth: usize,
        forced_action: Option<usize>,
        root_state: Option<&D::RootState>,
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

            let a = forced_action.unwrap_or_else(|| self.descent.select(&data, root_state));
            let moves = board.legal_moves();
            let s_prime = board.apply_move(&moves[a]);
            self.tree.record_action(thread_id, &s_prime, a);

            data.apply_virtual_loss(a);
            drop(data);

            let (v, outcome) = self.rollout(&s_prime, remaining_depth - 1, None, None, thread_id);

            let data = self.tree.get_data(board)
                .expect("node data disappeared mid-rollout");
            if let Some(o) = outcome {
                data.tick_outcome(o);
            }
            data.resolve_virtual_loss(a, self.mcts_params.gamma * v);

            return (-(self.mcts_params.gamma * v), outcome.map(|o| o.flip()));
        }

        let node = self.new_leaf_for(board, &self.services[thread_id], self.model_id);
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

    // ── tree-level operations exposed externally ──

    pub fn notify_move_applied(&self, board: &B) {
        if let Some(action) = self.tree.maybe_prune(board) {
            if self.tree.debug_prints() {
                let report = self.tree.memory_report();
                log::debug!("[mcts] pruned action={action}: {report}");
            }
        }
    }

    pub fn get_node_snapshot(&self, board: &B) -> Option<MCTSNode> {
        self.tree.get_data(board).map(|data| data.snapshot())
    }

    pub fn clear_tree(&self) {
        self.tree.clear();
    }

    pub fn get_all_nodes(&self) -> std::collections::HashMap<B, MCTSNode> {
        self.tree.iter_data()
            .map(|entry| (entry.key().clone(), entry.value().snapshot()))
            .collect()
    }

    /// Walk the tree greedily from `board`, picking `descent.best_child` at each
    /// node. Returns the WDL of the deepest reachable node, from root-player's POV.
    pub fn greedy_backup_wdl(&self, board: &B, max_depth: usize) -> [f32; 3] {
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

            let best_action = self.descent.best_child(&data);

            let moves = current.legal_moves();
            current = current.apply_move(&moves[best_action]);
            drop(data);
            depth += 1;
        }
    }
}
