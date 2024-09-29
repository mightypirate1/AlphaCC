extern crate pyo3;
extern crate rand_distr;
extern crate lru;

use std::num::NonZeroUsize;
use pyo3::prelude::*;
use lru::LruCache;

use crate::cc::board::Board;
use crate::cc::moves::find_all_moves;
use crate::cc::rollouts::nn_remote::NNRemote;
use crate::cc::rollouts::{MCTSNode, MCTSParams};
use crate::cc::pred_db::{NNPred, PredDBChannel};


#[pyclass(module="alpha_cc_engine")]
pub struct MCTS {
    nn_remote: NNRemote,
    nodes: LruCache<Board, MCTSNode>,
    mcts_params: MCTSParams,
}


impl MCTS {
    pub fn new(
        nn_remote: NNRemote,
        cache_size: usize,
        mcts_params: MCTSParams,
    ) -> Self {
        MCTS { 
            nn_remote,
            nodes: LruCache::new(NonZeroUsize::new(cache_size).unwrap()),
            mcts_params,
        }
    }

    fn rollout(
        board: Board,
        nn_remote: &mut NNRemote,
        nodes: &mut LruCache<Board, MCTSNode>,
        remaining_depth: usize,
        mcts_params: &MCTSParams,
        is_root: bool,
    ) -> f32 {
        let info = board.get_info();
        if info.game_over {
            return -info.reward;
        }
        
        // if we've seen this node before, we keep rolling
        if let Some(node) = nodes.get(&board) {
            if remaining_depth == 0 {
                return -node.v;
            }
            
            // prepare continued rollout
            let mut a = MCTS::find_best_action_for_node(
                node,
                mcts_params,
            );
            if is_root {
                a = MCTS::find_best_action_for_node(
                    &node.with_noised_pi(mcts_params),
                    mcts_params,
                );
            }
            
            let moves = find_all_moves(&board);
            let s_prime = board.apply(&moves[a]);
            
            // continue rollout
            let v = mcts_params.gamma * MCTS::rollout(
                s_prime,
                nn_remote,
                nodes,
                remaining_depth - 1,
                mcts_params,
                false,
            );
            
            // backprop rollout update
            nodes.get_mut(&board).unwrap().update_on_visit(a, v);
            return -v;
        }

        let nn_pred = nn_remote.fetch_pred(&board);
        MCTS::add_as_new_node(
            nodes,
            board,
            &nn_pred,
        );
        -nn_pred.value
    }

    fn find_best_action_for_node(node: &MCTSNode, mcts_params: &MCTSParams) -> usize {
        let c_puct_init = mcts_params.c_puct_init;
        let c_puct_base = mcts_params.c_puct_base;

        let sum_n = node.n.iter().sum::<u32>() as f32;
        let c_puct = c_puct_init + ((sum_n + c_puct_base + 1.0) / c_puct_base).ln();

        let mut best_action = 0;
        let mut best_u = f32::MIN;
        for i in 0..node.n.len() {
            let prior_weight = c_puct * sum_n.sqrt() / (1.0 + node.n[i] as f32);
            let u = node.q[i] + prior_weight * node.pi[i];
            if u > best_u {
                best_u = u;
                best_action = i;
            }
        }
        best_action
    }

    fn add_as_new_node(nodes: &mut LruCache<Board, MCTSNode>, board: Board, nn_pred: &NNPred) {
        let pi = nn_pred.pi.clone();
        let v = nn_pred.value;
        nodes.put(
            board,
            MCTSNode {
                n: vec![0; pi.len()],
                q: vec![0.0; pi.len()],
                pi,
                v,
            }
        );
    }
}


#[pymethods]
impl MCTS {
    #[allow(clippy::too_many_arguments)]
    #[new]
    fn create(
        url: String,
        channel: usize,
        cache_size: usize,
        gamma: f32,
        dirichlet_weight: f32,
        dirichlet_alpha: f32,
        c_puct_init: f32,
        c_puct_base: f32,
    ) -> MCTS {
        MCTS::new(
            NNRemote::new(PredDBChannel::new(&url, channel)),
            cache_size,
            MCTSParams {
                gamma,
                dirichlet_weight,
                dirichlet_alpha,
                c_puct_init,
                c_puct_base,
            },
        )
    }

    pub fn run(&mut self, board: &Board, rollout_depth: usize) -> f32 {
        MCTS::rollout(
            board.clone(),
            &mut self.nn_remote,
            &mut self.nodes,
            rollout_depth,
            &self.mcts_params,
            true,
        )
    }

    pub fn get_node(&mut self, board: &Board) -> MCTSNode {
        self.nodes.get(board).unwrap().clone()
    }
}
