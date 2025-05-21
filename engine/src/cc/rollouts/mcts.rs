extern crate pyo3;
extern crate rand_distr;
extern crate lru;

use std::num::NonZeroUsize;
use pyo3::prelude::*;
use rand::prelude::*;
use rand_distr::Dirichlet;
use lru::LruCache;

use crate::cc::board::Board;
use crate::cc::moves::find_all_moves;
use crate::cc::rollouts::nn_remote::NNRemote;
use crate::cc::rollouts::mcts_node::MCTSNode;
use crate::cc::pred_db::{NNPred, PredDBChannel};


#[pyclass(module="alpha_cc_engine")]
pub struct MCTS {
    nn_remote: NNRemote,
    nodes: LruCache<Board, MCTSNode>,
    mcts_params: MCTSParams,
}


#[derive(Clone)]
pub struct MCTSParams {
    pub gamma: f32,
    pub dirichlet_weight: f32,
    pub dirichlet_alpha: f32,
    pub c_puct_init: f32,
    pub c_puct_base: f32,
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
    ) -> f32 {
        let info = board.get_info();
        if info.game_over {
            return -info.reward;
        }
        
        // if we've seen this node before, we keep rolling
        if let Some(node) = nodes.get(&board) {
            if remaining_depth == 0 {
                return -node.rollout_value();
            }
    
            // prepare continued rollout
            let a = MCTS::find_best_action_for_node(
                node,
                &mcts_params.c_puct_init,
                &mcts_params.c_puct_base,
            );
            let s_prime = board.apply(&node.moves[a]);
            
            // continue rollout
            let v = MCTS::rollout(
                s_prime,
                nn_remote,
                nodes,
                remaining_depth - 1,
                mcts_params,
            );
            let gamma_v = mcts_params.gamma * v;
            
            // backprop rollout update
            nodes.get_mut(&board).unwrap().update_on_visit(a, gamma_v);
            return -gamma_v;
        }


        let nn_pred = nn_remote.fetch_pred(&board);
        MCTS::add_as_new_node(
            nodes,
            board,
            &nn_pred,
            mcts_params.dirichlet_weight,
            mcts_params.dirichlet_alpha,
        );
        -nn_pred.value
    }

    fn find_best_action_for_node(node: &MCTSNode, c_puct_init: &f32, c_puct_base: &f32) -> usize {
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

    fn add_as_new_node(
        nodes: &mut LruCache<Board, MCTSNode>,
        board: Board,
        nn_pred: &NNPred,
        dirichlet_weight: f32,
        dirichlet_alpha: f32,
) {
        let mut pi = nn_pred.pi.clone();
        let v = nn_pred.value;

        if dirichlet_weight > 0.0 && nn_pred.pi.len() > 1 {
            let alpha = nn_pred.pi.iter()
                .map(|x| x * dirichlet_alpha)
                .collect::<Vec<f32>>();
            match Dirichlet::new(&alpha) {
                Ok(dirichlet) => {
                    let noise = dirichlet.sample(&mut rand::thread_rng());
                    pi = pi.iter()
                        .zip(noise.iter())
                        .map(|(p, n)| p * (1.0 - dirichlet_weight) + n * dirichlet_weight)
                        .collect();
                },
                Err(e) => {
                    println!("Failed to create Dirichlet distribution: {}", e);
                }
            }
        }

        let moves = find_all_moves(&board);
        let node= MCTSNode::new_leaf(pi, v, moves);
        nodes.put(board,node);
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
        )
    }

    pub fn get_node(&mut self, board: &Board) -> MCTSNode {
        self.nodes.get(board).unwrap().clone()
    }
}
