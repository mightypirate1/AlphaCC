extern crate pyo3;
extern crate rand_distr;
extern crate lru;

use std::io::Error;
use std::collections::HashMap;
use std::num::NonZeroUsize;
use pyo3::prelude::*;
use rand::prelude::*;
use rand_distr::multi::Dirichlet;
use lru::LruCache;

use crate::cc::game::board::Board;
use crate::cc::game::moves::find_all_moves;
use crate::cc::predictions::{NNPred, PredictionClient, FetchStats};
use crate::cc::rollouts::mcts_node::MCTSNode;


#[pyclass(module="alpha_cc_engine")]
pub struct MCTS {
    client: PredictionClient,
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
        client: PredictionClient,
        cache_size: usize,
        mcts_params: MCTSParams,
    ) -> Self {
        MCTS {
            client,
            nodes: LruCache::new(NonZeroUsize::new(cache_size).unwrap()),
            mcts_params,
        }
    }

    fn rollout(
        board: Board,
        client: &mut PredictionClient,
        nodes: &mut LruCache<Board, MCTSNode>,
        remaining_depth: usize,
        mcts_params: &MCTSParams,
    ) -> Result<f32, Error> {
        let info = board.get_info();
        if info.game_over {
            return Ok(-info.reward);
        }

        // if we've seen this node before, we keep rolling
        if let Some(node) = nodes.get(&board) {
            if remaining_depth == 0 {
                return Ok(-node.rollout_value());
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
                client,
                nodes,
                remaining_depth - 1,
                mcts_params,
            )?;
            let gamma_v = mcts_params.gamma * v;

            // backprop rollout update
            nodes.get_mut(&board).unwrap().update_on_visit(a, gamma_v);
            return Ok(-gamma_v);
        }


        let nn_pred = client.fetch_pred(&board)?;
        MCTS::add_as_new_node(
            nodes,
            board,
            &nn_pred,
            mcts_params.dirichlet_weight,
            mcts_params.dirichlet_alpha,
        );
        Ok(-nn_pred.value())
    }

    fn find_best_action_for_node(node: &MCTSNode, c_puct_init: &f32, c_puct_base: &f32) -> usize {
        let sum_n = node.n.iter().sum::<u32>() as f32;
        let c_puct = c_puct_init + ((sum_n + c_puct_base + 1.0) / c_puct_base).ln();

        let mut best_action = 0;
        let mut best_u = f32::MIN;
        for i in 0..node.n.len() {
            let prior_weight = c_puct * sum_n.sqrt() / (1.0 + node.n[i] as f32);
            let u = node.get_q(i) + prior_weight * node.get_pi(i);
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
        let mut pi = nn_pred.pi();
        let v = nn_pred.value();

        if dirichlet_weight > 0.0 && pi.len() > 1 {
            let alpha = pi.iter()
                .map(|x| (x * dirichlet_alpha).max(f32::EPSILON))
                .collect::<Vec<f32>>();
            match Dirichlet::new(&alpha) {
                Ok(dirichlet) => {
                    let noise = dirichlet.sample(&mut rand::rng());
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
        let node = MCTSNode::new_leaf(pi, v, moves);
        nodes.put(board, node);
    }
}


#[pymethods]
impl MCTS {
    #[allow(clippy::too_many_arguments)]
    #[new]
    #[pyo3(signature = (nn_service_addr, channel, cache_size, gamma, dirichlet_weight, dirichlet_alpha, c_puct_init, c_puct_base))]
    fn create(
        nn_service_addr: String,
        channel: u32,
        cache_size: usize,
        gamma: f32,
        dirichlet_weight: f32,
        dirichlet_alpha: f32,
        c_puct_init: f32,
        c_puct_base: f32,
    ) -> MCTS {
        MCTS::new(
            PredictionClient::new(&nn_service_addr, channel),
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

    pub fn run(&mut self, board: &Board, rollout_depth: usize) -> Result<f32, Error> {
        MCTS::rollout(
            board.clone(),
            &mut self.client,
            &mut self.nodes,
            rollout_depth,
            &self.mcts_params,
        )
    }

    /// Run n_rollouts rollouts and return (pi, mean_value).
    /// pi is the temperature-weighted visit count distribution.
    #[pyo3(signature = (board, n_rollouts, rollout_depth, temperature=1.0))]
    pub fn run_rollouts<'py>(
        &mut self,
        py: Python<'py>,
        board: &Board,
        n_rollouts: usize,
        rollout_depth: usize,
        temperature: f32,
    ) -> Result<(Bound<'py, numpy::PyArray1<f32>>, f32), Error> {
        let mut value_sum = 0.0f64;
        for _ in 0..n_rollouts {
            let v = MCTS::rollout(
                board.clone(),
                &mut self.client,
                &mut self.nodes,
                rollout_depth,
                &self.mcts_params,
            )?;
            value_sum += (-v) as f64;
        }
        let mean_value = (value_sum / n_rollouts as f64) as f32;

        // Build pi from visit counts
        let pi = if let Some(node) = self.nodes.get(board) {
            let mut weights: Vec<f32> = node.n.iter().map(|&n| n as f32).collect();
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
                let n_moves = weights.len();
                vec![1.0 / n_moves as f32; n_moves]
            }
        } else {
            // No node found — uniform over legal moves
            let n_moves = find_all_moves(board).len();
            vec![1.0 / n_moves as f32; n_moves]
        };

        let pi_arr = numpy::IntoPyArray::into_pyarray(
            numpy::ndarray::Array1::from_vec(pi), py,
        );
        Ok((pi_arr, mean_value))
    }

    pub fn get_node(&mut self, board: &Board) -> Option<MCTSNode> {
        self.nodes.get(board).cloned()
    }

    /// Return all nodes in the live cache.
    pub fn get_nodes(&self) -> HashMap<Board, MCTSNode> {
        self.nodes.iter().map(|(b, n)| (b.clone(), n.clone())).collect()
    }

    pub fn clear_nodes(&mut self) {
        self.nodes.clear();
    }

    pub fn get_fetch_stats(&mut self) -> FetchStats {
        self.client.get_fetch_stats()
    }
}
