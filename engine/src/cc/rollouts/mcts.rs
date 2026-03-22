#[cfg(feature = "extension-module")]
extern crate pyo3;
extern crate rand_distr;

use std::io::Error;
use std::sync::Arc;
#[cfg(feature = "extension-module")]
use std::collections::HashMap;

#[cfg(feature = "extension-module")]
use pyo3::prelude::*;
use rand::prelude::*;
use rand_distr::multi::Dirichlet;

use crate::cc::game::board::Board;
use crate::cc::predictions::NNPred;
#[cfg(feature = "extension-module")]
use crate::cc::predictions::FetchStats;
use crate::cc::rollouts::mcts_node::MCTSNode;
use crate::cc::rollouts::tree::{NodeData, Tree};
use crate::nn::client::PredictClient;
use crate::nn::io;
use super::super::predictions::inference_utils::softmax;


struct ThreadContext {
    rt: tokio::runtime::Runtime,
    client: PredictClient,
}

#[cfg_attr(feature = "extension-module", pyo3::prelude::pyclass(module="alpha_cc_engine"))]
pub struct MCTS {
    tree: Arc<Tree>,
    thread_pool: Vec<ThreadContext>,
    model_id: u32,
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
        nn_service_addr: &str,
        model_id: u32,
        mcts_params: MCTSParams,
        n_threads: usize,
    ) -> Self {
        let contexts: Vec<ThreadContext> = (0..n_threads.max(1))
            .map(|_| {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("failed to create thread-local tokio runtime");
                let client = rt.block_on(PredictClient::connect(nn_service_addr))
                    .unwrap_or_else(|e| panic!("failed to connect to nn-service at {nn_service_addr}: {e}"));
                ThreadContext { rt, client }
            })
            .collect();

        MCTS {
            tree: Arc::new(Tree::new()),
            thread_pool: contexts,
            model_id,
            mcts_params,
        }
    }

    fn fetch_pred(
        rt: &tokio::runtime::Runtime,
        client: &PredictClient,
        model_id: u32,
        board: &Board,
    ) -> Result<NNPred, Error> {
        let (state_tensor, moves) = io::encode_request(board);
        let resp = rt
            .block_on(client.predict(state_tensor, moves, model_id))
            .map_err(|e| Error::other(format!("prediction failed: {e}")))?;
        let (logits, value) = io::decode_response(&resp);
        let pi = softmax(&logits);
        Ok(NNPred::new(pi, value))
    }

    /// Perform a single rollout from `board` down the tree.
    fn rollout(
        tree: &Tree,
        rt: &tokio::runtime::Runtime,
        client: &PredictClient,
        model_id: u32,
        board: Board,
        remaining_depth: usize,
        params: &MCTSParams,
    ) -> Result<f32, Error> {
        let info = board.get_info();
        if info.game_over {
            return Ok(-info.reward);
        }

        // Check if we have data for this board
        if let Some(data) = tree.get_data(&board) {
            if remaining_depth == 0 {
                return Ok(-data.get_v());
            }

            let a = find_best_action(&data, params.c_puct_init, params.c_puct_base);
            let s_prime = board.apply(&data.moves[a]);

            data.apply_virtual_loss(a);
            drop(data);

            let v = MCTS::rollout(tree, rt, client, model_id, s_prime, remaining_depth - 1, params)?;
            let gamma_v = params.gamma * v;

            if let Some(data) = tree.get_data(&board) {
                data.resolve_virtual_loss(a, gamma_v);
            }

            return Ok(-gamma_v);
        }

        // Leaf: fetch prediction
        let nn_pred = MCTS::fetch_pred(rt, client, model_id, &board)?;
        let v = nn_pred.value();
        let mut pi = nn_pred.pi();

        if params.dirichlet_weight > 0.0 && pi.len() > 1 {
            let alpha: Vec<f32> = pi.iter()
                .map(|x| (x * params.dirichlet_alpha).max(f32::EPSILON))
                .collect();
            if let Ok(dirichlet) = Dirichlet::new(&alpha) {
                let noise = dirichlet.sample(&mut rand::rng());
                pi = pi.iter()
                    .zip(noise.iter())
                    .map(|(p, n)| p * (1.0 - params.dirichlet_weight) + n * params.dirichlet_weight)
                    .collect();
            }
        }

        tree.insert_data(board, pi, v);

        Ok(-v)
    }

    fn run_rollouts_inner(
        &self,
        board: &Board,
        n_rollouts: usize,
        rollout_depth: usize,
        temperature: f32,
    ) -> Result<(Vec<f32>, f32), Error> {
        // Ensure root data exists (use first thread context)
        let ctx0 = &self.thread_pool[0];
        if self.tree.get_data(board).is_none() {
            let nn_pred = MCTS::fetch_pred(&ctx0.rt, &ctx0.client, self.model_id, board)?;
            self.tree.insert_data(board.clone(), nn_pred.pi(), nn_pred.value());
        }
        self.tree.set_root(board.clone());

        let n_threads = self.thread_pool.len().min(n_rollouts);
        let rollouts_per_thread = n_rollouts / n_threads;
        let remainder = n_rollouts % n_threads;

        let tree = &self.tree;
        let params = &self.mcts_params;
        let model_id = self.model_id;

        let value_sum: f64 = std::thread::scope(|s| {
            let handles: Vec<_> = self.thread_pool[..n_threads].iter().enumerate().map(|(i, ctx)| {
                let board = board.clone();
                let extra = if i < remainder { 1 } else { 0 };
                let count = rollouts_per_thread + extra;

                s.spawn(move || {
                    let mut local_sum = 0.0f64;
                    for _ in 0..count {
                        match MCTS::rollout(tree, &ctx.rt, &ctx.client, model_id, board.clone(), rollout_depth, params) {
                            Ok(v) => local_sum += (-v) as f64,
                            Err(e) => eprintln!("rollout error: {e}"),
                        }
                    }
                    local_sum
                })
            }).collect();

            handles.into_iter().map(|h: std::thread::ScopedJoinHandle<'_, f64>| h.join().unwrap()).sum()
        });

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
            let moves = crate::cc::game::moves::find_all_moves(board);
            let n = moves.len();
            vec![1.0 / n as f32; n]
        };

        Ok((pi, mean_value))
    }
}


fn find_best_action(data: &NodeData, c_puct_init: f32, c_puct_base: f32) -> usize {
    let sum_n: u32 = (0..data.num_actions()).map(|a| data.get_n(a)).sum();
    let sum_n_f = sum_n as f32;
    let c_puct = c_puct_init + ((sum_n_f + c_puct_base + 1.0) / c_puct_base).ln();

    let mut best_action = 0;
    let mut best_u = f32::MIN;
    for i in 0..data.num_actions() {
        let n_a = data.get_n(i) as f32;
        let prior_weight = c_puct * sum_n_f.sqrt() / (1.0 + n_a);
        let u = data.get_q(i) + prior_weight * data.get_pi(i);
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
    #[pyo3(signature = (nn_service_addr, channel, gamma, dirichlet_weight, dirichlet_alpha, c_puct_init, c_puct_base, n_threads=1))]
    fn create(
        nn_service_addr: String,
        channel: u32,
        gamma: f32,
        dirichlet_weight: f32,
        dirichlet_alpha: f32,
        c_puct_init: f32,
        c_puct_base: f32,
        n_threads: usize,
    ) -> MCTS {
        MCTS::new(
            &nn_service_addr,
            channel,
            MCTSParams {
                gamma,
                dirichlet_weight,
                dirichlet_alpha,
                c_puct_init,
                c_puct_base,
            },
            n_threads,
        )
    }

    pub fn run(&self, board: &Board, rollout_depth: usize) -> Result<f32, Error> {
        let (_, value) = self.run_rollouts_inner(board, 1, rollout_depth, 1.0)?;
        Ok(value)
    }

    #[pyo3(signature = (board, n_rollouts, rollout_depth, temperature=1.0))]
    pub fn run_rollouts<'py>(
        &self,
        py: Python<'py>,
        board: &Board,
        n_rollouts: usize,
        rollout_depth: usize,
        temperature: f32,
    ) -> Result<(Bound<'py, numpy::PyArray1<f32>>, f32), Error> {
        let (pi, mean_value) = self.run_rollouts_inner(board, n_rollouts, rollout_depth, temperature)?;
        let pi_arr = numpy::IntoPyArray::into_pyarray(
            numpy::ndarray::Array1::from_vec(pi), py,
        );
        Ok((pi_arr, mean_value))
    }

    pub fn advance_root(&self, action: usize) -> Option<Board> {
        self.tree.advance_root(action)
    }

    pub fn get_node(&self, board: &Board) -> Option<MCTSNode> {
        self.tree.get_data(board).map(|data| MCTSNode::from_node_data(&data))
    }

    pub fn get_nodes(&self) -> HashMap<Board, MCTSNode> {
        self.tree.iter_data()
            .map(|entry| (entry.key().clone(), MCTSNode::from_node_data(entry.value())))
            .collect()
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
