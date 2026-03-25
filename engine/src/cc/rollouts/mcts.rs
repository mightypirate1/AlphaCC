#[cfg(feature = "extension-module")]
extern crate pyo3;
extern crate rand_distr;

use std::io::Error;
use std::sync::Arc;
use std::time::Duration;
#[cfg(feature = "extension-module")]
use std::collections::HashMap;

#[cfg(feature = "extension-module")]
use pyo3::prelude::*;
use rand::prelude::*;
use rand_distr::multi::Dirichlet;

use crate::cc::game::board::Board;
#[cfg(feature = "extension-module")]
use crate::cc::predictions::FetchStats;
#[cfg(feature = "extension-module")]
use crate::cc::rollouts::mcts_node::MCTSNode;
use crate::cc::rollouts::tree::{NodeData, Tree};
use crate::cc::predictions::nn_remote::NNRemote;

const DEFAULT_PREDICT_TIMEOUT: Duration = Duration::from_secs(1);

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
        game_size: usize,
    ) -> Self {
        let services = (0..n_threads.max(1))
            .map(|_| NNRemote::connect(nn_service_addr, DEFAULT_PREDICT_TIMEOUT))
            .collect();
        MCTS {
            tree: Arc::new(Tree::new(Board::create(game_size))),
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
        root_noise: Option<&[f32]>,
    ) -> Result<f32, Error> {
        let info = board.get_info();
        if info.game_over {
            return Ok(-info.reward);
        }

        // Check if we have data for this board
        if let Some(data) = tree.get_data(board) {
            if remaining_depth == 0 {
                return Ok(-data.get_v());
            }

            let a = find_best_action(&data, params.c_puct_init, params.c_puct_base, root_noise, params.dirichlet_weight);
            let s_prime = board.apply(&data.moves[a]);

            data.apply_virtual_loss(a);
            drop(data);

            let result = MCTS::rollout(tree, service, model_id, &s_prime, remaining_depth - 1, params, None);

            let data = tree.get_data(board)
                .expect("node data disappeared mid-rollout");
            match &result {
                Ok(v) => data.resolve_virtual_loss(a, params.gamma * *v),
                Err(_) => data.resolve_virtual_loss(a, 0.0),
            }

            return result.map(|v| -(params.gamma * v));
        }

        // Leaf: fetch prediction and optionally apply Dirichlet noise to priors
        let nn_pred = service.predict(board, model_id)
            .map_err(|e| Error::other(format!("prediction failed: {e}")))?;
        let v = nn_pred.value();
        let mut pi = nn_pred.pi();

        if params.dirichlet_leaf_weight > 0.0 {
            apply_dirichlet_noise(&mut pi, params.dirichlet_leaf_weight, params.dirichlet_alpha);
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
        // TEMPORARY: verify tree root matches the board the runtime expects
        self.tree.assert_root_matches(board);

        // Ensure root data exists
        if self.tree.get_data(board).is_none() {
            let nn_pred = self.services[0].predict(board, self.model_id)
                .map_err(|e| Error::other(format!("root prediction failed: {e}")))?;
            self.tree.insert_data(board, nn_pred.pi(), nn_pred.value());
        }
        self.tree.set_root(board);

        // Sample Dirichlet noise once for root exploration (blending happens in find_best_action)
        let root_noise: Option<Vec<f32>> = if self.mcts_params.dirichlet_weight > 0.0 {
            self.tree.get_data(board).and_then(|data| {
                let pi: Vec<f32> = (0..data.num_actions()).map(|a| data.get_pi(a)).collect();
                if pi.len() <= 1 { return None; }
                let alpha: Vec<f32> = pi.iter()
                    .map(|x| (x * self.mcts_params.dirichlet_alpha).max(f32::EPSILON))
                    .collect();
                Dirichlet::new(&alpha).ok().map(|d| d.sample(&mut rand::rng()))
            })
        } else {
            None
        };
        let root_noise_ref = root_noise.as_deref();

        let n_threads = self.services.len().min(n_rollouts);
        let rollouts_per_thread = n_rollouts / n_threads;
        let remainder = n_rollouts % n_threads;

        let tree = &self.tree;
        let params = &self.mcts_params;
        let model_id = self.model_id;

        let (value_sum, n_ok): (f64, u64) = std::thread::scope(|s| {
            let handles: Vec<_> = self.services[..n_threads].iter().enumerate().map(|(i, svc)| {
                let board = board.clone();
                let extra = if i < remainder { 1 } else { 0 };
                let count = rollouts_per_thread + extra;

                s.spawn(move || {
                    let mut local_sum = 0.0f64;
                    let mut local_ok = 0u64;
                    for _ in 0..count {
                        match MCTS::rollout(tree, svc, model_id, &board, rollout_depth, params, root_noise_ref) {
                            Ok(v) => {
                                local_sum += (-v) as f64;
                                local_ok += 1;
                            }
                            Err(e) => eprintln!("rollout error: {e}"),
                        }
                    }
                    (local_sum, local_ok)
                })
            }).collect();

            handles.into_iter()
                .map(|h| h.join().unwrap())
                .fold((0.0, 0), |(s, n), (ds, dn)| (s + ds, n + dn))
        });

        let mean_value = if n_ok > 0 {
            (value_sum / n_ok as f64) as f32
        } else {
            0.0
        };

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


/// Sample Dirichlet noise and blend it into `pi` in-place.
/// Returns the original `pi` unmodified if there are fewer than 2 actions.
fn apply_dirichlet_noise(pi: &mut Vec<f32>, weight: f32, alpha_scale: f32) {
    if pi.len() <= 1 { return; }
    let alpha: Vec<f32> = pi.iter()
        .map(|x| (x * alpha_scale).max(f32::EPSILON))
        .collect();
    if let Ok(dirichlet) = Dirichlet::new(&alpha) {
        let noise = dirichlet.sample(&mut rand::rng());
        for (p, n) in pi.iter_mut().zip(noise.iter()) {
            *p = *p * (1.0 - weight) + n * weight;
        }
    }
}

fn find_best_action(data: &NodeData, c_puct_init: f32, c_puct_base: f32, noise: Option<&[f32]>, dirichlet_weight: f32) -> usize {
    let sum_n: u32 = (0..data.num_actions()).map(|a| data.get_n(a)).sum();
    let sum_n_f = sum_n as f32;
    let c_puct = c_puct_init + ((sum_n_f + c_puct_base + 1.0) / c_puct_base).ln();

    let mut best_action = 0;
    let mut best_u = f32::MIN;
    for i in 0..data.num_actions() {
        let n_a = data.get_n(i) as f32;
        let prior_weight = c_puct * sum_n_f.sqrt() / (1.0 + n_a);
        let pi = match noise {
            Some(n) => (1.0 - dirichlet_weight) * data.get_pi(i) + dirichlet_weight * n[i],
            None => data.get_pi(i),
        };
        let u = data.get_q(i) + prior_weight * pi;
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
    #[pyo3(signature = (nn_service_addr, channel, gamma, dirichlet_weight, dirichlet_leaf_weight, dirichlet_alpha, c_puct_init, c_puct_base, game_size, n_threads=1))]
    fn create(
        nn_service_addr: String,
        channel: u32,
        gamma: f32,
        dirichlet_weight: f32,
        dirichlet_leaf_weight: f32,
        dirichlet_alpha: f32,
        c_puct_init: f32,
        c_puct_base: f32,
        game_size: usize,
        n_threads: usize,
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
            game_size,
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

    pub fn advance_root(&self, action: usize) {
        self.tree.advance_root(action);
    }

    pub fn get_node(&self, board: &Board) -> Option<MCTSNode> {
        self.tree.get_data(board).map(|data| MCTSNode::from_node_data(&data))
    }

    pub fn get_nodes(&self) -> HashMap<Board, MCTSNode> {
        self.tree.iter_data()
            .map(|entry| (entry.key().clone(), MCTSNode::from_node_data(entry.value())))
            .collect()
    }

    pub fn clear_nodes(&self, board: &Board) {
        self.tree.clear(board.clone());
        for svc in &self.services {
            let _ = svc.reconnect();
        }
    }

    pub fn get_fetch_stats(&self) -> FetchStats {
        FetchStats {
            total_fetch_time_us: 0,
            total_fetches: 0,
        }
    }
}
