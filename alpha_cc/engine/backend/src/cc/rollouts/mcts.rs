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
use crate::cc::rollouts::nn_pred::NNPred;
use crate::cc::rollouts::pred_db::PredDB;



#[pyclass(module="alpha_cc_engine")]
pub struct MCTS {
    nn_remote: NNRemote,
    nodes: LruCache<Board, MCTSNode>,
    dirichlet_weight: f32,
    dirichlet_alpha: f32,
}

impl MCTS {
    pub fn new(nn_remote: NNRemote, cache_size: usize, dirichlet_weight: f32, dirichlet_alpha: f32) -> Self {
        MCTS { 
            nn_remote,
            nodes: LruCache::new(NonZeroUsize::new(cache_size).unwrap()),
            dirichlet_weight,
            dirichlet_alpha,
        }
    }

    fn rollout(
        board: Board,
        nn_remote: &mut NNRemote,
        nodes: &mut LruCache<Board, MCTSNode>,
        remaining_depth: usize,
        dirichlet_weight: f32,
        dirichlet_alpha: f32,
    ) -> f32 {
        let info = board.get_info();
        if info.game_over {
            return -info.reward as f32;
        }
        
        // if we haven't seen this board before, or we're at the end of the rollout
        let mb_node = nodes.get(&board);
        if mb_node.is_none() {
            let nn_pred = nn_remote.fetch_pred(&board);
            MCTS::add_as_new_node(nodes, board, &nn_pred, dirichlet_weight, dirichlet_alpha);
            return -nn_pred.value;
        }
        let node = mb_node.unwrap();

        if remaining_depth == 0 {
            return -node.v;
        }

        // prepare continued rollout
        let a = MCTS::find_best_action_for_node(node);
        let moves = find_all_moves(&board);
        
        // continue rollout
        let s_prime = board.apply(&moves[a]);
        let v = MCTS::rollout(
            s_prime,
            nn_remote,
            nodes,
            remaining_depth - 1,
            dirichlet_weight,
            dirichlet_alpha,
        );
        
        // backprop rollout update
        nodes.get_mut(&board).unwrap().update_on_visit(a, v);

        -v
    }

    fn find_best_action_for_node(node: &MCTSNode) -> usize {
        const C_PUCT_INIT: f32 = 2.5;
        const C_PUCT_BASE: f32 = 19652.0;
        let sum_n = node.n.iter().sum::<u32>() as f32;
        let c_puct = C_PUCT_INIT + ((sum_n + C_PUCT_BASE + 1.0) / C_PUCT_BASE).ln();

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

    fn add_as_new_node(nodes: &mut LruCache<Board, MCTSNode>, board: Board, nn_pred: &NNPred, dirichlet_weight: f32, dirichlet_alpha: f32) {
        let mut pi = nn_pred.pi.clone();
        let v = nn_pred.value;

        if dirichlet_weight > 0.0 {
            let alpha = nn_pred.pi.iter().map(|x| x * dirichlet_alpha).collect::<Vec<f32>>();
            let dirichlet = Dirichlet::new(&alpha).unwrap();
            let noise = dirichlet.sample(&mut rand::thread_rng());
            pi = pi.iter()
                .zip(noise.iter())
                .map(|(p, n)| p * (1.0 - dirichlet_weight) + n * dirichlet_weight)
                .collect();
        }

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
    #[new]
    fn create(url: String, cache_size: usize, dirichlet_weight: f32, dirichlet_alpha: f32) -> MCTS {
        MCTS::new(
            NNRemote::new(PredDB::new(&url)),
            cache_size,
            dirichlet_weight,
            dirichlet_alpha,
        )
    }

    pub fn run(&mut self, board: &Board, rollout_depth: usize) -> f32 {
        MCTS::rollout(board.clone(), &mut self.nn_remote, &mut self.nodes, rollout_depth, self.dirichlet_weight, self.dirichlet_alpha)
    }

    pub fn get_node(&mut self, board: &Board) -> MCTSNode {
        self.nodes.get(board).unwrap().clone()
    }
}
