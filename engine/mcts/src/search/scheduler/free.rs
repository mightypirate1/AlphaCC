use alpha_cc_core::Board;
use alpha_cc_nn::{softmax, PredictionSource};

use crate::mcts::{MCTSParams, RolloutResult, MCTS};
use crate::search::descent::puct::{PuctDescent, PuctParams};
use crate::search::descent::Descent;
use crate::search::scheduler::Scheduler;
use crate::stats::SearchStats;

#[derive(Clone)]
pub struct FreeConfig {
    /// Applied to the final visit-count distribution: pi ∝ N(a)^(1/T). T=1 is uniform.
    pub temperature: f32,
}

pub struct FreeScheduler<B, T, D>
where B: Board, T: PredictionSource<B>, D: Descent
{
    mcts: MCTS<B, T, D>,
    config: FreeConfig,
}

impl<B, T, D> FreeScheduler<B, T, D>
where B: Board, T: PredictionSource<B>, D: Descent
{
    pub fn new(mcts: MCTS<B, T, D>, config: FreeConfig) -> Self {
        Self { mcts, config }
    }
}

/// Preset: PUCT descent + free parallel rollouts.
impl<B, T> FreeScheduler<B, T, PuctDescent>
where B: Board, T: PredictionSource<B>
{
    pub fn build_puct_free(
        services: Vec<T>,
        model_id: u32,
        gamma: f32,
        puct: PuctParams,
        free: FreeConfig,
        pruning_tree: bool,
        debug_prints: bool,
    ) -> Self {
        let descent = PuctDescent::new(puct);
        let mcts = MCTS::new(
            services, model_id, MCTSParams { gamma }, descent, pruning_tree, debug_prints,
        );
        Self::new(mcts, free)
    }
}

impl<B, T, D> Scheduler<B, T, D> for FreeScheduler<B, T, D>
where B: Board, T: PredictionSource<B>, D: Descent
{
    fn mcts(&self) -> &MCTS<B, T, D> { &self.mcts }

    fn run(&self, board: &B, n_rollouts: usize, rollout_depth: usize) -> RolloutResult {
        self.mcts.ensure_root(board);

        let pi_logits = {
            let root = self.mcts.tree().get_data(board).unwrap();
            root.pi_logits.clone()
        };

        let root_state: Option<D::RootState> = {
            let root = self.mcts.tree().get_data(board).unwrap();
            self.mcts.descent().fresh_root_state(&root)
        };

        let n_threads = self.mcts.services().len().min(n_rollouts.max(1));
        let rollouts_per_thread = n_rollouts / n_threads;
        let remainder = n_rollouts % n_threads;

        let value_sum: f64 = std::thread::scope(|s| {
            let handles: Vec<_> = (0..n_threads).map(|thread_id| {
                let board = board.clone();
                let count = rollouts_per_thread + if thread_id < remainder { 1 } else { 0 };
                let mcts = &self.mcts;
                let root_state_ref = root_state.as_ref();
                s.spawn(move || {
                    let mut local_sum = 0.0f64;
                    for _ in 0..count {
                        mcts.begin_rollout(thread_id);
                        let (v, _outcome) = mcts.rollout(
                            &board, rollout_depth, None, root_state_ref, thread_id,
                        );
                        local_sum += (-v) as f64;
                    }
                    local_sum
                })
            }).collect();
            handles.into_iter().map(|h| h.join().unwrap()).sum()
        });

        self.mcts.finalize_rollouts(board);

        let mean_value = if n_rollouts > 0 { (value_sum / n_rollouts as f64) as f32 } else { 0.0 };

        let root = self.mcts.tree().get_data(board).unwrap();
        let num_actions = root.num_actions();

        let pi = {
            let mut weights: Vec<f32> = (0..num_actions).map(|a| root.get_n(a) as f32).collect();
            let t = self.config.temperature;
            if (t - 1.0).abs() > 1e-6 {
                let inv_t = 1.0 / t;
                let max_w = weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                if max_w > 0.0 {
                    for w in weights.iter_mut() {
                        *w = (*w / max_w).powf(inv_t);
                    }
                }
            }
            let sum: f32 = weights.iter().sum();
            if sum > 0.0 {
                weights.iter().map(|&w| w / sum).collect::<Vec<_>>()
            } else {
                vec![1.0 / num_actions as f32; num_actions]
            }
        };

        let prior_pi = softmax(&pi_logits);
        let sigma_qs = vec![0.0f32; num_actions];
        let search_stats = SearchStats::compute(&prior_pi, &pi, &pi_logits, &sigma_qs);

        let wdl = root.blended_wdl();
        let mcts_wdl = [wdl.win, wdl.draw, wdl.loss];
        drop(root);

        let greedy_backup_wdl = self.mcts.greedy_backup_wdl(board, rollout_depth);

        RolloutResult {
            pi,
            value: mean_value,
            mcts_wdl,
            greedy_backup_wdl,
            search_stats,
        }
    }
}
