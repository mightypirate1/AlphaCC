use alpha_cc_core::Board;
use alpha_cc_nn::{softmax, PredictionSource};

use crate::mcts::RolloutResult;
use crate::search::descent::Descent;
use crate::search::scheduler::{RootScheduler, SchedulerCtx};
use crate::stats::SearchStats;

#[derive(Clone)]
pub struct FreeConfig {
    /// Applied to the final visit-count distribution: pi ∝ N(a)^(1/T). T=1 is uniform.
    pub temperature: f32,
}

pub struct FreeScheduler {
    config: FreeConfig,
}

impl FreeScheduler {
    pub fn new(config: FreeConfig) -> Self { Self { config } }
}

impl<B, P, D> RootScheduler<B, P, D> for FreeScheduler
where B: Board, P: PredictionSource<B>, D: Descent
{
    type Config = FreeConfig;

    fn run(
        &self,
        ctx: SchedulerCtx<'_, B, P, D>,
        board: &B,
        n_rollouts: usize,
        rollout_depth: usize,
    ) -> RolloutResult {
        ctx.engine.ensure_root(board);

        // Snapshot prior logits for final stats + build root state once.
        let pi_logits = {
            let root = ctx.tree.get_data(board).unwrap();
            let pl = root.pi_logits.clone();
            drop(root);
            pl
        };

        let root_state: Option<D::RootState> = {
            let root = ctx.tree.get_data(board).unwrap();
            let s = ctx.descent.fresh_root_state(&root);
            drop(root);
            s
        };

        let n_threads = ctx.services.len().min(n_rollouts.max(1));
        let rollouts_per_thread = n_rollouts / n_threads;
        let remainder = n_rollouts % n_threads;

        let value_sum: f64 = std::thread::scope(|s| {
            let handles: Vec<_> = (0..n_threads).map(|thread_id| {
                let board = board.clone();
                let count = rollouts_per_thread + if thread_id < remainder { 1 } else { 0 };
                let engine = ctx.engine;
                let root_state_ref = root_state.as_ref();
                s.spawn(move || {
                    let mut local_sum = 0.0f64;
                    for _ in 0..count {
                        engine.begin_rollout(thread_id);
                        let (v, _outcome) = engine.run_single(
                            &board, rollout_depth, None, root_state_ref, thread_id,
                        );
                        local_sum += (-v) as f64;
                    }
                    local_sum
                })
            }).collect();
            handles.into_iter().map(|h| h.join().unwrap()).sum()
        });

        ctx.engine.finalize_rollouts(board);

        let mean_value = if n_rollouts > 0 { (value_sum / n_rollouts as f64) as f32 } else { 0.0 };

        // Read root for final pi + wdl + stats.
        let root = ctx.tree.get_data(board).unwrap();
        let num_actions = root.num_actions();

        // pi = normalized N(a)^(1/T)
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
        // No σ-shift for free scheduler; zero-vector keeps stats shape consistent.
        let sigma_qs = vec![0.0f32; num_actions];
        let search_stats = SearchStats::compute(&prior_pi, &pi, &pi_logits, &sigma_qs);

        let wdl = root.blended_wdl();
        let mcts_wdl = [wdl.win, wdl.draw, wdl.loss];
        drop(root);

        RolloutResult {
            pi,
            value: mean_value,
            mcts_wdl,
            greedy_backup_wdl: [0.0, 1.0, 0.0],
            search_stats,
        }
    }
}
