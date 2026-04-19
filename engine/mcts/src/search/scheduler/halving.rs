use alpha_cc_core::Board;
use alpha_cc_nn::{softmax, PredictionSource};

use crate::mcts::{MCTSParams, RolloutResult, MCTS};
use crate::noise;
use crate::search::descent::improved::{sigma, ImprovedPolicyDescent, SigmaParams};
use crate::search::descent::Descent;
use crate::search::scheduler::Scheduler;
use crate::stats::SearchStats;

#[derive(Clone)]
pub struct GumbelParams {
    pub all_at_least_once: bool,
    pub base_count: usize,
    pub floor_count: usize,
    pub keep_frac: f32,
}

#[derive(Clone)]
pub struct HalvingConfig {
    pub gumbel: GumbelParams,
    pub sigma: SigmaParams,
}

pub struct HalvingScheduler<B, T, D>
where B: Board, T: PredictionSource<B>, D: Descent
{
    mcts: MCTS<B, T, D>,
    config: HalvingConfig,
}

impl<B, T, D> HalvingScheduler<B, T, D>
where B: Board, T: PredictionSource<B>, D: Descent
{
    pub fn new(mcts: MCTS<B, T, D>, config: HalvingConfig) -> Self {
        Self { mcts, config }
    }
}

/// Preset: improved-policy descent + sequential-halving scheduling.
impl<B, T> HalvingScheduler<B, T, ImprovedPolicyDescent>
where B: Board, T: PredictionSource<B>
{
    pub fn build_improved_halving(
        services: Vec<T>,
        model_id: u32,
        gamma: f32,
        sigma: SigmaParams,
        gumbel: GumbelParams,
        pruning_tree: bool,
        debug_prints: bool,
    ) -> Self {
        let descent = ImprovedPolicyDescent::new(sigma.clone());
        let mcts = MCTS::new(
            services, model_id, MCTSParams { gamma }, descent, pruning_tree, debug_prints,
        );
        Self::new(mcts, HalvingConfig { gumbel, sigma })
    }
}

impl<B, T, D> Scheduler<B, T, D> for HalvingScheduler<B, T, D>
where B: Board, T: PredictionSource<B>, D: Descent
{
    fn mcts(&self) -> &MCTS<B, T, D> { &self.mcts }

    fn run(&self, board: &B, n_rollouts: usize, rollout_depth: usize) -> RolloutResult {
        let gumbel = &self.config.gumbel;
        let c_visit = self.config.sigma.c_visit;
        let c_scale = self.config.sigma.c_scale;

        self.mcts.ensure_root(board);

        let root = self.mcts.tree().get_data(board).unwrap();
        let num_actions = root.num_actions();
        let pi_logits = root.pi_logits.clone();
        drop(root);

        // Sample Gumbel noise and compute initial scores.
        let gumbels: Vec<f32> = (0..num_actions).map(|_| noise::gumbel()).collect();
        let gumbel_scores: Vec<f32> = (0..num_actions)
            .map(|a| gumbels[a] + pi_logits[a])
            .collect();

        let mut candidates: Vec<usize> = (0..num_actions).collect();
        candidates.sort_by(|&a, &b| gumbel_scores[b].partial_cmp(&gumbel_scores[a]).unwrap());
        if !gumbel.all_at_least_once {
            candidates.truncate(gumbel.base_count.min(num_actions));
        }

        let mut sim_budget = n_rollouts;

        while sim_budget > 0 && !candidates.is_empty() {
            let sims_this_round = candidates.len().min(sim_budget);
            let actions_this_round: Vec<usize> = candidates[..sims_this_round].to_vec();

            let n_threads = self.mcts.services().len().min(sims_this_round);
            let rollouts_per_thread = sims_this_round / n_threads;
            let remainder = sims_this_round % n_threads;

            std::thread::scope(|s| {
                let handles: Vec<_> = (0..n_threads).map(|thread_id| {
                    let board = board.clone();
                    let start = thread_id * rollouts_per_thread + thread_id.min(remainder);
                    let count = rollouts_per_thread + if thread_id < remainder { 1 } else { 0 };
                    let actions = &actions_this_round[start..start + count];
                    let mcts = &self.mcts;
                    s.spawn(move || {
                        for &action in actions {
                            mcts.begin_rollout(thread_id);
                            mcts.rollout(&board, rollout_depth, Some(action), None, thread_id);
                        }
                    })
                }).collect();
                for h in handles {
                    h.join().unwrap();
                }
            });

            self.mcts.finalize_rollouts(board);
            sim_budget -= sims_this_round;

            if sim_budget > 0 {
                let root = self.mcts.tree().get_data(board).unwrap();
                let n_max = (0..num_actions).map(|a| root.get_n(a)).max().unwrap_or(0);

                candidates.sort_by(|&a, &b| {
                    let sa = gumbels[a] + sigma(root.completed_q(a), root.get_n(a), n_max, c_visit, c_scale);
                    let sb = gumbels[b] + sigma(root.completed_q(b), root.get_n(b), n_max, c_visit, c_scale);
                    sb.partial_cmp(&sa).unwrap()
                });
                drop(root);

                let next_size = (candidates.len() as f32 * gumbel.keep_frac) as usize;
                let next_size = next_size.max(gumbel.floor_count).min(candidates.len());
                candidates.truncate(next_size);
            }
        }

        // Compute improved policy target over ALL actions.
        let root = self.mcts.tree().get_data(board).unwrap();
        let n_max = (0..num_actions).map(|a| root.get_n(a)).max().unwrap_or(0);

        let sigma_qs: Vec<f32> = (0..num_actions)
            .map(|a| sigma(root.completed_q(a), root.get_n(a), n_max, c_visit, c_scale))
            .collect();
        let improved_logits: Vec<f32> = (0..num_actions)
            .map(|a| pi_logits[a] + sigma_qs[a])
            .collect();
        let pi = softmax(&improved_logits);

        let prior_pi = softmax(&pi_logits);
        let search_stats = SearchStats::compute(&prior_pi, &pi, &pi_logits, &sigma_qs);

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

        let greedy_backup_wdl = self.mcts.greedy_backup_wdl(board, rollout_depth);

        RolloutResult {
            pi,
            value,
            mcts_wdl,
            greedy_backup_wdl,
            search_stats,
        }
    }
}
