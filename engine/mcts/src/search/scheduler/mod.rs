use alpha_cc_core::Board;
use alpha_cc_nn::PredictionSource;

use crate::mcts::{RolloutResult, MCTS};
use crate::search::descent::Descent;

pub mod free;
pub mod halving;

pub use free::{FreeConfig, FreeScheduler};
pub use halving::{GumbelParams, HalvingConfig, HalvingScheduler};

/// Top-level search entry point. A scheduler owns an `MCTS` engine and drives it.
/// Implementations decide how to organise rollouts at the root and how to extract
/// the final `(pi, value, stats)` from the tree afterwards.
pub trait Scheduler<B, T, D>: Send + Sync
where B: Board, T: PredictionSource<B>, D: Descent
{
    fn run(&self, board: &B, n_rollouts: usize, rollout_depth: usize) -> RolloutResult;
    fn mcts(&self) -> &MCTS<B, T, D>;
}
