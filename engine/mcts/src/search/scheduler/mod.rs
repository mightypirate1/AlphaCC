use alpha_cc_core::Board;
use alpha_cc_nn::PredictionSource;

use crate::mcts::RolloutResult;
use crate::outcome::Outcome;
use crate::search::descent::Descent;
use crate::tree::Tree;

pub mod free;
pub mod halving;

pub use free::{FreeConfig, FreeScheduler};
pub use halving::{GumbelParams, HalvingConfig, HalvingScheduler};

pub struct SchedulerCtx<'a, B, P, D>
where B: Board, P: PredictionSource<B>, D: Descent
{
    pub tree: &'a Tree<B>,
    pub services: &'a [P],
    pub model_id: u32,
    pub gamma: f32,
    pub descent: &'a D,
    pub engine: &'a (dyn RolloutEngine<B, D> + 'a),
}

/// Run a single rollout on behalf of a scheduler. Implemented by MCTS.
pub trait RolloutEngine<B: Board, D: Descent>: Send + Sync {
    fn run_single(
        &self,
        board: &B,
        rollout_depth: usize,
        forced_action: Option<usize>,
        root_state: Option<&D::RootState>,
        thread_id: usize,
    ) -> (f32, Option<Outcome>);

    /// Insert a fresh NN leaf for `board` if one isn't already in the tree.
    fn ensure_root(&self, board: &B);

    /// Advance the tree's path-tracking after a batch of parallel rollouts.
    fn finalize_rollouts(&self, board: &B);

    /// Mark the start of a rollout on `thread_id` (clears the per-thread path).
    fn begin_rollout(&self, thread_id: usize);
}

pub trait RootScheduler<B, P, D>: Send + Sync
where B: Board, P: PredictionSource<B>, D: Descent
{
    /// Config type used to construct concrete schedulers (see each impl's inherent
    /// `new(config)` constructor — kept inherent to avoid type-inference ambiguity
    /// when a scheduler impls the trait for many (B, P, D) triples).
    type Config: Clone + Send + Sync;

    fn run(
        &self,
        ctx: SchedulerCtx<'_, B, P, D>,
        board: &B,
        n_rollouts: usize,
        rollout_depth: usize,
    ) -> RolloutResult;
}
