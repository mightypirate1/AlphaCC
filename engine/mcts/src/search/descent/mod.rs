use crate::mcts_node::MCTSNode;

pub mod improved;
pub mod puct;

pub use improved::{ImprovedPolicyDescent, SigmaParams};
pub use puct::{DirichletParams, PuctDescent, PuctParams};

pub trait Descent: Send + Sync {
    /// Config type used to construct concrete descents (each impl provides an
    /// inherent `new(config)`; kept inherent to avoid forcing the trait into
    /// scope at call sites).
    type Config: Clone + Send + Sync;
    type RootState: Send + Sync;

    /// Called once per search at the root. `None` means this descent has no
    /// per-search root state (e.g. the improved-policy rule doesn't).
    fn fresh_root_state(&self, root: &MCTSNode) -> Option<Self::RootState>;

    /// Pick next action at `node` during rollout descent.
    /// `root` is Some only at the true root of the current rollout.
    fn select(&self, node: &MCTSNode, root: Option<&Self::RootState>) -> usize;

    /// Tie-breaker used when walking the tree greedily (e.g. for greedy_backup_wdl).
    fn best_child(&self, node: &MCTSNode) -> usize;
}
