pub mod mcts;
pub mod mcts_node;
pub mod outcome;
pub mod search;
pub mod stats;
pub mod tree;
pub mod noise;

// Back-compat re-exports so external callers can keep using
// `alpha_cc_mcts::descent::...` / `alpha_cc_mcts::scheduler::...`.
pub use search::descent;
pub use search::scheduler;

pub use descent::Descent;
pub use mcts::{MCTS, MCTSParams, RolloutResult};
pub use mcts_node::MCTSNode;
pub use outcome::Outcome;
pub use scheduler::{
    FreeConfig, FreeScheduler, GumbelParams, HalvingConfig, HalvingScheduler,
    RolloutEngine, RootScheduler, SchedulerCtx,
};
pub use stats::SearchStats;
