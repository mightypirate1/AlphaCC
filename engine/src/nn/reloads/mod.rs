mod reloader;
pub mod training_db;

pub use reloader::spawn_reloader;
pub use training_db::TrainingDBSource;

use std::collections::HashMap;
use anyhow::Result;

/// Trait for polling desired model state from an external source (e.g. Redis).
///
/// Implementors provide the desired version map and raw model bytes.
/// The reloader loop calls these methods periodically to reconcile
/// the ModelStore with the external desired state.
pub trait ModelSource: Send + Sync + 'static {
    /// Poll the desired configuration. Returns None if the source is unavailable.
    fn desired_versions(&self) -> Option<HashMap<usize, usize>>;
    /// Load raw model bytes for a given version.
    fn load_bytes(&self, version: usize) -> Result<Vec<u8>>;
}
