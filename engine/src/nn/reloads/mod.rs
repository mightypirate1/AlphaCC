mod reloader;

pub use reloader::spawn_reloader;

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
    /// Load raw model bytes for a given version and batch_size variant.
    fn load_bytes(&self, version: usize, batch_size: Option<usize>) -> Result<Vec<u8>>;

    /// Try to acquire a build lock for the given version.
    /// Returns the lock position: 1 = builder, 2+ = wait in queue.
    fn try_acquire_build_lock(&self, _version: usize) -> u64 { 1 }
    /// Release the build lock.
    fn release_build_lock(&self, _version: usize) {}
    /// Check whether another instance has finished building this version.
    fn is_build_complete(&self, _version: usize) -> bool { true }
    /// Mark the build as complete so other instances can proceed.
    fn mark_build_complete(&self, _version: usize) {}
}
