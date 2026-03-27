use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Shared flag indicating the service is temporarily unavailable (e.g. during model reload).
///
/// The reloader sets the gate to unavailable before GPU-intensive work
/// (model_from_bytes, compile_model) and back to available after the new
/// model is swapped in. The gRPC service checks the gate on each request
/// and closes the worker stream when unavailable, causing workers to
/// reconnect to a different replica.
#[derive(Clone)]
pub struct ServiceGate(Arc<AtomicBool>);

impl ServiceGate {
    pub fn new() -> Self {
        Self(Arc::new(AtomicBool::new(true))) // starts available
    }

    pub fn set_unavailable(&self) {
        self.0.store(false, Ordering::Release);
    }

    pub fn set_available(&self) {
        self.0.store(true, Ordering::Release);
    }

    pub fn is_available(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }

    /// Returns a guard that sets the gate to unavailable on creation
    /// and back to available when dropped.
    pub fn unavailable_guard(&self) -> UnavailableGuard {
        self.set_unavailable();
        UnavailableGuard(self.clone())
    }
}

/// RAII guard that restores the gate to available on drop.
pub struct UnavailableGuard(ServiceGate);

impl Drop for UnavailableGuard {
    fn drop(&mut self) {
        self.0.set_available();
    }
}
