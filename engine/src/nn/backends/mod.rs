pub mod cpu;
pub mod onnx;
pub mod respond;

use std::collections::HashMap;
use std::sync::Arc;
use arc_swap::ArcSwap;

use crate::nn::server::types::StateBytes;

pub struct VersionedModel<M> {
    pub model: M,
    pub version: usize,
}

/// A model slot that can be atomically swapped or cleared without locking other slots.
type ModelSlot<M> = ArcSwap<Option<VersionedModel<M>>>;

/// Fixed-size collection of model slots with lock-free per-slot access.
pub struct ModelStore<M> {
    slots: Box<[ModelSlot<M>]>,
}

impl<M> ModelStore<M> {
    pub fn new(models: Vec<VersionedModel<M>>, min_capacity: usize) -> Self {
        let n_models = models.len();
        let capacity = n_models.max(min_capacity);
        let mut slots: Vec<_> = models.into_iter()
            .map(|m| ArcSwap::new(Arc::new(Some(m))))
            .collect();
        slots.resize_with(capacity, || ArcSwap::new(Arc::new(None)));
        Self { slots: slots.into_boxed_slice() }
    }

    /// Lock-free load of a model slot. Returns a guard that derefs to `Option<VersionedModel<M>>`.
    pub fn load(&self, model_id: usize) -> arc_swap::Guard<Arc<Option<VersionedModel<M>>>> {
        self.slots[model_id].load()
    }

    /// Atomically swap in a new model at the given slot.
    pub fn set(&self, model_id: usize, model: VersionedModel<M>) {
        self.slots[model_id].store(Arc::new(Some(model)));
    }

    /// Atomically clear a model slot.
    pub fn drop_model(&self, model_id: usize) {
        self.slots[model_id].store(Arc::new(None));
    }

    /// Snapshot of which slots are populated and their versions.
    pub fn current_models(&self) -> HashMap<usize, usize> {
        self.slots.iter().enumerate()
            .filter_map(|(id, slot)| {
                let guard = slot.load();
                guard.as_ref().as_ref().map(|vm| (id, vm.version))
            })
            .collect()
    }

    pub fn len(&self) -> usize {
        self.slots.len()
    }
}

/// Trait that abstracts the encode → inference → decode → respond pipeline.
///
/// Each backend picks its own intermediate types (`Encoded` between
/// encoder↔inference, `Inferred` between inference↔decoder) while the
/// server drives the pipeline generically.
pub trait Backend: Send + Sync + 'static {
    type Model: Send + 'static;
    type Encoded: Send + 'static;
    type Inferred: Send + 'static;

    fn encode(&self, batch: Vec<StateBytes>) -> Self::Encoded;
    fn inference(&self, model_id: u32, input: Self::Encoded) -> Self::Inferred;
    fn decode(&self, output: Self::Inferred) -> Vec<(Vec<u8>, f32)>;
    fn respond(&self, pi_bytes: Vec<u8>, value: f32, move_bytes: Vec<u8>) -> (Vec<u8>, f32);
    fn compile_model(&self, model: Self::Model) -> anyhow::Result<Self::Model>;
    fn model_from_bytes(&self, bytes: &[u8]) -> anyhow::Result<Self::Model>;
    fn model_store(&self) -> &ModelStore<Self::Model>;
}
