use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crate::nn::backends::{Backend, VersionedModel};
use super::ModelSource;

/// Spawn a background thread that periodically polls `source` for desired model
/// versions, diffs against the current ModelStore state, and reconciles:
/// dropping removed models, loading + compiling new/changed ones.
///
/// Fully fault-tolerant — the loop never crashes. All errors are logged and skipped.
pub fn spawn_reloader<B: Backend>(
    backend: Arc<B>,
    source: impl ModelSource,
    poll_interval: Duration,
) -> JoinHandle<()> {
    thread::spawn(move || {
        pyo3::Python::attach(|_py| {});

        loop {
            thread::sleep(poll_interval);
            reload_cycle(&backend, &source);
        }
    })
}

fn reload_cycle<B: Backend>(backend: &Arc<B>, source: &impl ModelSource) {
    // 1. Poll desired state
    let desired = match source.desired_versions() {
        Some(d) => d,
        None => {
            eprintln!("[reloader] warning: source unavailable, skipping cycle");
            return;
        }
    };

    // 2. Snapshot current state
    let current = backend.model_store().current_models();

    // 3a. Remove models no longer in desired state
    for &id in current.keys() {
        if !desired.contains_key(&id) {
            backend.model_store().drop_model(id);
            eprintln!("[reloader] dropped model_id={id}");
        }
    }

    // 3b. Add/update models whose version differs
    for (&model_id, &desired_version) in &desired {
        let current_version = current.get(&model_id).copied();
        if current_version == Some(desired_version) {
            continue; // unchanged
        }

        // Load bytes
        let bytes = match source.load_bytes(desired_version) {
            Ok(b) => b,
            Err(e) => {
                eprintln!(
                    "[reloader] error loading bytes for model_id={model_id} version={desired_version}: {e}"
                );
                continue;
            }
        };

        // Construct model from bytes and compile
        let model = match backend.model_from_bytes(&bytes) {
            Ok(m) => m,
            Err(e) => {
                eprintln!(
                    "[reloader] model_from_bytes failed for model_id={model_id} version={desired_version}: {e}"
                );
                continue;
            }
        };
        let model = match backend.compile_model(model) {
            Ok(m) => m,
            Err(e) => {
                eprintln!(
                    "[reloader] compile_model failed for model_id={model_id} version={desired_version}: {e}"
                );
                continue;
            }
        };

        // Arc-swap into the store
        backend.model_store().set(
            model_id,
            VersionedModel { model, version: desired_version },
        );
        eprintln!(
            "[reloader] loaded model_id={model_id} version={desired_version} (was {:?})",
            current_version,
        );
    }
}
