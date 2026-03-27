use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crate::nn::backends::{Backend, VersionedModel};
use crate::nn::server::gate::ServiceGate;
use super::ModelSource;

static HEALTH_MARKED: AtomicBool = AtomicBool::new(false);

fn mark_healthy(health_file: &Option<String>) {
    if HEALTH_MARKED.swap(true, Ordering::Relaxed) {
        return; // already marked
    }
    if let Some(path) = health_file {
        if let Err(e) = std::fs::write(path, "ok") {
            eprintln!("[reloader] warning: failed to write health file {path}: {e}");
        } else {
            eprintln!("[reloader] health file written: {path}");
        }
    }
}

/// Spawn a background thread that periodically polls `source` for desired model
/// versions, diffs against the current ModelStore state, and reconciles:
/// dropping removed models, loading + compiling new/changed ones.
///
/// Fully fault-tolerant — the loop never crashes. All errors are logged and skipped.
pub fn spawn_reloader<B: Backend>(
    backend: Arc<B>,
    source: impl ModelSource,
    poll_interval: Duration,
    health_file: Option<String>,
    trt_cache_path: Option<String>,
    gate: ServiceGate,
    use_trt: bool,
) -> JoinHandle<()> {
    thread::spawn(move || {
        loop {
            thread::sleep(poll_interval);
            reload_cycle(&backend, &source, &health_file, &trt_cache_path, &gate, use_trt);
        }
    })
}

fn reload_cycle<B: Backend>(backend: &Arc<B>, source: &impl ModelSource, health_file: &Option<String>, trt_cache_path: &Option<String>, gate: &ServiceGate, use_trt: bool) {
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

        // --- TRT: coordinated build with gate closure ---
        // Without TRT, model_from_bytes is a lightweight CUDA load — no gate needed.
        let is_builder = if use_trt {
            let lock_position = if source.is_build_complete(desired_version) {
                0 // already built, skip straight to loading
            } else {
                source.try_acquire_build_lock(desired_version)
            };
            let builder = lock_position == 1;
            if builder {
                if let Some(path) = trt_cache_path {
                    eprintln!("[reloader] wiping TRT cache at {path} for version={desired_version}");
                    let _ = std::fs::remove_dir_all(path);
                    let _ = std::fs::create_dir_all(path);
                }
            } else {
                eprintln!(
                    "[reloader] waiting for another instance to build version={desired_version}..."
                );
                while !source.is_build_complete(desired_version) {
                    thread::sleep(Duration::from_secs(1));
                }
                let stagger = Duration::from_secs((1 + lock_position.saturating_sub(1)) * 5);
                eprintln!(
                    "[reloader] build complete for version={desired_version}, loading from cache (stagger={stagger:?})"
                );
                thread::sleep(stagger);
            }
            builder
        } else {
            false
        };

        // Close gate during GPU-heavy model loading (TRT compilation or cache load).
        // Without TRT the CUDA EP load is lightweight — no gate needed.
        let _unavailable_guard = if use_trt {
            eprintln!("[reloader] gate closed — stopping inference for model reload");
            Some(gate.unavailable_guard())
        } else {
            None
        };

        let model = match backend.model_from_bytes(&bytes) {
            Ok(m) => m,
            Err(e) => {
                eprintln!(
                    "[reloader] model_from_bytes failed for model_id={model_id} version={desired_version}: {e}"
                );
                if is_builder {
                    source.release_build_lock(desired_version);
                }
                continue;
            }
        };
        let model = match backend.compile_model(model) {
            Ok(m) => m,
            Err(e) => {
                eprintln!(
                    "[reloader] compile_model failed for model_id={model_id} version={desired_version}: {e}"
                );
                if is_builder {
                    source.release_build_lock(desired_version);
                }
                continue;
            }
        };

        if is_builder {
            source.mark_build_complete(desired_version);
            source.release_build_lock(desired_version);
        }

        // Arc-swap into the store
        backend.model_store().set(
            model_id,
            VersionedModel { model, version: desired_version },
        );
        eprintln!(
            "[reloader] loaded model_id={model_id} version={desired_version} (was {:?})",
            current_version,
        );
        mark_healthy(health_file);
        // _gate_guard dropped here → gate restored
        eprintln!("[reloader] gate open — resuming inference");
    }
}
