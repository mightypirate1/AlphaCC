use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crate::backends::{Backend, VersionedModel};
use super::ModelSource;

static HEALTH_MARKED: AtomicBool = AtomicBool::new(false);

fn mark_healthy(health_file: &Option<String>) {
    if HEALTH_MARKED.swap(true, Ordering::Relaxed) {
        return; // already marked
    }
    if let Some(path) = health_file {
        if let Err(e) = std::fs::write(path, "ok") {
            log::error!("[reloader] failed to write health file {path}: {e}");
        } else {
            log::info!("[reloader] health file written: {path}");
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
    use_trt: bool,
    batch_size_map: HashMap<usize, Option<usize>>,
) -> JoinHandle<()> {
    thread::spawn(move || {
        loop {
            thread::sleep(poll_interval);
            reload_cycle(&backend, &source, &health_file, &trt_cache_path, use_trt, &batch_size_map);
        }
    })
}

fn reload_cycle<B: Backend>(backend: &Arc<B>, source: &impl ModelSource, health_file: &Option<String>, trt_cache_path: &Option<String>, use_trt: bool, batch_size_map: &HashMap<usize, Option<usize>>) {
    // 1. Poll desired state
    let desired = match source.desired_versions() {
        Some(d) => d,
        None => {
            log::warn!("[reloader] source unavailable, skipping cycle");
            return;
        }
    };

    // 2. Snapshot current state
    let current = backend.model_store().current_models();

    // 3a. Remove models no longer in desired state
    for &id in current.keys() {
        if !desired.contains_key(&id) {
            backend.model_store().drop_model(id);
            log::info!("[reloader] dropped model_id={id}");
        }
    }

    // 3b. Add/update models whose version differs
    for (&model_id, &desired_version) in &desired {
        let current_version = current.get(&model_id).copied();
        if current_version == Some(desired_version) {
            continue; // unchanged
        }

        // Load bytes
        let batch_size = batch_size_map.get(&model_id).copied().flatten();
        let bytes = match source.load_bytes(desired_version, batch_size) {
            Ok(b) => b,
            Err(e) => {
                log::error!(
                    "[reloader] error loading bytes for model_id={model_id} version={desired_version} batch_size={batch_size:?}: {e}"
                );
                continue;
            }
        };

        // --- TRT: coordinated build (one replica compiles, others load from cache) ---
        let is_builder = if use_trt {
            let lock_position = if source.is_build_complete(desired_version) {
                0 // already built, skip straight to loading
            } else {
                source.try_acquire_build_lock(desired_version)
            };
            let builder = lock_position == 1;
            if builder {
                if let Some(path) = trt_cache_path {
                    log::info!("[reloader] wiping TRT cache at {path} for version={desired_version}");
                    let _ = std::fs::remove_dir_all(path);
                    let _ = std::fs::create_dir_all(path);
                }
            } else {
                log::info!(
                    "[reloader] waiting for another instance to build version={desired_version}..."
                );
                while !source.is_build_complete(desired_version) {
                    thread::sleep(Duration::from_secs(1));
                }
                log::info!("[reloader] build complete for version={desired_version}, loading from cache");
            }
            builder
        } else {
            false
        };

        let model = match backend.model_from_bytes(&bytes) {
            Ok(m) => m,
            Err(e) => {
                if is_builder {
                    source.release_build_lock(desired_version);
                }
                log::error!(
                    "[reloader] model_from_bytes failed for model_id={model_id} version={desired_version}: {e}"
                );
                continue;
            }
        };
        let model = match backend.compile_model(model) {
            Ok(m) => m,
            Err(e) => {
                if is_builder {
                    source.release_build_lock(desired_version);
                }
                log::error!(
                    "[reloader] compile_model failed for model_id={model_id} version={desired_version}: {e}"
                );
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
        log::info!(
            "[reloader] loaded model_id={model_id} version={desired_version} (was {:?})",
            current_version,
        );
        mark_healthy(health_file);
    }
}
