use std::collections::HashMap;

use anyhow::{Context, Result, anyhow};
use pyo3::prelude::*;

use super::ModelSource;

/// ModelSource that speaks the same Redis protocol as `alpha_cc.db.TrainingDB`.
///
/// Uses redis-py directly (no alpha_cc imports) to avoid pulling in the engine.
///
/// Redis schema (db 0):
/// - hset "update-models": channel(str) -> weight_index(str)
/// - key  "weights-{index:04}": dill.dumps(state_dict)
pub struct TrainingDBSource {
    conn: Py<PyAny>,
}

impl TrainingDBSource {
    /// Connect to Redis at `host`, db 0 (TRAINING).
    pub fn new(host: &str) -> Result<Self> {
        let conn = Python::attach(|py| -> PyResult<Py<PyAny>> {
            let redis = py.import("redis")?;
            let conn = redis.getattr("Redis")?.call1((host, 6379_u16, 0_u16))?;
            // Quick connectivity check
            conn.call_method0("ping")?;
            Ok(conn.unbind())
        })
        .context("failed to connect to Redis")?;
        Ok(Self { conn })
    }
}

impl ModelSource for TrainingDBSource {
    fn desired_versions(&self) -> Option<HashMap<usize, usize>> {
        Python::attach(|py| -> Option<HashMap<usize, usize>> {
            let result = match self.conn.call_method1(py, "hgetall", ("update-models",)) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("[TrainingDBSource] hgetall failed: {e}");
                    return None;
                }
            };
            let dict = match result.cast_bound::<pyo3::types::PyDict>(py) {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("[TrainingDBSource] unexpected return type: {e}");
                    return None;
                }
            };
            let mut map = HashMap::new();
            for (k, v) in dict.iter() {
                // Redis returns bytes, parse via str
                let channel: usize = match k.extract::<Vec<u8>>() {
                    Ok(b) => match std::str::from_utf8(&b).ok().and_then(|s| s.parse().ok()) {
                        Some(n) => n,
                        None => continue,
                    },
                    Err(_) => continue,
                };
                let version: usize = match v.extract::<Vec<u8>>() {
                    Ok(b) => match std::str::from_utf8(&b).ok().and_then(|s| s.parse().ok()) {
                        Some(n) => n,
                        None => continue,
                    },
                    Err(_) => continue,
                };
                map.insert(channel, version);
            }
            Some(map)
        })
    }

    fn load_bytes(&self, version: usize) -> Result<Vec<u8>> {
        let key = format!("weights-{version:04}");
        Python::attach(|py| -> Result<Vec<u8>> {
            let raw = self.conn.call_method1(py, "get", (&key,))
                .context("redis GET failed")?;
            if raw.is_none(py) {
                return Err(anyhow!("no weights found for version {version} (key={key})"));
            }
            let bytes: Vec<u8> = raw.extract(py)
                .context("failed to extract bytes from redis response")?;
            Ok(bytes)
        })
    }
}
