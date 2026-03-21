use std::collections::HashMap;
use std::sync::Mutex;

use anyhow::{Context, Result, anyhow};
use pyo3::prelude::*;
use redis::Commands;

use crate::nn::reloads::ModelSource;

/// Pure-Rust Redis client for the training database.
///
/// Speaks the same Redis key schema as `alpha_cc.db.TrainingDB` (Python) for
/// shared keys (`update-models`, `latest-weights-index`), and adds a parallel
/// `jit-weights-*` key family for JIT-traced CModule blobs.
///
/// Exposed to Python via PyO3 so the trainer can publish JIT weights.
/// Implements `ModelSource` so the tch backend reloader can consume them.
#[pyclass]
pub struct TrainingDBRs {
    conn: Mutex<redis::Connection>,
}

#[pymethods]
impl TrainingDBRs {
    #[new]
    #[pyo3(signature = (host = "localhost"))]
    fn new(host: &str) -> PyResult<Self> {
        let url = format!("redis://{host}:6379/0");
        let client = redis::Client::open(url.as_str())
            .map_err(|e| pyo3::exceptions::PyConnectionError::new_err(format!("redis client: {e}")))?;
        let conn = client.get_connection()
            .map_err(|e| pyo3::exceptions::PyConnectionError::new_err(format!("redis connect: {e}")))?;
        Ok(Self { conn: Mutex::new(conn) })
    }

    /// Publish JIT-traced model bytes at a specific index.
    fn jit_weights_publish(&self, index: u32, payload: &[u8], set_latest: bool) -> PyResult<()> {
        let mut conn = self.conn.lock().unwrap();
        let key = jit_weight_key(index);
        conn.set::<_, _, ()>(&key, payload)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("redis SET {key}: {e}")))?;
        if set_latest {
            conn.set::<_, _, ()>("jit-weights-latest", payload)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("redis SET jit-weights-latest: {e}")))?;
        }
        Ok(())
    }
}

impl TrainingDBRs {
    /// Non-PyO3 constructor for use from pure Rust (e.g. nn-service binary).
    pub fn from_host(host: &str) -> Result<Self> {
        let url = format!("redis://{host}:6379/0");
        let client = redis::Client::open(url.as_str())
            .context("failed to open redis client")?;
        let conn = client.get_connection()
            .context("failed to connect to redis")?;
        Ok(Self { conn: Mutex::new(conn) })
    }
}

impl ModelSource for TrainingDBRs {
    fn desired_versions(&self) -> Option<HashMap<usize, usize>> {
        let mut conn = self.conn.lock().unwrap();
        let raw: HashMap<String, String> = match conn.hgetall("update-models") {
            Ok(r) => r,
            Err(e) => {
                eprintln!("[TrainingDBRs] hgetall failed: {e}");
                return None;
            }
        };
        let mut map = HashMap::new();
        for (k, v) in raw {
            if let (Ok(channel), Ok(version)) = (k.parse::<usize>(), v.parse::<usize>()) {
                map.insert(channel, version);
            }
        }
        Some(map)
    }

    fn load_bytes(&self, version: usize) -> Result<Vec<u8>> {
        let key = jit_weight_key(version as u32);
        let mut conn = self.conn.lock().unwrap();
        let bytes: Option<Vec<u8>> = conn.get(&key)
            .context("redis GET failed")?;
        bytes.ok_or_else(|| anyhow!("no JIT weights for version {version} (key={key})"))
    }
}

fn jit_weight_key(index: u32) -> String {
    format!("jit-weights-{index:04}")
}
