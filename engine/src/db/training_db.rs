use std::collections::HashMap;
use std::sync::Mutex;

use anyhow::{Context, Result, anyhow};
use redis::Commands;

use crate::nn::reloads::ModelSource;

const WEIGHTS_KEY_PREFIX: &str = "weights";

/// Pure-Rust Redis client for the training database.
///
/// Speaks the same Redis key schema as `alpha_cc.db.TrainingDB` (Python):
/// `weights-{index:04}`, `weights-latest`, `update-models`, `latest-weights-index`.
///
/// Implements `ModelSource` so the reloader can poll and load model bytes.
pub struct TrainingDBRs {
    conn: Mutex<redis::Connection>,
}

impl TrainingDBRs {
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
        let key = weight_key(version as u32);
        let mut conn = self.conn.lock().unwrap();
        let bytes: Option<Vec<u8>> = conn.get(&key)
            .context("redis GET failed")?;
        bytes.ok_or_else(|| anyhow!("no weights for version {version} (key={key})"))
    }

    fn try_acquire_build_lock(&self, version: usize) -> u64 {
        let key = format!("trt-build-lock:{version}");
        let mut conn = self.conn.lock().unwrap();
        let val: u64 = redis::cmd("INCR").arg(&key).query(&mut *conn).unwrap_or(0);
        let _: () = redis::cmd("EXPIRE").arg(&key).arg(120).query(&mut *conn).unwrap_or(());
        val
    }

    fn release_build_lock(&self, version: usize) {
        let key = format!("trt-build-lock:{version}");
        let mut conn = self.conn.lock().unwrap();
        let _: () = redis::cmd("SET").arg(&key).arg("0").query(&mut *conn).unwrap_or(());
    }

    fn is_build_complete(&self, version: usize) -> bool {
        let key = format!("trt-build-done:{version}");
        let mut conn = self.conn.lock().unwrap();
        let exists: bool = conn.exists(&key).unwrap_or(false);
        exists
    }

    fn mark_build_complete(&self, version: usize) {
        let key = format!("trt-build-done:{version}");
        let mut conn = self.conn.lock().unwrap();
        // Mark complete with 300s TTL (long enough for all replicas to load)
        let _: () = redis::cmd("SET")
            .arg(&key).arg("1").arg("EX").arg(300)
            .query(&mut *conn)
            .unwrap_or(());
    }
}

fn weight_key(index: u32) -> String {
    format!("{}-{index:04}", WEIGHTS_KEY_PREFIX)
}
