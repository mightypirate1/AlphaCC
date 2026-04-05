use std::sync::Mutex;

use ort::session::Session;
use ort::value::Tensor;

use crate::backends::{Backend, DecodedPrediction, ModelStore, VersionedModel};
use crate::io;
use crate::server::types::StateBytes;

/// Thread-safe wrapper around `ort::Session` (same as OnnxSession but without CUDA).
pub struct CpuSession(Mutex<Session>);

impl CpuSession {
    pub fn lock(&self) -> std::sync::MutexGuard<'_, Session> {
        self.0.lock().unwrap()
    }
}

pub struct CpuBackend {
    models: ModelStore<CpuSession>,
    game_size: i64,
    verbose: bool,
}

impl CpuBackend {
    pub fn new(models: Vec<VersionedModel<CpuSession>>, game_size: i64, verbose: bool, max_models: usize) -> Self {
        Self {
            models: ModelStore::new(models, max_models),
            game_size,
            verbose,
        }
    }

    fn build_session(bytes: &[u8]) -> anyhow::Result<CpuSession> {
        log::info!("[cpu] building session from {} bytes", bytes.len());
        let session = Session::builder()
            .map_err(|e| anyhow::anyhow!("session builder: {e}"))?
            .commit_from_memory(bytes)
            .map_err(|e| anyhow::anyhow!("commit_from_memory: {e}"))?;
        Ok(CpuSession(Mutex::new(session)))
    }

    pub fn load_session_from_file(path: &str) -> anyhow::Result<CpuSession> {
        let session = Session::builder()
            .map_err(|e| anyhow::anyhow!("session builder: {e}"))?
            .commit_from_file(path)
            .map_err(|e| anyhow::anyhow!("commit_from_file: {e}"))?;
        Ok(CpuSession(Mutex::new(session)))
    }
}

/// CPU tensors: just Vec<f32> passed between pipeline stages.
pub struct CpuEncoded {
    pub data: Vec<f32>,
    pub batch_size: usize,
    pub game_size: usize,
}

pub struct CpuInferred {
    pub policy: Vec<f32>,
    pub value: Vec<f32>,
    pub batch_size: usize,
    pub game_size: usize,
}

impl Backend for CpuBackend {
    type Model = CpuSession;
    type Encoded = CpuEncoded;
    type Inferred = CpuInferred;

    fn encode(&self, batch: Vec<StateBytes>) -> CpuEncoded {
        let n = batch.len();
        let s = self.game_size as usize;
        let data: Vec<f32> = batch.iter()
            .flat_map(|item| io::state_bytes_as_f32s(item))
            .copied()
            .collect();
        CpuEncoded { data, batch_size: n, game_size: s }
    }

    fn inference(&self, model_id: u32, input: CpuEncoded) -> CpuInferred {
        let s = input.game_size;
        let n = input.batch_size;
        if self.verbose {
            log::debug!("Inference model_id={model_id} batch_size={n}");
        }

        loop {
            let guard = self.models.load(model_id as usize);
            if let Some(vm) = guard.as_ref().as_ref() {
                let mut session = vm.model.lock();

                let input_tensor = Tensor::from_array(([n, 2, s, s], input.data))
                    .expect("failed to create input tensor");

                let inputs = vec![
                    (std::borrow::Cow::from("input"), ort::session::SessionInputValue::from(input_tensor)),
                ];
                let outputs = session.run(inputs).expect("onnx run failed");

                let policy_value = outputs.get("policy").expect("missing policy output");
                let value_value = outputs.get("value").expect("missing value output");
                let (_, policy_slice) = policy_value.try_extract_tensor::<f32>().expect("extract policy");
                let (_, value_slice) = value_value.try_extract_tensor::<f32>().expect("extract value");

                return CpuInferred {
                    policy: policy_slice.to_vec(),
                    value: value_slice.to_vec(),
                    batch_size: n,
                    game_size: s,
                };
            }
            drop(guard);
            log::warn!("model_id={model_id} not loaded yet, waiting...");
            std::thread::sleep(std::time::Duration::from_secs(1));
        }
    }

    fn decode(&self, output: CpuInferred) -> Vec<DecodedPrediction> {
        let s4 = output.game_size.pow(4);
        let mut decoded = Vec::with_capacity(output.batch_size);
        for i in 0..output.batch_size {
            let v = output.value[i];
            let pi_row = &output.policy[i * s4..(i + 1) * s4];
            let pi_bytes: Vec<u8> = bytemuck::cast_slice(pi_row).to_vec();
            decoded.push((pi_bytes, v));
        }
        decoded
    }

    fn respond(&self, pi_bytes: Vec<u8>, value: f32, move_bytes: Vec<u8>) -> DecodedPrediction {
        crate::backends::respond::respond(&pi_bytes, value, &move_bytes, self.game_size as usize)
    }

    fn compile_model(&self, model: CpuSession) -> anyhow::Result<CpuSession> {
        Ok(model)
    }

    fn model_from_bytes(&self, bytes: &[u8]) -> anyhow::Result<CpuSession> {
        Self::build_session(bytes)
    }

    fn model_store(&self) -> &ModelStore<CpuSession> {
        &self.models
    }
}
