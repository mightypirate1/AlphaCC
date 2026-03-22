use tch::{CModule, Device, Tensor};

use crate::nn::backends::{Backend, ModelStore, VersionedModel};
use crate::nn::server::types::StateBytes;
use super::{encoder, inference, decoder};

pub struct TchBackend {
    pub models: ModelStore<CModule>,
    game_size: i64,
    device: Device,
    verbose: bool,
}

impl TchBackend {
    pub fn new(models: Vec<VersionedModel<CModule>>, game_size: i64, device: Device, verbose: bool, max_models: usize) -> Self {
        Self { models: ModelStore::new(models, max_models), game_size, device, verbose }
    }
}

impl Backend for TchBackend {
    type Model = CModule;
    type Encoded = Tensor;
    type Inferred = (Tensor, Tensor);

    fn encode(&self, batch: Vec<StateBytes>) -> Tensor {
        encoder::encode(batch, self.game_size, self.device)
    }

    fn inference(&self, model_id: u32, input: Tensor) -> (Tensor, Tensor) {
        if self.verbose {
            println!("Inference model_id={model_id} batch_size={}", input.size()[0]);
        }
        loop {
            let guard = self.models.load(model_id as usize);
            if let Some(vm) = guard.as_ref().as_ref() {
                return inference::nn_inference(&vm.model, input);
            }
            drop(guard);
            eprintln!("model_id={model_id} not loaded yet, waiting...");
            std::thread::sleep(std::time::Duration::from_secs(1));
        }
    }

    fn decode(&self, output: (Tensor, Tensor)) -> Vec<(Vec<u8>, f32)> {
        decoder::decode(output, self.game_size)
    }

    fn respond(&self, pi_bytes: Vec<u8>, value: f32, move_bytes: Vec<u8>) -> (Vec<u8>, f32) {
        crate::nn::backends::respond::respond(&pi_bytes, value, &move_bytes, self.game_size as usize)
    }

    fn compile_model(&self, model: CModule) -> anyhow::Result<CModule> {
        println!("torch.compile not supported for tch-rs JIT backend, skipping");
        Ok(model)
    }

    fn model_from_bytes(&self, bytes: &[u8]) -> anyhow::Result<CModule> {
        let mut cursor = std::io::Cursor::new(bytes);
        CModule::load_data_on_device(&mut cursor, self.device)
            .map_err(|e| anyhow::anyhow!("failed to load CModule from bytes: {e}"))
    }

    fn model_store(&self) -> &ModelStore<CModule> {
        &self.models
    }
}
