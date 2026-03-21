use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::nn::backends::{Backend, ModelStore, VersionedModel};
use crate::nn::server::types::StateBytes;
use super::{PyModel, PyTensor, encoder, inference, decoder};

pub struct PyTorchBackend {
    pub models: ModelStore<PyModel>,
    game_size: i64,
    verbose: bool,
}

impl PyTorchBackend {
    pub fn new(models: Vec<VersionedModel<PyModel>>, game_size: i64, verbose: bool, max_models: usize) -> Self {
        Self { models: ModelStore::new(models, max_models), game_size, verbose }
    }

    /// Set up a DefaultNet model on the given device, optionally compiled.
    pub fn setup_model(game_size: i64, compile_mode: Option<&str>) -> PyModel {
        Python::attach(|py| {
            let torch = py.import("torch").expect("failed to import torch");
            let device = torch.call_method1("device", ("cuda:0",)).expect("failed to create device");

            // Enable TF32
            torch.getattr("backends").unwrap().getattr("cuda").unwrap().getattr("matmul").unwrap()
                .setattr("allow_tf32", true).unwrap();
            torch.getattr("backends").unwrap().getattr("cudnn").unwrap()
                .setattr("allow_tf32", true).unwrap();
            torch.call_method1("set_float32_matmul_precision", ("high",)).unwrap();

            let nets_module = py.import("alpha_cc.nn.nets").expect("failed to import alpha_cc.nn.nets");
            let model = nets_module.getattr("DefaultNet").unwrap()
                .call1((game_size,)).expect("failed to create DefaultNet");
            model.call_method1("to", (&device,)).unwrap();
            model.call_method0("eval").unwrap();

            let model: Bound<'_, PyAny> = if let Some(mode) = compile_mode {
                let kwargs = PyDict::new(py);
                kwargs.set_item("mode", mode).unwrap();
                torch.call_method("compile", (&model,), Some(&kwargs))
                    .expect("torch.compile failed")
            } else {
                model
            };

            model.unbind()
        })
    }

    /// Load a model from a checkpoint path onto the given device, optionally compiled.
    pub fn setup_model_from_path(nn_path: &str, game_size: i64, compile_mode: Option<&str>) -> PyModel {
        Python::attach(|py| {
            let torch = py.import("torch").expect("failed to import torch");
            let device = torch.call_method1("device", ("cuda:0",)).expect("failed to create device");

            // Enable TF32
            torch.getattr("backends").unwrap().getattr("cuda").unwrap().getattr("matmul").unwrap()
                .setattr("allow_tf32", true).unwrap();
            torch.getattr("backends").unwrap().getattr("cudnn").unwrap()
                .setattr("allow_tf32", true).unwrap();
            torch.call_method1("set_float32_matmul_precision", ("high",)).unwrap();

            let nets_module = py.import("alpha_cc.nn.nets").expect("failed to import alpha_cc.nn.nets");
            let model = nets_module.getattr("DefaultNet").unwrap()
                .call1((game_size,))
                .expect("failed to create DefaultNet");
            model.call_method1("to", (&device,)).unwrap();
            let state_dict = torch.call_method1("load", (nn_path,))
                .expect("failed to load state dict from path");
            model.call_method1("load_state_dict", (&state_dict,))
                .expect("failed to load_state_dict");
            model.call_method0("eval").unwrap();

            let model: Bound<'_, PyAny> = if let Some(mode) = compile_mode {
                let kwargs = PyDict::new(py);
                kwargs.set_item("mode", mode).unwrap();
                torch.call_method("compile", (&model,), Some(&kwargs))
                    .expect("torch.compile failed")
            } else {
                model
            };

            model.unbind()
        })
    }
}

impl Backend for PyTorchBackend {
    type Model = PyModel;
    type Encoded = PyTensor;
    type Inferred = PyTensor;

    fn encode(&self, batch: Vec<StateBytes>) -> PyTensor {
        encoder::py_encode(batch, self.game_size)
    }

    fn inference(&self, model_id: u32, input: PyTensor) -> PyTensor {
        if self.verbose {
            Python::attach(|py| {
                let size = input.bind(py).call_method0("size").unwrap();
                println!("Inference model_id={model_id} batch_size={size}");
            });
        }
        // Wait for the model to be loaded (e.g. by the reloader on first startup)
        loop {
            let guard = self.models.load(model_id as usize);
            if let Some(vm) = guard.as_ref().as_ref() {
                return inference::py_inference(&vm.model, input);
            }
            drop(guard);
            eprintln!("model_id={model_id} not loaded yet, waiting...");
            std::thread::sleep(std::time::Duration::from_secs(1));
        }
    }

    fn decode(&self, output: PyTensor) -> Vec<(Vec<u8>, f32)> {
        decoder::py_decode(output, self.game_size)
    }

    fn respond(&self, pi_bytes: Vec<u8>, value: f32, move_bytes: Vec<u8>) -> (Vec<u8>, f32) {
        crate::nn::backends::respond::respond(&pi_bytes, value, &move_bytes, self.game_size as usize)
    }

    fn compile_model(&self, model: PyModel) -> anyhow::Result<PyModel> {
        let start = std::time::Instant::now();
        let compiled = Python::attach(|py| -> PyResult<Py<PyAny>> {
            let torch = py.import("torch")?;
            let kwargs = PyDict::new(py);
            kwargs.set_item("mode", "max-autotune-no-cudagraphs")?;
            Ok(torch.call_method("compile", (model.bind(py),), Some(&kwargs))?.unbind())
        }).map_err(|e| anyhow::anyhow!("torch.compile failed: {e}"))?;
        println!("torch.compile(mode=\"max-autotune-no-cudagraphs\") completed in {:.2?}", start.elapsed());
        Ok(compiled)
    }

    fn model_from_bytes(&self, bytes: &[u8]) -> anyhow::Result<PyModel> {
        let game_size = self.game_size;
        Python::attach(|py| -> PyResult<Py<PyAny>> {
            // Deserialize the dill blob into a state_dict
            let dill = py.import("dill")?;
            let py_bytes = pyo3::types::PyBytes::new(py, bytes);
            let state_dict = dill.call_method1("loads", (py_bytes,))?;

            // Create a fresh model and load the state_dict
            let torch = py.import("torch")?;
            let device = torch.call_method1("device", ("cuda:0",))?;
            let nets = py.import("alpha_cc.nn.nets")?;
            let model = nets.getattr("DefaultNet")?.call1((game_size,))?;
            model.call_method1("to", (&device,))?;
            model.call_method1("load_state_dict", (&state_dict,))?;
            model.call_method0("eval")?;
            Ok(model.unbind())
        }).map_err(|e| anyhow::anyhow!("model_from_bytes failed: {e}"))
    }

    fn model_store(&self) -> &ModelStore<PyModel> {
        &self.models
    }
}
