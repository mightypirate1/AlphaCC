use std::sync::{Mutex, OnceLock};

use ort::execution_providers::{CUDAExecutionProvider, TensorRTExecutionProvider};
use ort::memory::{Allocator, AllocationDevice, AllocatorType, MemoryInfo, MemoryType};
use ort::session::Session;
use ort::value::{DynTensor, DynValue, TensorElementType};

use crate::nn::backends::{Backend, ModelStore, VersionedModel};
use crate::nn::server::types::StateBytes;
use super::{encoder, inference, decoder};

// --- CUDA FFI (runtime-loaded to avoid hard link dep on libcudart) -------------

const CUDA_MEMCPY_HOST_TO_DEVICE: i32 = 1;
const CUDA_MEMCPY_DEVICE_TO_HOST: i32 = 2;

type CudaMemcpyFn = unsafe extern "C" fn(
    dst: *mut std::ffi::c_void,
    src: *const std::ffi::c_void,
    count: usize,
    kind: i32,
) -> i32;

fn load_cuda_memcpy() -> CudaMemcpyFn {
    unsafe {
        let handle = libc::dlopen(b"libcudart.so.12\0".as_ptr().cast(), libc::RTLD_NOW | libc::RTLD_GLOBAL);
        assert!(!handle.is_null(), "failed to dlopen libcudart.so");
        let sym = libc::dlsym(handle, b"cudaMemcpy\0".as_ptr().cast());
        assert!(!sym.is_null(), "failed to dlsym cudaMemcpy");
        std::mem::transmute(sym)
    }
}

static CUDA_MEMCPY: std::sync::OnceLock<CudaMemcpyFn> = std::sync::OnceLock::new();

fn cuda_memcpy() -> CudaMemcpyFn {
    *CUDA_MEMCPY.get_or_init(load_cuda_memcpy)
}

/// Copy `data` from host into a GPU-allocated `DynTensor`.
/// # Safety
/// `tensor` must have been allocated on a CUDA device with enough capacity for `data`.
pub(crate) unsafe fn copy_to_gpu_tensor(tensor: &mut DynTensor, data: &[f32]) {
    let gpu_ptr = tensor.data_ptr_mut();
    let size_bytes = data.len() * std::mem::size_of::<f32>();
    let ret = cuda_memcpy()(gpu_ptr, data.as_ptr().cast(), size_bytes, CUDA_MEMCPY_HOST_TO_DEVICE);
    assert!(ret == 0, "cudaMemcpy H2D failed with error code {ret}");
}

/// Copy GPU data into a pre-allocated CPU slice.
/// # Safety
/// `gpu_ptr` must point to GPU memory with at least `dst.len() * size_of::<f32>()` bytes.
pub(crate) unsafe fn cuda_memcpy_d2h(dst: &mut [f32], gpu_ptr: *const std::ffi::c_void) {
    let size_bytes = dst.len() * std::mem::size_of::<f32>();
    let ret = cuda_memcpy()(dst.as_mut_ptr().cast(), gpu_ptr, size_bytes, CUDA_MEMCPY_DEVICE_TO_HOST);
    assert!(ret == 0, "cudaMemcpy D2H failed with error code {ret}");
}

// --- Send+Sync wrappers for ort types (ONNX Runtime is thread-safe in C++) ----

struct SyncAllocator(Allocator);
unsafe impl Send for SyncAllocator {}
unsafe impl Sync for SyncAllocator {}

fn create_cuda_allocator(session: &Session) -> SyncAllocator {
    let mem_info = MemoryInfo::new(
        AllocationDevice::CUDA, 0, AllocatorType::Device, MemoryType::Default,
    ).expect("failed to create CUDA MemoryInfo");
    SyncAllocator(
        Allocator::new(session, mem_info)
            .expect("failed to create CUDA Allocator"),
    )
}

// --- OnnxSession --------------------------------------------------------------

/// Wrapper around `ort::Session` providing interior mutability.
/// `session.run_binding()` requires `&mut self` in ort 2.x, but ONNX Runtime
/// is internally thread-safe for inference.
pub struct OnnxSession(Mutex<Session>);

impl OnnxSession {
    pub fn new(session: Session) -> Self {
        Self(Mutex::new(session))
    }

    pub fn lock(&self) -> std::sync::MutexGuard<'_, Session> {
        self.0.lock().unwrap()
    }
}

// --- OnnxBackend --------------------------------------------------------------

pub struct OnnxBackend {
    models: ModelStore<OnnxSession>,
    /// Lazily initialized on the first model load. Requires a Session reference
    /// to create, so it can't be constructed when starting with zero models.
    cuda_allocator: OnceLock<SyncAllocator>,
    game_size: i64,
    verbose: bool,
    trt_cache_path: Option<String>,
}

impl OnnxBackend {
    pub fn new(models: Vec<VersionedModel<OnnxSession>>, game_size: i64, verbose: bool, max_models: usize, trt_cache_path: Option<String>) -> Self {
        let cuda_allocator = OnceLock::new();
        // Eagerly initialize if we have a model at startup
        if let Some(first) = models.first() {
            let session = first.model.lock();
            let _ = cuda_allocator.set(create_cuda_allocator(&session));
        }
        if let Some(path) = &trt_cache_path {
            eprintln!("[onnx] TensorRT engine cache enabled at: {path}");
        }
        Self {
            models: ModelStore::new(models, max_models),
            cuda_allocator,
            game_size,
            verbose,
            trt_cache_path,
        }
    }

    fn allocator(&self) -> &Allocator {
        loop {
            if let Some(a) = self.cuda_allocator.get() {
                return &a.0;
            }
            eprintln!("CUDA allocator not ready, waiting for first model load...");
            std::thread::sleep(std::time::Duration::from_secs(1));
        }
    }

    fn execution_providers(&self) -> Vec<ort::execution_providers::ExecutionProviderDispatch> {
        let trt = match &self.trt_cache_path {
            Some(path) => TensorRTExecutionProvider::default()
                .with_engine_cache(true)
                .with_engine_cache_path(path)
                .build(),
            None => TensorRTExecutionProvider::default().build(),
        };
        vec![trt, CUDAExecutionProvider::default().build()]
    }

    fn build_session(&self, bytes: &[u8]) -> anyhow::Result<OnnxSession> {
        eprintln!("[onnx] building session from {} bytes", bytes.len());
        let session = Session::builder()
            .map_err(|e| anyhow::anyhow!("session builder: {e}"))?
            .with_log_level(ort::logging::LogLevel::Verbose)
            .map_err(|e| anyhow::anyhow!("log level: {e}"))?
            .with_execution_providers(self.execution_providers())
            .map_err(|e| anyhow::anyhow!("execution providers: {e}"))?
            .commit_from_memory(bytes)
            .map_err(|e| anyhow::anyhow!("commit_from_memory: {e}"))?;
        Ok(OnnxSession::new(session))
    }

    pub fn load_session_from_file(path: &str, trt_cache_path: Option<&str>) -> anyhow::Result<OnnxSession> {
        let trt = match trt_cache_path {
            Some(cache) => TensorRTExecutionProvider::default()
                .with_engine_cache(true)
                .with_engine_cache_path(cache)
                .build(),
            None => TensorRTExecutionProvider::default().build(),
        };
        let session = Session::builder()
            .map_err(|e| anyhow::anyhow!("session builder: {e}"))?
            .with_execution_providers(vec![trt, CUDAExecutionProvider::default().build()])
            .map_err(|e| anyhow::anyhow!("execution providers: {e}"))?
            .commit_from_file(path)
            .map_err(|e| anyhow::anyhow!("commit_from_file: {e}"))?;
        Ok(OnnxSession::new(session))
    }
}

impl Backend for OnnxBackend {
    type Model = OnnxSession;
    type Encoded = DynValue;
    type Inferred = (DynTensor, DynTensor);

    fn encode(&self, batch: Vec<StateBytes>) -> DynValue {
        encoder::encode(batch, self.game_size, self.allocator())
    }


    fn inference(&self, model_id: u32, input: DynValue) -> (DynTensor, DynTensor) {
        let batch_size = match input.dtype() {
            ort::value::ValueType::Tensor { shape, .. } => shape[0] as usize,
            _ => panic!("expected tensor input"),
        };
        if self.verbose {
            println!("Inference model_id={model_id} batch_size={batch_size}");
        }
        loop {
            let guard = self.models.load(model_id as usize);
            if let Some(vm) = guard.as_ref().as_ref() {
                let mut session = vm.model.lock();
                return inference::nn_inference(&mut session, self.allocator(), &input, self.game_size, batch_size);
            }
            drop(guard);
            eprintln!("model_id={model_id} not loaded yet, waiting...");
            std::thread::sleep(std::time::Duration::from_secs(1));
        }
    }

    fn decode(&self, output: (DynTensor, DynTensor)) -> Vec<(Vec<u8>, f32)> {
        decoder::decode(output, self.game_size)
    }

    fn respond(&self, pi_bytes: Vec<u8>, value: f32, move_bytes: Vec<u8>) -> (Vec<u8>, f32) {
        crate::nn::backends::respond::respond(&pi_bytes, value, &move_bytes, self.game_size as usize)
    }

    fn compile_model(&self, model: OnnxSession) -> anyhow::Result<OnnxSession> {
        Ok(model)
    }

    fn model_from_bytes(&self, bytes: &[u8]) -> anyhow::Result<OnnxSession> {
        let session = self.build_session(bytes)?;
        // Initialize the CUDA allocator on first model load
        if self.cuda_allocator.get().is_none() {
            let alloc = create_cuda_allocator(&session.lock());
            let info = alloc.0.memory_info();
            eprintln!("[onnx] CUDA allocator device: {:?} (id={})", info.allocation_device(), info.device_id());
            let _ = self.cuda_allocator.set(alloc);
        }
        Ok(session)
    }

    fn model_store(&self) -> &ModelStore<OnnxSession> {
        &self.models
    }
}
