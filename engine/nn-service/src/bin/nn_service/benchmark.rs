use std::time::Instant;

use alpha_cc_nn::GameConfig;
use alpha_cc_nn_service::backends::{Backend, VersionedModel};
use alpha_cc_nn_service::backends::onnx::OnnxBackend;

type CudaDeviceSyncFn = unsafe extern "C" fn() -> i32;

fn load_cuda_device_sync() -> CudaDeviceSyncFn {
    unsafe {
        let handle = libc::dlopen(c"libcudart.so.12".as_ptr().cast(), libc::RTLD_NOW | libc::RTLD_GLOBAL);
        assert!(!handle.is_null(), "failed to dlopen libcudart.so.12");
        let sym = libc::dlsym(handle, c"cudaDeviceSynchronize".as_ptr().cast());
        assert!(!sym.is_null(), "failed to dlsym cudaDeviceSynchronize");
        std::mem::transmute(sym)
    }
}

static CUDA_SYNC: std::sync::OnceLock<CudaDeviceSyncFn> = std::sync::OnceLock::new();

fn cuda_sync() {
    let f = *CUDA_SYNC.get_or_init(load_cuda_device_sync);
    unsafe { f(); }
}

struct BenchResult {
    batch_size: usize,
    latencies_us: Vec<u64>,
}

impl BenchResult {
    fn median_us(&self) -> u64 {
        let mut sorted = self.latencies_us.clone();
        sorted.sort();
        sorted[sorted.len() / 2]
    }
    fn mean_us(&self) -> u64 {
        self.latencies_us.iter().sum::<u64>() / self.latencies_us.len() as u64
    }
    fn min_us(&self) -> u64 {
        *self.latencies_us.iter().min().unwrap()
    }
    fn max_us(&self) -> u64 {
        *self.latencies_us.iter().max().unwrap()
    }
    fn p99_us(&self) -> u64 {
        let mut sorted = self.latencies_us.clone();
        sorted.sort();
        sorted[(sorted.len() as f64 * 0.99) as usize]
    }
    fn throughput(&self) -> f64 {
        self.batch_size as f64 / (self.median_us() as f64 / 1_000_000.0)
    }
}

fn print_result(r: &BenchResult) {
    println!(
        "  batch={:>4}  median={:>7}μs  mean={:>7}μs  min={:>7}μs  max={:>7}μs  p99={:>7}μs  throughput={:>10.0} samples/s",
        r.batch_size, r.median_us(), r.mean_us(), r.min_us(), r.max_us(), r.p99_us(), r.throughput()
    );
}

/// Create a synthetic batch of `StateBytes` for the given board size.
fn make_batch(batch_size: usize, game_size: usize) -> Vec<Vec<u8>> {
    let floats_per_state = 2 * game_size * game_size;
    let bytes_per_state = floats_per_state * std::mem::size_of::<f32>();
    (0..batch_size)
        .map(|_| vec![0u8; bytes_per_state])
        .collect()
}

/// Run the full encode → inference → decode pipeline and return results.
fn bench_pipeline<B: Backend>(
    backend: &B,
    game_size: usize,
    batch_sizes: &[usize],
    warmup: usize,
    iters: usize,
) -> Vec<BenchResult> {
    let mut results = Vec::new();
    for &bs in batch_sizes {
        let batch = make_batch(bs, game_size);

        // Warmup
        for _ in 0..warmup {
            let encoded = backend.encode(batch.clone());
            let inferred = backend.inference(0, encoded);
            let _ = backend.decode(inferred);
        }
        cuda_sync();

        // Timed runs
        let mut latencies = Vec::with_capacity(iters);
        for _ in 0..iters {
            cuda_sync();
            let start = Instant::now();
            let encoded = backend.encode(batch.clone());
            let inferred = backend.inference(0, encoded);
            let _ = backend.decode(inferred);
            cuda_sync();
            latencies.push(start.elapsed().as_micros() as u64);
        }

        let r = BenchResult { batch_size: bs, latencies_us: latencies };
        print_result(&r);
        results.push(r);
    }
    results
}

pub fn run_benchmarks(
    nn_path: &str,
    game_size: usize,
    warmup: usize,
    iters: usize,
) -> anyhow::Result<()> {
    let batch_sizes = [1, 8, 32, 64, 128, 256, 512, 1024];
    let game_config = GameConfig::from_game::<alpha_cc_core::cc::CCBoard>(game_size);

    println!("Model: {nn_path}");
    println!("Warmup: {warmup}, Iterations: {iters}");
    println!();

    println!("=== ONNX (ort + TensorRT/CUDA EP) ===");
    let model = OnnxBackend::load_session_from_file(nn_path, None, true)
        .unwrap_or_else(|e| panic!("failed to load ONNX model: {e}"));
    let backend = OnnxBackend::new(
        vec![VersionedModel { model, version: 0 }], game_config, false, 1, None, true,
    );
    let results = bench_pipeline(&backend, game_size, &batch_sizes, warmup, iters);

    println!();
    println!("=== Throughput summary (samples/s) ===");
    println!("{:>6}  {:>16}", "batch", "throughput");
    println!("{}  {}", "-".repeat(6), "-".repeat(16));
    for r in &results {
        println!("{:>6}  {:>16.0}", r.batch_size, r.throughput());
    }
    println!();

    Ok(())
}
