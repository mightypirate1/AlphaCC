mod benchmark;

use clap::{Parser, Subcommand};

use alpha_cc_engine::nn::backends::VersionedModel;
use alpha_cc_engine::nn::backends::cpu::CpuBackend;
use alpha_cc_engine::nn::backends::onnx::{OnnxBackend, OnnxSession};
use alpha_cc_engine::nn::server::{config::ServerConfig, PredictServer, ServiceGate};

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run the prediction server.
    Server {
        /// Path(s) to ONNX model files for initial preload. If omitted, the server
        /// starts with empty model slots and waits for the reloader to populate
        /// them from Redis.
        #[arg(long)]
        nn_path: Vec<String>,
        /// Game board size.
        #[arg(long, default_value = "9")]
        game_size: usize,
        /// gRPC listen port.
        #[arg(long, default_value = "50055")]
        port: u16,
        /// Max batch size.
        #[arg(long, default_value = "128")]
        batch_size: usize,
        /// Max wait time (ms) for a full batch before flushing.
        #[arg(long, default_value = "5")]
        max_wait: u64,
        /// Channel buffer size for incoming gRPC requests.
        #[arg(long, default_value = "1024")]
        channel_buffer: usize,
        /// Pipeline buffer before inference (low = backpressure → larger batches).
        #[arg(long, default_value = "1")]
        intake_buffer: usize,
        /// Pipeline buffer after inference (high = GPU never stalls on downstream).
        #[arg(long, default_value = "4")]
        outtake_buffer: usize,
        /// Print batch size on each inference call.
        #[arg(long)]
        verbose: bool,
        /// Model reload poll frequency in seconds.
        #[arg(long, default_value = "5")]
        reload_freq: u64,
        /// Redis host for the model reload source.
        #[arg(long, default_value = "localhost")]
        redis_host: String,
        /// Maximum number of model slots.
        #[arg(long, default_value = "8")]
        max_models: usize,
        /// Path to shared TensorRT engine cache directory. Enables TRT engine
        /// caching and Redis-coordinated staggered reloads across replicas.
        #[arg(long)]
        trt_cache_path: Option<String>,
        /// Use TensorRT execution provider for optimized inference. Requires
        /// TRT compilation on model load (slow reload, fast steady-state).
        /// Without this flag, uses CUDA EP only (instant reload, ~30% slower).
        #[arg(long)]
        trt: bool,
        /// Zero-pad all batches to --batch-size so TRT only compiles
        /// kernels for one shape. Must match --onnx-compiled-batch-size
        /// in the trainer if the ONNX model uses a static batch dim.
        #[arg(long)]
        fixed_batch_size: bool,
        /// Run inference on CPU instead of GPU (no CUDA/TensorRT needed).
        #[arg(long)]
        cpu: bool,
    },
    /// Benchmark the ONNX inference pipeline.
    Bench {
        /// Path to ONNX model file.
        #[arg(long)]
        nn_path: String,
        /// Game board size.
        #[arg(long, default_value = "9")]
        game_size: usize,
        /// Number of warmup iterations.
        #[arg(long, default_value = "10")]
        warmup: usize,
        /// Number of timed iterations.
        #[arg(long, default_value = "100")]
        iters: usize,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Server { nn_path, game_size, port, batch_size, max_wait, channel_buffer, intake_buffer, outtake_buffer, verbose, reload_freq, redis_host, max_models, trt_cache_path, trt, fixed_batch_size, cpu } => {
            let config = ServerConfig {
                port,
                game_size,
                batcher: alpha_cc_engine::nn::server::config::BatcherConfig {
                    max_batch_size: batch_size,
                    max_wait: std::time::Duration::from_millis(max_wait),
                    channel_buffer,
                },
                pipeline: alpha_cc_engine::nn::server::config::PipelineConfig {
                    intake_buffer,
                    outtake_buffer,
                },
            };
            let rt = tokio::runtime::Runtime::new()?;
            if cpu {
                rt.block_on(run_server_cpu(config, nn_path, verbose, reload_freq, redis_host, max_models))
            } else {
                let fixed = if fixed_batch_size { Some(batch_size) } else { None };
                rt.block_on(run_server_gpu(config, nn_path, verbose, reload_freq, redis_host, max_models, trt_cache_path, trt, fixed))
            }
        }
        Command::Bench { nn_path, game_size, warmup, iters } => {
            benchmark::run_benchmarks(&nn_path, game_size, warmup, iters)
        }
    }
}

async fn run_server_gpu(
    config: ServerConfig,
    nn_paths: Vec<String>,
    verbose: bool,
    reload_freq: u64,
    redis_host: String,
    max_models: usize,
    trt_cache_path: Option<String>,
    use_trt: bool,
    fixed_batch_size: Option<usize>,
) -> anyhow::Result<()> {
    let addr = format!("[::]:{}", config.port);
    let game_size = config.game_size as i64;
    let n_models = nn_paths.len();
    let poll_interval = std::time::Duration::from_secs(reload_freq);

    let models: Vec<VersionedModel<OnnxSession>> = nn_paths.iter().map(|path| {
        let model = OnnxBackend::load_session_from_file(path, trt_cache_path.as_deref())
            .unwrap_or_else(|e| panic!("failed to load ONNX model {path}: {e}"));
        VersionedModel { model, version: 0 }
    }).collect();
    println!("Loaded {n_models} onnx model(s) (GPU)");
    let reloader_trt_cache_path = trt_cache_path.clone();
    let backend = OnnxBackend::new(models, game_size, verbose, max_models, trt_cache_path, use_trt, fixed_batch_size);
    let gate = ServiceGate::new();
    let server = PredictServer::new(config, backend, gate.clone());

    let source = alpha_cc_engine::db::TrainingDBRs::from_host(&redis_host)
        .expect("failed to connect to Redis for model reloading");
    let _reloader = alpha_cc_engine::nn::reloads::spawn_reloader(
        server.backend(), source, poll_interval, Some("/tmp/healthy".to_string()), reloader_trt_cache_path, gate, use_trt,
    );
    println!("Model reloader started (poll every {reload_freq}s, redis={redis_host})");

    server.serve(&addr).await
        .map_err(|e| anyhow::anyhow!("error: {e}"))
}

async fn run_server_cpu(
    config: ServerConfig,
    nn_paths: Vec<String>,
    verbose: bool,
    reload_freq: u64,
    redis_host: String,
    max_models: usize,
) -> anyhow::Result<()> {
    use alpha_cc_engine::nn::backends::cpu::CpuBackend;

    let addr = format!("[::]:{}", config.port);
    let game_size = config.game_size as i64;
    let n_models = nn_paths.len();
    let poll_interval = std::time::Duration::from_secs(reload_freq);

    let models = nn_paths.iter().map(|path| {
        let model = CpuBackend::load_session_from_file(path)
            .unwrap_or_else(|e| panic!("failed to load ONNX model {path}: {e}"));
        VersionedModel { model, version: 0 }
    }).collect();
    println!("Loaded {n_models} onnx model(s) (CPU)");
    let backend = CpuBackend::new(models, game_size, verbose, max_models);
    let gate = ServiceGate::new();
    let server = PredictServer::new(config, backend, gate.clone());

    let source = alpha_cc_engine::db::TrainingDBRs::from_host(&redis_host)
        .expect("failed to connect to Redis for model reloading");
    let _reloader = alpha_cc_engine::nn::reloads::spawn_reloader(
        server.backend(), source, poll_interval, Some("/tmp/healthy".to_string()), None, gate, false,
    );
    println!("Model reloader started (poll every {reload_freq}s, redis={redis_host})");

    server.serve(&addr).await
        .map_err(|e| anyhow::anyhow!("error: {e}"))
}
