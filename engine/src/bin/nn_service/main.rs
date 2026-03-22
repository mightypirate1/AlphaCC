mod benchmark;

use clap::{Parser, Subcommand, ValueEnum};

use alpha_cc_engine::nn::backends::{self, VersionedModel};
use alpha_cc_engine::nn::server::config;
use alpha_cc_engine::nn::server::{config::ServerConfig, PredictServer};

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Check CUDA availability.
    Test,
    /// Run the prediction server.
    Server {
        /// Path(s) to model files for initial preload. If omitted, the server
        /// starts with empty model slots and waits for the reloader to populate
        /// them from Redis.
        #[arg(long)]
        nn_path: Vec<String>,
        /// Backend to use for inference.
        #[arg(long, value_enum, default_value = "pytorch")]
        backend: BackendChoice,
        /// torch.compile mode (only used with --backend pytorch).
        #[arg(long)]
        compile_mode: Option<String>,
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
        /// Inter-stage pipeline channel buffer size.
        #[arg(long, default_value = "4")]
        pipeline_buffer_size: usize,
        /// Print batch size on each inference call.
        #[arg(long)]
        verbose: bool,
        /// Model reload poll frequency in seconds.
        #[arg(long, default_value = "5")]
        reload_freq: u64,
        /// Redis host for the model reload source.
        #[arg(long, default_value = "localhost")]
        redis_host: String,
        /// Maximum number of model slots (must be >= number of nn_paths and any
        /// channels the reloader might add).
        #[arg(long, default_value = "8")]
        max_models: usize,
    },
    /// Benchmark inference backends.
    Bench {
        /// Path to model file.
        #[arg(long)]
        nn_path: String,
        /// Backend to benchmark.
        #[arg(long, value_enum)]
        backend: BackendChoice,
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

#[derive(Clone, ValueEnum)]
enum BackendChoice {
    /// tch-rs CModule JIT backend.
    Tch,
    /// ONNX backend.
    Onnx,
    /// pyo3 DefaultNet + optional torch.compile backend.
    Pytorch,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    alpha_cc_engine::nn::load_cuda();
    // Initialize the Python interpreter on the main thread before any
    // worker/reloader threads call Python::attach(). This ensures CPython's
    // thread-state bookkeeping is set up correctly.
    pyo3::Python::attach(|_py| {});
    let cli = Cli::parse();

    match cli.command {
        Command::Test => {
            println!("CUDA is_available: {}", tch::Cuda::is_available());
            println!("cuDNN is_available: {}", tch::Cuda::cudnn_is_available());
            println!("CUDA device_count: {}", tch::Cuda::device_count());
            Ok(())
        }
        Command::Server { nn_path, backend, compile_mode, game_size, port, batch_size, max_wait, channel_buffer, pipeline_buffer_size, verbose, reload_freq, redis_host, max_models } => {
            let config = ServerConfig {
                port,
                game_size,
                device: config::detect_device(),
                batcher: config::BatcherConfig {
                    max_batch_size: batch_size,
                    max_wait: std::time::Duration::from_millis(max_wait),
                    channel_buffer,
                },
                pipeline: config::PipelineConfig {
                    buffer_size: pipeline_buffer_size,
                },
            };
            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(run_server(config, nn_path, backend, compile_mode, verbose, reload_freq, redis_host, max_models))
        }
        Command::Bench { nn_path, backend, game_size, warmup, iters } => {
            benchmark::run_benchmarks(&nn_path, backend, game_size, warmup, iters)
        }
    }
}

async fn run_server(
    config: ServerConfig,
    nn_paths: Vec<String>,
    backend_choice: BackendChoice,
    compile_mode: Option<String>,
    verbose: bool,
    reload_freq: u64,
    redis_host: String,
    max_models: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let addr = format!("[::]:{}", config.port);
    let game_size = config.game_size as i64;
    let n_models = nn_paths.len();
    let poll_interval = std::time::Duration::from_secs(reload_freq);

    match backend_choice {
        BackendChoice::Onnx => {
            use backends::onnx::{OnnxBackend, OnnxSession};
            let models: Vec<VersionedModel<OnnxSession>> = nn_paths.iter().map(|path| {
                let model = OnnxBackend::load_session_from_file(path)
                    .unwrap_or_else(|e| panic!("failed to load ONNX model {path}: {e}"));
                VersionedModel { model, version: 0 }
            }).collect();
            println!("Loaded {n_models} onnx model(s)");
            let backend = OnnxBackend::new(models, game_size, verbose, max_models);
            let server = PredictServer::new(config, backend);

            let source = alpha_cc_engine::db::TrainingDBRs::from_host(&redis_host)
                .expect("failed to connect to Redis for model reloading");
            let _reloader = alpha_cc_engine::nn::reloads::spawn_reloader(
                server.backend(), source, poll_interval,
            );
            println!("Model reloader started (poll every {reload_freq}s, redis={redis_host})");

            server.serve(&addr).await
        }
        BackendChoice::Tch => {
            use tch::CModule;
            let models: Vec<_> = nn_paths.iter().map(|path| {
                let model = CModule::load_on_device(path, config.device)
                    .unwrap_or_else(|e| panic!("failed to load model {path}: {e}"));
                VersionedModel { model, version: 0 }
            }).collect();
            println!("Loaded {n_models} tch model(s)");
            let backend = backends::torchrs::TchBackend::new(models, game_size, config.device, verbose, max_models);
            let server = PredictServer::new(config, backend);

            let source = alpha_cc_engine::db::TrainingDBRs::from_host(&redis_host)
                .expect("failed to connect to Redis for model reloading");
            let _reloader = alpha_cc_engine::nn::reloads::spawn_reloader(
                server.backend(), source, poll_interval,
            );
            println!("Model reloader started (poll every {reload_freq}s, redis={redis_host})");

            server.serve(&addr).await
        }
        BackendChoice::Pytorch => {
            use backends::pytorch::PyTorchBackend;
            let models: Vec<_> = nn_paths.iter().map(|path| {
                let model = PyTorchBackend::setup_model_from_path(path, game_size, compile_mode.as_deref());
                VersionedModel { model, version: 0 }
            }).collect();
            println!("Loaded {n_models} pytorch model(s)");
            let backend = PyTorchBackend::new(models, game_size, verbose, max_models);
            let server = PredictServer::new(config, backend);

            let source = alpha_cc_engine::nn::reloads::TrainingDBSource::new(&redis_host)
                .expect("failed to connect to Redis for model reloading");
            let _reloader = alpha_cc_engine::nn::reloads::spawn_reloader(
                server.backend(), source, poll_interval,
            );
            println!("Model reloader started (poll every {reload_freq}s, redis={redis_host})");

            server.serve(&addr).await
        }
    }
}
