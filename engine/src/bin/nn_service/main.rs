mod benchmark;

use std::collections::HashMap;

use clap::{Parser, Subcommand};

use alpha_cc_engine::nn::backends::VersionedModel;
use alpha_cc_engine::nn::backends::onnx::{OnnxBackend, OnnxSession};
use alpha_cc_engine::nn::server::config::{
    BatcherConfig, PipelineChannelConfig, PipelineConfig, ServerConfig,
};
use alpha_cc_engine::nn::server::PredictServer;

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run the prediction server.
    Server {
        /// Path(s) to ONNX model files for initial preload.
        #[arg(long)]
        nn_path: Vec<String>,
        /// Game board size.
        #[arg(long, default_value = "9")]
        game_size: usize,
        /// gRPC listen port.
        #[arg(long, default_value = "50055")]
        port: u16,
        /// Primary pipeline: max batch size.
        #[arg(long, default_value = "128")]
        batch_size: usize,
        /// Primary pipeline: min wait time (ms).
        #[arg(long, default_value = "1")]
        min_wait: u64,
        /// Primary pipeline: max wait time (ms).
        #[arg(long, default_value = "10000")]
        max_wait: u64,
        /// Secondary pipeline: max batch size. Enables a separate pipeline for
        /// tournament channels (model_ids 1+). If omitted, all model_ids share
        /// the primary pipeline.
        #[arg(long)]
        secondary_batch_size: Option<usize>,
        /// Secondary pipeline: min wait time (ms).
        #[arg(long, default_value = "1")]
        secondary_min_wait: u64,
        /// Secondary pipeline: max wait time (ms).
        #[arg(long, default_value = "10000")]
        secondary_max_wait: u64,
        /// Half-life in seconds for adaptive wait.
        /// Time for the error between current and ideal wait to halve.
        #[arg(long, default_value = "2.0")]
        half_life: f64,
        /// Channel buffer size for incoming gRPC requests.
        #[arg(long, default_value = "1024")]
        channel_buffer: usize,
        /// Pipeline buffer before inference.
        #[arg(long, default_value = "2")]
        intake_buffer: usize,
        /// Pipeline buffer after inference.
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
        /// Path to TensorRT engine cache directory.
        #[arg(long)]
        trt_cache_path: Option<String>,
        /// Use TensorRT execution provider.
        #[arg(long)]
        trt: bool,
        /// Zero-pad all batches to their pipeline's batch-size.
        #[arg(long)]
        fixed_batch_size: bool,
        /// Run inference on CPU instead of GPU.
        #[arg(long)]
        cpu: bool,
    },
    /// Benchmark the ONNX inference pipeline.
    Bench {
        #[arg(long)]
        nn_path: String,
        #[arg(long, default_value = "9")]
        game_size: usize,
        #[arg(long, default_value = "10")]
        warmup: usize,
        #[arg(long, default_value = "100")]
        iters: usize,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Server {
            nn_path, game_size, port, batch_size, min_wait, max_wait,
            secondary_batch_size, secondary_min_wait,
            secondary_max_wait, half_life, channel_buffer, intake_buffer,
            outtake_buffer, verbose, reload_freq, redis_host, max_models,
            trt_cache_path, trt, fixed_batch_size, cpu,
        } => {
            let pad_item_len = 2 * game_size * game_size * std::mem::size_of::<f32>();
            let pipeline_cfg = PipelineConfig { intake_buffer, outtake_buffer };

            let mut pipelines = vec![PipelineChannelConfig {
                batcher: BatcherConfig {
                    max_batch_size: batch_size,
                    min_wait: std::time::Duration::from_millis(min_wait),
                    max_wait: std::time::Duration::from_millis(max_wait),
                    channel_buffer,
                    half_life,
                    pad_to_max: fixed_batch_size,
                    pad_item_len,
                },
                pipeline: pipeline_cfg.clone(),
                model_ids: vec![0],
                weight_batch_size: if fixed_batch_size { Some(batch_size) } else { None },
            }];

            if let Some(sec_bs) = secondary_batch_size {
                // One pipeline per secondary model_id — each has different weights.
                for model_id in 1..max_models as u32 {
                    pipelines.push(PipelineChannelConfig {
                        batcher: BatcherConfig {
                            max_batch_size: sec_bs,
                            min_wait: std::time::Duration::from_millis(secondary_min_wait),
                            max_wait: std::time::Duration::from_millis(secondary_max_wait),
                            channel_buffer,
                            half_life,
                            pad_to_max: fixed_batch_size,
                            pad_item_len,
                        },
                        pipeline: pipeline_cfg.clone(),
                        model_ids: vec![model_id],
                        weight_batch_size: if fixed_batch_size { Some(sec_bs) } else { None },
                    });
                }
            } else {
                // No secondary — primary pipeline handles all model_ids.
                pipelines[0].model_ids = (0..max_models as u32).collect();
            }

            let config = ServerConfig { port, game_size, pipelines };

            let rt = tokio::runtime::Runtime::new()?;
            if cpu {
                rt.block_on(run_server_cpu(config, nn_path, verbose, reload_freq, redis_host, max_models))
            } else {
                rt.block_on(run_server_gpu(config, nn_path, verbose, reload_freq, redis_host, max_models, trt_cache_path, trt))
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
) -> anyhow::Result<()> {
    let addr = format!("[::]:{}", config.port);
    let game_size = config.game_size as i64;
    let n_models = nn_paths.len();
    let poll_interval = std::time::Duration::from_secs(reload_freq);

    // Build batch_size_map from pipeline configs for the reloader.
    let mut batch_size_map: HashMap<usize, Option<usize>> = HashMap::new();
    for pipeline_cfg in &config.pipelines {
        for &model_id in &pipeline_cfg.model_ids {
            batch_size_map.insert(model_id as usize, pipeline_cfg.weight_batch_size);
        }
    }

    let models: Vec<VersionedModel<OnnxSession>> = nn_paths.iter().map(|path| {
        let model = OnnxBackend::load_session_from_file(path, trt_cache_path.as_deref())
            .unwrap_or_else(|e| panic!("failed to load ONNX model {path}: {e}"));
        VersionedModel { model, version: 0 }
    }).collect();
    println!("Loaded {n_models} onnx model(s) (GPU)");
    let reloader_trt_cache_path = trt_cache_path.clone();
    let backend = OnnxBackend::new(models, game_size, verbose, max_models, trt_cache_path, use_trt);
    let server = PredictServer::new(config, backend);

    let source = alpha_cc_engine::db::TrainingDBRs::from_host(&redis_host)
        .expect("failed to connect to Redis for model reloading");
    let _reloader = alpha_cc_engine::nn::reloads::spawn_reloader(
        server.backend(), source, poll_interval, Some("/tmp/healthy".to_string()), reloader_trt_cache_path, use_trt, batch_size_map,
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
    let server = PredictServer::new(config, backend);

    let source = alpha_cc_engine::db::TrainingDBRs::from_host(&redis_host)
        .expect("failed to connect to Redis for model reloading");
    let _reloader = alpha_cc_engine::nn::reloads::spawn_reloader(
        server.backend(), source, poll_interval, Some("/tmp/healthy".to_string()), None, false, HashMap::new(),
    );
    println!("Model reloader started (poll every {reload_freq}s, redis={redis_host})");

    server.serve(&addr).await
        .map_err(|e| anyhow::anyhow!("error: {e}"))
}
