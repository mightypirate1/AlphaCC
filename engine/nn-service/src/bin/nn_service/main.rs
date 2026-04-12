mod benchmark;

#[cfg(feature = "jemalloc")]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

use std::collections::HashMap;

use clap::{Args, Parser, Subcommand};

use alpha_cc_nn::Game;
use alpha_cc_nn_service::backends::{Backend, VersionedModel};
use alpha_cc_nn_service::backends::onnx::OnnxBackend;
use alpha_cc_nn_service::server::config::{
    BatcherConfig, PipelineChannelConfig, PipelineConfig, ServerConfig,
};
use alpha_cc_nn_service::server::PredictServer;

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

/// Shared arguments for all serve modes.
#[derive(Args)]
struct ServeArgs {
    /// Path(s) to ONNX model files for initial preload.
    #[arg(long)]
    nn_path: Vec<String>,
    /// Game identifier (e.g. "cc:9", "cc:5").
    #[arg(long, default_value = "cc:9")]
    game: String,
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
}

#[derive(Subcommand)]
enum Command {
    /// Run the prediction server with Redis-based model reloading.
    Server {
        #[command(flatten)]
        common: ServeArgs,
        /// Model reload poll frequency in seconds.
        #[arg(long, default_value = "5")]
        reload_freq: u64,
        /// Redis host for the model reload source.
        #[arg(long, default_value = "localhost")]
        redis_host: String,
    },
    /// Run the prediction server with static weight files (no Redis).
    ServeStatic {
        #[command(flatten)]
        common: ServeArgs,
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

fn build_server_config(args: &ServeArgs) -> ServerConfig {
    let game_config = Game::parse(&args.game).config();
    let pad_item_len = game_config.pad_item_len();
    let pipeline_cfg = PipelineConfig { intake_buffer: args.intake_buffer, outtake_buffer: args.outtake_buffer };

    let mut pipelines = vec![PipelineChannelConfig {
        batcher: BatcherConfig {
            max_batch_size: args.batch_size,
            min_wait: std::time::Duration::from_millis(args.min_wait),
            max_wait: std::time::Duration::from_millis(args.max_wait),
            channel_buffer: args.channel_buffer,
            half_life: args.half_life,
            pad_to_max: args.fixed_batch_size,
            pad_item_len,
        },
        pipeline: pipeline_cfg.clone(),
        model_ids: vec![0],
        weight_batch_size: if args.fixed_batch_size { Some(args.batch_size) } else { None },
    }];

    if let Some(sec_bs) = args.secondary_batch_size {
        for model_id in 1..args.max_models as u32 {
            pipelines.push(PipelineChannelConfig {
                batcher: BatcherConfig {
                    max_batch_size: sec_bs,
                    min_wait: std::time::Duration::from_millis(args.secondary_min_wait),
                    max_wait: std::time::Duration::from_millis(args.secondary_max_wait),
                    channel_buffer: args.channel_buffer,
                    half_life: args.half_life,
                    pad_to_max: args.fixed_batch_size,
                    pad_item_len,
                },
                pipeline: pipeline_cfg.clone(),
                model_ids: vec![model_id],
                weight_batch_size: if args.fixed_batch_size { Some(sec_bs) } else { None },
            });
        }
    } else {
        pipelines[0].model_ids = (0..args.max_models as u32).collect();
    }

    ServerConfig { port: args.port, game_config, pipelines }
}

/// Load initial models from file paths into the backend's model store.
fn load_models<B: Backend>(backend: &B, paths: &[String], static_mode: bool) {
    for (i, path) in paths.iter().enumerate() {
        log::info!("Loading model {i}: {path}");
        let model = backend.load_model_from_file(path)
            .unwrap_or_else(|e| panic!("failed to load model {path}: {e}"));
        let version = if static_mode { i } else { 0 };
        backend.model_store().set(i, VersionedModel { model, version });
    }
}

async fn run_server<B: Backend>(
    config: ServerConfig,
    backend: B,
    reload_freq: u64,
    redis_host: String,
) -> anyhow::Result<()> {
    let addr = format!("[::]:{}", config.port);
    let poll_interval = std::time::Duration::from_secs(reload_freq);

    let mut batch_size_map: HashMap<usize, Option<usize>> = HashMap::new();
    for pipeline_cfg in &config.pipelines {
        for &model_id in &pipeline_cfg.model_ids {
            batch_size_map.insert(model_id as usize, pipeline_cfg.weight_batch_size);
        }
    }

    let (trt_cache_path, use_trt) = backend.trt_config();
    let server = PredictServer::new(config, backend, false);

    let source = alpha_cc_nn_service::db::TrainingDBRs::from_host(&redis_host)
        .expect("failed to connect to Redis for model reloading");
    let _reloader = alpha_cc_nn_service::reloads::spawn_reloader(
        server.backend(), source, poll_interval, Some("/tmp/healthy".to_string()),
        trt_cache_path, use_trt, batch_size_map,
    );
    log::info!("Model reloader started (poll every {reload_freq}s, redis={redis_host})");

    server.serve(&addr).await
        .map_err(|e| anyhow::anyhow!("error: {e}"))
}

async fn run_static<B: Backend>(
    config: ServerConfig,
    backend: B,
) -> anyhow::Result<()> {
    let addr = format!("[::]:{}", config.port);
    let server = PredictServer::new(config, backend, true);

    let _ = std::fs::write("/tmp/healthy", "ok");
    log::info!("Serving static model(s) (no reloader)");

    server.serve(&addr).await
        .map_err(|e| anyhow::anyhow!("error: {e}"))
}

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format(|buf, record| {
            use std::io::Write;
            let level = record.level();
            let style = buf.default_level_style(level);
            writeln!(buf, "{} {style}[{level}]{style:#} {}", buf.timestamp_seconds(), record.args())
        })
        .init();
    let cli = Cli::parse();

    match cli.command {
        Command::Server { common, reload_freq, redis_host } => {
            let config = build_server_config(&common);
            let rt = tokio::runtime::Runtime::new()?;
            if common.cpu {
                use alpha_cc_nn_service::backends::cpu::CpuBackend;
                let backend = CpuBackend::new(vec![], config.game_config.clone(), common.verbose, common.max_models);
                load_models(&backend, &common.nn_path, false);
                log::info!("Loaded {} model(s) (CPU)", common.nn_path.len());
                rt.block_on(run_server(config, backend, reload_freq, redis_host))
            } else {
                let backend = OnnxBackend::new(vec![], config.game_config.clone(), common.verbose, common.max_models, common.trt_cache_path, common.trt);
                load_models(&backend, &common.nn_path, false);
                log::info!("Loaded {} model(s) (GPU)", common.nn_path.len());
                rt.block_on(run_server(config, backend, reload_freq, redis_host))
            }
        }
        Command::ServeStatic { mut common } => {
            common.max_models = common.max_models.max(common.nn_path.len().max(1));
            let config = build_server_config(&common);
            let rt = tokio::runtime::Runtime::new()?;
            if common.cpu {
                use alpha_cc_nn_service::backends::cpu::CpuBackend;
                let backend = CpuBackend::new(vec![], config.game_config.clone(), common.verbose, common.max_models);
                load_models(&backend, &common.nn_path, true);
                log::info!("Loaded {} model(s) (CPU, static)", common.nn_path.len());
                rt.block_on(run_static(config, backend))
            } else {
                let backend = OnnxBackend::new(vec![], config.game_config.clone(), common.verbose, common.max_models, common.trt_cache_path, common.trt);
                load_models(&backend, &common.nn_path, true);
                log::info!("Loaded {} model(s) (GPU, static)", common.nn_path.len());
                rt.block_on(run_static(config, backend))
            }
        }
        Command::Bench { nn_path, game_size, warmup, iters } => {
            benchmark::run_benchmarks(&nn_path, game_size, warmup, iters)
        }
    }
}
