mod app;
mod agent;
mod cc;
mod game;
mod input;
mod modal;
mod renderer;
mod sidebar;
mod status_bar;
mod styling;
mod theme;
mod visual;

use std::time::Duration;

use clap::Parser;

use alpha_cc_core::cc::CCBoard;
use crate::agent::MctsVariant;
use crate::app::{App, AppConfig, PlayerConfig};
use crate::cc::HexRenderer;

/// Interactive AlphaCC game client.
///
/// Each player is configured with --p1 / --p2:
///   human                — click to play
///   ai:<channel>         — AI on channel, default MCTS (puct-free)
///   ai:<channel>:puct    — AI using PUCT + free rollouts
///   ai:<channel>:gumbel  — AI using Gumbel top-k + sequential halving
///
/// Examples:
///   tui                                  # human vs AI on channel 0
///   tui --p1 ai:0:puct --p2 ai:1:gumbel # PUCT vs Gumbel
///   tui --p1 ai:3 --p2 human            # play as P2
#[derive(Parser)]
#[command(name = "tui", about = "Interactive AlphaCC game client")]
struct Cli {
    /// Player 1 config: "human", "ai:<channel>", "ai:<channel>:puct", or "ai:<channel>:gumbel"
    #[arg(long, default_value = "human")]
    p1: String,

    /// Player 2 config: "human", "ai:<channel>", "ai:<channel>:puct", or "ai:<channel>:gumbel"
    #[arg(long, default_value = "ai:0")]
    p2: String,

    /// Board size (3, 5, 7, or 9)
    #[arg(long, default_value = "7")]
    board_size: u8,

    /// nn-service gRPC address
    #[arg(long, default_value = "http://localhost:50055")]
    nn_addr: String,

    /// Seconds the AI gets per move
    #[arg(long, default_value = "5.0")]
    think_time: f64,

    /// MCTS threads per AI
    #[arg(long, default_value = "4")]
    n_threads: usize,

    /// Max MCTS rollout depth
    #[arg(long, default_value = "100")]
    rollout_depth: usize,

    /// MCTS gamma
    #[arg(long, default_value = "1.0")]
    gamma: f32,

    // ── puct-free params ──

    /// MCTS c_puct_init (PUCT exploration constant)
    #[arg(long, default_value = "2.0")]
    c_puct_init: f32,

    /// MCTS c_puct_base (PUCT exploration base)
    #[arg(long, default_value = "10000.0")]
    c_puct_base: f32,

    /// MCTS temperature (applied to visit-count distribution)
    #[arg(long, default_value = "1.0")]
    temperature: f32,

    /// Dirichlet noise weight (0 = disabled)
    #[arg(long, default_value = "0.0")]
    dirichlet_weight: f32,

    /// Dirichlet noise alpha
    #[arg(long, default_value = "0.15")]
    dirichlet_alpha: f32,

    // ── improved-halving (gumbel) params ──

    /// Gumbel MCTS c_visit (σ transform)
    #[arg(long, default_value = "50.0")]
    c_visit: f32,

    /// Gumbel MCTS c_scale (σ transform)
    #[arg(long, default_value = "1.0")]
    c_scale: f32,

    /// Enable MCTS tree pruning
    #[arg(long)]
    pruning_tree: bool,
}

fn parse_mcts_variant(s: &str) -> anyhow::Result<MctsVariant> {
    match s.trim().to_lowercase().as_str() {
        "puct" | "puct-free" => Ok(MctsVariant::PuctFree),
        "gumbel" | "halving" | "improved-halving" => Ok(MctsVariant::ImprovedHalving),
        other => anyhow::bail!("Unknown MCTS variant '{other}'. Use 'puct' or 'gumbel'"),
    }
}

fn parse_player(s: &str) -> anyhow::Result<PlayerConfig> {
    let s = s.trim();
    if s.eq_ignore_ascii_case("human") {
        return Ok(PlayerConfig::Human);
    }
    if let Some(rest) = s.strip_prefix("ai:") {
        let mut parts = rest.splitn(2, ':');
        let channel: u32 = parts.next().unwrap_or("").parse()
            .map_err(|_| anyhow::anyhow!("Invalid channel in '{s}'. Expected ai:<number>[:<variant>]"))?;
        let mcts = match parts.next() {
            Some(v) => parse_mcts_variant(v)?,
            None => MctsVariant::PuctFree,
        };
        return Ok(PlayerConfig::Ai { channel, mcts });
    }
    if let Ok(channel) = s.parse::<u32>() {
        return Ok(PlayerConfig::Ai { channel, mcts: MctsVariant::PuctFree });
    }
    anyhow::bail!("Invalid player config: '{s}'. Use 'human' or 'ai:<channel>[:<variant>]'")
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let cli = Cli::parse();

    let config = AppConfig {
        p1: parse_player(&cli.p1)?,
        p2: parse_player(&cli.p2)?,
        board_size: cli.board_size,
        nn_addr: cli.nn_addr,
        think_time: Duration::from_secs_f64(cli.think_time),
        n_threads: cli.n_threads,
        rollout_depth: cli.rollout_depth,
        gamma: cli.gamma,
        c_puct_init: cli.c_puct_init,
        c_puct_base: cli.c_puct_base,
        temperature: cli.temperature,
        dirichlet_weight: cli.dirichlet_weight,
        dirichlet_alpha: cli.dirichlet_alpha,
        c_visit: cli.c_visit,
        c_scale: cli.c_scale,
        pruning_tree: cli.pruning_tree,
    };

    let board = CCBoard::create(cli.board_size as usize);
    let renderer = HexRenderer::new(cli.board_size);
    let mut app = App::new(config, board, renderer);
    app.run()
}
