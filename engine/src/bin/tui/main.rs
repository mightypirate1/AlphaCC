use std::time::Duration;

use clap::Parser;

use alpha_cc_engine::tui::app::{App, AppConfig, PlayerConfig};

/// Interactive AlphaCC game client.
///
/// Each player is configured with --p1 / --p2:
///   human       — click to play
///   ai:<channel> — AI using nn-service channel (e.g. ai:0)
///
/// Examples:
///   tui                              # human vs AI on channel 0
///   tui --p1 ai:0 --p2 ai:1         # AI vs AI
///   tui --p1 ai:3 --p2 human        # play as P2
///   tui --p1 human --p2 human        # two humans
#[derive(Parser)]
#[command(name = "tui", about = "Interactive AlphaCC game client")]
struct Cli {
    /// Player 1 config: "human" or "ai:<channel>"
    #[arg(long, default_value = "human")]
    p1: String,

    /// Player 2 config: "human" or "ai:<channel>"
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

    /// MCTS c_puct_init
    #[arg(long, default_value = "2.0")]
    c_puct_init: f32,

    /// MCTS c_puct_base
    #[arg(long, default_value = "10000.0")]
    c_puct_base: f32,

    /// Enable MCTS tree pruning
    #[arg(long)]
    pruning_tree: bool,
}

fn parse_player(s: &str) -> anyhow::Result<PlayerConfig> {
    let s = s.trim();
    if s.eq_ignore_ascii_case("human") {
        return Ok(PlayerConfig::Human);
    }
    if let Some(ch) = s.strip_prefix("ai:") {
        let channel: u32 = ch.parse()
            .map_err(|_| anyhow::anyhow!("Invalid channel in '{s}'. Expected ai:<number>"))?;
        return Ok(PlayerConfig::Ai { channel });
    }
    // bare number → ai:<n>
    if let Ok(channel) = s.parse::<u32>() {
        return Ok(PlayerConfig::Ai { channel });
    }
    anyhow::bail!("Invalid player config: '{s}'. Use 'human' or 'ai:<channel>'")
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
        pruning_tree: cli.pruning_tree,
    };

    let mut app = App::new(config);
    app.run()
}
