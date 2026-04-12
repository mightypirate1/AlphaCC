use alpha_cc_core::cc::CCBoard;
use crate::game_config::GameConfig;

/// Identifies a game variant. Parseable from CLI strings like `"cc:9"` or `"cc:5"`.
///
/// This is the canonical way to specify which game is being played — shared
/// across nn-service (`--game`), trainer (`--game`), and workers (`--game`).
#[derive(Clone, Debug)]
pub enum Game {
    CC { size: usize },
    // Chess, // future
}

impl Game {
    /// Parse a game string like `"cc:9"`, `"cc:5"`, `"cc"` (defaults to 9).
    pub fn parse(s: &str) -> Self {
        let parts: Vec<&str> = s.splitn(2, ':').collect();
        match parts[0] {
            "cc" => {
                let size: usize = parts.get(1)
                    .unwrap_or(&"9")
                    .parse()
                    .unwrap_or_else(|e| panic!("invalid board size in game string '{s}': {e}"));
                Game::CC { size }
            }
            other => panic!("unknown game '{other}'. Supported: cc"),
        }
    }

    /// Build the encoding config for this game variant.
    pub fn config(&self) -> GameConfig {
        match self {
            Game::CC { size } => GameConfig::from_game::<CCBoard>(*size),
        }
    }
}

impl std::fmt::Display for Game {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Game::CC { size } => write!(f, "cc:{size}"),
        }
    }
}
