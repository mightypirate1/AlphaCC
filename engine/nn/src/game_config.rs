use crate::board_encoding::BoardEncoding;

/// Encoding metadata extracted from a `BoardEncoding` impl at startup.
///
/// Used by nn-service pipeline stages that need shape information without
/// holding a board instance. Construct via `GameConfig::from_game::<G>(board_size)`.
#[derive(Clone)]
pub struct GameConfig {
    pub name: &'static str,
    pub state_channels: usize,
    pub board_size: usize,
    pub move_bytes: usize,
    pub policy_size: usize,
    pub policy_shape: Vec<usize>,
    pub move_to_policy_index: fn(&[u8], usize) -> usize,
}

impl GameConfig {
    pub fn from_game<G: BoardEncoding>(board_size: usize) -> Self {
        let policy_size = G::policy_size(board_size);
        let policy_shape = G::policy_shape(board_size);
        Self {
            name: std::any::type_name::<G>(),
            state_channels: G::STATE_CHANNELS,
            board_size,
            move_bytes: G::MOVE_BYTES,
            policy_size,
            policy_shape,
            move_to_policy_index: G::move_to_policy_index,
        }
    }

    /// Byte size of a single zero-padded state tensor item.
    pub fn pad_item_len(&self) -> usize {
        self.state_channels * self.board_size * self.board_size * std::mem::size_of::<f32>()
    }
}

impl std::fmt::Debug for GameConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GameConfig")
            .field("name", &self.name)
            .field("board_size", &self.board_size)
            .field("state_channels", &self.state_channels)
            .field("policy_shape", &self.policy_shape)
            .finish()
    }
}
