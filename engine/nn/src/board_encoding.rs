use alpha_cc_core::{Board, Move};

/// How a game encodes itself for neural network consumption.
///
/// Provides the mapping between game state and tensor representations:
/// - State encoding: board → input tensor channels
/// - Move encoding: moves → wire format bytes
/// - Policy indexing: move bytes → flat index into the policy output tensor
pub trait BoardEncoding: Board {
    /// Number of input channels in the state tensor (e.g. 2 for CC: one plane per player).
    const STATE_CHANNELS: usize;

    /// Bytes per move in the wire format (e.g. 4 for `[fx, fy, tx, ty]`).
    const MOVE_BYTES: usize;

    /// Fill a pre-allocated `[STATE_CHANNELS * s * s]` buffer with the encoded board state.
    fn encode_state(&self, buf: &mut [f32]);

    /// Encode a single move into its wire-format bytes.
    fn encode_move(mv: &Move<Self::Coord>, buf: &mut [u8]);

    /// Total flat size of the policy output tensor (e.g. `s^4` for from-to games).
    fn policy_size(board_size: usize) -> usize;

    /// Map encoded move bytes to a flat index into the policy output tensor.
    fn move_to_policy_index(move_bytes: &[u8], board_size: usize) -> usize;
}
