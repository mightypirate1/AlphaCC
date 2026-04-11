use alpha_cc_core::Board;
use crate::proto::PredictResponse;

/// Encode a board into the wire format for the prediction service.
///
/// Returns `(state_tensor_bytes, moves_bytes)`:
/// - `state_tensor_bytes`: one-hot `[2, s, s]` as flattened little-endian f32s.
///   Channel 0 = current player (matrix == 1), channel 1 = opponent (matrix == 2).
///   Matches `state_tensor()` in `alpha_cc.state.state_tensors`.
/// - `moves_bytes`: each legal move as 4 consecutive bytes `[fx, fy, tx, ty]`.
///   Matches `action_indexer()` in `alpha_cc.engine.engine_utils`.
pub fn encode_request(board: &Board) -> (Vec<u8>, Vec<u8>) {
    let s = board.get_size() as usize;
    let matrix = board.get_matrix();

    // One-hot encode into [2, s, s] flattened f32s, then to bytes.
    let mut tensor_data = vec![0.0f32; 2 * s * s];
    #[allow(clippy::needless_range_loop)]
    for x in 0..s {
        for y in 0..s {
            let idx = x * s + y;
            match matrix[x][y] {
                1 => tensor_data[idx] = 1.0,
                2 => tensor_data[s * s + idx] = 1.0,
                _ => {}
            }
        }
    }
    let state_bytes: &[u8] = bytemuck::cast_slice(&tensor_data);
    let state_bytes = state_bytes.to_vec();

    // Encode legal moves: 4 bytes per move.
    let moves = board.get_moves();
    let mut moves_bytes = Vec::with_capacity(moves.len() * 4);
    for m in &moves {
        moves_bytes.push(m.from_coord.x);
        moves_bytes.push(m.from_coord.y);
        moves_bytes.push(m.to_coord.x);
        moves_bytes.push(m.to_coord.y);
    }

    (state_bytes, moves_bytes)
}


/// Decode a prediction response into `(pi_logits, wdl_logits)`.
///
/// - `pi_logits`: one f32 per legal move, same order as `Board::get_moves()`.
///   Apply softmax to get probabilities.
/// - `wdl_logits`: 3 f32s (win, draw, loss). Apply softmax to get probabilities.
pub fn decode_response(response: &PredictResponse) -> (Vec<f32>, [f32; 3]) {
    let pi_logits: &[f32] = bytemuck::cast_slice(&response.pi_logits);
    let wdl_logits: &[f32] = bytemuck::cast_slice(&response.wdl_logits);
    (pi_logits.to_vec(), [wdl_logits[0], wdl_logits[1], wdl_logits[2]])
}

/// Decode state tensor bytes back into f32s (zero-copy view).
///
/// Returns `2 * s * s` elements in `[2, s, s]` layout.
/// Use with `Tensor::from_slice(floats).reshape([2, s, s])`.
pub fn state_bytes_as_f32s(state_bytes: &[u8]) -> &[f32] {
    bytemuck::cast_slice(state_bytes)
}

/// Decode moves bytes back into coordinate tuples.
///
/// Returns `(from_x, from_y, to_x, to_y)` per move.
/// Index into the `[s^4]` policy tensor with `fx*s³ + fy*s² + tx*s + ty`.
pub fn moves_bytes_to_coords(moves_bytes: &[u8]) -> Vec<(u8, u8, u8, u8)> {
    moves_bytes
        .chunks_exact(4)
        .map(|c| (c[0], c[1], c[2], c[3]))
        .collect()
}
