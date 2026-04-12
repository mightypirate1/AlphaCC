use alpha_cc_nn::BoardEncoding;
use crate::proto::PredictResponse;

/// Encode a board into the wire format for the prediction service.
///
/// Returns `(state_tensor_bytes, moves_bytes)`:
/// - `state_tensor_bytes`: flattened `[channels, s, s]` as little-endian f32s.
/// - `moves_bytes`: each legal move as `MOVE_BYTES` consecutive bytes.
pub fn encode_request<B: BoardEncoding>(board: &B) -> (Vec<u8>, Vec<u8>) {
    let (s, _) = board.get_sizes();
    let s = s as usize;

    let mut tensor_data = vec![0.0f32; B::STATE_CHANNELS * s * s];
    board.encode_state(&mut tensor_data);
    let state_bytes: Vec<u8> = bytemuck::cast_slice(&tensor_data).to_vec();

    let moves = board.legal_moves();
    let mut moves_bytes = vec![0u8; moves.len() * B::MOVE_BYTES];
    for (i, mv) in moves.iter().enumerate() {
        B::encode_move(mv, &mut moves_bytes[i * B::MOVE_BYTES..]);
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
