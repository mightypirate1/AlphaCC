use alpha_cc_nn::GameConfig;

/// Extract pi logits for legal moves from the full policy tensor.
/// WDL logits pass through unchanged.
pub fn respond(pi_bytes: &[u8], wdl_bytes: Vec<u8>, move_bytes: &[u8], config: &GameConfig) -> super::DecodedPrediction {
    let pi_row: &[f32] = bytemuck::cast_slice(pi_bytes);
    let move_byte_size = config.move_bytes;
    let logits: Vec<f32> = move_bytes.chunks_exact(move_byte_size).map(|mb| {
        let idx = (config.move_to_policy_index)(mb, config.board_size);
        pi_row[idx]
    }).collect();
    let logit_bytes: Vec<u8> = bytemuck::cast_slice(&logits).to_vec();
    (logit_bytes, wdl_bytes)
}
