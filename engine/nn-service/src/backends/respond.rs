use crate::io;

/// Extract pi logits for legal moves from the full policy tensor.
/// WDL logits pass through unchanged.
pub fn respond(pi_bytes: &[u8], wdl_bytes: Vec<u8>, move_bytes: &[u8], game_size: usize) -> super::DecodedPrediction {
    let s = game_size;
    let pi_row: &[f32] = bytemuck::cast_slice(pi_bytes);
    let coords = io::moves_bytes_to_coords(move_bytes);
    let logits: Vec<f32> = coords.iter().map(|&(fx, fy, tx, ty)| {
        let idx = (fx as usize) * s * s * s
                + (fy as usize) * s * s
                + (tx as usize) * s
                + (ty as usize);
        pi_row[idx]
    }).collect();
    let logit_bytes: Vec<u8> = bytemuck::cast_slice(&logits).to_vec();
    (logit_bytes, wdl_bytes)
}
