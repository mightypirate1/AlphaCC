use ort::value::DynTensor;

use super::backend::cuda_memcpy_d2h;

/// Decode GPU-resident output tensors to (pi_bytes, value) pairs.
/// This is where D2H transfer happens — run in the decoder thread
/// so the inference thread can immediately process the next batch.
pub fn decode((pis, vs): (DynTensor, DynTensor), game_size: i64) -> Vec<(Vec<u8>, f32)> {
    let s4 = (game_size * game_size * game_size * game_size) as usize;

    // Get shapes from metadata (no D2H needed)
    let batch_size = match vs.dtype() {
        ort::value::ValueType::Tensor { shape, .. } => shape[0] as usize,
        _ => panic!("expected tensor"),
    };

    // D2H: copy GPU data to CPU vecs
    let mut pi_flat = vec![0.0f32; batch_size * s4];
    let mut v_flat = vec![0.0f32; batch_size];
    unsafe {
        cuda_memcpy_d2h(&mut pi_flat, pis.data_ptr());
        cuda_memcpy_d2h(&mut v_flat, vs.data_ptr());
    }

    let mut decoded = Vec::with_capacity(batch_size);
    for i in 0..batch_size {
        let v = v_flat[i];
        let pi_row = &pi_flat[i * s4..(i + 1) * s4];
        let pi_bytes: Vec<u8> = bytemuck::cast_slice(pi_row).to_vec();
        decoded.push((pi_bytes, v));
    }
    decoded
}
