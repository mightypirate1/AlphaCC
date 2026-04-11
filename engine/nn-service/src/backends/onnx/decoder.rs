use ort::value::DynTensor;

use super::backend::cuda_memcpy_d2h;

/// Decode GPU-resident output tensors to (pi_bytes, wdl_bytes) pairs.
/// This is where D2H transfer happens — run in the decoder thread
/// so the inference thread can immediately process the next batch.
pub fn decode((pis, wdls): (DynTensor, DynTensor), game_size: i64) -> Vec<crate::backends::DecodedPrediction> {
    let s4 = (game_size * game_size * game_size * game_size) as usize;

    // Get batch size from WDL tensor shape: (batch, 3)
    let batch_size = match wdls.dtype() {
        ort::value::ValueType::Tensor { shape, .. } => shape[0] as usize,
        _ => panic!("expected tensor"),
    };

    // D2H: copy GPU data to CPU vecs
    let mut pi_flat = vec![0.0f32; batch_size * s4];
    let mut wdl_flat = vec![0.0f32; batch_size * 3];
    unsafe {
        cuda_memcpy_d2h(&mut pi_flat, pis.data_ptr());
        cuda_memcpy_d2h(&mut wdl_flat, wdls.data_ptr());
    }

    let mut decoded = Vec::with_capacity(batch_size);
    for i in 0..batch_size {
        let pi_row = &pi_flat[i * s4..(i + 1) * s4];
        let wdl_row = &wdl_flat[i * 3..(i + 1) * 3];
        let pi_bytes: Vec<u8> = bytemuck::cast_slice(pi_row).to_vec();
        let wdl_bytes: Vec<u8> = bytemuck::cast_slice(wdl_row).to_vec();
        decoded.push((pi_bytes, wdl_bytes));
    }
    decoded
}
