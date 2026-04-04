use ort::memory::Allocator;
use ort::value::{DynTensor, DynValue, TensorElementType};

use alpha_cc_nn::io;
use super::backend::copy_to_gpu_tensor;

/// Encode a batch of state bytes into a GPU tensor.
///
/// 1. Collect flat f32s on CPU
/// 2. Allocate empty GPU tensor via CUDA allocator
/// 3. cudaMemcpy H2D into the GPU tensor
pub fn encode(batch: Vec<Vec<u8>>, game_size: i64, allocator: &Allocator) -> DynValue {
    let n = batch.len();
    let s = game_size as usize;
    let flat: Vec<f32> = batch.iter()
        .flat_map(|item| io::state_bytes_as_f32s(item))
        .copied()
        .collect();

    let mut gpu_tensor = DynTensor::new(allocator, TensorElementType::Float32, [n, 2, s, s])
        .expect("failed to allocate GPU tensor");
    unsafe { copy_to_gpu_tensor(&mut gpu_tensor, &flat) };
    gpu_tensor.into_dyn()
}
