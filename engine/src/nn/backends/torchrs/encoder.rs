use tch::{Device, Tensor};

use crate::nn::io;

pub fn encode(batch: Vec<Vec<u8>>, game_size: i64, device: Device) -> Tensor {
    let n = batch.len() as i64;
    let flat: Vec<f32> = batch.iter()
        .flat_map(|item| io::state_bytes_as_f32s(item))
        .copied()
        .collect();
    Tensor::from_slice(&flat)
        .reshape([n, 2, game_size, game_size])
        .to(device)
}
