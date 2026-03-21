use tch::{Device, Tensor};

pub fn decode((pis, vs): (Tensor, Tensor), game_size: i64) -> Vec<(Vec<u8>, f32)> {
    let batch_size = pis.size()[0] as usize;
    let pis = pis.to(Device::Cpu);
    let vs = vs.to(Device::Cpu);
    let pi_flat: Vec<f32> = Vec::<f32>::try_from(&pis.ravel()).unwrap();
    let s4 = (game_size * game_size * game_size * game_size) as usize;

    let mut decoded = Vec::with_capacity(batch_size);
    for i in 0..batch_size {
        let v = vs.double_value(&[i as i64]) as f32;
        let pi_row = &pi_flat[i * s4..(i + 1) * s4];
        let pi_bytes: Vec<u8> = bytemuck::cast_slice(pi_row).to_vec();
        decoded.push((pi_bytes, v));
    }
    decoded
}
