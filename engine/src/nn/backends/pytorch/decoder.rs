use pyo3::prelude::*;

use super::PyTensor;

pub fn py_decode(output: PyTensor, game_size: i64) -> Vec<(Vec<u8>, f32)> {
    Python::attach(|py| {
        let output = output.bind(py);

        // output is a tuple (pi_logits, values)
        let pi = output.get_item(0).expect("failed to get pi from output");
        let vs = output.get_item(1).expect("failed to get values from output");

        // Flatten [B, S, S, S, S] -> [B, S^4], move to CPU, convert to contiguous numpy
        let flat = pi.call_method1("flatten", (1,)).unwrap();
        let cpu = flat.call_method0("detach").unwrap()
            .call_method0("cpu").unwrap();
        let np_pi = cpu.call_method0("numpy").unwrap();

        let vs_cpu = vs.call_method0("detach").unwrap()
            .call_method0("cpu").unwrap();
        let np_vs = vs_cpu.call_method0("numpy").unwrap();

        // Extract raw bytes via numpy's buffer protocol — avoids Python list iteration.
        let pi_bytes: Vec<u8> = np_pi.call_method0("tobytes").unwrap()
            .extract::<Vec<u8>>().unwrap();
        let vs_bytes: Vec<u8> = np_vs.call_method0("tobytes").unwrap()
            .extract::<Vec<u8>>().unwrap();

        let s4 = (game_size * game_size * game_size * game_size) as usize;
        let pi_floats: &[f32] = bytemuck::cast_slice(&pi_bytes);
        let vs_floats: &[f32] = bytemuck::cast_slice(&vs_bytes);

        pi_floats.chunks_exact(s4).zip(vs_floats.iter()).map(|(pi_row, &v)| {
            let pi_bytes: Vec<u8> = bytemuck::cast_slice(pi_row).to_vec();
            (pi_bytes, v)
        }).collect()
    })
}
