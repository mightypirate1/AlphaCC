use pyo3::prelude::*;

use crate::nn::io;
use super::PyTensor;

pub fn py_encode(batch: Vec<Vec<u8>>, game_size: i64) -> PyTensor {
    let n = batch.len() as i64;
    let flat: Vec<f32> = batch.iter()
        .flat_map(|item| io::state_bytes_as_f32s(item))
        .copied()
        .collect();

    Python::attach(|py| {
        let torch = py.import("torch").expect("failed to import torch");

        // torch.frombuffer: zero-copy view into the byte buffer, then .to() copies to GPU.
        let flat_bytes: &[u8] = bytemuck::cast_slice(&flat);
        let py_bytes = pyo3::types::PyBytes::new(py, flat_bytes);
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("dtype", torch.getattr("float32").unwrap()).unwrap();
        let tensor = torch.call_method("frombuffer", (py_bytes,), Some(&kwargs))
            .expect("torch.frombuffer failed");
        let tensor = tensor.call_method1("reshape", ((n, 2i64, game_size, game_size),))
            .expect("failed to reshape");
        let tensor = tensor.call_method1("to", ("cuda:0",))
            .expect("failed to move to device");

        // # Alternative: numpy.frombuffer → torch.from_numpy (triggers "not writable" warning)
        // let np = py.import("numpy").expect("failed to import numpy");
        // let kwargs = pyo3::types::PyDict::new(py);
        // kwargs.set_item("dtype", np.getattr("float32").unwrap()).unwrap();
        // let np_arr = np.call_method("frombuffer", (py_bytes,), Some(&kwargs))
        //     .expect("numpy.frombuffer failed");
        // let np_arr = np_arr.call_method1("reshape", ((n, 2i64, game_size, game_size),))
        //     .expect("failed to reshape");
        // let tensor = torch.call_method1("from_numpy", (&np_arr,))
        //     .expect("torch.from_numpy failed");
        // let tensor = tensor.call_method1("to", ("cuda:0",))
        //     .expect("failed to move to device");

        tensor.unbind()
    })
}
