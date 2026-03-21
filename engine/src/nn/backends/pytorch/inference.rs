use pyo3::prelude::*;

use super::{PyModel, PyTensor};

/// Run inference on a batch tensor using the Python model.
/// Returns the raw output tuple as a `Py<PyAny>`.
pub fn py_inference(model: &PyModel, input: PyTensor) -> PyTensor {
    Python::attach(|py| {
        let torch = py.import("torch").expect("failed to import torch");
        let no_grad = torch.getattr("no_grad").unwrap();

        let ctx = no_grad.call0().unwrap();
        ctx.call_method0("__enter__").unwrap();

        let output = model.bind(py).call1((input.bind(py),))
            .expect("model forward pass failed");

        ctx.call_method1("__exit__", (py.None(), py.None(), py.None())).unwrap();

        output.unbind()
    })
}
