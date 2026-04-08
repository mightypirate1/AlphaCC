use std::ffi::CString;

use ort::AsPointer;
use ort::memory::Allocator;
use ort::session::Session;
use ort::value::{DynTensor, DynValue, TensorElementType};

/// Run inference, returning GPU-resident output tensors.
///
/// Input is already on GPU (from encoder). Outputs are pre-allocated on GPU
/// and bound via the raw C API so we retain ownership. The decoder thread
/// handles D2H.
pub fn nn_inference(
    session: &mut Session,
    allocator: &Allocator,
    input: &DynValue,
    game_size: i64,
    batch_size: usize,
) -> (DynTensor, DynTensor) {
    let s = game_size as usize;

    let mut binding = session.create_binding().expect("failed to create IoBinding");
    binding.bind_input("input", input).expect("failed to bind input");

    // Pre-allocate output tensors on GPU — shapes must match model outputs
    let policy_out = DynTensor::new(allocator, TensorElementType::Float32, [batch_size, s, s, s, s])
        .expect("failed to allocate policy output");
    let value_out = DynTensor::new(allocator, TensorElementType::Float32, [batch_size, 3])
        .expect("failed to allocate value output");

    // Bind outputs via raw C API so we keep ownership (the safe bind_output moves the value)
    unsafe {
        let policy_name = CString::new("policy").unwrap();
        let value_name = CString::new("value").unwrap();
        let bind_output = ort::api().BindOutput;
        let _ = bind_output(binding.ptr_mut(), policy_name.as_ptr(), policy_out.ptr());
        let _ = bind_output(binding.ptr_mut(), value_name.as_ptr(), value_out.ptr());
    }

    session.run_binding(&binding).expect("onnx run_binding failed");

    (policy_out, value_out)
}
