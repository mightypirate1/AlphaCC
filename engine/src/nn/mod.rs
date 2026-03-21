pub mod backends;
pub mod client;
pub mod io;
pub mod reloads;
pub mod server;

/// Generated protobuf types and gRPC service definitions.
pub mod proto {
    tonic::include_proto!("predict");
}

/// Force-load the CUDA backend for libtorch.
///
/// Libtorch discovers CUDA lazily via `dlopen`, which sometimes fails
/// to find `libtorch_cuda.so` even when `LD_LIBRARY_PATH` is correct.
/// Call this once at startup (before any `tch::Cuda` or `Device` calls)
/// to eagerly load the library.
pub fn load_cuda() {
    unsafe {
        let name = c"libtorch_cuda.so";
        let handle = libc::dlopen(name.as_ptr(), libc::RTLD_LAZY | libc::RTLD_GLOBAL);
        if handle.is_null() {
            let err = std::ffi::CStr::from_ptr(libc::dlerror());
            eprintln!("Warning: failed to preload libtorch_cuda.so: {}", err.to_string_lossy());
        }
    }
}
