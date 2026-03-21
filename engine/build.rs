fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_prost_build::compile_protos("proto/predict.proto")?;

    // Bake the libtorch lib directory into the binary's rpath so it can
    // find .so files at runtime without LD_LIBRARY_PATH.
    if let Ok(libtorch) = std::env::var("LIBTORCH") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}/lib", libtorch);
    }

    Ok(())
}
