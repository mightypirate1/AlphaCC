use pyo3_stub_gen::Result;
use std::fs;
use std::path::Path;

fn main() -> Result<()> {
    let stub = alpha_cc_engine::stub_info()?;
    stub.generate()?;

    // Move the generated .pyi into the package directory next to Cargo.toml
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let pkg_dir = manifest_dir.join("alpha_cc_engine");
    let generated = manifest_dir.join("alpha_cc_engine.pyi");
    if generated.exists() {
        fs::rename(&generated, pkg_dir.join("__init__.pyi"))?;
    }

    Ok(())
}
