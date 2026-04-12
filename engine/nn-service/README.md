# nn-service

gRPC inference server for ONNX models with batched GPU inference pipeline.

## Building

```bash
cd engine
cargo build --release -p alpha-cc-nn-service --bin nn-service
```

## Running locally (outside Docker)

### Library dependencies

The binary dynamically loads `libonnxruntime.so` at runtime. Install it via pip:

```bash
pip install onnxruntime-gpu
```

Set the library path to point to the onnxruntime `.so` (note the version suffix):

```bash
export ORT_DYLIB_PATH=$(find $(python3 -c "import onnxruntime; print(onnxruntime.__file__.rsplit('/', 1)[0])") -name "libonnxruntime.so.*" | head -1)
```

CUDA EP requires cuDNN and CUDA runtime libs on `LD_LIBRARY_PATH`. These are typically installed as pip dependencies of `onnxruntime-gpu` (via `nvidia-cudnn-cu12` etc). Important: use the libs from the **same venv** as onnxruntime-gpu to avoid version mismatches.

```bash
VENV_SITE=$(../.venv/bin/python3 -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH=$(find "$VENV_SITE/nvidia" -name "lib" -type d | tr '\n' ':')${VENV_SITE}/onnxruntime/capi:$LD_LIBRARY_PATH
```

### Serving static weight files (no Redis)

```bash
./target/release/nn-service serve-static \
    --nn-path model_a.onnx \
    --nn-path model_b.onnx \
    --game "cc:9" \
    --batch-size 32
```

Each `--nn-path` is assigned to channel 0, 1, 2... in order. No Redis required.

Omit `--trt` to use CUDA EP only (instant load, no TRT compilation).

For eval with few concurrent requests, lower `--max-wait` (e.g. `--max-wait 10` for 10ms)
to avoid the batcher waiting for a full batch that never arrives.

### Serving with Redis model reloading (training mode)

```bash
./target/release/nn-service server \
    --redis-host localhost \
    --reload-freq 3 \
    --game-size 7 \
    --batch-size 192 \
    --fixed-batch-size \
    --trt \
    --trt-cache-path /tmp/trt-cache
```

### Exporting weights to ONNX

Use the `export-onnx` command from the Python package:

```bash
python -m alpha_cc.entrypoints.eval_weights export-onnx \
    model.pt model.onnx \
    --size 7 \
    --batch-size 32  # optional: fixed batch dim
```
