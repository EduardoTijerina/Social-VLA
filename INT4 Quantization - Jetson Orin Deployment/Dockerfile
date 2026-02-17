# Dockerfile for INT4 quantization of Nemotron-3-Nano-30B
# Tested on: NVIDIA DGX Spark (GB10 Blackwell, 128GB unified memory)
#
# Build:
#   docker build -t nemotron-quantize .
#
# Run:
#   docker run -it \
#     --gpus all \
#     --shm-size=64g \
#     --memory=115g \
#     --memory-swap=130g \
#     --ulimit memlock=-1 \
#     --ulimit stack=67108864 \
#     -v /path/to/nemotron_merged_fp16:/workspace/nemotron_merged_fp16 \
#     -v /path/to/output:/workspace/output \
#     nemotron-quantize

FROM nvcr.io/nvidia/tensorrt-llm/release:latest

# Install quantization dependencies
RUN pip install --no-cache-dir \
    psutil \
    safetensors \
    && pip install --no-cache-dir "numpy<2"

# Install mamba-ssm for NemotronH model loading (if needed for validation)
RUN pip install --no-cache-dir mamba-ssm causal-conv1d --no-build-isolation || true

WORKDIR /workspace

# Copy quantization script
COPY quantize_int4.py /workspace/quantize_int4.py

# Default command
CMD ["bash"]
