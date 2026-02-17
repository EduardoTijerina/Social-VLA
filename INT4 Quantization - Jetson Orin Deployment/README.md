# INT4 Quantization for Fine-Tuned Nemotron-3-Nano-30B

INT4 weight-only quantization pipeline for deploying a fine-tuned NVIDIA Nemotron-3-Nano-30B (NemotronH) model on edge hardware (NVIDIA Jetson Orin).

## Overview

This pipeline takes a fine-tuned Nemotron-3-Nano-30B model (59 GB in bf16) and quantizes it to INT4 weight-only format (~16.5 GB), making it deployable on memory-constrained devices like the Jetson Orin.

### Pipeline Summary

```
SFT + DPO Fine-Tuning (bf16)  ──►  Merged Model (59 GB)  ──►  INT4 Quantized (~16.5 GB)
         ▲                                                              │
    Training Container                                          Deployment on
    (DGX Spark)                                                 Jetson Orin
```

### Why Manual INT4 Quantization?

Standard quantization tools failed on this specific hardware + model combination:

| Tool | Issue |
|------|-------|
| **ModelOpt AWQ** | CUDA `IndexKernel` crashes on Blackwell sm120 — INT4 AWQ not in HW support matrix |
| **TRT-LLM export** | `NemotronH` not in supported model architecture list; `AssertionError: The model is not supported` |
| **AutoAWQ** | Deprecated; `nemotron_h` not in `AWQ_CAUSAL_LM_MODEL_MAP` |
| **llm-compressor** | NumPy 2.x incompatibility with container's PyTorch build |
| **CPU-based AWQ** | NemotronH `modeling_nemotron_h.py` has hardcoded `torch.cuda.stream()` calls; cannot run on CPU |

The manual approach is architecture-agnostic, runs entirely on CPU (no CUDA kernel dependencies), and produces standard GPTQ-compatible safetensors output.

## Environment

### Quantization Host

| Component | Specification |
|-----------|--------------|
| System | NVIDIA DGX Spark |
| GPU | NVIDIA GB10 (Blackwell, sm120) |
| Memory | 128 GB unified (CPU + GPU shared) |
| CUDA | 13.0 |
| Driver | 580.95.05 |

### Container Configuration

```bash
docker run -it \
  --gpus all \
  --shm-size=64g \
  --memory=115g \
  --memory-swap=130g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v /path/to/merged_model:/workspace/nemotron_merged_fp16 \
  -v /path/to/output:/workspace/output \
  --name nemotron-quantize \
  nvcr.io/nvidia/tensorrt-llm/release:latest \
  bash
```

**Critical container settings:**

| Parameter | Value | Reason |
|-----------|-------|--------|
| `--shm-size` | `64g` | Default 64 MB causes PyTorch to crash during model loading with mmap'd safetensors |
| `--memory` | `115g` | Leave headroom for host OS and display server |
| `--ulimit memlock` | `-1` (unlimited) | Required for GPU unified memory pinning on GB10 |

### Dependencies

```
torch>=2.8.0
safetensors>=0.7.0
psutil>=5.9.0
```

Install inside the container:

```bash
pip install psutil safetensors
```

### Deployment Target

| Component | Specification |
|-----------|--------------|
| Device | NVIDIA Jetson Orin |
| Architecture | Ampere (sm87) |
| Supported formats | INT4 weight-only, INT8 |

## Usage

### Basic Usage

```bash
python quantize_int4.py \
  --input_dir /workspace/nemotron_merged_fp16 \
  --output_dir /workspace/nemotron_int4_manual
```

### From ModelOpt Checkpoint

If you ran TRT-LLM's `quantize.py` and it saved a `.pth` but failed at export:

```bash
python quantize_int4.py \
  --input_dir /workspace/nemotron_int4_wo \
  --output_dir /workspace/nemotron_int4_manual
```

### Custom Group Size

```bash
python quantize_int4.py \
  --input_dir /workspace/nemotron_merged_fp16 \
  --output_dir /workspace/nemotron_int4_g64 \
  --group_size 64
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--input_dir` | (required) | Path to merged model directory |
| `--output_dir` | (required) | Output directory for quantized model |
| `--group_size` | 128 | Columns per quantization group |
| `--shard_size_gb` | 4.0 | Max safetensors shard size |

## Quantization Method

### Algorithm

INT4 weight-only quantization with per-group symmetric scaling:

1. **Grouping**: Each weight matrix `(rows, cols)` is divided into groups of `group_size` along the column dimension.
2. **Scale computation**: Per-group scale factor `s = max(|w_group|) / 7`.
3. **Quantization**: `q = clamp(round(w / s), -8, 7)` mapping to signed 4-bit range.
4. **Packing**: 8 INT4 values are packed into each INT32 container using bit shifts.

### What Gets Quantized

- All 2D weight matrices with `cols >= group_size` (6,029 layers in Nemotron-3-Nano-30B)
- Biases, layer norms, embeddings, and small weights are kept in original precision

### Output Format

Each quantized layer produces two tensors:

| Tensor | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| `*.qweight` | `(rows, cols // 8)` | int32 | 8 INT4 values packed per int32 |
| `*.scales` | `(rows, cols // group_size)` | float16 | Per-group scaling factors |

### Dequantization

To recover approximate weights at inference time:

```python
def dequantize_int4(qweight, scales, group_size=128):
    """Unpack INT4 weights and apply scales."""
    rows, packed_cols = qweight.shape
    cols = packed_cols * 8

    # Unpack 8 INT4 values from each INT32
    unpacked = torch.zeros(rows, cols, dtype=torch.int8)
    for i in range(8):
        unpacked[:, i::8] = ((qweight >> (i * 4)) & 0xF).to(torch.int8)

    # Convert unsigned 4-bit to signed (values > 7 become negative)
    unpacked = torch.where(unpacked > 7, unpacked - 16, unpacked)

    # Apply per-group scales
    unpacked = unpacked.float().reshape(-1, group_size)
    scales_flat = scales.float().reshape(-1, 1)
    dequantized = (unpacked * scales_flat).reshape(rows, cols)

    return dequantized
```

## Memory Profile

Observed memory usage during quantization on DGX Spark (128 GB unified memory):

| Stage | RAM Used | Available |
|-------|----------|-----------|
| Baseline | 4.8 GB | 122.6 GB |
| State dict loaded | 68.1 GB | 59.3 GB |
| Mid-quantization (3000/6243) | 78.1 GB | 49.4 GB |
| Quantization complete | 26.9 GB | 100.6 GB |
| Safetensors saved | 11.9 GB | 115.5 GB |

Peak memory usage is ~88 GB during quantization (original + quantized tensors coexist briefly). The script includes a safety abort that terminates if available RAM drops below 10 GB.

## Output Structure

```
nemotron_int4_manual/
├── config.json                    # Model architecture config
├── configuration_nemotron_h.py    # Custom model class
├── modeling_nemotron_h.py         # Custom model implementation
├── tokenizer.json                 # Tokenizer
├── tokenizer_config.json          # Tokenizer config
├── special_tokens_map.json        # Special tokens
├── generation_config.json         # Generation defaults
├── chat_template.jinja            # Chat template
├── quantize_config.json           # Quantization metadata
├── model.safetensors.index.json   # Shard index
├── model-00001.safetensors        # ~3.8 GB
├── model-00002.safetensors        # ~3.8 GB
├── model-00003.safetensors        # ~3.8 GB
├── model-00004.safetensors        # ~3.8 GB
└── model-00005.safetensors        # ~0.5 GB
```

## Troubleshooting

### Container crashes during model loading

Increase `--shm-size`. The default 64 MB is insufficient for large model loading:

```bash
docker run --shm-size=64g ...
```

Verify inside the container:

```bash
df -h /dev/shm  # Should show 64G, not 64M
```

### Out of memory during quantization

1. Stop other containers: `docker stop <container_id>`
2. Close desktop applications (Firefox, VS Code) on the host
3. Check memory before running: `free -h`
4. The script will auto-abort if available RAM drops below 10 GB

### ModelOpt CUDA crashes on Blackwell

This is expected. The GB10 (sm120) does not support INT4 AWQ CUDA kernels. Use this manual quantization script instead.

## License

MIT
