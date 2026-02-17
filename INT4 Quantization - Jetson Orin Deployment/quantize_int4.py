"""
INT4 Weight-Only Quantization for NVIDIA Nemotron-3-Nano-30B (NemotronH)
=========================================================================

Performs INT4 weight-only quantization with group-wise scaling on a
fine-tuned Nemotron-3-Nano-30B model. Produces GPTQ-compatible safetensors
output suitable for deployment on NVIDIA Jetson Orin via vLLM or custom
inference runtimes.

Target pipeline:
    SFT + DPO fine-tuning (bf16) -> Merged model (59GB) -> INT4 quantized (~16GB)

Hardware tested:
    - Quantization host: NVIDIA DGX Spark (GB10 Blackwell, 128GB unified memory)
    - Deployment target: NVIDIA Jetson Orin

Why manual INT4 instead of ModelOpt/AutoAWQ:
    - ModelOpt AWQ: CUDA index kernels crash on Blackwell sm120 (unsupported HW)
    - TRT-LLM export: NemotronH architecture not in supported model matrix
    - AutoAWQ: Deprecated; no NemotronH support in model map
    - Manual INT4-WO: Architecture-agnostic, runs on CPU, no special CUDA kernels

Author: Eduardo (University of Colorado Bioengineering / SAR Research)
Date: February 2026
"""

import torch
import gc
import os
import sys
import json
import shutil
import psutil
import argparse
import time
from pathlib import Path
from safetensors.torch import save_file


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_GROUP_SIZE = 128
DEFAULT_SHARD_SIZE_GB = 4.0
RAM_ABORT_THRESHOLD_GB = 10.0


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def report_memory(stage: str, abort_below_gb: float = RAM_ABORT_THRESHOLD_GB):
    """Print current memory usage and abort if available RAM is critically low."""
    mem = psutil.virtual_memory()
    used_gb = mem.used / 1e9
    total_gb = mem.total / 1e9
    avail_gb = mem.available / 1e9
    gpu_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

    print(
        f"[{stage}] "
        f"RAM: {used_gb:.1f}/{total_gb:.1f} GB | "
        f"Available: {avail_gb:.1f} GB | "
        f"GPU: {gpu_gb:.1f} GB"
    )

    if avail_gb < abort_below_gb:
        print(f"ABORTING: Available RAM ({avail_gb:.1f} GB) below {abort_below_gb} GB safety threshold.")
        sys.exit(1)


def pack_int4(qweight: torch.Tensor) -> torch.Tensor:
    """
    Pack INT4 values into INT32 containers.

    Takes a (rows, cols) tensor of int8 values in [-8, 7] range,
    masks to 4-bit unsigned representation, and packs 8 values per int32.

    Args:
        qweight: Quantized weight tensor, shape (rows, cols). cols must be divisible by 8.

    Returns:
        Packed tensor of shape (rows, cols // 8) with dtype int32.
    """
    rows, cols = qweight.shape
    assert cols % 8 == 0, f"Columns ({cols}) must be divisible by 8 for INT4 packing."

    qweight = qweight.to(torch.int32) & 0xF  # mask to unsigned 4-bit

    packed = torch.zeros(rows, cols // 8, dtype=torch.int32)
    for i in range(8):
        packed |= qweight[:, i::8] << (i * 4)

    return packed


def quantize_weight_int4(
    weight: torch.Tensor,
    group_size: int = DEFAULT_GROUP_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a single weight matrix to INT4 with per-group symmetric scaling.

    The weight matrix is divided into groups of `group_size` along the column
    dimension. Each group is independently scaled to fit the INT4 range [-8, 7].

    Args:
        weight: Original weight tensor, shape (rows, cols), any float dtype.
        group_size: Number of columns per quantization group.

    Returns:
        packed: INT4-packed weights, shape (rows, cols // 8), dtype int32.
        scales: Per-group scaling factors, shape (rows, cols // group_size), dtype float16.
    """
    rows, cols = weight.shape

    # Pad columns to be divisible by group_size
    if cols % group_size != 0:
        padded_cols = ((cols + group_size - 1) // group_size) * group_size
        padded = torch.zeros(rows, padded_cols, dtype=weight.dtype)
        padded[:, :cols] = weight
        weight = padded
        cols = padded_cols

    # Also ensure cols is divisible by 8 (for INT4 packing)
    if cols % 8 != 0:
        padded_cols = ((cols + 7) // 8) * 8
        padded = torch.zeros(rows, padded_cols, dtype=weight.dtype)
        padded[:, :cols] = weight
        weight = padded
        cols = padded_cols

    weight_float = weight.float().reshape(-1, group_size)

    # Symmetric quantization: scale = max(|w|) / 7
    w_max = weight_float.abs().amax(dim=1, keepdim=True).clamp(min=1e-6)
    scale = w_max / 7.0

    # Quantize to INT4 range [-8, 7]
    qweight = torch.clamp(torch.round(weight_float / scale), -8, 7).to(torch.int8)

    # Reshape scale back to (rows, num_groups)
    num_groups = cols // group_size
    scale = scale.squeeze(1).reshape(rows, num_groups).to(torch.float16)

    # Reshape quantized weights and pack
    qweight = qweight.reshape(rows, cols)
    packed = pack_int4(qweight)

    return packed, scale


# ---------------------------------------------------------------------------
# Main quantization pipeline
# ---------------------------------------------------------------------------

def quantize_model(
    input_path: str,
    output_dir: str,
    group_size: int = DEFAULT_GROUP_SIZE,
    shard_size_gb: float = DEFAULT_SHARD_SIZE_GB,
):
    """
    Quantize a full model checkpoint from bf16/fp16 to INT4 weight-only.

    This function:
      1. Loads the state dict from a .pth or safetensors checkpoint.
      2. Quantizes all 2D weight matrices with cols >= group_size.
      3. Saves the result as sharded safetensors with GPTQ-compatible naming.
      4. Copies tokenizer and config files from the source directory.

    Args:
        input_path: Path to the merged model directory (containing safetensors or .pth).
        output_dir: Directory to write the quantized model.
        group_size: Quantization group size (default 128).
        shard_size_gb: Maximum shard size in GB for output safetensors.
    """
    start_time = time.time()
    report_memory("START")

    # ------------------------------------------------------------------
    # Phase 1: Load state dict
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("PHASE 1: Loading model state dict")
    print(f"{'='*70}")

    pth_file = os.path.join(input_path, "modelopt_model.0.pth")
    if os.path.exists(pth_file):
        print(f"Loading from: {pth_file}")
        state = torch.load(pth_file, map_location="cpu")
    else:
        # Try loading from safetensors via transformers
        print(f"Loading from safetensors in: {input_path}")
        from safetensors.torch import load_file
        index_file = os.path.join(input_path, "model.safetensors.index.json")
        if os.path.exists(index_file):
            with open(index_file) as f:
                index = json.load(f)
            state = {}
            loaded_files = set()
            for key, fname in index["weight_map"].items():
                if fname not in loaded_files:
                    shard = load_file(os.path.join(input_path, fname))
                    state.update(shard)
                    loaded_files.add(fname)
                    del shard
                    gc.collect()
        else:
            raise FileNotFoundError(
                f"No model files found in {input_path}. "
                "Expected modelopt_model.0.pth or safetensors with index."
            )

    total_keys = len(state)
    print(f"Loaded {total_keys} tensors")
    report_memory("STATE LOADED")

    # ------------------------------------------------------------------
    # Phase 2: Quantize weight matrices
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("PHASE 2: INT4 weight-only quantization")
    print(f"  Group size: {group_size}")
    print(f"{'='*70}")

    output_state = {}
    quantized_count = 0
    skipped_count = 0

    for idx, (key, tensor) in enumerate(list(state.items())):
        if idx % 500 == 0:
            print(f"  Processing tensor {idx}/{total_keys}...")
            gc.collect()
            report_memory(f"PROGRESS {idx}/{total_keys}")

        # Quantize 2D weight matrices large enough for grouping
        if "weight" in key and tensor.dim() == 2 and tensor.shape[1] >= group_size:
            packed, scales = quantize_weight_int4(tensor, group_size)

            base = key.replace(".weight", "")
            output_state[f"{base}.qweight"] = packed
            output_state[f"{base}.scales"] = scales
            quantized_count += 1

            # Free original tensor immediately
            del state[key]
        else:
            output_state[key] = tensor
            skipped_count += 1

    del state
    gc.collect()

    print(f"\n  Quantized: {quantized_count} weight matrices")
    print(f"  Skipped:   {skipped_count} tensors (biases, norms, small weights)")
    report_memory("QUANTIZATION DONE")

    # ------------------------------------------------------------------
    # Phase 3: Save as sharded safetensors
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("PHASE 3: Saving quantized model")
    print(f"{'='*70}")

    os.makedirs(output_dir, exist_ok=True)
    shard_limit = shard_size_gb * 1e9

    shard_size = 0
    shard_idx = 0
    current_shard = {}
    index_map = {}

    for key, tensor in output_state.items():
        tensor_bytes = tensor.numel() * tensor.element_size()

        if shard_size + tensor_bytes > shard_limit and current_shard:
            fname = f"model-{shard_idx + 1:05d}.safetensors"
            save_file(current_shard, os.path.join(output_dir, fname))
            print(f"  Saved {fname} ({shard_size / 1e9:.1f} GB)")
            for k in current_shard:
                index_map[k] = fname
            current_shard = {}
            shard_size = 0
            shard_idx += 1
            gc.collect()

        current_shard[key] = tensor
        shard_size += tensor_bytes

    # Save final shard
    if current_shard:
        fname = f"model-{shard_idx + 1:05d}.safetensors"
        save_file(current_shard, os.path.join(output_dir, fname))
        print(f"  Saved {fname} ({shard_size / 1e9:.1f} GB)")
        for k in current_shard:
            index_map[k] = fname

    del output_state, current_shard
    gc.collect()

    # Write safetensors index
    total_size = sum(
        os.path.getsize(os.path.join(output_dir, f))
        for f in os.listdir(output_dir)
        if f.endswith(".safetensors")
    )
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": index_map,
    }
    with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)

    # ------------------------------------------------------------------
    # Phase 4: Copy config and tokenizer files
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("PHASE 4: Copying configuration files")
    print(f"{'='*70}")

    # Determine source directory for configs
    src_dir = input_path
    # If input was a .pth inside a dir, look for configs in the merged model dir
    merged_dir = os.path.join(os.path.dirname(input_path), "nemotron_merged_fp16")
    if os.path.exists(merged_dir):
        src_dir = merged_dir

    config_files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
        "configuration_nemotron_h.py",
        "modeling_nemotron_h.py",
        "chat_template.jinja",
    ]

    for fname in config_files:
        src_path = os.path.join(src_dir, fname)
        if os.path.exists(src_path):
            shutil.copy2(src_path, os.path.join(output_dir, fname))
            print(f"  Copied {fname}")
        else:
            print(f"  Skipped {fname} (not found in {src_dir})")

    # Write quantization config
    quant_config = {
        "quant_method": "gptq",
        "bits": 4,
        "group_size": group_size,
        "desc_act": False,
        "sym": True,
        "packing": "int32_8x",
        "source_dtype": "bfloat16",
        "quantizer": "manual_int4_wo",
        "notes": (
            "INT4 weight-only quantization with per-group symmetric scaling. "
            "Weights are packed 8xINT4 into INT32 containers. "
            "Scales are stored as float16 with shape (rows, cols // group_size)."
        ),
    }
    with open(os.path.join(output_dir, "quantize_config.json"), "w") as f:
        json.dump(quant_config, f, indent=2)
    print("  Wrote quantize_config.json")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print("QUANTIZATION COMPLETE")
    print(f"{'='*70}")
    print(f"  Output directory: {output_dir}")
    print(f"  Total size:       {total_size / 1e9:.1f} GB")
    print(f"  Layers quantized: {quantized_count}")
    print(f"  Group size:       {group_size}")
    print(f"  Time elapsed:     {elapsed / 60:.1f} minutes")
    print(f"  Files: {sorted(os.listdir(output_dir))}")
    report_memory("COMPLETE")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="INT4 weight-only quantization for Nemotron-3-Nano-30B",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quantize from merged bf16 safetensors directory
  python quantize_int4.py --input_dir /workspace/nemotron_merged_fp16 --output_dir /workspace/nemotron_int4

  # Quantize from ModelOpt .pth checkpoint (after failed TRT-LLM export)
  python quantize_int4.py --input_dir /workspace/nemotron_int4_wo --output_dir /workspace/nemotron_int4

  # Custom group size
  python quantize_int4.py --input_dir ./merged_model --output_dir ./quantized --group_size 64
        """,
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to merged model directory (safetensors or containing modelopt_model.0.pth)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save quantized model",
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=DEFAULT_GROUP_SIZE,
        help=f"Quantization group size (default: {DEFAULT_GROUP_SIZE})",
    )
    parser.add_argument(
        "--shard_size_gb",
        type=float,
        default=DEFAULT_SHARD_SIZE_GB,
        help=f"Max shard size in GB (default: {DEFAULT_SHARD_SIZE_GB})",
    )

    args = parser.parse_args()

    quantize_model(
        input_path=args.input_dir,
        output_dir=args.output_dir,
        group_size=args.group_size,
        shard_size_gb=args.shard_size_gb,
    )


if __name__ == "__main__":
    main()
