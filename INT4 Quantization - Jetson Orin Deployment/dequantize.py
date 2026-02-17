"""
Dequantization utilities for INT4 weight-only quantized models.

Provides functions to unpack INT4 weights and apply per-group scales
for inference or quality validation.

Usage:
    python dequantize.py --model_dir /workspace/nemotron_int4_manual --layer backbone.layers.0.mixer.in_proj
"""

import torch
import json
import os
import argparse
from safetensors.torch import load_file


def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    """
    Unpack 8 INT4 values from each INT32 container.

    Args:
        packed: Packed tensor of shape (rows, cols // 8), dtype int32.

    Returns:
        Unpacked tensor of shape (rows, cols), dtype int8, values in [-8, 7].
    """
    rows, packed_cols = packed.shape
    cols = packed_cols * 8

    unpacked = torch.zeros(rows, cols, dtype=torch.int8)
    for i in range(8):
        unsigned = ((packed >> (i * 4)) & 0xF).to(torch.int8)
        # Convert unsigned 4-bit [0,15] to signed [-8, 7]
        signed = torch.where(unsigned > 7, unsigned - 16, unsigned)
        unpacked[:, i::8] = signed

    return unpacked


def dequantize_int4(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """
    Dequantize INT4 packed weights using per-group scales.

    Args:
        qweight: Packed INT4 weights, shape (rows, cols // 8), dtype int32.
        scales: Per-group scales, shape (rows, cols // group_size), dtype float16.
        group_size: Number of columns per quantization group.

    Returns:
        Dequantized weight matrix, shape (rows, cols), dtype float32.
    """
    unpacked = unpack_int4(qweight).float()
    rows, cols = unpacked.shape

    # Reshape to groups and apply scales
    unpacked = unpacked.reshape(-1, group_size)
    scales_flat = scales.float().reshape(-1, 1)
    dequantized = (unpacked * scales_flat).reshape(rows, cols)

    return dequantized


def load_quantized_layer(
    model_dir: str,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load a specific quantized layer from sharded safetensors.

    Args:
        model_dir: Path to quantized model directory.
        layer_name: Base layer name (e.g., 'backbone.layers.0.mixer.in_proj').

    Returns:
        Tuple of (qweight, scales) tensors.
    """
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)

    qweight_key = f"{layer_name}.qweight"
    scales_key = f"{layer_name}.scales"

    weight_map = index["weight_map"]
    if qweight_key not in weight_map:
        raise KeyError(f"Layer '{layer_name}' not found. Available layers: "
                       f"{[k.replace('.qweight','') for k in weight_map if k.endswith('.qweight')][:5]}...")

    # Load from appropriate shards
    qweight_shard = load_file(os.path.join(model_dir, weight_map[qweight_key]))
    scales_shard = load_file(os.path.join(model_dir, weight_map[scales_key]))

    return qweight_shard[qweight_key], scales_shard[scales_key]


def main():
    parser = argparse.ArgumentParser(description="Dequantize and inspect INT4 layers")
    parser.add_argument("--model_dir", type=str, required=True, help="Quantized model directory")
    parser.add_argument("--layer", type=str, required=True, help="Layer name to inspect")
    parser.add_argument("--group_size", type=int, default=128, help="Group size used during quantization")
    args = parser.parse_args()

    print(f"Loading layer: {args.layer}")
    qweight, scales = load_quantized_layer(args.model_dir, args.layer)

    print(f"  qweight shape: {qweight.shape}, dtype: {qweight.dtype}")
    print(f"  scales shape:  {scales.shape}, dtype: {scales.dtype}")

    # Dequantize
    weight = dequantize_int4(qweight, scales, args.group_size)
    print(f"  Dequantized shape: {weight.shape}")
    print(f"  Value range: [{weight.min():.4f}, {weight.max():.4f}]")
    print(f"  Mean: {weight.mean():.6f}, Std: {weight.std():.4f}")


if __name__ == "__main__":
    main()
