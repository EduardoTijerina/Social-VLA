"""
Pre-flight memory test for INT4 quantization.

Verifies that the system has sufficient memory to load and quantize the model
without crashing. Run this BEFORE the full quantization to ensure safety.

Usage:
    python memory_test.py --model_dir /workspace/nemotron_merged_fp16
"""

import os
import sys
import torch
import gc
import psutil
import argparse


def report_memory(stage: str):
    """Print memory usage and abort if critically low."""
    mem = psutil.virtual_memory()
    gpu_alloc = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    gpu_reserved = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0.0

    print(f"\n{'='*60}")
    print(f"[{stage}]")
    print(f"  RAM: {mem.used/1e9:.1f} GB used / {mem.total/1e9:.1f} GB total ({mem.percent}%)")
    print(f"  RAM available: {mem.available/1e9:.1f} GB")
    if torch.cuda.is_available():
        print(f"  GPU allocated: {gpu_alloc:.1f} GB | reserved: {gpu_reserved:.1f} GB")
    print(f"{'='*60}")

    if mem.available < 10e9:
        print("ABORTING: Less than 10 GB RAM available!")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Memory test for quantization")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to model directory")
    args = parser.parse_args()

    report_memory("BASELINE")

    # Check model size on disk
    model_size = sum(
        os.path.getsize(os.path.join(args.model_dir, f))
        for f in os.listdir(args.model_dir)
        if f.endswith((".safetensors", ".pth", ".bin"))
    )
    print(f"\nModel size on disk: {model_size / 1e9:.1f} GB")

    mem = psutil.virtual_memory()
    # Need ~1.5x model size for loading + quantization overhead
    required = model_size * 1.5
    if mem.available < required:
        print(f"WARNING: May not have enough RAM.")
        print(f"  Available: {mem.available/1e9:.1f} GB")
        print(f"  Estimated need: {required/1e9:.1f} GB")
        print(f"  Consider stopping other containers and applications.")
        sys.exit(1)
    else:
        print(f"  Available RAM ({mem.available/1e9:.1f} GB) exceeds estimated need ({required/1e9:.1f} GB)")

    # Check shared memory
    shm_stat = os.statvfs("/dev/shm")
    shm_total = shm_stat.f_blocks * shm_stat.f_frsize / 1e9
    print(f"\n/dev/shm size: {shm_total:.1f} GB")
    if shm_total < 1.0:
        print("WARNING: Shared memory is very small (< 1 GB).")
        print("  This will cause PyTorch to crash during model loading.")
        print("  Restart container with: --shm-size=64g")
        sys.exit(1)

    # Test loading a state dict
    print("\nTesting state dict loading...")
    pth_file = os.path.join(args.model_dir, "modelopt_model.0.pth")
    if os.path.exists(pth_file):
        state = torch.load(pth_file, map_location="cpu")
    else:
        from safetensors.torch import load_file
        index_file = os.path.join(args.model_dir, "model.safetensors.index.json")
        if os.path.exists(index_file):
            import json
            with open(index_file) as f:
                index = json.load(f)
            first_shard = list(set(index["weight_map"].values()))[0]
            state = load_file(os.path.join(args.model_dir, first_shard))
            print(f"  Loaded first shard: {first_shard} ({len(state)} tensors)")
        else:
            print("  No model files found to test.")
            sys.exit(1)

    report_memory("AFTER PARTIAL LOAD")
    del state
    gc.collect()

    print("\nMemory test PASSED - quantization should be safe to run.")


if __name__ == "__main__":
    main()
