#!/usr/bin/env python3
"""
Convert FSDP distributed checkpoints to standard PEFT format.

FSDP saves checkpoints as sharded .distcp files which PEFT cannot load directly.
This script loads the FSDP checkpoint and saves it as a standard adapter_model.safetensors.
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from safetensors.torch import save_file


def convert_fsdp_checkpoint(checkpoint_path: Path, output_path: Path = None) -> Path:
    """
    Convert an FSDP checkpoint to PEFT-compatible format.

    Args:
        checkpoint_path: Path to the FSDP checkpoint directory
        output_path: Where to save the converted checkpoint (default: in-place)

    Returns:
        Path to the converted checkpoint
    """
    checkpoint_path = Path(checkpoint_path)

    if output_path is None:
        output_path = checkpoint_path
    else:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

    # Check if already converted
    if (output_path / "adapter_model.safetensors").exists():
        print(f"Already converted: {output_path}")
        return output_path

    fsdp_dir = checkpoint_path / "pytorch_model_fsdp_0"
    if not fsdp_dir.exists():
        raise ValueError(f"No FSDP checkpoint found at {fsdp_dir}")

    print(f"Loading FSDP checkpoint from {fsdp_dir}...")

    # Use dcp_to_torch_save to convert distributed checkpoint to regular state dict
    # This creates a temporary file, then we load it
    temp_file = checkpoint_path / "_temp_converted.pt"

    try:
        dcp_to_torch_save(fsdp_dir, temp_file)
        state_dict = torch.load(temp_file, map_location="cpu", weights_only=True)
    finally:
        if temp_file.exists():
            temp_file.unlink()

    # Extract only the LoRA weights (keys containing "lora_")
    lora_state_dict = {}
    for key, value in state_dict.items():
        if "lora_" in key:
            # Clean up the key name - remove any FSDP prefixes
            clean_key = key
            # Handle potential FSDP key prefixes like "_fsdp_wrapped_module."
            if "_fsdp_wrapped_module." in clean_key:
                clean_key = clean_key.replace("_fsdp_wrapped_module.", "")
            if "_orig_mod." in clean_key:
                clean_key = clean_key.replace("_orig_mod.", "")
            # Ensure proper prefix for PEFT
            if not clean_key.startswith("base_model.model."):
                clean_key = f"base_model.model.{clean_key}"
            lora_state_dict[clean_key] = value

    if not lora_state_dict:
        print("Warning: No LoRA weights found in checkpoint!")
        print(f"Available keys: {list(state_dict.keys())[:20]}...")
        raise ValueError("No LoRA weights found in FSDP checkpoint")

    print(f"Found {len(lora_state_dict)} LoRA weight tensors")

    # Save as safetensors
    output_file = output_path / "adapter_model.safetensors"
    save_file(lora_state_dict, output_file)
    print(f"Saved adapter weights to {output_file}")

    # Copy adapter_config.json if converting to a different location
    if output_path != checkpoint_path:
        config_src = checkpoint_path / "adapter_config.json"
        if config_src.exists():
            shutil.copy2(config_src, output_path / "adapter_config.json")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Convert FSDP checkpoints to PEFT format")
    parser.add_argument("checkpoint_path", type=Path, help="Path to FSDP checkpoint directory")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output path (default: in-place conversion)")
    parser.add_argument("--batch", action="store_true",
                        help="Convert all checkpoints in a directory")

    args = parser.parse_args()

    if args.batch:
        # Convert all checkpoint-* directories
        checkpoint_dirs = sorted(args.checkpoint_path.glob("checkpoint-*"))
        print(f"Found {len(checkpoint_dirs)} checkpoints to convert")

        for ckpt_dir in checkpoint_dirs:
            if (ckpt_dir / "pytorch_model_fsdp_0").exists():
                print(f"\nConverting {ckpt_dir.name}...")
                try:
                    convert_fsdp_checkpoint(ckpt_dir)
                except Exception as e:
                    print(f"  Error: {e}")
            else:
                print(f"Skipping {ckpt_dir.name} (no FSDP checkpoint)")
    else:
        convert_fsdp_checkpoint(args.checkpoint_path, args.output)


if __name__ == "__main__":
    main()
