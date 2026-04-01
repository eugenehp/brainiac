#!/usr/bin/env python3
"""
Convert BrainIAC PyTorch checkpoints to safetensors format for brainiac-rs.

Usage:
    # Convert backbone (SimCLR pretrained ViT)
    python scripts/export_safetensors.py \
        --input checkpoints/BrainIAC.ckpt \
        --output brainiac_backbone.safetensors

    # Convert downstream task head (e.g., brain age)
    python scripts/export_safetensors.py \
        --input checkpoints/brainage_model.ckpt \
        --output brainage_head.safetensors \
        --head-only

    # Convert full downstream model (backbone + head)
    python scripts/export_safetensors.py \
        --input checkpoints/brainage_model.ckpt \
        --output brainage_full.safetensors \
        --full
"""

import argparse
import torch
from safetensors.torch import save_file


def extract_backbone_weights(state_dict):
    """Extract ViT backbone weights, stripping 'backbone.' prefix."""
    backbone = {}
    for key, value in state_dict.items():
        if key.startswith("backbone.backbone."):
            # SimCLR checkpoint: backbone.backbone.X → X
            new_key = key[len("backbone.backbone."):]
            backbone[new_key] = value
        elif key.startswith("backbone."):
            # Direct backbone checkpoint: backbone.X → X
            new_key = key[len("backbone."):]
            backbone[new_key] = value
    return backbone


def extract_head_weights(state_dict):
    """Extract classifier head weights."""
    head = {}
    for key, value in state_dict.items():
        if key.startswith("classifier."):
            new_key = key[len("classifier."):]
            head[new_key] = value
        elif key.startswith("fc."):
            head[key] = value
    return head


def main():
    parser = argparse.ArgumentParser(
        description="Convert BrainIAC PyTorch checkpoints to safetensors"
    )
    parser.add_argument("--input", required=True, help="Input .ckpt file")
    parser.add_argument("--output", required=True, help="Output .safetensors file")
    parser.add_argument(
        "--head-only",
        action="store_true",
        help="Extract only the classifier head",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Extract both backbone and head (keep prefixes)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print all weight keys",
    )
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.input}")
    ckpt = torch.load(args.input, map_location="cpu", weights_only=False)

    # Handle different checkpoint formats
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    print(f"  Total keys: {len(state_dict)}")

    if args.verbose:
        for k, v in sorted(state_dict.items()):
            print(f"  {k:70s}  {list(v.shape)}")

    if args.head_only:
        weights = extract_head_weights(state_dict)
        print(f"  Extracted {len(weights)} head weight tensors")
    elif args.full:
        # Keep all weights, just strip Lightning prefix
        weights = {}
        for k, v in state_dict.items():
            # Strip common Lightning wrapper prefixes
            new_k = k
            for prefix in ["model.", "net."]:
                if new_k.startswith(prefix):
                    new_k = new_k[len(prefix):]
                    break
            weights[new_k] = v
        print(f"  Extracted {len(weights)} total weight tensors")
    else:
        weights = extract_backbone_weights(state_dict)
        print(f"  Extracted {len(weights)} backbone weight tensors")

    if not weights:
        print("WARNING: No weights extracted! Check the checkpoint format.")
        print("Available keys:")
        for k in sorted(state_dict.keys())[:20]:
            print(f"  {k}")
        return

    # Convert to float32 for safetensors
    weights_f32 = {}
    for k, v in weights.items():
        weights_f32[k] = v.float().contiguous()

    print(f"Saving to: {args.output}")
    save_file(weights_f32, args.output)

    # Verify
    from safetensors import safe_open
    with safe_open(args.output, framework="pt") as f:
        keys = f.keys()
        print(f"  Verified: {len(keys)} tensors in {args.output}")
        if args.verbose:
            for k in sorted(keys):
                t = f.get_tensor(k)
                print(f"    {k:70s}  {list(t.shape)}")

    print("Done!")


if __name__ == "__main__":
    main()
