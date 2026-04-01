#!/usr/bin/env python3
"""
Convert BrainIAC checkpoints to safetensors and prepare HuggingFace repo.

Usage:
    # From a complete BrainIAC.ckpt (SimCLR checkpoint):
    python scripts/convert_and_publish.py --ckpt /path/to/BrainIAC.ckpt --output hf_repo/

    # From a Lightning downstream checkpoint (contains backbone + head):
    python scripts/convert_and_publish.py --ckpt /path/to/brainage.ckpt --output hf_repo/ --task brainage

    # Create a dummy model for testing (random weights, correct shapes):
    python scripts/convert_and_publish.py --dummy --output hf_repo/

    # Publish to HuggingFace:
    huggingface-cli login
    huggingface-cli upload eugenehp/brainiac hf_repo/ .
"""

import argparse
import json
import os
import sys

import torch
from safetensors.torch import save_file


def create_monai_vit():
    """Create a MONAI ViT with the exact BrainIAC architecture."""
    from monai.networks.nets import ViT
    return ViT(
        in_channels=1,
        img_size=(96, 96, 96),
        patch_size=(16, 16, 16),
        hidden_size=768,
        mlp_dim=3072,
        num_layers=12,
        num_heads=12,
        save_attn=False,
    )


def extract_backbone_from_simclr(ckpt_path):
    """Extract backbone weights from a SimCLR checkpoint."""
    print(f"Loading SimCLR checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    state_dict = ckpt.get("state_dict", ckpt)

    # Try various prefix patterns
    backbone = {}
    prefixes = ["backbone.backbone.", "backbone.", "encoder.", "model.backbone.", "model."]

    for prefix in prefixes:
        matched = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
        if matched:
            # Verify it looks like a ViT backbone
            if any("blocks" in k for k in matched) and any("patch_embedding" in k for k in matched):
                backbone = matched
                print(f"  Matched prefix '{prefix}' → {len(backbone)} tensors")
                break

    if not backbone:
        # Maybe keys are already unprefixed
        if any("blocks" in k for k in state_dict) and any("patch_embedding" in k for k in state_dict):
            backbone = state_dict
            print(f"  No prefix needed → {len(backbone)} tensors")
        else:
            print("ERROR: Could not find backbone weights!")
            print("Available key prefixes:")
            prefixes_seen = set()
            for k in state_dict.keys():
                parts = k.split(".")
                for i in range(1, min(4, len(parts))):
                    prefixes_seen.add(".".join(parts[:i]) + ".")
            for p in sorted(prefixes_seen)[:20]:
                print(f"  {p}")
            sys.exit(1)

    # Filter to only backbone keys (skip cross_attn, projection heads, etc.)
    # Keep: patch_embedding.*, blocks.*.{norm1,norm2,attn,mlp}.*, norm.*
    filtered = {}
    for k, v in backbone.items():
        if any(k.startswith(p) for p in ["patch_embedding.", "blocks.", "norm."]):
            # Skip cross-attention layers (not used in BrainIAC inference)
            if "cross_attn" in k or "norm_cross_attn" in k:
                continue
            filtered[k] = v

    print(f"  Filtered to {len(filtered)} backbone tensors")
    return filtered


def extract_head_from_lightning(ckpt_path):
    """Extract classifier head from a Lightning checkpoint."""
    print(f"Loading Lightning checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)

    head = {}
    for k, v in state_dict.items():
        for prefix in ["model.classifier.", "classifier.", "model.fc.", "fc."]:
            if k.startswith(prefix):
                new_k = "fc." + k[len(prefix):]
                head[new_k] = v
                break

    print(f"  Extracted {len(head)} head tensors")
    return head


def create_dummy_weights():
    """Create random weights matching BrainIAC architecture."""
    print("Creating dummy MONAI ViT weights...")
    model = create_monai_vit()

    # Small random weights for testing
    torch.manual_seed(42)
    with torch.no_grad():
        for p in model.parameters():
            p.data = torch.randn_like(p) * 0.02

    state_dict = model.state_dict()
    # Filter out cross_attn
    filtered = {k: v for k, v in state_dict.items()
                if "cross_attn" not in k and "norm_cross_attn" not in k}

    print(f"  Created {len(filtered)} tensors")
    return filtered


def save_safetensors(tensors, path):
    """Save tensors as safetensors, converting to f32."""
    f32_tensors = {k: v.float().contiguous() for k, v in tensors.items()}
    save_file(f32_tensors, path)
    size_mb = os.path.getsize(path) / 1e6
    print(f"  Saved {path} ({len(f32_tensors)} tensors, {size_mb:.1f} MB)")


def write_config(output_dir):
    """Write model config.json."""
    config = {
        "model_type": "brainiac-vit",
        "architecture": "MONAI ViT-B/16³",
        "img_size": 96,
        "patch_size": 16,
        "in_channels": 1,
        "hidden_size": 768,
        "mlp_dim": 3072,
        "num_layers": 12,
        "num_heads": 12,
        "num_patches": 216,
        "norm_eps": 1e-6,
        "pretraining": "SimCLR",
        "input_format": "NIfTI (.nii.gz), skull-stripped, registered, 96×96×96",
        "preprocessing": "trilinear resize to 96³ + z-score normalization (nonzero voxels)"
    }
    path = os.path.join(output_dir, "config.json")
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Wrote {path}")


def write_readme(output_dir):
    """Write HuggingFace model card README.md."""
    readme = """---
license: other
license_name: mass-general-brigham-non-commercial
license_link: LICENSE
tags:
  - brain
  - mri
  - neuroimaging
  - vit
  - foundation-model
  - medical-imaging
library_name: brainiac-rs
pipeline_tag: feature-extraction
---

# BrainIAC — Brain Imaging Adaptive Core

**A generalizable foundation model for analysis of human brain MRI**

BrainIAC is a Vision Transformer (ViT-B/16) pretrained with SimCLR on structural brain MRI scans.
Published in [Nature Neuroscience](https://www.nature.com/articles/s41593-026-02202-6) (2026).

## Model Details

| Property | Value |
|----------|-------|
| Architecture | MONAI ViT-B/16³ (3D) |
| Parameters | 88.4M |
| Input | 96×96×96 single-channel brain MRI |
| Patches | 216 (6×6×6 grid, 16³ voxel patches) |
| Hidden dim | 768 |
| Layers | 12 transformer blocks |
| Heads | 12 attention heads |
| MLP dim | 3072 |
| Pretraining | SimCLR contrastive learning |
| Output | 768-dim feature vector (first patch token) |

## Files

- `backbone.safetensors` — Pretrained ViT backbone weights
- `config.json` — Model configuration
- `LICENSE` — Non-commercial academic research license

## Downstream Tasks

The backbone can be fine-tuned for:
- **Brain age prediction** (regression)
- **IDH mutation classification** (binary, dual-scan FLAIR+T1CE)
- **MCI classification** (binary)
- **Glioma overall survival** (binary, quad-scan T1+T1CE+T2+FLAIR)
- **MR sequence classification** (4-class: T1/T2/FLAIR/T1CE)
- **Time-to-stroke prediction** (regression)
- **Tumor segmentation** (UNETR decoder)

## Usage with brainiac-rs (Rust)

```bash
cargo run --release --bin infer -- \\
    --weights backbone.safetensors \\
    --input brain_t1.nii.gz
```

```rust
use brainiac::{BrainiacEncoder, TaskType};

let (encoder, _) = BrainiacEncoder::<B>::load(
    "backbone.safetensors", None,
    TaskType::FeatureExtraction, 1, device,
)?;
let features = encoder.encode_nifti(Path::new("brain.nii.gz"))?;
// features: Vec<f32> with 768 dimensions
```

## Usage with Python

```python
import torch
from monai.networks.nets import ViT
from safetensors.torch import load_file

model = ViT(in_channels=1, img_size=(96,96,96), patch_size=(16,16,16),
            hidden_size=768, mlp_dim=3072, num_layers=12, num_heads=12)

weights = load_file("backbone.safetensors")
model.load_state_dict(weights, strict=False)
model.eval()

# features[0][:, 0] gives the 768-dim feature vector
features = model(preprocessed_mri)
```

## Preprocessing

Input MRI volumes must be:
1. Skull-stripped (HD-BET recommended)
2. Registered to standard space (MNI152)
3. Bias field corrected (N4)
4. Resized to 96×96×96 voxels (trilinear)
5. Z-score normalized (nonzero voxels only)

## Citation

```bibtex
@article{tak2026generalizable,
  title={A generalizable foundation model for analysis of human brain MRI},
  author={Tak, Divyanshu and Gormosa, B.A. and Zapaishchykova, A. and others},
  journal={Nature Neuroscience},
  year={2026},
  publisher={Springer Nature},
  doi={10.1038/s41593-026-02202-6}
}
```

## License

This model is licensed for **non-commercial academic research use only**.
Commercial use requires a separate license from Mass General Brigham.
See [LICENSE](LICENSE) for details.
"""
    path = os.path.join(output_dir, "README.md")
    with open(path, "w") as f:
        f.write(readme.strip() + "\n")
    print(f"  Wrote {path}")


def write_license(output_dir, source_license_path=None):
    """Write LICENSE file."""
    if source_license_path and os.path.exists(source_license_path):
        import shutil
        dest = os.path.join(output_dir, "LICENSE")
        shutil.copy(source_license_path, dest)
        print(f"  Copied {source_license_path} → {dest}")
    else:
        # Use the BrainIAC license text
        license_text = """BrainIAC Model License

Copyright (c) 2026 Mass General Brigham

This software and model weights are provided for non-commercial academic
research use only. Commercial use is not permitted without a separate
license from Mass General Brigham.

For commercial licensing inquiries, please contact the Mass General Brigham
Office of Technology Development.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
"""
        path = os.path.join(output_dir, "LICENSE")
        with open(path, "w") as f:
            f.write(license_text)
        print(f"  Wrote {path}")


def main():
    parser = argparse.ArgumentParser(description="Convert BrainIAC to HuggingFace format")
    parser.add_argument("--ckpt", help="Path to BrainIAC .ckpt file")
    parser.add_argument("--task", help="Task name for downstream head (brainage, mci, idh, etc.)")
    parser.add_argument("--dummy", action="store_true", help="Create dummy weights for testing")
    parser.add_argument("--output", default="hf_repo", help="Output directory")
    parser.add_argument("--license", default=None, help="Path to LICENSE file to copy")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.dummy:
        backbone = create_dummy_weights()
    elif args.ckpt:
        backbone = extract_backbone_from_simclr(args.ckpt)
    else:
        print("ERROR: Specify --ckpt or --dummy")
        sys.exit(1)

    # Save backbone
    save_safetensors(backbone, os.path.join(args.output, "backbone.safetensors"))

    # Save downstream head if available
    if args.ckpt and args.task:
        head = extract_head_from_lightning(args.ckpt)
        if head:
            save_safetensors(head, os.path.join(args.output, f"{args.task}_head.safetensors"))

    # Write HF repo files
    write_config(args.output)
    write_readme(args.output)

    # License
    license_path = args.license
    if not license_path:
        # Try to find BrainIAC LICENSE
        for candidate in [
            "/Users/Shared/BrainIAC/LICENSE",
            os.path.join(os.path.dirname(__file__), "..", "..", "BrainIAC", "LICENSE"),
        ]:
            if os.path.exists(candidate):
                license_path = candidate
                break
    write_license(args.output, license_path)

    print(f"\n✓ HuggingFace repo prepared in {args.output}/")
    print(f"\nTo publish:")
    print(f"  huggingface-cli login")
    print(f"  huggingface-cli repo create eugenehp/brainiac --type model")
    print(f"  huggingface-cli upload eugenehp/brainiac {args.output}/ .")


if __name__ == "__main__":
    main()
