#!/usr/bin/env python3
"""
Generate numerical parity test vectors for brainiac-rs.

Creates a small ViT model with deterministic weights, runs inference on a
synthetic input, and saves intermediate activations + final output at every
stage. The Rust tests load these vectors and compare.

Output directory: tests/vectors/
Files:
  config.json              - model config
  weights.safetensors      - model weights (backbone only)
  input_volume.bin         - raw f32 input [1, 1, 96, 96, 96]
  patch_embed_output.bin   - after patch embedding [1, 216, 768]
  pos_embed_output.bin     - after adding CLS + pos embed [1, 217, 768]
  block_0_output.bin       - after transformer block 0 [1, 217, 768]
  block_11_output.bin      - after transformer block 11 [1, 217, 768]
  norm_output.bin          - after final layernorm [1, 217, 768]
  cls_token_output.bin     - final CLS features [1, 768]
  layernorm_input.bin      - small LN test input [1, 4, 8]
  layernorm_output.bin     - small LN test output [1, 4, 8]
  attention_input.bin      - small attn test input [1, 4, 768]
  attention_output.bin     - small attn test output [1, 4, 768]
"""

import os
import json
import struct
import numpy as np
import torch
from monai.networks.nets import ViT
from safetensors.torch import save_file

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'tests', 'vectors')
os.makedirs(OUT_DIR, exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)

def save_f32(path, data):
    """Save a numpy array as raw f32 binary."""
    arr = data.astype(np.float32).flatten()
    with open(path, 'wb') as f:
        f.write(struct.pack(f'{len(arr)}f', *arr))
    print(f"  {os.path.basename(path)}: shape={list(data.shape)}, {len(arr)*4} bytes")

def save_shape(path, shape):
    """Save shape as JSON alongside binary."""
    with open(path, 'w') as f:
        json.dump(list(shape), f)

# ── 1. Create MONAI ViT with deterministic small weights ─────────────────────

print("Creating MONAI ViT model...")
model = ViT(
    in_channels=1,
    img_size=(96, 96, 96),
    patch_size=(16, 16, 16),
    hidden_size=768,
    mlp_dim=3072,
    num_layers=12,
    num_heads=12,
    save_attn=True,
)

# Scale down weights for numerical stability in tests
with torch.no_grad():
    for name, param in model.named_parameters():
        param.data = torch.randn_like(param) * 0.02

model.eval()

# ── 2. Save weights as safetensors ───────────────────────────────────────────

print("Saving weights...")
state = model.state_dict()
weights = {}
for k, v in state.items():
    weights[k] = v.float().contiguous()

save_file(weights, os.path.join(OUT_DIR, 'weights.safetensors'))
print(f"  Saved {len(weights)} weight tensors")

# Print key names for debugging
for k, v in sorted(weights.items()):
    print(f"    {k:60s} {list(v.shape)}")

# ── 3. Save config ───────────────────────────────────────────────────────────

config = {
    "img_size": 96,
    "patch_size": 16,
    "in_channels": 1,
    "hidden_size": 768,
    "mlp_dim": 3072,
    "num_layers": 12,
    "num_heads": 12,
    "norm_eps": 1e-6,
}
with open(os.path.join(OUT_DIR, 'config.json'), 'w') as f:
    json.dump(config, f, indent=2)

# ── 4. Generate synthetic input ──────────────────────────────────────────────

print("\nGenerating input volume...")
torch.manual_seed(123)
input_vol = torch.randn(1, 1, 96, 96, 96) * 0.1
save_f32(os.path.join(OUT_DIR, 'input_volume.bin'), input_vol.numpy())

# ── 5. Run forward pass capturing intermediates ──────────────────────────────

print("\nRunning forward pass with intermediates...")

with torch.no_grad():
    # Step 1: Patch embedding
    # MONAI ViT's patch_embedding includes cls_token + position_embeddings
    x = input_vol

    # Get patch embeddings manually
    patch_embed = model.patch_embedding.patch_embeddings(x)  # Conv3d
    # patch_embed: [1, 768, 6, 6, 6]
    patch_embed_flat = patch_embed.flatten(2).transpose(1, 2)  # [1, 216, 768]
    save_f32(os.path.join(OUT_DIR, 'patch_embed_output.bin'), patch_embed_flat.numpy())
    save_shape(os.path.join(OUT_DIR, 'patch_embed_output.shape.json'), patch_embed_flat.shape)

    # Step 2: Add position embeddings (NO CLS token in MONAI ViT non-classification)
    embeddings = patch_embed_flat + model.patch_embedding.position_embeddings
    save_f32(os.path.join(OUT_DIR, 'pos_embed_output.bin'), embeddings.numpy())
    save_shape(os.path.join(OUT_DIR, 'pos_embed_output.shape.json'), embeddings.shape)

    # Step 3: Transformer blocks
    hidden = embeddings
    for i, block in enumerate(model.blocks):
        hidden = block(hidden)
        if i == 0:
            save_f32(os.path.join(OUT_DIR, 'block_0_output.bin'), hidden.numpy())
            save_shape(os.path.join(OUT_DIR, 'block_0_output.shape.json'), hidden.shape)
        if i == 11:
            save_f32(os.path.join(OUT_DIR, 'block_11_output.bin'), hidden.numpy())
            save_shape(os.path.join(OUT_DIR, 'block_11_output.shape.json'), hidden.shape)

    # Step 4: Final norm
    normed = model.norm(hidden)
    save_f32(os.path.join(OUT_DIR, 'norm_output.bin'), normed.numpy())
    save_shape(os.path.join(OUT_DIR, 'norm_output.shape.json'), normed.shape)

    # Step 5: First patch token (BrainIAC uses features[:, 0])
    first_patch = normed[:, 0]  # [1, 768]
    save_f32(os.path.join(OUT_DIR, 'first_patch_output.bin'), first_patch.numpy())
    save_shape(os.path.join(OUT_DIR, 'first_patch_output.shape.json'), first_patch.shape)

    # Also run full model forward to verify
    full_output = model(input_vol)
    full_first = full_output[0][:, 0]  # [1, 768]
    save_f32(os.path.join(OUT_DIR, 'full_forward_first_patch.bin'), full_first.numpy())

    # Verify our manual decomposition matches full forward
    diff = (first_patch - full_first).abs().max().item()
    print(f"\n  Manual vs full forward max diff: {diff:.2e}")
    assert diff < 1e-5, f"Manual decomposition doesn't match! diff={diff}"

# ── 6. Small component tests ─────────────────────────────────────────────────

print("\nGenerating component test vectors...")

# LayerNorm test
torch.manual_seed(99)
ln_input = torch.randn(1, 4, 768) * 0.5
with torch.no_grad():
    # Use the first block's norm1 for testing
    ln_output = model.blocks[0].norm1(ln_input)
save_f32(os.path.join(OUT_DIR, 'layernorm_input.bin'), ln_input.numpy())
save_f32(os.path.join(OUT_DIR, 'layernorm_output.bin'), ln_output.numpy())
save_shape(os.path.join(OUT_DIR, 'layernorm_shape.json'), ln_input.shape)

# Attention test (single block)
torch.manual_seed(77)
attn_input = torch.randn(1, 8, 768) * 0.1
with torch.no_grad():
    # Run through first block only
    block0_output = model.blocks[0](attn_input)
save_f32(os.path.join(OUT_DIR, 'block0_test_input.bin'), attn_input.numpy())
save_f32(os.path.join(OUT_DIR, 'block0_test_output.bin'), block0_output.numpy())
save_shape(os.path.join(OUT_DIR, 'block0_test_shape.json'), attn_input.shape)

# ── 7. Preprocessing test vectors ────────────────────────────────────────────

print("\nGenerating preprocessing test vectors...")

# Create a simple 8x8x8 volume, resize to 4x4x4 via trilinear
torch.manual_seed(55)
preproc_input = torch.randn(8, 8, 8).abs() * 100  # positive values
save_f32(os.path.join(OUT_DIR, 'preproc_input_8x8x8.bin'), preproc_input.numpy())

# Use torch trilinear interpolation as reference
preproc_4d = preproc_input.unsqueeze(0).unsqueeze(0)  # [1,1,8,8,8]
resized = torch.nn.functional.interpolate(preproc_4d, size=(4,4,4), mode='trilinear', align_corners=False)
save_f32(os.path.join(OUT_DIR, 'preproc_resized_4x4x4.bin'), resized.squeeze().numpy())

# NormalizeIntensity (nonzero, channel_wise) reference
from monai.transforms import NormalizeIntensity
normalizer = NormalizeIntensity(nonzero=True, channel_wise=True)
normed_vol = normalizer(resized.squeeze(0))  # [1, 4, 4, 4]
save_f32(os.path.join(OUT_DIR, 'preproc_normalized_4x4x4.bin'), normed_vol.squeeze().numpy())

print("\n✓ All parity vectors generated in", OUT_DIR)
print(f"  Total files: {len(os.listdir(OUT_DIR))}")
