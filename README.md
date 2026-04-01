# brainiac

Pure-Rust inference and training for the [BrainIAC](https://www.nature.com/articles/s41593-026-02202-6) (Brain Imaging Adaptive Core) foundation model, built on [Burn 0.20](https://burn.dev).

BrainIAC is a Vision Transformer (ViT-B/16) pretrained with SimCLR on structural brain MRI. This crate provides numerically-verified inference, fine-tuning, and all downstream tasks — no Python or PyTorch required.

## Features

- **768-dim feature extraction** from brain MRI volumes
- **7 downstream tasks**: brain age, MCI, IDH mutation, glioma survival, MR sequence classification, time-to-stroke, tumor segmentation
- **Training**: full fine-tuning pipeline with Adam, cosine annealing, data augmentation, and checkpointing
- **NIfTI reader/writer** (.nii / .nii.gz) with automatic decompression
- **Preprocessing** matching MONAI's pipeline (trilinear resize + z-score normalization)
- **Data augmentation** matching MONAI's transforms (affine, flip, noise, smooth, contrast)
- **Saliency maps** from ViT attention weights → NIfTI output
- **Batch inference** with CSV in/out and full metrics (MAE, AUC, accuracy)
- **MRI preprocessing** (bias field correction, skull stripping)
- **Checkpoint export** (safetensors round-trip: load PyTorch → train in Rust → save safetensors)
- **Multi-backend**: CPU (NdArray) or GPU (wgpu/Metal/Vulkan)
- **Numerically verified**: max error 4.03e-5 vs Python MONAI across full 12-layer forward pass

## Benchmarks

Measured on Apple Silicon (M-series) with 96×96×96 brain MRI volumes, 88.4M parameter ViT-B/16³:

### Inference

| Component | CPU (NdArray) | GPU (Metal) | Speedup |
|-----------|--------------|-------------|---------|
| Patch embedding | 69.8 ms | 3.1 ms | **22.5×** |
| Single ViT block | 31.8 ms | 1.4 ms | **22.7×** |
| Full backbone (12 layers) | 270.1 ms | 22.0 ms | **12.3×** |
| End-to-end (preprocess + forward) | 267.0 ms | 22.4 ms | **11.9×** |
| **Throughput** | **3.7 scans/sec** | **44.7 scans/sec** | **12.0×** |

### Model Loading

| Method | CPU | GPU (Metal) |
|--------|-----|-------------|
| `from_weights` (direct build) | 831 ms | 598 ms |
| init + load_weights (zero-init then overwrite) | 1108 ms | 1375 ms |
| **Speedup** | **1.3×** | **2.3×** |

`from_weights` builds the model directly from safetensors data, skipping ~75 redundant zero-tensor GPU allocations. This is the default load path.

Current Metal benchmark for the optimized path is ~242 ms for `from_weights` vs ~944 ms for init + load, measured on the real BrainIAC weights.

```bash
# Run benchmarks yourself:
cargo run --release --example bench_all                               # CPU
cargo run --release --no-default-features --features metal --example bench_all  # Metal GPU
```

## Numerical Parity

Verified against Python MONAI ViT at every pipeline stage:

| Stage | Max Error | Mean Error |
|-------|-----------|------------|
| Patch embedding (Conv3d → [216, 768]) | 1.67e-6 | 1.14e-7 |
| Position embedding addition | 1.67e-6 | 1.14e-7 |
| LayerNorm | 5.75e-5 | 1.25e-5 |
| Single transformer block | 1.84e-3 | 2.16e-4 |
| Block 0 (full embed → block pipeline) | 2.67e-2 | 7.14e-4 |
| **Full 12-layer forward** | **4.03e-5** | **9.20e-6** |

```bash
# Generate reference vectors and run parity tests:
python3 scripts/generate_parity_vectors.py
cargo test --release --test parity -- --nocapture
```

## Quick Start

### 1. Convert weights

BrainIAC checkpoints are PyTorch `.ckpt` files. Convert to safetensors:

```bash
pip install torch safetensors monai
python scripts/export_safetensors.py \
    --input BrainIAC.ckpt \
    --output brainiac_backbone.safetensors
```

### 2. Run inference

```bash
# Feature extraction (default)
cargo run --release --bin infer -- \
    --weights brainiac_backbone.safetensors \
    --input brain_t1.nii.gz

# Brain age prediction
cargo run --release --bin infer -- \
    --weights brainiac_backbone.safetensors \
    --head brainage_head.safetensors \
    --task brain_age \
    --input brain_t1.nii.gz

# IDH mutation (dual scan: FLAIR + T1CE)
cargo run --release --bin infer -- \
    --weights brainiac_backbone.safetensors \
    --head idh_head.safetensors \
    --task idh \
    --input scan_flair.nii.gz \
    --input scan_t1ce.nii.gz

# GPU inference (Metal)
cargo run --release --no-default-features --features metal --bin infer -- \
    --weights brainiac_backbone.safetensors \
    --input brain_t1.nii.gz
```

### 3. Use as a library

```rust
use brainiac::{BrainiacEncoder, TaskType};
use std::path::Path;

// Load model
let (encoder, timings) = BrainiacEncoder::<B>::load(
    "brainiac_backbone.safetensors",
    None,  // no task head = feature extraction
    TaskType::FeatureExtraction,
    1,
    device,
)?;

// Extract features
let features = encoder.encode_nifti(Path::new("brain.nii.gz"))?;
println!("Feature dim: {}", features.len()); // 768

// Or run a downstream task
let output = encoder.infer_nifti(Path::new("brain.nii.gz"))?;
println!("{}", output);
```

## Training (Fine-tuning)

Fine-tune the BrainIAC backbone on downstream tasks, matching the Python Lightning training pipeline.

### CLI Training

```bash
# Brain age prediction (regression)
cargo run --release --bin train -- \
    --weights brainiac_backbone.safetensors \
    --task brain_age \
    --train-csv data/csvs/brainage_train.csv \
    --val-csv data/csvs/brainage_val.csv \
    --root-dir data/images \
    --epochs 200 \
    --lr 0.001

# MCI classification with frozen backbone (linear probing)
cargo run --release --bin train -- \
    --weights brainiac_backbone.safetensors \
    --task mci \
    --train-csv data/csvs/mci_train.csv \
    --val-csv data/csvs/mci_val.csv \
    --root-dir data/images \
    --freeze

# IDH mutation (dual-scan training)
cargo run --release --bin train -- \
    --weights brainiac_backbone.safetensors \
    --task idh \
    --train-csv data/csvs/idh_train.csv \
    --val-csv data/csvs/idh_val.csv \
    --root-dir data/images

# Overall survival (quad-scan training)
cargo run --release --bin train -- \
    --weights brainiac_backbone.safetensors \
    --task survival \
    --train-csv data/csvs/survival_train.csv \
    --val-csv data/csvs/survival_val.csv \
    --root-dir data/images

# MR sequence classification (4-class)
cargo run --release --bin train -- \
    --weights brainiac_backbone.safetensors \
    --task sequence \
    --train-csv data/csvs/sequence_train.csv \
    --val-csv data/csvs/sequence_val.csv \
    --root-dir data/images
```

### JSON Config Training

```bash
cargo run --release --bin train -- --config data/sample_train_config.json
```

See `data/sample_train_config.json` for the full config format.

## Hugging Face Model Hub

Pretrained weights and fine-tuned task models are published at:

https://huggingface.co/eugenehp/brainiac

Included artifacts:
- `backbone.safetensors`
- `brainage.safetensors`
- `idh.safetensors`
- `mci.safetensors`
- `stroke.safetensors`

### Training Features

| Feature | Details |
|---------|---------|
| Optimizer | Adam (β₁=0.9, β₂=0.999, weight decay) |
| LR Schedule | Cosine annealing with warm restarts (T₀=50, T_mult=2) |
| Loss functions | MSE (regression), BCE with logits (binary), Cross-entropy (multiclass) |
| Metrics | MAE (regression), AUC-ROC (binary), Accuracy (multiclass) |
| Data augmentation | Affine, L-R flip, Gaussian noise/smooth, gamma contrast |
| Freeze backbone | `--freeze` flag for linear probing |
| Multi-scan | Dual-scan (IDH) and quad-scan (survival) training |
| Checkpointing | Best model saved as .safetensors by validation metric |

## Build Options

```bash
# CPU (default — NdArray with Rayon multi-threading)
cargo build --release

# CPU with Apple Accelerate BLAS (macOS, fastest on Apple Silicon)
cargo build --release --features blas-accelerate

# GPU via Metal (macOS) — 12× faster inference
cargo build --release --no-default-features --features metal

# GPU via Vulkan (Linux/Windows)
cargo build --release --no-default-features --features vulkan
```

## Architecture

The model is a MONAI ViT-B/16 adapted for 3D volumetric input:

| Component | Details |
|-----------|---------|
| Input | 96×96×96 single-channel brain MRI |
| Patch embedding | Conv3d(1, 768, kernel=16³, stride=16³) → 216 patches |
| Positional embedding | Learned, 216 positions (no CLS token, per MONAI) |
| Transformer | 12 layers, 12 heads, hidden=768, MLP=3072, ~88M params |
| Output | First patch token → 768-dim features |

Downstream tasks use a linear head on top of the features. Multi-scan tasks (IDH: 2 scans, survival: 4 scans) process each scan through the shared backbone and mean-pool features before the head.

## Supported Tasks

| Task | Type | Input | Output |
|------|------|-------|--------|
| Feature extraction | Embedding | Single T1 | 768-dim vector |
| Brain age | Regression | Single T1 | Age in months |
| MCI classification | Binary | Single T1 | Healthy vs MCI |
| Time-to-stroke | Regression | Single T1 | Days since stroke |
| MR sequence | 4-class | Single scan | T1/T2/FLAIR/T1CE |
| IDH mutation | Binary | FLAIR + T1CE | Wild-type vs mutant |
| Overall survival | Binary | T1+T1CE+T2+FLAIR | Short vs long survival |
| Tumor segmentation | Segmentation | FLAIR | Binary mask |

## Project Structure

```
src/
├── lib.rs                  # Public API
├── config.rs               # Model, task & training configuration
├── nifti.rs                # NIfTI-1 reader + writer (.nii/.nii.gz)
├── preprocessing.rs        # Resize + z-score normalization
├── augmentation.rs         # Data augmentation (affine, flip, noise, smooth, contrast)
├── mri_preprocess.rs       # MRI preprocessing (bias correction, skull stripping)
├── weights.rs              # Safetensors loader + weight assignment
├── checkpoint.rs           # Model checkpoint saving (safetensors export)
├── encoder.rs              # High-level load/infer API
├── data.rs                 # CSV parsing + dataset loaders (single/dual/quad)
├── losses.rs               # MSE, BCE, CrossEntropy loss functions
├── metrics.rs              # MAE, AUC-ROC, accuracy metrics
├── training.rs             # Training loop (Adam + cosine annealing + validation)
├── batch_inference.rs      # Batch CSV inference with metrics + output
├── saliency.rs             # ViT attention → 3D saliency maps → NIfTI
├── model/
│   ├── patch_embed_3d.rs   # 3D patch embedding (Conv3d as unfolded matmul)
│   ├── vit.rs              # LayerNorm, Attention, MLP, ViTBlock
│   ├── backbone.rs         # Full ViT backbone (first-patch feature extraction)
│   ├── classifier.rs       # Task heads (single/dual/quad scan)
│   └── segmentation.rs     # UNETR-style segmentation decoder
├── bin/
│   ├── infer.rs            # CLI inference
│   ├── train.rs            # CLI training / fine-tuning
│   └── download_weights.rs # Weight download helper
examples/
├── extract_features.rs     # Feature extraction example
├── train_brainage.rs       # Brain age fine-tuning example
├── bench_all.rs            # Comprehensive benchmark (CPU + GPU)
└── benchmark.rs            # Simple throughput benchmark
scripts/
├── export_safetensors.py   # PyTorch → safetensors converter
└── generate_parity_vectors.py  # Generate numerical parity test vectors
tests/
├── model_test.rs           # Model component tests (8 tests)
├── nifti_test.rs           # Preprocessing tests (3 tests)
└── parity.rs               # Numerical parity vs Python MONAI (6 tests)
data/
└── sample_train_config.json # Example training config
```

## Testing

```bash
# Run all 34 tests
cargo test --release

# Parity tests only (requires python3 scripts/generate_parity_vectors.py first)
cargo test --release --test parity -- --nocapture

# Model component tests
cargo test --release --test model_test

# Benchmarks
cargo run --release --example bench_all
```

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

Apache-2.0. The original BrainIAC model weights are licensed for non-commercial academic research use only — see the [BrainIAC repository](https://github.com/YourUsername/BrainIAC_V2) for details.
