/// Load pretrained BrainIAC weights from safetensors files.
///
/// BrainIAC weights originate as PyTorch Lightning SimCLR checkpoints.
/// Use `scripts/export_safetensors.py` to convert `.ckpt` → `.safetensors`.
///
/// MONAI ViT key patterns (after stripping `backbone.` prefix):
/// ```text
/// patch_embedding.cls_token                     [1, 1, 768]
/// patch_embedding.position_embeddings           [1, 217, 768]
/// patch_embedding.patch_embeddings.0.weight     [768, 1, 16, 16, 16]
/// patch_embedding.patch_embeddings.0.bias       [768]
/// blocks.{i}.norm1.weight                       [768]
/// blocks.{i}.norm1.bias                         [768]
/// blocks.{i}.attn.qkv.weight                   [2304, 768]
/// blocks.{i}.attn.qkv.bias                     [2304]
/// blocks.{i}.attn.out_proj.weight               [768, 768]
/// blocks.{i}.attn.out_proj.bias                 [768]
/// blocks.{i}.norm2.weight                       [768]
/// blocks.{i}.norm2.bias                         [768]
/// blocks.{i}.mlp.linear1.weight                 [3072, 768]
/// blocks.{i}.mlp.linear1.bias                   [3072]
/// blocks.{i}.mlp.linear2.weight                 [768, 3072]
/// blocks.{i}.mlp.linear2.bias                   [768]
/// norm.weight                                   [768]
/// norm.bias                                     [768]
/// ```

use std::collections::HashMap;
use anyhow::{Result, bail};
use burn::prelude::*;

/// Raw weight store loaded from safetensors.
pub struct WeightMap {
    pub tensors: HashMap<String, (Vec<f32>, Vec<usize>)>,
}

impl WeightMap {
    /// Load all tensors from a safetensors file.
    pub fn from_file(path: &str) -> Result<Self> {
        let bytes = std::fs::read(path)
            .map_err(|e| anyhow::anyhow!("failed to read {}: {}", path, e))?;
        let st = safetensors::SafeTensors::deserialize(&bytes)?;
        let mut tensors = HashMap::with_capacity(st.len());

        for (raw_key, view) in st.tensors() {
            // Strip common prefixes from PyTorch Lightning checkpoints
            let key = raw_key
                .strip_prefix("model.")
                .or_else(|| raw_key.strip_prefix("backbone."))
                .unwrap_or(raw_key.as_str())
                .to_string();

            let shape: Vec<usize> = view.shape().to_vec();
            let data = view.data();

            let f32s: Vec<f32> = match view.dtype() {
                safetensors::Dtype::BF16 => data
                    .chunks_exact(2)
                    .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
                    .collect(),
                safetensors::Dtype::F16 => data
                    .chunks_exact(2)
                    .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
                    .collect(),
                safetensors::Dtype::F32 => data
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect(),
                other => bail!("unsupported dtype {:?} for key {}", other, key),
            };

            tensors.insert(key, (f32s, shape));
        }

        Ok(Self { tensors })
    }

    /// Take a tensor by key, removing it from the map (zero-copy).
    pub fn take<B: Backend, const N: usize>(
        &mut self,
        key: &str,
        device: &B::Device,
    ) -> Result<Tensor<B, N>> {
        let (data, shape) = self.tensors.remove(key)
            .ok_or_else(|| anyhow::anyhow!("weight key not found: {}", key))?;
        if shape.len() != N {
            bail!("rank mismatch for {}: expected {}, got {}", key, N, shape.len());
        }
        Ok(Tensor::<B, N>::from_data(TensorData::new(data, shape), device))
    }

    /// Take a tensor and flatten to 2D [first_dim, rest].
    pub fn take_2d_flat<B: Backend>(
        &mut self,
        key: &str,
        device: &B::Device,
    ) -> Result<Tensor<B, 2>> {
        let (data, shape) = self.tensors.remove(key)
            .ok_or_else(|| anyhow::anyhow!("weight key not found: {}", key))?;
        let first = shape[0];
        let rest: usize = shape[1..].iter().product();
        Ok(Tensor::<B, 2>::from_data(TensorData::new(data, [first, rest]), device))
    }

    /// Check if a key exists.
    pub fn has(&self, key: &str) -> bool {
        self.tensors.contains_key(key)
    }

    /// Print all remaining keys with shapes.
    pub fn print_keys(&self) {
        let mut keys: Vec<&str> = self.tensors.keys().map(String::as_str).collect();
        keys.sort();
        for k in keys {
            let (_, s) = &self.tensors[k];
            eprintln!("  {k:70}  {s:?}");
        }
    }

    /// Number of remaining tensors.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }
}

// ── Weight assignment helpers ─────────────────────────────────────────────────

/// Set a burn Linear's weight (PyTorch [out, in] → burn [in, out]) and bias.
pub fn set_linear_wb<B: Backend>(
    linear: &mut burn::nn::Linear<B>,
    w: Tensor<B, 2>,
    b: Tensor<B, 1>,
) {
    linear.weight = linear.weight.clone().map(|_| w.transpose());
    if let Some(ref bias) = linear.bias {
        linear.bias = Some(bias.clone().map(|_| b));
    }
}

/// Set a burn Linear's weight only (no bias).
pub fn set_linear_w<B: Backend>(
    linear: &mut burn::nn::Linear<B>,
    w: Tensor<B, 2>,
) {
    linear.weight = linear.weight.clone().map(|_| w.transpose());
}

use crate::model::vit::LayerNorm;
use crate::model::backbone::ViTBackbone;

/// Set LayerNorm weight and bias.
pub fn set_layernorm<B: Backend>(
    norm: &mut LayerNorm<B>,
    w: Tensor<B, 1>,
    b: Tensor<B, 1>,
) {
    norm.weight = norm.weight.clone().map(|_| w);
    norm.bias = norm.bias.clone().map(|_| b);
}

/// Load backbone weights from a WeightMap into a ViTBackbone.
///
/// Handles two naming conventions:
/// - MONAI direct: `patch_embedding.patch_embeddings.weight` (no `.0.`)
/// - SimCLR wrapped: `patch_embedding.patch_embeddings.0.weight` (with `.0.`)
pub fn load_backbone_weights<B: Backend>(
    wm: &mut WeightMap,
    model: &mut ViTBackbone<B>,
    device: &B::Device,
) -> Result<()> {
    // ── Position embeddings [1, 216, 768] ────────────────────────
    if let Ok(t) = wm.take::<B, 3>("patch_embedding.position_embeddings", device) {
        model.pos_embed = model.pos_embed.clone().map(|_| t);
    }

    // ── Patch embedding Conv3d ────────────────────────────────────
    // Try both MONAI key variants
    let pe_w_key = if wm.has("patch_embedding.patch_embeddings.weight") {
        "patch_embedding.patch_embeddings.weight"
    } else {
        "patch_embedding.patch_embeddings.0.weight"
    };
    if let Ok(w) = wm.take_2d_flat::<B>(pe_w_key, device) {
        model.patch_embed.proj_weight = model.patch_embed.proj_weight.clone().map(|_| w);
    }
    let pe_b_key = if wm.has("patch_embedding.patch_embeddings.bias") {
        "patch_embedding.patch_embeddings.bias"
    } else {
        "patch_embedding.patch_embeddings.0.bias"
    };
    if let Ok(b) = wm.take::<B, 1>(pe_b_key, device) {
        model.patch_embed.proj_bias = model.patch_embed.proj_bias.clone().map(|_| b);
    }

    // ── Transformer blocks ───────────────────────────────────────
    for (i, block) in model.blocks.iter_mut().enumerate() {
        let p = format!("blocks.{i}");

        // norm1
        if let (Ok(w), Ok(b)) = (
            wm.take::<B, 1>(&format!("{p}.norm1.weight"), device),
            wm.take::<B, 1>(&format!("{p}.norm1.bias"), device),
        ) {
            set_layernorm(&mut block.norm1, w, b);
        }

        // Attention: fused QKV (MONAI has no QKV bias by default)
        if let Ok(w) = wm.take::<B, 2>(&format!("{p}.attn.qkv.weight"), device) {
            if let Ok(b) = wm.take::<B, 1>(&format!("{p}.attn.qkv.bias"), device) {
                set_linear_wb(&mut block.attn.qkv, w, b);
            } else {
                set_linear_w(&mut block.attn.qkv, w);
            }
        }

        // Attention: output projection
        if let (Ok(w), Ok(b)) = (
            wm.take::<B, 2>(&format!("{p}.attn.out_proj.weight"), device),
            wm.take::<B, 1>(&format!("{p}.attn.out_proj.bias"), device),
        ) {
            set_linear_wb(&mut block.attn.proj, w, b);
        }

        // norm2
        if let (Ok(w), Ok(b)) = (
            wm.take::<B, 1>(&format!("{p}.norm2.weight"), device),
            wm.take::<B, 1>(&format!("{p}.norm2.bias"), device),
        ) {
            set_layernorm(&mut block.norm2, w, b);
        }

        // MLP: MONAI uses linear1/linear2 naming
        if let (Ok(w), Ok(b)) = (
            wm.take::<B, 2>(&format!("{p}.mlp.linear1.weight"), device),
            wm.take::<B, 1>(&format!("{p}.mlp.linear1.bias"), device),
        ) {
            set_linear_wb(&mut block.mlp.fc1, w, b);
        }

        if let (Ok(w), Ok(b)) = (
            wm.take::<B, 2>(&format!("{p}.mlp.linear2.weight"), device),
            wm.take::<B, 1>(&format!("{p}.mlp.linear2.bias"), device),
        ) {
            set_linear_wb(&mut block.mlp.fc2, w, b);
        }
    }

    // ── Final norm ───────────────────────────────────────────────
    if let (Ok(w), Ok(b)) = (
        wm.take::<B, 1>("norm.weight", device),
        wm.take::<B, 1>("norm.bias", device),
    ) {
        set_layernorm(&mut model.norm, w, b);
    }

    Ok(())
}

/// Load a simple linear classifier head weights.
pub fn load_classifier_weights<B: Backend>(
    wm: &mut WeightMap,
    linear: &mut burn::nn::Linear<B>,
    prefix: &str,
    device: &B::Device,
) -> Result<()> {
    if let (Ok(w), Ok(b)) = (
        wm.take::<B, 2>(&format!("{prefix}.weight"), device),
        wm.take::<B, 1>(&format!("{prefix}.bias"), device),
    ) {
        set_linear_wb(linear, w, b);
    }
    Ok(())
}
