/// ViT Backbone for BrainIAC — matches MONAI ViT exactly.
///
/// Architecture (NO CLS token, unlike standard ViT):
///   1. 3D Patch Embedding (Conv3d kernel=16³) → [B, 216, 768]
///   2. Add positional embeddings [1, 216, 768]
///   3. 12× ViT Encoder Blocks (pre-norm: LN → Attn → Res → LN → MLP → Res)
///   4. Final LayerNorm
///   5. Take first patch token as feature → [B, 768]
///
/// MONAI ViT (non-classification mode) does NOT have a CLS token.
/// Position embeddings: [1, 216, 768] (patches only, no CLS position).
/// BrainIAC Python takes `features[0][:, 0]` = first patch token.

use burn::prelude::*;
use burn::module::{Param, ParamId};

use crate::config::ModelConfig;
use crate::model::patch_embed_3d::PatchEmbed3d;
use crate::model::vit::{ViTBlock, LayerNorm};

/// BrainIAC ViT backbone — extracts a 768-dim feature from a 96³ brain volume.
#[derive(Module, Debug)]
pub struct ViTBackbone<B: Backend> {
    pub patch_embed: PatchEmbed3d<B>,
    /// Positional embeddings: [1, num_patches, hidden_dim]
    pub pos_embed: Param<Tensor<B, 3>>,
    /// Transformer encoder blocks
    pub blocks: Vec<ViTBlock<B>>,
    /// Final layer norm
    pub norm: LayerNorm<B>,
    // Config
    pub hidden_dim: usize,
    pub num_patches: usize,
}

impl<B: Backend> ViTBackbone<B> {
    /// Create a new backbone with zero-initialized weights.
    pub fn new(cfg: &ModelConfig, device: &B::Device) -> Self {
        let num_patches = cfg.num_patches();

        let patch_embed = PatchEmbed3d::new(
            cfg.in_channels,
            cfg.hidden_size,
            cfg.img_size,
            cfg.patch_size,
            device,
        );

        let pos_embed = Param::initialized(
            ParamId::new(),
            Tensor::zeros([1, num_patches, cfg.hidden_size], device),
        );

        let blocks: Vec<ViTBlock<B>> = (0..cfg.num_layers)
            .map(|_| ViTBlock::new(
                cfg.hidden_size,
                cfg.num_heads,
                cfg.mlp_dim,
                cfg.norm_eps,
                device,
            ))
            .collect();

        let norm = LayerNorm::new(cfg.hidden_size, cfg.norm_eps, device);

        Self {
            patch_embed,
            pos_embed,
            blocks,
            norm,
            hidden_dim: cfg.hidden_size,
            num_patches,
        }
    }

    /// Build a backbone directly from a WeightMap, skipping zero-init.
    ///
    /// This is significantly faster on GPU because it avoids allocating
    /// ~75 zero tensors on the device only to immediately overwrite them.
    /// Instead, each tensor is created once from the weight data.
    pub fn from_weights(
        cfg: &ModelConfig,
        wm: &mut crate::weights::WeightMap,
        device: &B::Device,
    ) -> anyhow::Result<Self> {
        use crate::weights;
        let num_patches = cfg.num_patches();

        // Patch embedding
        let pe_w_key = if wm.has("patch_embedding.patch_embeddings.weight") {
            "patch_embedding.patch_embeddings.weight"
        } else {
            "patch_embedding.patch_embeddings.0.weight"
        };
        let pe_b_key = if wm.has("patch_embedding.patch_embeddings.bias") {
            "patch_embedding.patch_embeddings.bias"
        } else {
            "patch_embedding.patch_embeddings.0.bias"
        };
        let proj_weight = wm.take_2d_flat::<B>(pe_w_key, device)?;
        let proj_bias = wm.take::<B, 1>(pe_b_key, device)?;

        let patch_embed = PatchEmbed3d {
            proj_weight: Param::initialized(ParamId::new(), proj_weight),
            proj_bias: Param::initialized(ParamId::new(), proj_bias),
            hidden_dim: cfg.hidden_size,
            in_channels: cfg.in_channels,
            patch_size: cfg.patch_size,
            img_size: cfg.img_size,
        };

        // Position embeddings
        let pos = wm.take::<B, 3>("patch_embedding.position_embeddings", device)?;
        let pos_embed = Param::initialized(ParamId::new(), pos);

        // Transformer blocks
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let p = format!("blocks.{i}");

            // Build each component from weights directly
            let norm1_w = wm.take::<B, 1>(&format!("{p}.norm1.weight"), device)?;
            let norm1_b = wm.take::<B, 1>(&format!("{p}.norm1.bias"), device)?;
            let mut norm1 = LayerNorm::new(cfg.hidden_size, cfg.norm_eps, device);
            norm1.weight = Param::initialized(ParamId::new(), norm1_w);
            norm1.bias = Param::initialized(ParamId::new(), norm1_b);

            let qkv_w = wm.take::<B, 2>(&format!("{p}.attn.qkv.weight"), device)?;
            let mut attn = crate::model::vit::ViTAttention::new(cfg.hidden_size, cfg.num_heads, device);
            weights::set_linear_w(&mut attn.qkv, qkv_w);
            if let Ok(qkv_b) = wm.take::<B, 1>(&format!("{p}.attn.qkv.bias"), device) {
                if let Some(ref bias) = attn.qkv.bias {
                    attn.qkv.bias = Some(bias.clone().map(|_| qkv_b));
                }
            }
            let proj_w = wm.take::<B, 2>(&format!("{p}.attn.out_proj.weight"), device)?;
            let proj_b = wm.take::<B, 1>(&format!("{p}.attn.out_proj.bias"), device)?;
            weights::set_linear_wb(&mut attn.proj, proj_w, proj_b);

            let norm2_w = wm.take::<B, 1>(&format!("{p}.norm2.weight"), device)?;
            let norm2_b = wm.take::<B, 1>(&format!("{p}.norm2.bias"), device)?;
            let mut norm2 = LayerNorm::new(cfg.hidden_size, cfg.norm_eps, device);
            norm2.weight = Param::initialized(ParamId::new(), norm2_w);
            norm2.bias = Param::initialized(ParamId::new(), norm2_b);

            let fc1_w = wm.take::<B, 2>(&format!("{p}.mlp.linear1.weight"), device)?;
            let fc1_b = wm.take::<B, 1>(&format!("{p}.mlp.linear1.bias"), device)?;
            let fc2_w = wm.take::<B, 2>(&format!("{p}.mlp.linear2.weight"), device)?;
            let fc2_b = wm.take::<B, 1>(&format!("{p}.mlp.linear2.bias"), device)?;
            let mut mlp = crate::model::vit::ViTMlp::new(cfg.hidden_size, cfg.mlp_dim, device);
            weights::set_linear_wb(&mut mlp.fc1, fc1_w, fc1_b);
            weights::set_linear_wb(&mut mlp.fc2, fc2_w, fc2_b);

            blocks.push(crate::model::vit::ViTBlock { norm1, attn, norm2, mlp });
        }

        // Final norm
        let norm_w = wm.take::<B, 1>("norm.weight", device)?;
        let norm_b = wm.take::<B, 1>("norm.bias", device)?;
        let mut norm = LayerNorm::new(cfg.hidden_size, cfg.norm_eps, device);
        norm.weight = Param::initialized(ParamId::new(), norm_w);
        norm.bias = Param::initialized(ParamId::new(), norm_b);

        Ok(Self {
            patch_embed,
            pos_embed,
            blocks,
            norm,
            hidden_dim: cfg.hidden_size,
            num_patches,
        })
    }

    /// Forward pass: volume data → first-patch features [B, hidden_dim].
    ///
    /// Matches MONAI ViT: no CLS token, takes features[0][:, 0].
    pub fn forward(&self, volume_data: &[f32], batch_size: usize, device: &B::Device) -> Tensor<B, 2> {
        let b = batch_size;

        // 1. Patch embedding: → [B, N, D]
        let x = self.patch_embed.forward_from_flat(volume_data, batch_size, device);

        // 2. Add positional embeddings
        let x = x + self.pos_embed.val();

        // 3. Transformer blocks
        let mut x = x;
        for block in &self.blocks {
            x = block.forward(x);
        }

        // 4. Final norm
        let x = self.norm.forward(x);

        // 5. Take first patch token: [B, N, D] → [B, D]
        x.narrow(1, 0, 1).reshape([b, self.hidden_dim])
    }

    /// Forward returning all hidden states and per-layer attention weights.
    pub fn forward_with_attn(
        &self,
        volume_data: &[f32],
        batch_size: usize,
        device: &B::Device,
    ) -> (Tensor<B, 2>, Vec<Tensor<B, 4>>) {
        let b = batch_size;

        let x = self.patch_embed.forward_from_flat(volume_data, batch_size, device);
        let x = x + self.pos_embed.val();

        let mut x = x;
        let mut all_attn = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            let (out, attn) = block.forward_with_attn(x);
            x = out;
            all_attn.push(attn);
        }

        let x = self.norm.forward(x);
        let features = x.narrow(1, 0, 1).reshape([b, self.hidden_dim]);

        (features, all_attn)
    }

    /// Forward returning the full patch token sequence [B, N, D] (for segmentation).
    pub fn forward_all_tokens(
        &self,
        volume_data: &[f32],
        batch_size: usize,
        device: &B::Device,
    ) -> Tensor<B, 3> {
        let x = self.patch_embed.forward_from_flat(volume_data, batch_size, device);
        let x = x + self.pos_embed.val();

        let mut x = x;
        for block in &self.blocks {
            x = block.forward(x);
        }
        self.norm.forward(x)
    }
}

/// Describe model size.
pub fn describe_model(cfg: &ModelConfig) -> String {
    let n_patches = cfg.num_patches();
    let params_per_block = {
        let d = cfg.hidden_size;
        let mlp = cfg.mlp_dim;
        2 * d + (3 * d * d + 3 * d) + (d * d + d) + 2 * d + (d * mlp + mlp) + (mlp * d + d)
    };
    let total_block_params = params_per_block * cfg.num_layers;
    let patch_params = cfg.hidden_size * cfg.kernel_volume() + cfg.hidden_size;
    let pos_params = n_patches * cfg.hidden_size;
    let norm_params = 2 * cfg.hidden_size;
    let total = total_block_params + patch_params + pos_params + norm_params;

    format!(
        "ViT-B/16³ ({}×{}×{} → {} patches, {} layers, {} heads, {:.1}M params)",
        cfg.img_size, cfg.img_size, cfg.img_size,
        n_patches,
        cfg.num_layers,
        cfg.num_heads,
        total as f64 / 1e6,
    )
}
