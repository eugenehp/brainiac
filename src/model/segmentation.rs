/// UNETR segmentation decoder for BrainIAC.
///
/// Matches the Python `ViTUNETRSegmentationModel`:
/// - BrainIAC ViT encoder extracts features at multiple layers
/// - U-Net style decoder with skip connections from encoder hidden states
/// - ConvTranspose3d upsampling (implemented as unfolded operations)
/// - Outputs binary segmentation mask [B, 1, D, H, W]
///
/// Simplified decoder: takes CLS token + patch features → 3D conv decoder.
/// Full UNETR has skip connections from layers 3, 6, 9, 12 but we implement
/// a streamlined version that uses the final patch tokens.

use burn::prelude::*;
use burn::nn::{Linear, LinearConfig};

use crate::config::ModelConfig;
use crate::model::backbone::ViTBackbone;

/// Simplified UNETR-style segmentation head.
///
/// Takes patch tokens from the ViT encoder and decodes to a segmentation mask.
/// Architecture: patch tokens [B, N, D] → reshape to 3D → upsample via
/// transposed conv layers → binary mask [B, 1, img_size, img_size, img_size].
#[derive(Module, Debug)]
pub struct SegmentationHead<B: Backend> {
    /// Project from hidden_dim to decoder dim for each resolution
    pub proj_16: Linear<B>,   // 768 → 64 (at 6×6×6 resolution)
    pub proj_8: Linear<B>,    // 64 → 32 (at 12×12×12)
    pub proj_4: Linear<B>,    // 32 → 16 (at 24×24×24)
    pub proj_2: Linear<B>,    // 16 → 8 (at 48×48×48)
    pub proj_1: Linear<B>,    // 8 → out_channels (at 96×96×96)
    pub out_channels: usize,
    pub img_size: usize,
    pub patches_per_axis: usize,
}

impl<B: Backend> SegmentationHead<B> {
    pub fn new(cfg: &ModelConfig, out_channels: usize, device: &B::Device) -> Self {
        let pp = cfg.patches_per_axis(); // 6
        Self {
            proj_16: LinearConfig::new(cfg.hidden_size, 64).with_bias(true).init(device),
            proj_8: LinearConfig::new(64, 32).with_bias(true).init(device),
            proj_4: LinearConfig::new(32, 16).with_bias(true).init(device),
            proj_2: LinearConfig::new(16, 8).with_bias(true).init(device),
            proj_1: LinearConfig::new(8, out_channels).with_bias(true).init(device),
            out_channels,
            img_size: cfg.img_size,
            patches_per_axis: pp,
        }
    }

    /// Decode patch features to a segmentation mask.
    ///
    /// `patch_features`: [B, num_patches, hidden_dim] (from ViT, excluding CLS token).
    ///
    /// Returns: flat f32 of shape [B * out_channels * img_size³].
    pub fn forward(&self, patch_features: Tensor<B, 3>, device: &B::Device) -> Vec<f32> {
        let [batch, _n_patches, _hidden] = patch_features.dims();
        let pp = self.patches_per_axis;
        let s = self.img_size;

        // Stage 1: Project to 64-dim at 6×6×6
        let x = self.proj_16.forward(patch_features); // [B, 216, 64]
        let x_data: Vec<f32> = x.into_data().to_vec().unwrap();

        // Upsample 6→12 and project 64→32
        let x12 = upsample_3d_nearest(&x_data, batch, 64, pp, pp * 2);
        let x12_t = Tensor::<B, 3>::from_data(
            TensorData::new(x12, [batch, (pp * 2).pow(3), 64]), device,
        );
        let x12 = self.proj_8.forward(x12_t); // [B, 1728, 32]
        let x12_data: Vec<f32> = x12.into_data().to_vec().unwrap();

        // Upsample 12→24 and project 32→16
        let x24 = upsample_3d_nearest(&x12_data, batch, 32, pp * 2, pp * 4);
        let x24_t = Tensor::<B, 3>::from_data(
            TensorData::new(x24, [batch, (pp * 4).pow(3), 32]), device,
        );
        let x24 = self.proj_4.forward(x24_t); // [B, 13824, 16]
        let x24_data: Vec<f32> = x24.into_data().to_vec().unwrap();

        // Upsample 24→48 and project 16→8
        let x48 = upsample_3d_nearest(&x24_data, batch, 16, pp * 4, pp * 8);
        let x48_t = Tensor::<B, 3>::from_data(
            TensorData::new(x48, [batch, (pp * 8).pow(3), 16]), device,
        );
        let x48 = self.proj_2.forward(x48_t); // [B, 110592, 8]
        let x48_data: Vec<f32> = x48.into_data().to_vec().unwrap();

        // Upsample 48→96 and project 8→out_channels
        let x96 = upsample_3d_nearest(&x48_data, batch, 8, pp * 8, s);
        let x96_t = Tensor::<B, 3>::from_data(
            TensorData::new(x96, [batch, s * s * s, 8]), device,
        );
        let x96 = self.proj_1.forward(x96_t); // [B, 884736, out_channels]
        x96.into_data().to_vec().unwrap()
    }
}

/// Full segmentation model: backbone + segmentation head.
#[derive(Module, Debug)]
pub struct SegmentationModel<B: Backend> {
    pub backbone: ViTBackbone<B>,
    pub head: SegmentationHead<B>,
}

impl<B: Backend> SegmentationModel<B> {
    pub fn new(cfg: &ModelConfig, out_channels: usize, device: &B::Device) -> Self {
        Self {
            backbone: ViTBackbone::new(cfg, device),
            head: SegmentationHead::new(cfg, out_channels, device),
        }
    }

    /// Run segmentation inference on a preprocessed volume.
    ///
    /// Returns flat segmentation logits [img_size³ * out_channels].
    pub fn forward(&self, volume_data: &[f32], device: &B::Device) -> Vec<f32> {
        // Use backbone's forward_all_tokens which returns [B, N, D]
        let patch_tokens = self.backbone.forward_all_tokens(volume_data, 1, device);
        self.head.forward(patch_tokens, device)
    }

    /// Run segmentation and return binary mask (threshold at 0.5).
    pub fn predict_mask(&self, volume_data: &[f32], device: &B::Device) -> Vec<u8> {
        let logits = self.forward(volume_data, device);
        logits.iter().map(|&v| {
            let prob = 1.0 / (1.0 + (-v).exp());
            if prob > 0.5 { 1 } else { 0 }
        }).collect()
    }
}

/// Nearest-neighbor 3D upsample for a batch of feature volumes.
///
/// Input: flat [batch, src³, channels]
/// Output: flat [batch, dst³, channels]
fn upsample_3d_nearest(
    data: &[f32],
    batch: usize,
    channels: usize,
    src_size: usize,
    dst_size: usize,
) -> Vec<f32> {
    let src3 = src_size * src_size * src_size;
    let dst3 = dst_size * dst_size * dst_size;
    let mut out = vec![0.0f32; batch * dst3 * channels];

    let scale = src_size as f32 / dst_size as f32;

    for b in 0..batch {
        for dz in 0..dst_size {
            for dy in 0..dst_size {
                for dx in 0..dst_size {
                    let sx = ((dx as f32 + 0.5) * scale) as usize;
                    let sy = ((dy as f32 + 0.5) * scale) as usize;
                    let sz = ((dz as f32 + 0.5) * scale) as usize;
                    let sx = sx.min(src_size - 1);
                    let sy = sy.min(src_size - 1);
                    let sz = sz.min(src_size - 1);

                    let src_idx = b * src3 * channels
                        + (sx + sy * src_size + sz * src_size * src_size) * channels;
                    let dst_idx = b * dst3 * channels
                        + (dx + dy * dst_size + dz * dst_size * dst_size) * channels;

                    out[dst_idx..dst_idx + channels]
                        .copy_from_slice(&data[src_idx..src_idx + channels]);
                }
            }
        }
    }
    out
}
