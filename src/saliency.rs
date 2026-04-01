/// Saliency map generation from ViT attention weights.
///
/// Matches the Python `extract_attention_map()` function:
/// 1. Forward pass through ViT, collecting per-layer attention weights
/// 2. Select a layer (default: last)
/// 3. Average attention across heads → CLS-to-patch attention vector
/// 4. Reshape 1D attention → 3D volume (6×6×6 for 96³ input)
/// 5. Trilinear upsample to full image resolution (96×96×96)
/// 6. Normalize to [0, 1]
/// 7. Optionally save as NIfTI

use std::path::Path;
use anyhow::Result;
use burn::prelude::*;

use crate::config::ModelConfig;
use crate::model::backbone::ViTBackbone;
use crate::nifti;
use crate::preprocessing;

/// Generated saliency map.
#[derive(Debug, Clone)]
pub struct SaliencyMap {
    /// 3D saliency values normalized to [0, 1], shape [img_size³].
    pub data: Vec<f32>,
    /// Volume dimensions.
    pub dims: [usize; 3],
    /// Which layer was used (-1 = last).
    pub layer_index: usize,
}

impl SaliencyMap {
    /// Save as NIfTI .nii.gz file.
    pub fn save_nifti(&self, path: &Path) -> Result<()> {
        let voxel_size = [1.0f32, 1.0, 1.0]; // identity affine
        nifti::write_nifti(path, &self.data, self.dims, voxel_size)
    }
}

/// Generate a saliency map from a preprocessed volume using ViT attention.
///
/// `backbone`: loaded ViT backbone with pretrained weights.
/// `volume_data`: preprocessed flat f32 volume [C*D*H*W].
/// `layer_idx`: transformer layer to visualize. Negative indices count from end.
/// `cfg`: model config for dimensions.
pub fn generate_saliency<B: Backend>(
    backbone: &ViTBackbone<B>,
    volume_data: &[f32],
    layer_idx: isize,
    cfg: &ModelConfig,
    device: &B::Device,
) -> Result<SaliencyMap> {
    let num_layers = cfg.num_layers;
    let patches_per_dim = cfg.patches_per_axis();
    let num_patches = cfg.num_patches();
    let img_size = cfg.img_size;

    // Resolve negative layer index
    let layer = if layer_idx < 0 {
        (num_layers as isize + layer_idx) as usize
    } else {
        layer_idx as usize
    };
    if layer >= num_layers {
        anyhow::bail!("Layer index {} out of range (model has {} layers)", layer, num_layers);
    }

    // Forward pass with attention
    let (_features, attn_maps) = backbone.forward_with_attn(volume_data, 1, device);

    // attn_maps[layer]: [1, num_heads, num_patches, num_patches]
    // MONAI ViT has no CLS token, so seq_len = num_patches
    let layer_attn = &attn_maps[layer];
    let attn_data: Vec<f32> = layer_attn.clone().into_data().to_vec().unwrap();

    let num_heads = cfg.num_heads;
    let seq_len = num_patches;

    // Average across heads: mean over dim 1
    let mut avg_attn = vec![0.0f32; seq_len * seq_len];
    for h in 0..num_heads {
        for i in 0..seq_len {
            for j in 0..seq_len {
                avg_attn[i * seq_len + j] += attn_data[h * seq_len * seq_len + i * seq_len + j];
            }
        }
    }
    for v in &mut avg_attn {
        *v /= num_heads as f32;
    }

    // Extract first-patch attention to all patches: row 0
    // (BrainIAC uses features[:, 0] so we use attention from patch 0)
    let mut cls_attn = Vec::with_capacity(num_patches);
    for j in 0..seq_len {
        cls_attn.push(avg_attn[j]); // row 0, col j
    }

    // Reshape to 3D: [patches_per_dim, patches_per_dim, patches_per_dim]
    // Then upsample to [img_size, img_size, img_size] via trilinear interpolation
    let upsampled = trilinear_upsample_3d(
        &cls_attn,
        [patches_per_dim, patches_per_dim, patches_per_dim],
        [img_size, img_size, img_size],
    );

    // Normalize to [0, 1]
    let min_val = upsampled.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = upsampled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let range = (max_val - min_val).max(1e-8);
    let normalized: Vec<f32> = upsampled.iter().map(|&v| (v - min_val) / range).collect();

    Ok(SaliencyMap {
        data: normalized,
        dims: [img_size, img_size, img_size],
        layer_index: layer,
    })
}

/// Generate saliency from a NIfTI file path.
pub fn generate_saliency_from_nifti<B: Backend>(
    backbone: &ViTBackbone<B>,
    nifti_path: &Path,
    layer_idx: isize,
    cfg: &ModelConfig,
    device: &B::Device,
) -> Result<SaliencyMap> {
    let volume = nifti::read_nifti(nifti_path)?;
    let preprocessed = preprocessing::preprocess(&volume, cfg.img_size);
    generate_saliency(backbone, &preprocessed, layer_idx, cfg, device)
}

/// Trilinear upsample a 3D volume from src_dims to dst_dims.
fn trilinear_upsample_3d(
    data: &[f32],
    src_dims: [usize; 3],
    dst_dims: [usize; 3],
) -> Vec<f32> {
    let [sx, sy, sz] = src_dims;
    let [dx, dy, dz] = dst_dims;
    let mut out = vec![0.0f32; dx * dy * dz];

    for oz in 0..dz {
        for oy in 0..dy {
            for ox in 0..dx {
                // Map output coord to source coord (half-pixel centers)
                let fx = (ox as f64 + 0.5) * sx as f64 / dx as f64 - 0.5;
                let fy = (oy as f64 + 0.5) * sy as f64 / dy as f64 - 0.5;
                let fz = (oz as f64 + 0.5) * sz as f64 / dz as f64 - 0.5;

                let fx = fx.max(0.0).min((sx - 1) as f64);
                let fy = fy.max(0.0).min((sy - 1) as f64);
                let fz = fz.max(0.0).min((sz - 1) as f64);

                let x0 = fx.floor() as usize;
                let y0 = fy.floor() as usize;
                let z0 = fz.floor() as usize;
                let x1 = (x0 + 1).min(sx - 1);
                let y1 = (y0 + 1).min(sy - 1);
                let z1 = (z0 + 1).min(sz - 1);

                let xd = (fx - x0 as f64) as f32;
                let yd = (fy - y0 as f64) as f32;
                let zd = (fz - z0 as f64) as f32;

                let idx = |x: usize, y: usize, z: usize| x + y * sx + z * sx * sy;

                let c000 = data[idx(x0, y0, z0)];
                let c100 = data[idx(x1, y0, z0)];
                let c010 = data[idx(x0, y1, z0)];
                let c110 = data[idx(x1, y1, z0)];
                let c001 = data[idx(x0, y0, z1)];
                let c101 = data[idx(x1, y0, z1)];
                let c011 = data[idx(x0, y1, z1)];
                let c111 = data[idx(x1, y1, z1)];

                let c00 = c000 * (1.0 - xd) + c100 * xd;
                let c01 = c001 * (1.0 - xd) + c101 * xd;
                let c10 = c010 * (1.0 - xd) + c110 * xd;
                let c11 = c011 * (1.0 - xd) + c111 * xd;
                let c0 = c00 * (1.0 - yd) + c10 * yd;
                let c1 = c01 * (1.0 - yd) + c11 * yd;
                let val = c0 * (1.0 - zd) + c1 * zd;

                out[ox + oy * dx + oz * dx * dy] = val;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trilinear_upsample_uniform() {
        let data = vec![1.0f32; 8]; // 2×2×2 all ones
        let up = trilinear_upsample_3d(&data, [2, 2, 2], [4, 4, 4]);
        assert_eq!(up.len(), 64);
        for &v in &up {
            assert!((v - 1.0).abs() < 1e-5, "expected 1.0, got {}", v);
        }
    }

    #[test]
    fn test_trilinear_upsample_gradient() {
        // 2×1×1 with values [0, 1] → 4×1×1 should interpolate
        let data = vec![0.0, 1.0];
        let up = trilinear_upsample_3d(&data, [2, 1, 1], [4, 1, 1]);
        assert_eq!(up.len(), 4);
        // Should be monotonically increasing
        for i in 1..4 {
            assert!(up[i] >= up[i-1], "should be monotonic: {:?}", up);
        }
    }
}
