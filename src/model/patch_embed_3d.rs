/// 3D patch embedding for volumetric brain MRI.
///
/// Converts a brain volume [B, 1, D, H, W] into patch tokens [B, N, hidden_dim]
/// via a flattened 3D convolution (kernel = patch_size³, stride = patch_size³).
///
/// For BrainIAC with 96³ input and patch_size=16:
///   96/16 = 6 patches per axis → 6×6×6 = 216 patches
///   kernel_vol = 1 × 16 × 16 × 16 = 4096
///   Output: [B, 216, 768]

use burn::prelude::*;
use burn::module::{Param, ParamId};

/// 3D Patch Embedding via unfolded linear projection.
///
/// Instead of using burn's Conv3d (which may not be available on all backends),
/// we manually extract non-overlapping 3D patches and project with a linear layer.
/// This matches MONAI's Conv3d(in_channels, hidden_size, kernel=patch, stride=patch).
#[derive(Module, Debug)]
pub struct PatchEmbed3d<B: Backend> {
    /// Projection weight: [hidden_dim, in_channels * patch³]
    pub proj_weight: Param<Tensor<B, 2>>,
    /// Projection bias: [hidden_dim]
    pub proj_bias: Param<Tensor<B, 1>>,
    // Config
    pub hidden_dim: usize,
    pub in_channels: usize,
    pub patch_size: usize,
    pub img_size: usize,
}

impl<B: Backend> PatchEmbed3d<B> {
    pub fn new(
        in_channels: usize,
        hidden_dim: usize,
        img_size: usize,
        patch_size: usize,
        device: &B::Device,
    ) -> Self {
        let kernel_vol = in_channels * patch_size * patch_size * patch_size;
        Self {
            proj_weight: Param::initialized(
                ParamId::new(),
                Tensor::zeros([hidden_dim, kernel_vol], device),
            ),
            proj_bias: Param::initialized(
                ParamId::new(),
                Tensor::zeros([hidden_dim], device),
            ),
            hidden_dim,
            in_channels,
            patch_size,
            img_size,
        }
    }

    /// Number of patches per axis.
    pub fn patches_per_axis(&self) -> usize {
        self.img_size / self.patch_size
    }

    /// Total number of patches.
    pub fn num_patches(&self) -> usize {
        let p = self.patches_per_axis();
        p * p * p
    }

    /// Forward pass: extract 3D patches and project.
    ///
    /// Input: `volume` [B, C, D, H, W] as flat f32 data packed into a 3D tensor
    ///        via reshape. We expect the caller to provide [B, n_voxels] where
    ///        n_voxels = C * D * H * W.
    ///
    /// Actually we work with [B, C*D*H*W] and extract patches manually.
    ///
    /// Returns: [B, N, hidden_dim] where N = num_patches.
    pub fn forward_from_flat(&self, volume_data: &[f32], batch_size: usize, device: &B::Device) -> Tensor<B, 3> {
        let ps = self.patch_size;
        let c = self.in_channels;
        let s = self.img_size;
        let pp = self.patches_per_axis();
        let n_patches = self.num_patches();
        let kernel_vol = c * ps * ps * ps;

        // Extract patches: [B, N, kernel_vol]
        let mut patches = vec![0.0f32; batch_size * n_patches * kernel_vol];

        for bi in 0..batch_size {
            for pz in 0..pp {
                for py in 0..pp {
                    for px in 0..pp {
                        let patch_idx = pz * pp * pp + py * pp + px;
                        for cc in 0..c {
                            for dz in 0..ps {
                                for dy in 0..ps {
                                    for dx in 0..ps {
                                        let src_x = px * ps + dx;
                                        let src_y = py * ps + dy;
                                        let src_z = pz * ps + dz;
                                        // Volume layout: [B, C, D, H, W]
                                        // flat index: bi * (C*D*H*W) + cc * (D*H*W) + z*(H*W) + y*W + x
                                        let src_idx = bi * (c * s * s * s)
                                            + cc * (s * s * s)
                                            + src_z * (s * s)
                                            + src_y * s
                                            + src_x;
                                        // Patch layout: [B, N, C*ps*ps*ps]
                                        // Following PyTorch Conv3d weight layout:
                                        // kernel index: cc * (ps*ps*ps) + dz*(ps*ps) + dy*ps + dx
                                        let kern_idx = cc * (ps * ps * ps)
                                            + dz * (ps * ps)
                                            + dy * ps
                                            + dx;
                                        let dst_idx = bi * (n_patches * kernel_vol)
                                            + patch_idx * kernel_vol
                                            + kern_idx;
                                        patches[dst_idx] = volume_data[src_idx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        let patches_tensor = Tensor::<B, 3>::from_data(
            TensorData::new(patches, [batch_size, n_patches, kernel_vol]),
            device,
        );

        // Linear projection: [B, N, kernel_vol] @ [kernel_vol, hidden_dim] + bias
        let weight_t = self.proj_weight.val().transpose(); // [kernel_vol, hidden_dim]
        let bias = self.proj_bias.val()
            .unsqueeze_dim::<2>(0)
            .unsqueeze_dim::<3>(0); // [1, 1, hidden_dim]

        patches_tensor.matmul(weight_t.unsqueeze::<3>()) + bias
    }
}
