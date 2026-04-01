/// MRI preprocessing pipeline matching MONAI's validation transforms:
///
/// 1. Trilinear resize to target size (96×96×96)
/// 2. Z-score normalization (nonzero voxels only, channel-wise)
///
/// This matches the Python pipeline:
/// ```python
/// Resized(keys=["image"], spatial_size=(96,96,96), mode="trilinear")
/// NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True)
/// ```

use crate::nifti::NiftiVolume;

/// Preprocess a NIfTI volume for BrainIAC inference.
///
/// Returns a flat f32 array of shape [1, target, target, target] (with channel dim).
pub fn preprocess(volume: &NiftiVolume, target_size: usize) -> Vec<f32> {
    let resized = trilinear_resize(&volume.data, volume.dims, target_size);
    normalize_intensity(&resized)
}

/// Trilinear interpolation to resize a 3D volume.
///
/// Input: flat array with dims [nx, ny, nz] (row-major: x + y*nx + z*nx*ny).
/// Output: flat array with dims [target, target, target].
pub fn trilinear_resize(
    data: &[f32],
    src_dims: [usize; 3],
    target: usize,
) -> Vec<f32> {
    let [sx, sy, sz] = src_dims;
    let n = target * target * target;
    let mut out = vec![0.0f32; n];

    // Scale factors (align_corners=False style, matching PyTorch/MONAI default)
    let scale_x = sx as f64 / target as f64;
    let scale_y = sy as f64 / target as f64;
    let scale_z = sz as f64 / target as f64;

    for tz in 0..target {
        for ty in 0..target {
            for tx in 0..target {
                // Map target coordinate to source coordinate (center-aligned)
                let fx = (tx as f64 + 0.5) * scale_x - 0.5;
                let fy = (ty as f64 + 0.5) * scale_y - 0.5;
                let fz = (tz as f64 + 0.5) * scale_z - 0.5;

                // Clamp to valid range
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

                // Fetch 8 corners
                let c000 = data[x0 + y0 * sx + z0 * sx * sy];
                let c100 = data[x1 + y0 * sx + z0 * sx * sy];
                let c010 = data[x0 + y1 * sx + z0 * sx * sy];
                let c110 = data[x1 + y1 * sx + z0 * sx * sy];
                let c001 = data[x0 + y0 * sx + z1 * sx * sy];
                let c101 = data[x1 + y0 * sx + z1 * sx * sy];
                let c011 = data[x0 + y1 * sx + z1 * sx * sy];
                let c111 = data[x1 + y1 * sx + z1 * sx * sy];

                // Trilinear interpolation
                let c00 = c000 * (1.0 - xd) + c100 * xd;
                let c01 = c001 * (1.0 - xd) + c101 * xd;
                let c10 = c010 * (1.0 - xd) + c110 * xd;
                let c11 = c011 * (1.0 - xd) + c111 * xd;

                let c0 = c00 * (1.0 - yd) + c10 * yd;
                let c1 = c01 * (1.0 - yd) + c11 * yd;

                let val = c0 * (1.0 - zd) + c1 * zd;

                out[tx + ty * target + tz * target * target] = val;
            }
        }
    }

    out
}

/// Z-score normalize intensity values (nonzero voxels only).
///
/// Matches MONAI's `NormalizeIntensityd(nonzero=True, channel_wise=True)`:
/// - Compute mean and std over nonzero voxels
/// - Normalize all voxels: (v - mean) / std
/// - Zero voxels remain zero
fn normalize_intensity(data: &[f32]) -> Vec<f32> {
    // Compute mean over nonzero voxels
    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut count = 0u64;

    for &v in data {
        if v != 0.0 {
            sum += v as f64;
            sum_sq += (v as f64) * (v as f64);
            count += 1;
        }
    }

    if count == 0 {
        return data.to_vec();
    }

    let mean = sum / count as f64;
    let variance = sum_sq / count as f64 - mean * mean;
    let std = variance.sqrt().max(1e-8);

    data.iter()
        .map(|&v| {
            if v != 0.0 {
                ((v as f64 - mean) / std) as f32
            } else {
                0.0
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resize_identity() {
        // 4×4×4 volume resized to 4×4×4 should be ~identity
        let mut data = vec![0.0f32; 64];
        data[0] = 1.0;
        data[63] = 2.0;
        let resized = trilinear_resize(&data, [4, 4, 4], 4);
        assert_eq!(resized.len(), 64);
        // Corner values should be close to original
        assert!((resized[0] - 1.0).abs() < 0.5);
        assert!((resized[63] - 2.0).abs() < 0.5);
    }

    #[test]
    fn test_resize_upsample() {
        // 2×2×2 uniform volume → 4×4×4 should stay uniform
        let data = vec![5.0f32; 8];
        let resized = trilinear_resize(&data, [2, 2, 2], 4);
        assert_eq!(resized.len(), 64);
        for &v in &resized {
            assert!((v - 5.0).abs() < 1e-5, "expected 5.0, got {}", v);
        }
    }

    #[test]
    fn test_normalize_nonzero() {
        let data = vec![0.0, 10.0, 20.0, 30.0, 0.0];
        let normed = normalize_intensity(&data);
        // mean of nonzero = 20, std = sqrt(200/3)
        assert_eq!(normed[0], 0.0); // zero stays zero
        assert_eq!(normed[4], 0.0);
        // Nonzero values should have mean ~0
        let nz: Vec<f32> = normed.iter().filter(|&&v| v != 0.0).copied().collect();
        let mean: f32 = nz.iter().sum::<f32>() / nz.len() as f32;
        assert!(mean.abs() < 1e-5, "mean should be ~0, got {}", mean);
    }
}
