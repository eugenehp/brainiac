/// Data augmentation for BrainIAC training.
///
/// Matches MONAI's training transforms:
/// - RandAffined (rotation, translation, scale)
/// - RandFlipd (left-right flip, axis 2)
/// - RandGaussianNoised (std=0.05, prob=0.2)
/// - RandGaussianSmoothd (prob=0.2)
/// - RandAdjustContrastd (gamma 0.7-1.3, prob=0.2)
///
/// Uses a simple seeded RNG for reproducibility.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Simple seeded RNG based on hashing.
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed.wrapping_add(0x9e3779b97f4a7c15) }
    }

    fn next_u64(&mut self) -> u64 {
        let mut h = DefaultHasher::new();
        self.state.hash(&mut h);
        self.state = self.state.wrapping_add(1);
        h.finish()
    }

    /// Uniform f32 in [0, 1).
    pub fn f32(&mut self) -> f32 {
        (self.next_u64() & 0xFFFFFF) as f32 / 16777216.0
    }

    /// Uniform f32 in [lo, hi).
    pub fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.f32() * (hi - lo)
    }

    /// Standard normal (Box-Muller).
    pub fn normal(&mut self) -> f32 {
        let u1 = self.f32().max(1e-10);
        let u2 = self.f32();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }

    /// Returns true with given probability.
    pub fn chance(&mut self, prob: f32) -> bool {
        self.f32() < prob
    }
}

/// Augmentation configuration matching MONAI defaults.
#[derive(Debug, Clone)]
pub struct AugConfig {
    /// Probability of applying affine transform.
    pub affine_prob: f32,
    /// Rotation range (radians) for each axis.
    pub rotate_range: f32,
    /// Translation range (voxels) for each axis.
    pub translate_range: f32,
    /// Scale range (fraction, e.g. 0.1 means [0.9, 1.1]).
    pub scale_range: f32,
    /// Probability of left-right flip.
    pub flip_prob: f32,
    /// Probability of adding Gaussian noise.
    pub noise_prob: f32,
    /// Gaussian noise std.
    pub noise_std: f32,
    /// Probability of Gaussian smoothing.
    pub smooth_prob: f32,
    /// Probability of contrast adjustment.
    pub contrast_prob: f32,
    /// Gamma range for contrast [lo, hi].
    pub gamma_range: [f32; 2],
}

impl Default for AugConfig {
    fn default() -> Self {
        Self {
            affine_prob: 0.5,
            rotate_range: 0.1,
            translate_range: 5.0,
            scale_range: 0.1,
            flip_prob: 0.5,
            noise_prob: 0.2,
            noise_std: 0.05,
            smooth_prob: 0.2,
            contrast_prob: 0.2,
            gamma_range: [0.7, 1.3],
        }
    }
}

/// Apply augmentations to a 3D volume in-place.
///
/// `data`: flat f32 volume of shape [size³].
/// `size`: volume dimension (isotropic).
/// `rng`: seeded RNG.
/// `config`: augmentation config.
pub fn augment_volume(data: &mut [f32], size: usize, rng: &mut Rng, config: &AugConfig) {
    // 1. Random left-right flip (axis 2 = x-axis in RAS)
    if rng.chance(config.flip_prob) {
        flip_axis(data, size, 0); // flip x
    }

    // 2. Random affine (simplified: just apply small perturbation via resampling)
    if rng.chance(config.affine_prob) {
        let rx = rng.range(-config.rotate_range, config.rotate_range);
        let ry = rng.range(-config.rotate_range, config.rotate_range);
        let rz = rng.range(-config.rotate_range, config.rotate_range);
        let tx = rng.range(-config.translate_range, config.translate_range);
        let ty = rng.range(-config.translate_range, config.translate_range);
        let tz = rng.range(-config.translate_range, config.translate_range);
        let sx = 1.0 + rng.range(-config.scale_range, config.scale_range);
        let sy = 1.0 + rng.range(-config.scale_range, config.scale_range);
        let sz = 1.0 + rng.range(-config.scale_range, config.scale_range);
        apply_affine(data, size, [rx, ry, rz], [tx, ty, tz], [sx, sy, sz]);
    }

    // 3. Random Gaussian noise
    if rng.chance(config.noise_prob) {
        for v in data.iter_mut() {
            *v += rng.normal() * config.noise_std;
        }
    }

    // 4. Random Gaussian smoothing
    if rng.chance(config.smooth_prob) {
        let sigma = rng.range(0.5, 1.5);
        gaussian_smooth_3d(data, size, sigma);
    }

    // 5. Random contrast adjustment (gamma)
    if rng.chance(config.contrast_prob) {
        let gamma = rng.range(config.gamma_range[0], config.gamma_range[1]);
        // Normalize to [0,1], apply gamma, then restore range
        let min_v = data.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_v = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = (max_v - min_v).max(1e-8);
        for v in data.iter_mut() {
            let normed = ((*v - min_v) / range).max(0.0);
            *v = normed.powf(gamma) * range + min_v;
        }
    }
}

/// Flip a 3D volume along the given axis (0=x, 1=y, 2=z).
fn flip_axis(data: &mut [f32], size: usize, axis: usize) {
    let s = size;
    let mut buf = data.to_vec();
    for z in 0..s {
        for y in 0..s {
            for x in 0..s {
                let (fx, fy, fz) = match axis {
                    0 => (s - 1 - x, y, z),
                    1 => (x, s - 1 - y, z),
                    _ => (x, y, s - 1 - z),
                };
                buf[x + y * s + z * s * s] = data[fx + fy * s + fz * s * s];
            }
        }
    }
    data.copy_from_slice(&buf);
}

/// Apply a simple affine transform (rotation + translation + scale) via trilinear resampling.
fn apply_affine(
    data: &mut [f32],
    size: usize,
    rotation: [f32; 3],
    translation: [f32; 3],
    scale: [f32; 3],
) {
    let s = size as f32;
    let center = s / 2.0;
    let [rx, ry, rz] = rotation;
    let [tx, ty, tz] = translation;
    let [sx, sy, sz] = scale;

    // Precompute rotation matrix (small angle approximation for speed)
    let (cx, csx) = (rx.cos(), rx.sin());
    let (cy, csy) = (ry.cos(), ry.sin());
    let (cz, csz) = (rz.cos(), rz.sin());

    // Rotation matrix R = Rz * Ry * Rx
    let r00 = cy * cz;
    let r01 = csx * csy * cz - cx * csz;
    let r02 = cx * csy * cz + csx * csz;
    let r10 = cy * csz;
    let r11 = csx * csy * csz + cx * cz;
    let r12 = cx * csy * csz - csx * cz;
    let r20 = -csy;
    let r21 = csx * cy;
    let r22 = cx * cy;

    let src = data.to_vec();
    let si = size;

    for oz in 0..si {
        for oy in 0..si {
            for ox in 0..si {
                // Center, apply inverse transform
                let px = (ox as f32 - center) / sx;
                let py = (oy as f32 - center) / sy;
                let pz = (oz as f32 - center) / sz;

                // Inverse rotation (transpose)
                let rx = r00 * px + r10 * py + r20 * pz;
                let ry = r01 * px + r11 * py + r21 * pz;
                let rz = r02 * px + r12 * py + r22 * pz;

                let fx = rx + center - tx;
                let fy = ry + center - ty;
                let fz = rz + center - tz;

                // Trilinear sample
                data[ox + oy * si + oz * si * si] = trilinear_sample(&src, si, fx, fy, fz);
            }
        }
    }
}

/// Trilinear sample from a flat 3D volume.
fn trilinear_sample(data: &[f32], size: usize, x: f32, y: f32, z: f32) -> f32 {
    let s = size as f32;
    let x = x.max(0.0).min(s - 1.001);
    let y = y.max(0.0).min(s - 1.001);
    let z = z.max(0.0).min(s - 1.001);

    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let z0 = z.floor() as usize;
    let x1 = (x0 + 1).min(size - 1);
    let y1 = (y0 + 1).min(size - 1);
    let z1 = (z0 + 1).min(size - 1);

    let xd = x - x0 as f32;
    let yd = y - y0 as f32;
    let zd = z - z0 as f32;

    let idx = |x: usize, y: usize, z: usize| x + y * size + z * size * size;

    let c00 = data[idx(x0, y0, z0)] * (1.0 - xd) + data[idx(x1, y0, z0)] * xd;
    let c01 = data[idx(x0, y0, z1)] * (1.0 - xd) + data[idx(x1, y0, z1)] * xd;
    let c10 = data[idx(x0, y1, z0)] * (1.0 - xd) + data[idx(x1, y1, z0)] * xd;
    let c11 = data[idx(x0, y1, z1)] * (1.0 - xd) + data[idx(x1, y1, z1)] * xd;
    let c0 = c00 * (1.0 - yd) + c10 * yd;
    let c1 = c01 * (1.0 - yd) + c11 * yd;
    c0 * (1.0 - zd) + c1 * zd
}

/// Separable 3D Gaussian smoothing (sigma in voxels).
fn gaussian_smooth_3d(data: &mut [f32], size: usize, sigma: f32) {
    let radius = (2.0 * sigma).ceil() as isize;
    let ksize = (2 * radius + 1) as usize;
    let mut kernel = vec![0.0f32; ksize];
    let mut ksum = 0.0f32;
    for i in 0..ksize {
        let x = i as f32 - radius as f32;
        kernel[i] = (-0.5 * (x / sigma) * (x / sigma)).exp();
        ksum += kernel[i];
    }
    for k in &mut kernel { *k /= ksum; }

    // Separable: x, y, z passes
    let s = size;
    let mut tmp = vec![0.0f32; s * s * s];

    // X pass
    for z in 0..s {
        for y in 0..s {
            for x in 0..s {
                let mut sum = 0.0f32;
                for ki in 0..ksize {
                    let xi = x as isize + ki as isize - radius;
                    if xi >= 0 && xi < s as isize {
                        sum += data[xi as usize + y * s + z * s * s] * kernel[ki];
                    }
                }
                tmp[x + y * s + z * s * s] = sum;
            }
        }
    }
    // Y pass
    for z in 0..s {
        for y in 0..s {
            for x in 0..s {
                let mut sum = 0.0f32;
                for ki in 0..ksize {
                    let yi = y as isize + ki as isize - radius;
                    if yi >= 0 && yi < s as isize {
                        sum += tmp[x + yi as usize * s + z * s * s] * kernel[ki];
                    }
                }
                data[x + y * s + z * s * s] = sum;
            }
        }
    }
    // Z pass
    let src = data.to_vec();
    for z in 0..s {
        for y in 0..s {
            for x in 0..s {
                let mut sum = 0.0f32;
                for ki in 0..ksize {
                    let zi = z as isize + ki as isize - radius;
                    if zi >= 0 && zi < s as isize {
                        sum += src[x + y * s + zi as usize * s * s] * kernel[ki];
                    }
                }
                data[x + y * s + z * s * s] = sum;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flip_axis() {
        let mut data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]; // 2×2×2
        flip_axis(&mut data, 2, 0);
        assert_eq!(data[0], 1.0); // x=0 → was x=1
        assert_eq!(data[1], 0.0);
    }

    #[test]
    fn test_rng_range() {
        let mut rng = Rng::new(42);
        for _ in 0..100 {
            let v = rng.range(0.0, 1.0);
            assert!(v >= 0.0 && v < 1.0);
        }
    }

    #[test]
    fn test_augment_preserves_size() {
        let mut rng = Rng::new(123);
        let mut data = vec![1.0f32; 8]; // 2×2×2
        let cfg = AugConfig::default();
        augment_volume(&mut data, 2, &mut rng, &cfg);
        assert_eq!(data.len(), 8);
    }
}
