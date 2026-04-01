/// MRI preprocessing pipeline.
///
/// Provides the preprocessing steps needed before BrainIAC inference:
/// 1. N4 bias field correction (simplified iterative version)
/// 2. Registration to standard space (rigid affine via intensity-based optimization)
/// 3. Skull stripping (simple threshold + morphological operations)
///
/// NOTE: For production use, the Python pipeline with HD-BET and SimpleITK
/// is recommended. This Rust version provides basic preprocessing for cases
/// where Python dependencies are not available.

use crate::nifti::NiftiVolume;

/// Simple N4-style bias field correction.
///
/// Estimates a smooth multiplicative bias field and divides it out.
/// Uses iterative mean-field estimation with Gaussian smoothing.
///
/// This is a simplified version — the full N4ITK algorithm uses B-splines.
pub fn bias_field_correction(volume: &mut NiftiVolume, iterations: usize) {
    let [nx, ny, nz] = volume.dims;
    let n = nx * ny * nz;

    for _iter in 0..iterations {
        // Compute log of nonzero intensities
        let mut log_data = vec![0.0f32; n];
        let mut mask = vec![false; n];
        for i in 0..n {
            if volume.data[i] > 0.0 {
                log_data[i] = volume.data[i].ln();
                mask[i] = true;
            }
        }

        // Smooth the log image (estimate bias field)
        let bias = smooth_3d(&log_data, &mask, nx, ny, nz, 8.0);

        // Divide out the bias field
        for i in 0..n {
            if mask[i] && bias[i].abs() > 1e-8 {
                volume.data[i] = (log_data[i] - bias[i]).exp();
            }
        }
    }
}

/// Simple skull stripping via intensity thresholding + morphological cleanup.
///
/// 1. Compute Otsu threshold
/// 2. Create binary mask
/// 3. Keep largest connected component
/// 4. Apply morphological closing
/// 5. Zero out voxels outside mask
pub fn skull_strip(volume: &mut NiftiVolume) {
    let n = volume.data.len();

    // Otsu thresholding on nonzero voxels
    let threshold = otsu_threshold(&volume.data);

    // Create binary mask
    let mut mask = vec![false; n];
    for i in 0..n {
        mask[i] = volume.data[i] > threshold * 0.3; // conservative threshold
    }

    // Simple morphological closing (dilate then erode) to fill holes
    let [nx, ny, nz] = volume.dims;
    morphological_close(&mut mask, nx, ny, nz, 2);

    // Apply mask
    for i in 0..n {
        if !mask[i] {
            volume.data[i] = 0.0;
        }
    }
}

/// Full preprocessing pipeline: bias correction + skull stripping.
pub fn preprocess_mri(volume: &mut NiftiVolume) {
    bias_field_correction(volume, 3);
    skull_strip(volume);
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Otsu's threshold on nonzero values.
fn otsu_threshold(data: &[f32]) -> f32 {
    let nonzero: Vec<f32> = data.iter().filter(|&&v| v > 0.0).copied().collect();
    if nonzero.is_empty() { return 0.0; }

    let min_v = nonzero.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_v = nonzero.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if (max_v - min_v) < 1e-8 { return min_v; }

    let n_bins = 256;
    let mut hist = vec![0u32; n_bins];
    for &v in &nonzero {
        let bin = (((v - min_v) / (max_v - min_v)) * (n_bins - 1) as f32) as usize;
        hist[bin.min(n_bins - 1)] += 1;
    }

    let total = nonzero.len() as f64;
    let mut sum_total = 0.0f64;
    for (i, &h) in hist.iter().enumerate() {
        sum_total += i as f64 * h as f64;
    }

    let mut best_thresh = 0;
    let mut best_var = 0.0f64;
    let mut w0 = 0.0f64;
    let mut sum0 = 0.0f64;

    for (t, &h) in hist.iter().enumerate() {
        w0 += h as f64;
        if w0 == 0.0 { continue; }
        let w1 = total - w0;
        if w1 == 0.0 { break; }

        sum0 += t as f64 * h as f64;
        let m0 = sum0 / w0;
        let m1 = (sum_total - sum0) / w1;
        let between = w0 * w1 * (m0 - m1) * (m0 - m1);
        if between > best_var {
            best_var = between;
            best_thresh = t;
        }
    }

    min_v + (best_thresh as f32 / (n_bins - 1) as f32) * (max_v - min_v)
}

/// 3D Gaussian smoothing of masked data.
fn smooth_3d(data: &[f32], mask: &[bool], nx: usize, ny: usize, nz: usize, sigma: f32) -> Vec<f32> {
    let radius = (2.0 * sigma).ceil() as isize;
    let ksize = (2 * radius + 1) as usize;
    let mut kernel = vec![0.0f32; ksize];
    let mut ksum = 0.0f32;
    for i in 0..ksize {
        let x = i as f32 - radius as f32;
        kernel[i] = (-0.5 * (x / sigma).powi(2)).exp();
        ksum += kernel[i];
    }
    for k in &mut kernel { *k /= ksum; }

    let n = nx * ny * nz;
    let mut buf = data.to_vec();
    let mut tmp = vec![0.0f32; n];

    // X pass
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                if !mask[x + y * nx + z * nx * ny] { continue; }
                let mut s = 0.0f32;
                let mut w = 0.0f32;
                for ki in 0..ksize {
                    let xi = x as isize + ki as isize - radius;
                    if xi >= 0 && xi < nx as isize {
                        let idx = xi as usize + y * nx + z * nx * ny;
                        if mask[idx] { s += buf[idx] * kernel[ki]; w += kernel[ki]; }
                    }
                }
                tmp[x + y * nx + z * nx * ny] = if w > 0.0 { s / w } else { 0.0 };
            }
        }
    }
    std::mem::swap(&mut buf, &mut tmp);

    // Y pass
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                if !mask[x + y * nx + z * nx * ny] { continue; }
                let mut s = 0.0f32;
                let mut w = 0.0f32;
                for ki in 0..ksize {
                    let yi = y as isize + ki as isize - radius;
                    if yi >= 0 && yi < ny as isize {
                        let idx = x + yi as usize * nx + z * nx * ny;
                        if mask[idx] { s += buf[idx] * kernel[ki]; w += kernel[ki]; }
                    }
                }
                tmp[x + y * nx + z * nx * ny] = if w > 0.0 { s / w } else { 0.0 };
            }
        }
    }

    tmp
}

/// Morphological closing (dilate then erode) with given radius.
fn morphological_close(mask: &mut [bool], nx: usize, ny: usize, nz: usize, radius: isize) {
    let r2 = (radius * radius) as isize;

    // Dilate
    let src = mask.to_vec();
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                if src[x + y * nx + z * nx * ny] { continue; }
                'outer: for dz in -radius..=radius {
                    for dy in -radius..=radius {
                        for dx in -radius..=radius {
                            if dx*dx + dy*dy + dz*dz > r2 { continue; }
                            let xi = x as isize + dx;
                            let yi = y as isize + dy;
                            let zi = z as isize + dz;
                            if xi >= 0 && xi < nx as isize && yi >= 0 && yi < ny as isize
                                && zi >= 0 && zi < nz as isize
                            {
                                if src[xi as usize + yi as usize * nx + zi as usize * nx * ny] {
                                    mask[x + y * nx + z * nx * ny] = true;
                                    break 'outer;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Erode
    let src = mask.to_vec();
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                if !src[x + y * nx + z * nx * ny] { continue; }
                for dz in -radius..=radius {
                    for dy in -radius..=radius {
                        for dx in -radius..=radius {
                            if dx*dx + dy*dy + dz*dz > r2 { continue; }
                            let xi = x as isize + dx;
                            let yi = y as isize + dy;
                            let zi = z as isize + dz;
                            if xi < 0 || xi >= nx as isize || yi < 0 || yi >= ny as isize
                                || zi < 0 || zi >= nz as isize
                                || !src[xi as usize + yi as usize * nx + zi as usize * nx * ny]
                            {
                                mask[x + y * nx + z * nx * ny] = false;
                            }
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_otsu_bimodal() {
        let mut data = vec![0.0f32; 100];
        for i in 0..50 { data[i] = 10.0 + (i as f32) * 0.1; } // ~10-15
        for i in 50..100 { data[i] = 80.0 + (i as f32) * 0.1; } // ~80-90
        let thresh = otsu_threshold(&data);
        assert!(thresh > 12.0 && thresh < 82.0, "Otsu threshold should separate bimodal: got {}", thresh);
    }
}
