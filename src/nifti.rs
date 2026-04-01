/// NIfTI-1 (.nii / .nii.gz) reader for volumetric brain MRI.
///
/// Reads 3D volumes from NIfTI-1 format files, supporting:
/// - Uncompressed .nii files
/// - Gzip-compressed .nii.gz files
/// - Float32 (DT_FLOAT32 = 16) and Int16 (DT_INT16 = 4) data types
///
/// The reader extracts volume dimensions and voxel data, returning
/// a flat f32 array in row-major order [x + y*nx + z*nx*ny].

use std::io::Read;
use std::path::Path;
use anyhow::{Context, Result, bail};

/// Parsed NIfTI volume.
#[derive(Debug, Clone)]
pub struct NiftiVolume {
    /// Voxel data as flat f32 array, row-major [x, y, z].
    pub data: Vec<f32>,
    /// Volume dimensions [nx, ny, nz].
    pub dims: [usize; 3],
    /// Voxel sizes in mm [dx, dy, dz].
    pub pixdim: [f32; 3],
}

impl NiftiVolume {
    /// Total number of voxels.
    pub fn n_voxels(&self) -> usize {
        self.dims[0] * self.dims[1] * self.dims[2]
    }

    /// Index into the flat data array.
    pub fn idx(&self, x: usize, y: usize, z: usize) -> usize {
        x + y * self.dims[0] + z * self.dims[0] * self.dims[1]
    }

    /// Get voxel value at (x, y, z).
    pub fn get(&self, x: usize, y: usize, z: usize) -> f32 {
        self.data[self.idx(x, y, z)]
    }
}

/// Read a NIfTI-1 file (.nii or .nii.gz).
///
/// Supports DT_FLOAT32 (datatype=16) and DT_INT16 (datatype=4).
/// Other data types will return an error.
pub fn read_nifti(path: &Path) -> Result<NiftiVolume> {
    let raw = std::fs::read(path)
        .with_context(|| format!("failed to read: {}", path.display()))?;

    // Decompress if gzipped
    let bytes = if is_gzipped(&raw) {
        let mut decoder = flate2::read::GzDecoder::new(&raw[..]);
        let mut decompressed = Vec::new();
        decoder.read_to_end(&mut decompressed)
            .with_context(|| format!("failed to decompress: {}", path.display()))?;
        decompressed
    } else {
        raw
    };

    if bytes.len() < 352 {
        bail!("NIfTI file too small: {} bytes (need at least 352)", bytes.len());
    }

    // Parse header
    let sizeof_hdr = i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    if sizeof_hdr != 348 {
        bail!("Invalid NIfTI-1 header: sizeof_hdr = {} (expected 348)", sizeof_hdr);
    }

    // dim field at offset 40: [ndim, nx, ny, nz, nt, ...]
    let dim_off = 40;
    let ndim = i16::from_le_bytes([bytes[dim_off], bytes[dim_off + 1]]) as usize;
    if ndim < 3 {
        bail!("Expected at least 3D volume, got ndim={}", ndim);
    }
    let nx = i16::from_le_bytes([bytes[dim_off + 2], bytes[dim_off + 3]]) as usize;
    let ny = i16::from_le_bytes([bytes[dim_off + 4], bytes[dim_off + 5]]) as usize;
    let nz = i16::from_le_bytes([bytes[dim_off + 6], bytes[dim_off + 7]]) as usize;

    // datatype at offset 70
    let datatype = i16::from_le_bytes([bytes[70], bytes[71]]);
    // bitpix at offset 72
    let bitpix = i16::from_le_bytes([bytes[72], bytes[73]]) as usize;

    // pixdim at offset 76: [qfac, dx, dy, dz, ...]
    let pixdim_off = 76;
    let dx = f32::from_le_bytes([
        bytes[pixdim_off + 4], bytes[pixdim_off + 5],
        bytes[pixdim_off + 6], bytes[pixdim_off + 7],
    ]);
    let dy = f32::from_le_bytes([
        bytes[pixdim_off + 8], bytes[pixdim_off + 9],
        bytes[pixdim_off + 10], bytes[pixdim_off + 11],
    ]);
    let dz = f32::from_le_bytes([
        bytes[pixdim_off + 12], bytes[pixdim_off + 13],
        bytes[pixdim_off + 14], bytes[pixdim_off + 15],
    ]);

    // vox_offset at offset 108
    let vox_offset = f32::from_le_bytes([bytes[108], bytes[109], bytes[110], bytes[111]]) as usize;

    // scl_slope and scl_inter at offsets 112, 116
    let scl_slope = f32::from_le_bytes([bytes[112], bytes[113], bytes[114], bytes[115]]);
    let scl_inter = f32::from_le_bytes([bytes[116], bytes[117], bytes[118], bytes[119]]);
    let apply_scaling = scl_slope != 0.0 && (scl_slope != 1.0 || scl_inter != 0.0);

    let n_voxels = nx * ny * nz;
    let data_start = vox_offset;

    let mut data = Vec::with_capacity(n_voxels);

    match (datatype, bitpix) {
        // DT_FLOAT32
        (16, 32) => {
            if bytes.len() < data_start + n_voxels * 4 {
                bail!(
                    "NIfTI file truncated: need {} bytes for float32 data, have {}",
                    data_start + n_voxels * 4,
                    bytes.len()
                );
            }
            for i in 0..n_voxels {
                let off = data_start + i * 4;
                let v = f32::from_le_bytes([
                    bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3],
                ]);
                data.push(v);
            }
        }
        // DT_INT16
        (4, 16) => {
            if bytes.len() < data_start + n_voxels * 2 {
                bail!(
                    "NIfTI file truncated: need {} bytes for int16 data, have {}",
                    data_start + n_voxels * 2,
                    bytes.len()
                );
            }
            for i in 0..n_voxels {
                let off = data_start + i * 2;
                let v = i16::from_le_bytes([bytes[off], bytes[off + 1]]);
                data.push(v as f32);
            }
        }
        // DT_UINT8
        (2, 8) => {
            if bytes.len() < data_start + n_voxels {
                bail!(
                    "NIfTI file truncated: need {} bytes for uint8 data, have {}",
                    data_start + n_voxels,
                    bytes.len()
                );
            }
            for i in 0..n_voxels {
                data.push(bytes[data_start + i] as f32);
            }
        }
        // DT_FLOAT64
        (64, 64) => {
            if bytes.len() < data_start + n_voxels * 8 {
                bail!(
                    "NIfTI file truncated: need {} bytes for float64 data, have {}",
                    data_start + n_voxels * 8,
                    bytes.len()
                );
            }
            for i in 0..n_voxels {
                let off = data_start + i * 8;
                let v = f64::from_le_bytes([
                    bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3],
                    bytes[off + 4], bytes[off + 5], bytes[off + 6], bytes[off + 7],
                ]);
                data.push(v as f32);
            }
        }
        _ => {
            bail!(
                "Unsupported NIfTI datatype={} bitpix={} (supported: float32, int16, uint8, float64)",
                datatype, bitpix
            );
        }
    }

    // Apply scaling if needed
    if apply_scaling {
        for v in &mut data {
            *v = *v * scl_slope + scl_inter;
        }
    }

    Ok(NiftiVolume {
        data,
        dims: [nx, ny, nz],
        pixdim: [dx, dy, dz],
    })
}

/// Check if data starts with gzip magic bytes.
fn is_gzipped(data: &[u8]) -> bool {
    data.len() >= 2 && data[0] == 0x1f && data[1] == 0x8b
}

// ── NIfTI Writer ──────────────────────────────────────────────────────────────

/// Write a 3D float32 volume as NIfTI-1 (.nii or .nii.gz).
///
/// `volume`: flat f32 data in row-major order [x + y*nx + z*nx*ny].
/// `dims`: [nx, ny, nz].
/// `voxel_size`: voxel dimensions in mm [dx, dy, dz].
pub fn write_nifti(path: &Path, volume: &[f32], dims: [usize; 3], voxel_size: [f32; 3]) -> Result<()> {
    let [nx, ny, nz] = dims;
    let expected = nx * ny * nz;
    if volume.len() != expected {
        bail!("Volume has {} voxels, expected {} ({}×{}×{})", volume.len(), expected, nx, ny, nz);
    }

    let header = build_nifti1_header(dims, voxel_size);
    let extension = [0u8; 4]; // 4 bytes padding to reach offset 352
    let data_bytes: Vec<u8> = volume.iter().flat_map(|v| v.to_le_bytes()).collect();

    let path_str = path.to_string_lossy();
    if path_str.ends_with(".gz") {
        let file = std::fs::File::create(path)
            .with_context(|| format!("failed to create {}", path.display()))?;
        let mut gz = flate2::write::GzEncoder::new(file, flate2::Compression::default());
        use std::io::Write;
        gz.write_all(&header)?;
        gz.write_all(&extension)?;
        gz.write_all(&data_bytes)?;
        gz.finish()?;
    } else {
        let mut file = std::fs::File::create(path)
            .with_context(|| format!("failed to create {}", path.display()))?;
        use std::io::Write;
        file.write_all(&header)?;
        file.write_all(&extension)?;
        file.write_all(&data_bytes)?;
    }

    Ok(())
}

/// Build a minimal NIfTI-1 header (348 bytes) for a float32 3D volume.
fn build_nifti1_header(dims: [usize; 3], voxel_size: [f32; 3]) -> [u8; 348] {
    let mut hdr = [0u8; 348];
    let [nx, ny, nz] = dims;
    let [dx, dy, dz] = voxel_size;

    // sizeof_hdr = 348
    hdr[0..4].copy_from_slice(&348i32.to_le_bytes());

    // dim[0..7] at offset 40
    let d = 40;
    hdr[d..d+2].copy_from_slice(&3i16.to_le_bytes());
    hdr[d+2..d+4].copy_from_slice(&(nx as i16).to_le_bytes());
    hdr[d+4..d+6].copy_from_slice(&(ny as i16).to_le_bytes());
    hdr[d+6..d+8].copy_from_slice(&(nz as i16).to_le_bytes());
    hdr[d+8..d+10].copy_from_slice(&1i16.to_le_bytes());

    // datatype=16 (FLOAT32), bitpix=32
    hdr[70..72].copy_from_slice(&16i16.to_le_bytes());
    hdr[72..74].copy_from_slice(&32i16.to_le_bytes());

    // pixdim at offset 76
    let p = 76;
    hdr[p..p+4].copy_from_slice(&1.0f32.to_le_bytes());
    hdr[p+4..p+8].copy_from_slice(&dx.to_le_bytes());
    hdr[p+8..p+12].copy_from_slice(&dy.to_le_bytes());
    hdr[p+12..p+16].copy_from_slice(&dz.to_le_bytes());

    // vox_offset = 352.0
    hdr[108..112].copy_from_slice(&352.0f32.to_le_bytes());

    // scl_slope=1.0, scl_inter=0.0
    hdr[112..116].copy_from_slice(&1.0f32.to_le_bytes());
    hdr[116..120].copy_from_slice(&0.0f32.to_le_bytes());

    // sform_code = 1 (Scanner Anat)
    hdr[254..256].copy_from_slice(&1i16.to_le_bytes());

    // srow_x/y/z — identity affine with voxel sizes
    hdr[280..284].copy_from_slice(&dx.to_le_bytes());
    hdr[296+4..296+8].copy_from_slice(&dy.to_le_bytes());
    hdr[312+8..312+12].copy_from_slice(&dz.to_le_bytes());

    // magic = "n+1\0"
    hdr[344..348].copy_from_slice(b"n+1\0");

    hdr
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_gzipped() {
        assert!(is_gzipped(&[0x1f, 0x8b, 0x08]));
        assert!(!is_gzipped(&[0x00, 0x00]));
        assert!(!is_gzipped(&[0x1f]));
    }

    #[test]
    fn test_nifti_volume_idx() {
        let vol = NiftiVolume {
            data: vec![0.0; 27],
            dims: [3, 3, 3],
            pixdim: [1.0, 1.0, 1.0],
        };
        assert_eq!(vol.idx(0, 0, 0), 0);
        assert_eq!(vol.idx(2, 0, 0), 2);
        assert_eq!(vol.idx(0, 1, 0), 3);
        assert_eq!(vol.idx(0, 0, 1), 9);
        assert_eq!(vol.idx(2, 2, 2), 26);
    }
}
