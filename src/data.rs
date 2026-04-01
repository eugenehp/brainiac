/// Dataset loading for BrainIAC training.
///
/// Reads CSV files with (pat_id, label) columns and loads NIfTI volumes
/// from disk, applying preprocessing (resize + z-score normalization).
///
/// Supports single, dual, and quad image datasets matching the Python codebase.

use std::path::{Path, PathBuf};
use anyhow::{Result, Context};

use crate::nifti;
use crate::preprocessing;

/// A single training sample: preprocessed volume data + label.
#[derive(Debug, Clone)]
pub struct Sample {
    /// Preprocessed volume data, flat f32 of shape [C * D * H * W].
    pub data: Vec<f32>,
    /// Ground truth label (regression value or class index).
    pub label: f32,
    /// Patient ID for tracking.
    pub pat_id: String,
}

/// A multi-scan training sample (for dual/quad tasks).
#[derive(Debug, Clone)]
pub struct MultiSample {
    /// Multiple preprocessed volumes, each flat f32 of shape [C * D * H * W].
    pub scans: Vec<Vec<f32>>,
    /// Ground truth label.
    pub label: f32,
    /// Patient ID.
    pub pat_id: String,
}

/// CSV record for single-scan tasks (brain age, MCI, stroke, sequence).
#[derive(Debug, Clone)]
pub struct CsvRecord {
    pub pat_id: String,
    pub label: f32,
}

/// Parse a simple CSV with header "pat_id,label".
pub fn parse_csv(path: &Path) -> Result<Vec<CsvRecord>> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read CSV: {}", path.display()))?;
    let mut records = Vec::new();
    for (i, line) in content.lines().enumerate() {
        if i == 0 { continue; } // skip header
        let line = line.trim();
        if line.is_empty() { continue; }
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 2 {
            anyhow::bail!("CSV line {} has fewer than 2 columns: '{}'", i + 1, line);
        }
        let pat_id = parts[0].trim().to_string();
        let label: f32 = parts[1].trim().parse()
            .with_context(|| format!("Failed to parse label on line {}: '{}'", i + 1, parts[1]))?;
        records.push(CsvRecord { pat_id, label });
    }
    Ok(records)
}

/// Dataset for single-scan tasks.
///
/// Loads NIfTI volumes lazily and caches preprocessed data.
pub struct SingleScanDataset {
    pub records: Vec<CsvRecord>,
    pub root_dir: PathBuf,
    pub img_size: usize,
    cache: Vec<Option<Vec<f32>>>,
}

impl SingleScanDataset {
    pub fn new(csv_path: &Path, root_dir: &Path, img_size: usize) -> Result<Self> {
        let records = parse_csv(csv_path)?;
        let n = records.len();
        Ok(Self {
            records,
            root_dir: root_dir.to_path_buf(),
            img_size,
            cache: vec![None; n],
        })
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Get a sample by index, loading and caching the NIfTI if needed.
    pub fn get(&mut self, idx: usize) -> Result<Sample> {
        let record = &self.records[idx];

        if self.cache[idx].is_none() {
            let nifti_path = self.root_dir.join(format!("{}.nii.gz", record.pat_id));
            let volume = nifti::read_nifti(&nifti_path)
                .with_context(|| format!("Failed to load NIfTI for patient {}", record.pat_id))?;
            let preprocessed = preprocessing::preprocess(&volume, self.img_size);
            self.cache[idx] = Some(preprocessed);
        }

        Ok(Sample {
            data: self.cache[idx].as_ref().unwrap().clone(),
            label: record.label,
            pat_id: record.pat_id.clone(),
        })
    }

    /// Clear the cache to free memory.
    pub fn clear_cache(&mut self) {
        self.cache = vec![None; self.records.len()];
    }
}

/// Dataset for dual-scan tasks (IDH: FLAIR + T1CE).
pub struct DualScanDataset {
    pub records: Vec<CsvRecord>,
    pub root_dir: PathBuf,
    pub img_size: usize,
    pub suffixes: [String; 2],
}

impl DualScanDataset {
    pub fn new(
        csv_path: &Path,
        root_dir: &Path,
        img_size: usize,
        suffixes: [&str; 2],
    ) -> Result<Self> {
        let records = parse_csv(csv_path)?;
        Ok(Self {
            records,
            root_dir: root_dir.to_path_buf(),
            img_size,
            suffixes: [suffixes[0].to_string(), suffixes[1].to_string()],
        })
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    pub fn get(&self, idx: usize) -> Result<MultiSample> {
        let record = &self.records[idx];
        let mut scans = Vec::with_capacity(2);
        for suffix in &self.suffixes {
            let path = self.root_dir.join(format!("{}_{}.nii.gz", record.pat_id, suffix));
            let volume = nifti::read_nifti(&path)
                .with_context(|| format!("Failed to load {} for {}", suffix, record.pat_id))?;
            scans.push(preprocessing::preprocess(&volume, self.img_size));
        }
        Ok(MultiSample {
            scans,
            label: record.label,
            pat_id: record.pat_id.clone(),
        })
    }
}

/// Dataset for quad-scan tasks (Overall Survival: T1CE + T1 + T2 + FLAIR).
pub struct QuadScanDataset {
    pub records: Vec<CsvRecord>,
    pub root_dir: PathBuf,
    pub img_size: usize,
    pub suffixes: [String; 4],
}

impl QuadScanDataset {
    pub fn new(
        csv_path: &Path,
        root_dir: &Path,
        img_size: usize,
        suffixes: [&str; 4],
    ) -> Result<Self> {
        let records = parse_csv(csv_path)?;
        Ok(Self {
            records,
            root_dir: root_dir.to_path_buf(),
            img_size,
            suffixes: [
                suffixes[0].to_string(), suffixes[1].to_string(),
                suffixes[2].to_string(), suffixes[3].to_string(),
            ],
        })
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    pub fn get(&self, idx: usize) -> Result<MultiSample> {
        let record = &self.records[idx];
        let mut scans = Vec::with_capacity(4);
        for suffix in &self.suffixes {
            let path = self.root_dir.join(format!("{}_{}.nii.gz", record.pat_id, suffix));
            let volume = nifti::read_nifti(&path)
                .with_context(|| format!("Failed to load {} for {}", suffix, record.pat_id))?;
            scans.push(preprocessing::preprocess(&volume, self.img_size));
        }
        Ok(MultiSample {
            scans,
            label: record.label,
            pat_id: record.pat_id.clone(),
        })
    }
}

/// Shuffle indices for an epoch.
pub fn shuffled_indices(n: usize, seed: u64) -> Vec<usize> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut indices: Vec<usize> = (0..n).collect();
    // Simple Fisher-Yates with seeded hash
    for i in (1..n).rev() {
        let mut hasher = DefaultHasher::new();
        seed.hash(&mut hasher);
        i.hash(&mut hasher);
        let h = hasher.finish() as usize;
        let j = h % (i + 1);
        indices.swap(i, j);
    }
    indices
}
