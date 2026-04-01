/// High-level BrainIAC encoder API.
///
/// Provides a simple interface for loading the model and running inference:
///
/// ```rust,ignore
/// use brainiac::BrainiacEncoder;
///
/// let encoder = BrainiacEncoder::<B>::load(
///     "brainiac_backbone.safetensors",
///     None, // optional task head weights
///     TaskType::FeatureExtraction,
///     1, // num_classes
///     &device,
/// )?;
///
/// let features = encoder.encode_nifti(Path::new("brain.nii.gz"))?;
/// ```

use std::path::Path;
use std::time::Instant;
use anyhow::Result;
use burn::prelude::*;

use crate::config::{ModelConfig, TaskType};
use crate::model::classifier::{BrainiacModel, InferenceOutput};
use crate::nifti;
use crate::preprocessing;
use crate::weights::{WeightMap, load_classifier_weights};

/// High-level BrainIAC encoder / inference wrapper.
pub struct BrainiacEncoder<B: Backend> {
    pub model: BrainiacModel<B>,
    pub config: ModelConfig,
    pub device: B::Device,
}

/// Timing info from model loading.
#[derive(Debug, Clone)]
pub struct LoadTimings {
    pub weights_ms: f64,
    pub model_init_ms: f64,
    pub total_ms: f64,
}

impl<B: Backend> BrainiacEncoder<B> {
    /// Load a BrainIAC model from safetensors weights.
    ///
    /// - `backbone_weights`: path to backbone .safetensors
    /// - `head_weights`: optional path to downstream task head .safetensors
    /// - `task`: what kind of inference to run
    /// - `num_classes`: output dimension (1 for regression/binary, 4 for sequence)
    pub fn load(
        backbone_weights: &str,
        head_weights: Option<&str>,
        task: TaskType,
        num_classes: usize,
        device: B::Device,
    ) -> Result<(Self, LoadTimings)> {
        let t0 = Instant::now();
        let config = ModelConfig::default();

        // Load weights from disk into CPU memory first (fast)
        let t_weights = Instant::now();
        let mut wm = WeightMap::from_file(backbone_weights)?;
        eprintln!("Loaded {} backbone weight tensors", wm.len());
        let disk_ms = t_weights.elapsed().as_secs_f64() * 1000.0;

        // Build backbone directly from weight data — avoids double allocation
        // (zero-init + overwrite). 2.3× faster on GPU, 1.3× faster on CPU.
        let t_init = Instant::now();
        let backbone = crate::model::backbone::ViTBackbone::from_weights(
            &config, &mut wm, &device,
        )?;
        let head = burn::nn::LinearConfig::new(config.hidden_size, num_classes)
            .with_bias(true)
            .init(&device);
        let mut model = BrainiacModel {
            backbone,
            head,
            task,
            num_classes,
            hidden_dim: config.hidden_size,
            dropout_rate: config.dropout,
        };
        let init_ms = t_init.elapsed().as_secs_f64() * 1000.0;

        let t_load = Instant::now();

        // Load task head weights if provided
        if let Some(head_path) = head_weights {
            let mut head_wm = WeightMap::from_file(head_path)?;
            eprintln!("Loaded {} head weight tensors", head_wm.len());
            load_classifier_weights(&mut head_wm, &mut model.head, "fc", &device)?;
        }
        let load_ms = t_load.elapsed().as_secs_f64() * 1000.0;
        let total_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let timings = LoadTimings {
            weights_ms: disk_ms + load_ms,
            model_init_ms: init_ms,
            total_ms,
        };

        Ok((Self { model, config, device }, timings))
    }

    /// Model description string.
    pub fn describe(&self) -> String {
        crate::model::backbone::describe_model(&self.config)
    }

    /// Preprocess a NIfTI file and return the flat volume data ready for inference.
    pub fn preprocess_nifti(&self, path: &Path) -> Result<Vec<f32>> {
        let volume = nifti::read_nifti(path)?;
        let preprocessed = preprocessing::preprocess(&volume, self.config.img_size);
        Ok(preprocessed)
    }

    /// Run inference on a single NIfTI file.
    pub fn infer_nifti(&self, path: &Path) -> Result<InferenceOutput> {
        let data = self.preprocess_nifti(path)?;
        Ok(self.model.infer(&[&data], &self.device))
    }

    /// Run inference on multiple NIfTI files (for dual/quad tasks).
    pub fn infer_multi_nifti(&self, paths: &[&Path]) -> Result<InferenceOutput> {
        let data: Vec<Vec<f32>> = paths
            .iter()
            .map(|p| self.preprocess_nifti(p))
            .collect::<Result<Vec<_>>>()?;
        let refs: Vec<&[f32]> = data.iter().map(|d| d.as_slice()).collect();
        Ok(self.model.infer(&refs, &self.device))
    }

    /// Extract 768-dim feature vector from a NIfTI file.
    pub fn encode_nifti(&self, path: &Path) -> Result<Vec<f32>> {
        let data = self.preprocess_nifti(path)?;
        let features = self.model.extract_features(&data, &self.device);
        Ok(features.to_data().to_vec().unwrap())
    }

    /// Extract features with attention weights (for saliency maps).
    pub fn encode_with_attention(&self, path: &Path) -> Result<(Vec<f32>, Vec<Vec<f32>>)> {
        let data = self.preprocess_nifti(path)?;
        let (features, attn_maps) = self.model.backbone.forward_with_attn(&data, 1, &self.device);

        let feat_vec: Vec<f32> = features.to_data().to_vec().unwrap();
        let attn_vecs: Vec<Vec<f32>> = attn_maps
            .into_iter()
            .map(|a| a.to_data().to_vec().unwrap())
            .collect();

        Ok((feat_vec, attn_vecs))
    }
}
