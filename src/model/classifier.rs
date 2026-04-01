/// Downstream task classifier heads for BrainIAC.
///
/// Mirrors the Python model classes:
/// - `SingleScanModel`:  backbone → dropout → linear
/// - `SingleScanModelBP`: N scans → backbone each → mean pool → dropout → linear
/// - `SingleScanModelQuad`: 4 scans → backbone each → mean pool → dropout → linear
///
/// All share the same pattern: extract CLS tokens, optionally pool, then classify.

use burn::prelude::*;
use burn::nn::{Linear, LinearConfig};

use crate::config::{ModelConfig, TaskType};
use crate::model::backbone::ViTBackbone;

/// A complete BrainIAC model: backbone + task-specific head.
pub struct BrainiacModel<B: Backend> {
    pub backbone: ViTBackbone<B>,
    pub head: Linear<B>,
    pub task: TaskType,
    pub num_classes: usize,
    pub hidden_dim: usize,
    pub dropout_rate: f64,
}

impl<B: Backend> BrainiacModel<B> {
    /// Create a new model (weights initialized to zero).
    pub fn new(cfg: &ModelConfig, task: TaskType, num_classes: usize, device: &B::Device) -> Self {
        let backbone = ViTBackbone::new(cfg, device);
        let head = LinearConfig::new(cfg.hidden_size, num_classes)
            .with_bias(true)
            .init(device);

        Self {
            backbone,
            head,
            task,
            num_classes,
            hidden_dim: cfg.hidden_size,
            dropout_rate: cfg.dropout,
        }
    }

    /// Run inference on a single preprocessed volume.
    ///
    /// `volume_data`: flat f32 of shape [1, C, D, H, W] (preprocessed).
    ///
    /// Returns raw logits [1, num_classes].
    pub fn predict_single(
        &self,
        volume_data: &[f32],
        device: &B::Device,
    ) -> Tensor<B, 2> {
        let features = self.backbone.forward(volume_data, 1, device);
        // No dropout at inference
        self.head.forward(features)
    }

    /// Run inference on multiple scans (dual or quad) with mean pooling.
    ///
    /// `scan_data`: Vec of flat f32 arrays, each [1, C, D, H, W].
    ///
    /// Returns raw logits [1, num_classes].
    pub fn predict_multi(
        &self,
        scan_data: &[&[f32]],
        device: &B::Device,
    ) -> Tensor<B, 2> {
        // Extract features for each scan
        let features: Vec<Tensor<B, 2>> = scan_data
            .iter()
            .map(|data| self.backbone.forward(data, 1, device))
            .collect();

        // Stack and mean pool: [N_scans, D] → [1, D]
        let stacked: Tensor<B, 3> = Tensor::stack(features, 0); // [N_scans, 1, D]
        let pooled = stacked.mean_dim(0); // [1, 1, D]
        let pooled = pooled.reshape([1, self.hidden_dim]); // [1, D]

        // No dropout at inference
        self.head.forward(pooled)
    }

    /// Extract CLS token features without the classification head.
    ///
    /// Returns [1, hidden_dim].
    pub fn extract_features(
        &self,
        volume_data: &[f32],
        device: &B::Device,
    ) -> Tensor<B, 2> {
        self.backbone.forward(volume_data, 1, device)
    }

    /// Run inference and return interpreted output.
    pub fn infer(
        &self,
        scan_data: &[&[f32]],
        device: &B::Device,
    ) -> InferenceOutput {
        let logits = match self.task {
            TaskType::FeatureExtraction => {
                return InferenceOutput::Features(
                    self.extract_features(scan_data[0], device)
                        .to_data().to_vec().unwrap()
                );
            }
            TaskType::Regression
            | TaskType::BinaryClassification
            | TaskType::MulticlassClassification => {
                self.predict_single(scan_data[0], device)
            }
            TaskType::DualBinaryClassification => {
                assert!(scan_data.len() == 2, "Dual task requires 2 scans");
                self.predict_multi(scan_data, device)
            }
            TaskType::QuadBinaryClassification => {
                assert!(scan_data.len() == 4, "Quad task requires 4 scans");
                self.predict_multi(scan_data, device)
            }
        };

        let logits_vec: Vec<f32> = logits.to_data().to_vec().unwrap();

        match self.task {
            TaskType::Regression => {
                InferenceOutput::Regression(logits_vec[0])
            }
            TaskType::BinaryClassification
            | TaskType::DualBinaryClassification
            | TaskType::QuadBinaryClassification => {
                let prob = sigmoid(logits_vec[0]);
                InferenceOutput::BinaryClassification {
                    probability: prob,
                    predicted_class: if prob > 0.5 { 1 } else { 0 },
                }
            }
            TaskType::MulticlassClassification => {
                let probs = softmax_vec(&logits_vec);
                let predicted = probs
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                InferenceOutput::MulticlassClassification {
                    probabilities: probs,
                    predicted_class: predicted,
                }
            }
            TaskType::FeatureExtraction => unreachable!(),
        }
    }
}

/// Structured inference output.
#[derive(Debug, Clone)]
pub enum InferenceOutput {
    /// Regression: raw predicted value.
    Regression(f32),
    /// Binary classification: probability and predicted class (0 or 1).
    BinaryClassification {
        probability: f32,
        predicted_class: usize,
    },
    /// Multi-class classification: per-class probabilities and predicted class.
    MulticlassClassification {
        probabilities: Vec<f32>,
        predicted_class: usize,
    },
    /// Feature extraction: raw 768-dim feature vector.
    Features(Vec<f32>),
}

impl std::fmt::Display for InferenceOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InferenceOutput::Regression(v) => write!(f, "Regression: {:.4}", v),
            InferenceOutput::BinaryClassification { probability, predicted_class } => {
                write!(f, "Binary: class={} (p={:.4})", predicted_class, probability)
            }
            InferenceOutput::MulticlassClassification { probabilities, predicted_class } => {
                write!(f, "Multiclass: class={} (probs={:?})", predicted_class, probabilities)
            }
            InferenceOutput::Features(feats) => {
                let mean: f32 = feats.iter().sum::<f32>() / feats.len() as f32;
                write!(f, "Features: dim={}, mean={:.6}", feats.len(), mean)
            }
        }
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn softmax_vec(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}
