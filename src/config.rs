/// Model and runtime configuration for BrainIAC inference.
///
/// Field names match the MONAI ViT hyperparameters used in the Python codebase.

use serde::Deserialize;

/// BrainIAC ViT-B backbone configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    /// Input image size (isotropic, single value for all 3 dims).
    #[serde(default = "default_img_size")]
    pub img_size: usize,

    /// Spatial patch size (isotropic).
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,

    /// Number of input channels (1 for structural MRI).
    #[serde(default = "default_in_channels")]
    pub in_channels: usize,

    /// Transformer hidden dimension.
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,

    /// MLP intermediate dimension.
    #[serde(default = "default_mlp_dim")]
    pub mlp_dim: usize,

    /// Number of transformer layers.
    #[serde(default = "default_num_layers")]
    pub num_layers: usize,

    /// Number of attention heads.
    #[serde(default = "default_num_heads")]
    pub num_heads: usize,

    /// Layer norm epsilon.
    #[serde(default = "default_norm_eps")]
    pub norm_eps: f64,

    /// Dropout rate for downstream heads.
    #[serde(default = "default_dropout")]
    pub dropout: f64,
}

fn default_img_size() -> usize { 96 }
fn default_patch_size() -> usize { 16 }
fn default_in_channels() -> usize { 1 }
fn default_hidden_size() -> usize { 768 }
fn default_mlp_dim() -> usize { 3072 }
fn default_num_layers() -> usize { 12 }
fn default_num_heads() -> usize { 12 }
fn default_norm_eps() -> f64 { 1e-6 }
fn default_dropout() -> f64 { 0.2 }

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            img_size: default_img_size(),
            patch_size: default_patch_size(),
            in_channels: default_in_channels(),
            hidden_size: default_hidden_size(),
            mlp_dim: default_mlp_dim(),
            num_layers: default_num_layers(),
            num_heads: default_num_heads(),
            norm_eps: default_norm_eps(),
            dropout: default_dropout(),
        }
    }
}

impl ModelConfig {
    /// Load from a JSON config file.
    pub fn from_json(path: &str) -> anyhow::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&json)?)
    }

    /// Number of patches per spatial axis.
    pub fn patches_per_axis(&self) -> usize {
        self.img_size / self.patch_size
    }

    /// Total number of 3D patches (no CLS token).
    pub fn num_patches(&self) -> usize {
        let p = self.patches_per_axis();
        p * p * p
    }

    /// Total sequence length (patches only, no CLS in MONAI ViT).
    pub fn seq_len(&self) -> usize {
        self.num_patches()
    }

    /// Attention head dimension.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }

    /// Conv3d kernel volume: in_channels × patch_size³.
    pub fn kernel_volume(&self) -> usize {
        self.in_channels * self.patch_size * self.patch_size * self.patch_size
    }
}

/// Downstream task type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskType {
    /// Single-scan regression (brain age, time-to-stroke).
    Regression,
    /// Single-scan binary classification (MCI).
    BinaryClassification,
    /// Single-scan multi-class classification (MR sequence).
    MulticlassClassification,
    /// Dual-scan binary classification (IDH mutation: FLAIR + T1CE).
    DualBinaryClassification,
    /// Quad-scan binary classification (overall survival: T1 + T1CE + T2 + FLAIR).
    QuadBinaryClassification,
    /// Feature extraction only (no head).
    FeatureExtraction,
}

/// Full inference configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct InferenceConfig {
    /// Model architecture config.
    #[serde(default)]
    pub model: ModelConfig,
    /// Task type.
    pub task: TaskType,
    /// Number of output classes (1 for regression/binary, 4 for sequence classification).
    #[serde(default = "default_num_classes")]
    pub num_classes: usize,
}

fn default_num_classes() -> usize { 1 }

// ── Training configuration ────────────────────────────────────────────────────

/// Training hyperparameters matching the Python config_finetune.yml.
#[derive(Debug, Clone, Deserialize)]
pub struct TrainConfig {
    /// Task type for training.
    pub task: TaskType,

    /// Number of output classes.
    #[serde(default = "default_num_classes")]
    pub num_classes: usize,

    /// Maximum training epochs.
    #[serde(default = "default_max_epochs")]
    pub max_epochs: usize,

    /// Batch size.
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,

    /// Learning rate.
    #[serde(default = "default_lr")]
    pub lr: f64,

    /// Weight decay (L2 regularization).
    #[serde(default = "default_weight_decay")]
    pub weight_decay: f64,

    /// Freeze backbone weights (linear probing).
    #[serde(default)]
    pub freeze_backbone: bool,

    /// Path to backbone weights (.safetensors).
    pub backbone_weights: String,

    /// Training CSV file path.
    pub train_csv: String,

    /// Validation CSV file path.
    pub val_csv: String,

    /// Root directory for NIfTI images.
    pub root_dir: String,

    /// Directory to save checkpoints.
    #[serde(default = "default_save_dir")]
    pub save_dir: String,

    /// Model architecture config.
    #[serde(default)]
    pub model: ModelConfig,

    /// Log metrics every N batches.
    #[serde(default = "default_log_interval")]
    pub log_interval: usize,

    /// Cosine annealing T_0 (restart period in epochs).
    #[serde(default = "default_cosine_t0")]
    pub cosine_t0: usize,

    /// Number of dataloader workers.
    #[serde(default = "default_num_workers")]
    pub num_workers: usize,
}

fn default_max_epochs() -> usize { 200 }
fn default_batch_size() -> usize { 16 }
fn default_lr() -> f64 { 1e-3 }
fn default_weight_decay() -> f64 { 1e-4 }
fn default_save_dir() -> String { "./checkpoints".to_string() }
fn default_log_interval() -> usize { 10 }
fn default_cosine_t0() -> usize { 50 }
fn default_num_workers() -> usize { 4 }
