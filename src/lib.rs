//! # brainiac — BrainIAC Brain MRI Foundation Model inference in Rust
//!
//! Pure-Rust inference for the BrainIAC (Brain Imaging Adaptive Core)
//! foundation model, built on [Burn 0.20](https://burn.dev).
//!
//! BrainIAC is a Vision Transformer (ViT-B/16) pretrained with SimCLR
//! on structural brain MRI, supporting multiple downstream tasks:
//!
//! - Brain age prediction (regression)
//! - IDH mutation classification (binary, dual-scan)
//! - Mild cognitive impairment classification (binary)
//! - Diffuse glioma overall survival prediction (binary, quad-scan)
//! - MR sequence classification (4-class)
//! - Time-to-stroke prediction (regression)
//! - Feature extraction (768-dim embedding)
//!
//! ## Quick start
//!
//! ```rust,ignore
//! use brainiac::{BrainiacEncoder, TaskType};
//!
//! // Feature extraction
//! let (encoder, _timings) = BrainiacEncoder::<B>::load(
//!     "brainiac_backbone.safetensors",
//!     None,
//!     TaskType::FeatureExtraction,
//!     1,
//!     device,
//! )?;
//!
//! let features = encoder.encode_nifti(Path::new("brain_t1.nii.gz"))?;
//! println!("Feature dim: {}", features.len()); // 768
//! ```

pub mod config;
pub mod nifti;
pub mod preprocessing;
pub mod weights;
pub mod model;
pub mod encoder;
pub mod data;
pub mod losses;
pub mod metrics;
pub mod training;
pub mod saliency;
pub mod augmentation;
pub mod mri_preprocess;
pub mod checkpoint;
pub mod batch_inference;
pub mod fast_load;

// Flat re-exports for ergonomic API
pub use config::{ModelConfig, TaskType, InferenceConfig, TrainConfig};
pub use encoder::BrainiacEncoder;
pub use model::classifier::{BrainiacModel, InferenceOutput};
pub use nifti::{NiftiVolume, read_nifti};
pub use preprocessing::preprocess;
pub use saliency::SaliencyMap;
pub use augmentation::AugConfig;
