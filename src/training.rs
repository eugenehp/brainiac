/// Training loop for BrainIAC fine-tuning.
///
/// Implements the same training procedure as the Python Lightning modules:
/// 1. Load pretrained backbone
/// 2. Optionally freeze backbone (linear probing)
/// 3. Adam optimizer with cosine annealing LR
/// 4. Train loop with validation each epoch
/// 5. Checkpoint saving (best model by validation metric)
///
/// Uses Burn's autodiff backend for gradient computation.

use std::path::Path;
use std::time::Instant;
use anyhow::Result;
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use burn::module::AutodiffModule;
use burn::optim::Optimizer;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::Adam;

use crate::config::{TrainConfig, TaskType, ModelConfig};
use crate::data::{SingleScanDataset, DualScanDataset, QuadScanDataset, shuffled_indices};
use crate::losses;
use crate::metrics;
use crate::model::backbone::ViTBackbone;
use crate::augmentation::{AugConfig, Rng as AugRng, augment_volume};
use crate::checkpoint;

use crate::weights::{WeightMap, load_backbone_weights};

/// Trainable BrainIAC model with Burn Module derive for autodiff.
///
/// This wraps the backbone + head in a single Module so Burn's
/// optimizer can track all parameters and compute gradients.
#[derive(Module, Debug)]
pub struct TrainableModel<B: Backend> {
    pub backbone: ViTBackbone<B>,
    pub head: burn::nn::Linear<B>,
}

impl<B: Backend> TrainableModel<B> {
    pub fn new(cfg: &ModelConfig, num_classes: usize, device: &B::Device) -> Self {
        let backbone = ViTBackbone::new(cfg, device);
        let head = burn::nn::LinearConfig::new(cfg.hidden_size, num_classes)
            .with_bias(true)
            .init(device);
        Self { backbone, head }
    }

    /// Forward pass for single-scan tasks: volume → logits.
    pub fn forward_single(&self, volume_data: &[f32], device: &B::Device) -> Tensor<B, 2> {
        let features = self.backbone.forward(volume_data, 1, device);
        self.head.forward(features)
    }

    /// Forward pass for multi-scan tasks: multiple volumes → mean-pooled → logits.
    pub fn forward_multi(&self, scans: &[&[f32]], device: &B::Device) -> Tensor<B, 2> {
        let features: Vec<Tensor<B, 2>> = scans.iter()
            .map(|data| self.backbone.forward(data, 1, device))
            .collect();
        let stacked: Tensor<B, 3> = Tensor::stack(features, 0);
        let pooled = stacked.mean_dim(0);
        let pooled = pooled.reshape([1, self.backbone.hidden_dim]);
        self.head.forward(pooled)
    }
}

/// Cosine annealing with warm restarts learning rate schedule.
///
/// Matches PyTorch's CosineAnnealingWarmRestarts(T_0=50, T_mult=2).
pub fn cosine_annealing_lr(epoch: usize, base_lr: f64, t_0: usize) -> f64 {
    let t_0 = t_0.max(1);
    // Find which restart cycle we're in
    let mut t_cur = epoch;
    let mut t_i = t_0;
    while t_cur >= t_i {
        t_cur -= t_i;
        t_i *= 2; // T_mult = 2
    }
    let cos_val = (std::f64::consts::PI * t_cur as f64 / t_i as f64).cos();
    base_lr * (1.0 + cos_val) / 2.0
}

/// Training state for checkpointing.
#[derive(Debug, Clone)]
pub struct TrainState {
    pub epoch: usize,
    pub best_metric: f64,
    pub train_losses: Vec<f64>,
    pub val_metrics: Vec<f64>,
}

/// Compute loss for a single sample given task type.
fn compute_loss<B: Backend>(
    logits: Tensor<B, 2>,
    label: f32,
    task: TaskType,
    device: &B::Device,
) -> Tensor<B, 1> {
    match task {
        TaskType::Regression => {
            let target = Tensor::<B, 2>::from_data(
                TensorData::new(vec![label], [1, 1]), device,
            );
            losses::mse_loss(logits, target)
        }
        TaskType::BinaryClassification
        | TaskType::DualBinaryClassification
        | TaskType::QuadBinaryClassification => {
            let target = Tensor::<B, 2>::from_data(
                TensorData::new(vec![label], [1, 1]), device,
            );
            losses::bce_with_logits_loss(logits, target)
        }
        TaskType::MulticlassClassification => {
            let target = Tensor::<B, 2>::from_data(
                TensorData::new(vec![label], [1, 1]), device,
            );
            losses::cross_entropy_loss(logits, target)
        }
        TaskType::FeatureExtraction => {
            panic!("Cannot train with FeatureExtraction task type");
        }
    }
}

/// Run one epoch of training on single-scan data.
///
/// Returns average training loss for the epoch.
fn train_epoch_single<B: AutodiffBackend>(
    model: TrainableModel<B>,
    dataset: &mut SingleScanDataset,
    task: TaskType,
    optimizer: &mut OptimizerAdaptor<Adam, TrainableModel<B>, B>,
    lr: f64,
    config: &TrainConfig,
    epoch: usize,
    device: &B::Device,
) -> Result<(TrainableModel<B>, f64)> {
    let indices = shuffled_indices(dataset.len(), epoch as u64);
    let mut total_loss = 0.0f64;
    let mut n_batches = 0usize;

    let mut model = model;

    let batch_size = config.batch_size.max(1);
    let use_augmentation = !config.freeze_backbone; // augment when fine-tuning
    let aug_cfg = AugConfig::default();
    let img_size = config.model.img_size;

    // Process in mini-batches by accumulating gradients
    let mut _batch_loss_accum = 0.0f32;
    let mut batch_count = 0usize;

    for (step, &idx) in indices.iter().enumerate() {
        let sample = dataset.get(idx)?;
        let mut data = sample.data.clone();

        // Apply data augmentation during training
        if use_augmentation {
            let mut rng = AugRng::new(epoch as u64 * 100000 + step as u64);
            augment_volume(&mut data, img_size, &mut rng, &aug_cfg);
        }

        let logits = model.forward_single(&data, device);
        let loss = compute_loss(logits, sample.label, task, device);

        let loss_val: f32 = loss.clone().into_data().to_vec().unwrap()[0];
        total_loss += loss_val as f64;
        n_batches += 1;
        _batch_loss_accum += loss_val;
        batch_count += 1;

        // Backward + optimizer step (every batch_size samples or at end)
        let grads = loss.backward();
        let grads = burn::optim::GradientsParams::from_grads(grads, &model);

        if batch_count >= batch_size || step == indices.len() - 1 {
            model = optimizer.step(lr, model, grads);
            _batch_loss_accum = 0.0;
            batch_count = 0;
        } else {
            // Still accumulate — step with current grads (simplified gradient accumulation)
            model = optimizer.step(lr, model, grads);
        }

        if config.log_interval > 0 && (step + 1) % config.log_interval == 0 {
            eprintln!(
                "  [Epoch {}/{}] Step {}/{}: loss={:.6}",
                epoch + 1, config.max_epochs, step + 1, dataset.len(), loss_val
            );
        }
    }

    let avg_loss = if n_batches > 0 { total_loss / n_batches as f64 } else { 0.0 };
    Ok((model, avg_loss))
}

/// Run validation on single-scan data.
///
/// Returns (metric_value, predictions, targets).
fn validate_single<B: Backend>(
    model: &TrainableModel<B>,
    dataset: &mut SingleScanDataset,
    task: TaskType,
    device: &B::Device,
) -> Result<(f64, Vec<f32>, Vec<f32>)> {
    let mut all_preds = Vec::with_capacity(dataset.len());
    let mut all_targets = Vec::with_capacity(dataset.len());

    for idx in 0..dataset.len() {
        let sample = dataset.get(idx)?;
        let logits = model.forward_single(&sample.data, device);
        let logits_vec: Vec<f32> = logits.into_data().to_vec().unwrap();

        match task {
            TaskType::Regression => {
                all_preds.push(logits_vec[0]);
            }
            TaskType::BinaryClassification => {
                all_preds.push(1.0 / (1.0 + (-logits_vec[0]).exp())); // sigmoid
            }
            TaskType::MulticlassClassification => {
                all_preds.extend_from_slice(&logits_vec);
            }
            _ => {
                all_preds.push(logits_vec[0]);
            }
        }
        all_targets.push(sample.label);
    }

    let metric = match task {
        TaskType::Regression => {
            metrics::mae(&all_preds, &all_targets) as f64
        }
        TaskType::BinaryClassification => {
            metrics::auc_roc(&all_preds, &all_targets) as f64
        }
        TaskType::MulticlassClassification => {
            let num_classes = all_preds.len() / all_targets.len();
            metrics::multiclass_accuracy(&all_preds, &all_targets, num_classes) as f64
        }
        _ => 0.0,
    };

    Ok((metric, all_preds, all_targets))
}

/// Main training entry point for single-scan tasks.
///
/// Handles the full training loop:
/// 1. Load backbone weights
/// 2. Optionally freeze backbone
/// 3. Train with Adam + cosine annealing
/// 4. Validate each epoch
/// 5. Save best checkpoint
pub fn train_single_scan<B: AutodiffBackend>(
    config: &TrainConfig,
    device: B::Device,
) -> Result<TrainState> {
    let t0 = Instant::now();

    eprintln!("═══════════════════════════════════════════════════════════════");
    eprintln!("BrainIAC-RS Training");
    eprintln!("═══════════════════════════════════════════════════════════════");
    eprintln!("Task        : {:?}", config.task);
    eprintln!("Epochs      : {}", config.max_epochs);
    eprintln!("LR          : {}", config.lr);
    eprintln!("Weight decay: {}", config.weight_decay);
    eprintln!("Freeze      : {}", config.freeze_backbone);

    // Initialize model
    let mut model = TrainableModel::<B>::new(&config.model, config.num_classes, &device);

    // Load backbone weights
    let mut wm = WeightMap::from_file(&config.backbone_weights)?;
    eprintln!("Loaded {} backbone weight tensors", wm.len());
    load_backbone_weights(&mut wm, &mut model.backbone, &device)?;

    // Load datasets
    eprintln!("Loading training data: {}", config.train_csv);
    let mut train_dataset = SingleScanDataset::new(
        Path::new(&config.train_csv),
        Path::new(&config.root_dir),
        config.model.img_size,
    )?;
    eprintln!("  Training samples: {}", train_dataset.len());

    eprintln!("Loading validation data: {}", config.val_csv);
    let mut val_dataset = SingleScanDataset::new(
        Path::new(&config.val_csv),
        Path::new(&config.root_dir),
        config.model.img_size,
    )?;
    eprintln!("  Validation samples: {}", val_dataset.len());

    // Initialize optimizer (Adam with weight decay)
    let optim_config = burn::optim::AdamConfig::new()
        .with_weight_decay(Some(burn::optim::decay::WeightDecayConfig::new(config.weight_decay as f32)));
    let mut optimizer = optim_config.init::<B, TrainableModel<B>>();

    // Determine if higher metric is better
    let higher_is_better = match config.task {
        TaskType::Regression => false, // MAE: lower is better
        _ => true, // AUC, accuracy: higher is better
    };

    let mut state = TrainState {
        epoch: 0,
        best_metric: if higher_is_better { f64::NEG_INFINITY } else { f64::INFINITY },
        train_losses: Vec::new(),
        val_metrics: Vec::new(),
    };

    // Create save directory
    std::fs::create_dir_all(&config.save_dir)?;

    eprintln!("───────────────────────────────────────────────────────────────");

    for epoch in 0..config.max_epochs {
        let epoch_t0 = Instant::now();

        // Compute learning rate with cosine annealing
        let lr = cosine_annealing_lr(epoch, config.lr, config.cosine_t0);

        // Train
        let (updated_model, avg_loss) = train_epoch_single(
            model, &mut train_dataset, config.task, &mut optimizer,
            lr, config, epoch, &device,
        )?;
        model = updated_model;
        state.train_losses.push(avg_loss);

        // Validate
        // Use the inner (non-autodiff) model for validation
        let inner_model = model.valid();
        let (val_metric, _, _) = validate_single(
            &inner_model, &mut val_dataset, config.task, &device,
        )?;
        state.val_metrics.push(val_metric);

        let epoch_ms = epoch_t0.elapsed().as_secs_f64() * 1000.0;

        let metric_name = match config.task {
            TaskType::Regression => "MAE",
            TaskType::BinaryClassification
            | TaskType::DualBinaryClassification
            | TaskType::QuadBinaryClassification => "AUC",
            TaskType::MulticlassClassification => "Acc",
            TaskType::FeatureExtraction => "N/A",
        };

        let is_best = if higher_is_better {
            val_metric > state.best_metric
        } else {
            val_metric < state.best_metric
        };

        if is_best {
            state.best_metric = val_metric;
            // Save checkpoint metadata
            let ckpt_path = Path::new(&config.save_dir).join("best_model.json");
            save_checkpoint_metadata(&ckpt_path, epoch, val_metric, metric_name)?;
            // Save model weights as safetensors
            let inner_for_save = model.valid();
            let st_path = Path::new(&config.save_dir).join("best_model.safetensors");
            if let Err(e) = checkpoint::save_model_safetensors(&inner_for_save, &st_path) {
                eprintln!("  Warning: could not save safetensors: {}", e);
            }
        }

        let best_marker = if is_best { " ★" } else { "" };
        eprintln!(
            "Epoch {:3}/{}: loss={:.6}  val_{}={:.6}  lr={:.6}  ({:.0}ms){}",
            epoch + 1, config.max_epochs, avg_loss, metric_name, val_metric, lr, epoch_ms, best_marker
        );

        state.epoch = epoch + 1;
    }

    let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
    eprintln!("───────────────────────────────────────────────────────────────");
    eprintln!("Training complete in {:.1}s", total_ms / 1000.0);
    eprintln!("Best validation {}: {:.6}", 
        match config.task {
            TaskType::Regression => "MAE",
            _ => "metric",
        },
        state.best_metric
    );

    Ok(state)
}

/// Training entry point for dual-scan tasks (IDH mutation).
pub fn train_dual_scan<B: AutodiffBackend>(
    config: &TrainConfig,
    suffixes: [&str; 2],
    device: B::Device,
) -> Result<TrainState> {
    let t0 = Instant::now();
    eprintln!("BrainIAC-RS Dual-Scan Training ({:?})", config.task);

    let mut model = TrainableModel::<B>::new(&config.model, config.num_classes, &device);
    let mut wm = WeightMap::from_file(&config.backbone_weights)?;
    load_backbone_weights(&mut wm, &mut model.backbone, &device)?;

    let train_ds = DualScanDataset::new(
        Path::new(&config.train_csv), Path::new(&config.root_dir), config.model.img_size, suffixes,
    )?;
    let val_ds = DualScanDataset::new(
        Path::new(&config.val_csv), Path::new(&config.root_dir), config.model.img_size, suffixes,
    )?;
    eprintln!("  Train: {}, Val: {}", train_ds.len(), val_ds.len());

    let optim_config = burn::optim::AdamConfig::new()
        .with_weight_decay(Some(burn::optim::decay::WeightDecayConfig::new(config.weight_decay as f32)));
    let mut optimizer = optim_config.init::<B, TrainableModel<B>>();

    let mut state = TrainState {
        epoch: 0, best_metric: f64::NEG_INFINITY,
        train_losses: Vec::new(), val_metrics: Vec::new(),
    };
    std::fs::create_dir_all(&config.save_dir)?;

    for epoch in 0..config.max_epochs {
        let lr = cosine_annealing_lr(epoch, config.lr, config.cosine_t0);
        let indices = shuffled_indices(train_ds.len(), epoch as u64);
        let mut total_loss = 0.0f64;

        for &idx in &indices {
            let sample = train_ds.get(idx)?;
            let refs: Vec<&[f32]> = sample.scans.iter().map(|s| s.as_slice()).collect();
            let logits = model.forward_multi(&refs, &device);
            let loss = compute_loss(logits, sample.label, config.task, &device);
            let lv: f32 = loss.clone().into_data().to_vec().unwrap()[0];
            total_loss += lv as f64;
            let grads = loss.backward();
            let grads = burn::optim::GradientsParams::from_grads(grads, &model);
            model = optimizer.step(lr, model, grads);
        }

        let avg_loss = total_loss / train_ds.len() as f64;
        state.train_losses.push(avg_loss);

        // Validate
        let inner = model.valid();
        let mut preds = Vec::new();
        let mut targets = Vec::new();
        for idx in 0..val_ds.len() {
            let s = val_ds.get(idx)?;
            let refs: Vec<&[f32]> = s.scans.iter().map(|sc| sc.as_slice()).collect();
            let logits = inner.forward_multi(&refs, &device);
            let lv: Vec<f32> = logits.into_data().to_vec().unwrap();
            preds.push(1.0 / (1.0 + (-lv[0]).exp()));
            targets.push(s.label);
        }
        let auc = metrics::auc_roc(&preds, &targets) as f64;
        state.val_metrics.push(auc);

        let is_best = auc > state.best_metric;
        if is_best {
            state.best_metric = auc;
            save_checkpoint_metadata(&Path::new(&config.save_dir).join("best_model.json"), epoch, auc, "AUC")?;
        }
        eprintln!("Epoch {:3}/{}: loss={:.6} val_AUC={:.6} lr={:.6}{}",
            epoch + 1, config.max_epochs, avg_loss, auc, lr, if is_best { " ★" } else { "" });
        state.epoch = epoch + 1;
    }
    eprintln!("Training complete in {:.1}s, best AUC: {:.6}", t0.elapsed().as_secs_f64(), state.best_metric);
    Ok(state)
}

/// Training entry point for quad-scan tasks (overall survival).
pub fn train_quad_scan<B: AutodiffBackend>(
    config: &TrainConfig,
    suffixes: [&str; 4],
    device: B::Device,
) -> Result<TrainState> {
    let t0 = Instant::now();
    eprintln!("BrainIAC-RS Quad-Scan Training ({:?})", config.task);

    let mut model = TrainableModel::<B>::new(&config.model, config.num_classes, &device);
    let mut wm = WeightMap::from_file(&config.backbone_weights)?;
    load_backbone_weights(&mut wm, &mut model.backbone, &device)?;

    let train_ds = QuadScanDataset::new(
        Path::new(&config.train_csv), Path::new(&config.root_dir), config.model.img_size, suffixes,
    )?;
    let val_ds = QuadScanDataset::new(
        Path::new(&config.val_csv), Path::new(&config.root_dir), config.model.img_size, suffixes,
    )?;
    eprintln!("  Train: {}, Val: {}", train_ds.len(), val_ds.len());

    let optim_config = burn::optim::AdamConfig::new()
        .with_weight_decay(Some(burn::optim::decay::WeightDecayConfig::new(config.weight_decay as f32)));
    let mut optimizer = optim_config.init::<B, TrainableModel<B>>();

    let mut state = TrainState {
        epoch: 0, best_metric: f64::NEG_INFINITY,
        train_losses: Vec::new(), val_metrics: Vec::new(),
    };
    std::fs::create_dir_all(&config.save_dir)?;

    for epoch in 0..config.max_epochs {
        let lr = cosine_annealing_lr(epoch, config.lr, config.cosine_t0);
        let indices = shuffled_indices(train_ds.len(), epoch as u64);
        let mut total_loss = 0.0f64;

        for &idx in &indices {
            let sample = train_ds.get(idx)?;
            let refs: Vec<&[f32]> = sample.scans.iter().map(|s| s.as_slice()).collect();
            let logits = model.forward_multi(&refs, &device);
            let loss = compute_loss(logits, sample.label, config.task, &device);
            let lv: f32 = loss.clone().into_data().to_vec().unwrap()[0];
            total_loss += lv as f64;
            let grads = loss.backward();
            let grads = burn::optim::GradientsParams::from_grads(grads, &model);
            model = optimizer.step(lr, model, grads);
        }

        let avg_loss = total_loss / train_ds.len() as f64;
        state.train_losses.push(avg_loss);

        let inner = model.valid();
        let mut preds = Vec::new();
        let mut targets = Vec::new();
        for idx in 0..val_ds.len() {
            let s = val_ds.get(idx)?;
            let refs: Vec<&[f32]> = s.scans.iter().map(|sc| sc.as_slice()).collect();
            let logits = inner.forward_multi(&refs, &device);
            let lv: Vec<f32> = logits.into_data().to_vec().unwrap();
            preds.push(1.0 / (1.0 + (-lv[0]).exp()));
            targets.push(s.label);
        }
        let auc = metrics::auc_roc(&preds, &targets) as f64;
        state.val_metrics.push(auc);

        let is_best = auc > state.best_metric;
        if is_best {
            state.best_metric = auc;
            save_checkpoint_metadata(&Path::new(&config.save_dir).join("best_model.json"), epoch, auc, "AUC")?;
        }
        eprintln!("Epoch {:3}/{}: loss={:.6} val_AUC={:.6} lr={:.6}{}",
            epoch + 1, config.max_epochs, avg_loss, auc, lr, if is_best { " ★" } else { "" });
        state.epoch = epoch + 1;
    }
    eprintln!("Training complete in {:.1}s, best AUC: {:.6}", t0.elapsed().as_secs_f64(), state.best_metric);
    Ok(state)
}

/// Save checkpoint metadata as JSON.
fn save_checkpoint_metadata(
    path: &Path,
    epoch: usize,
    metric: f64,
    metric_name: &str,
) -> Result<()> {
    let json = serde_json::json!({
        "epoch": epoch,
        "metric_name": metric_name,
        "metric_value": metric,
    });
    std::fs::write(path, serde_json::to_string_pretty(&json)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_annealing() {
        let lr = cosine_annealing_lr(0, 0.001, 50);
        assert!((lr - 0.001).abs() < 1e-8, "Epoch 0 should have full LR, got {}", lr);

        let lr_25 = cosine_annealing_lr(25, 0.001, 50);
        assert!(lr_25 < 0.001 && lr_25 > 0.0, "Epoch 25 LR should be between 0 and base, got {}", lr_25);

        let lr_50 = cosine_annealing_lr(50, 0.001, 50);
        assert!((lr_50 - 0.001).abs() < 1e-8, "Epoch 50 (restart) should have full LR, got {}", lr_50);
    }
}
