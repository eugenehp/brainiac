/// BrainIAC fine-tuning CLI.
///
/// Fine-tune the BrainIAC backbone on downstream tasks.
///
/// Build:
///   cargo build --release
///
/// Usage:
///   train --config train_config.json
///
///   # Or specify everything on the command line:
///   train --weights backbone.safetensors \
///         --task brain_age \
///         --train-csv data/train.csv \
///         --val-csv data/val.csv \
///         --root-dir data/images \
///         --epochs 200 \
///         --lr 0.001

use clap::Parser;
use brainiac::config::{TrainConfig, TaskType, ModelConfig};
use brainiac::training;

// ── Backend (Autodiff wrapping NdArray or Wgpu) ──────────────────────────────
#[cfg(feature = "ndarray")]
mod backend {
    pub type InnerB = burn::backend::NdArray;
    pub type B = burn::backend::Autodiff<InnerB>;
    pub type Device = burn::backend::ndarray::NdArrayDevice;
    pub fn device() -> Device { Device::Cpu }
    pub const NAME: &str = "CPU (NdArray + Autodiff)";
}

#[cfg(all(feature = "wgpu", not(feature = "ndarray")))]
mod backend {
    pub type InnerB = burn::backend::Wgpu;
    pub type B = burn::backend::Autodiff<InnerB>;
    pub type Device = burn::backend::wgpu::WgpuDevice;
    pub fn device() -> Device { Device::DefaultDevice }
    pub const NAME: &str = "GPU (wgpu + Autodiff)";
}

use backend::{B, device};

// ── CLI ──────────────────────────────────────────────────────────────────────
#[derive(Parser, Debug)]
#[command(name = "brainiac-train")]
#[command(about = "BrainIAC fine-tuning on downstream tasks")]
struct Args {
    /// Path to JSON training config file.
    #[arg(long)]
    config: Option<String>,

    /// Path to backbone .safetensors weights.
    #[arg(long)]
    weights: Option<String>,

    /// Task: brain_age, mci, stroke, sequence, idh, survival
    #[arg(long)]
    task: Option<String>,

    /// Training CSV file.
    #[arg(long)]
    train_csv: Option<String>,

    /// Validation CSV file.
    #[arg(long)]
    val_csv: Option<String>,

    /// Root directory for NIfTI images.
    #[arg(long)]
    root_dir: Option<String>,

    /// Maximum epochs.
    #[arg(long)]
    epochs: Option<usize>,

    /// Learning rate.
    #[arg(long)]
    lr: Option<f64>,

    /// Weight decay.
    #[arg(long)]
    weight_decay: Option<f64>,

    /// Freeze backbone (linear probing).
    #[arg(long)]
    freeze: bool,

    /// Checkpoint save directory.
    #[arg(long)]
    save_dir: Option<String>,

    /// Log every N steps.
    #[arg(long)]
    log_interval: Option<usize>,
}

fn parse_task(s: &str) -> (TaskType, usize) {
    match s {
        "brain_age" | "brainage" => (TaskType::Regression, 1),
        "stroke" | "timetostroke" => (TaskType::Regression, 1),
        "mci" => (TaskType::BinaryClassification, 1),
        "idh" => (TaskType::DualBinaryClassification, 1),
        "survival" | "os" => (TaskType::QuadBinaryClassification, 1),
        "sequence" | "multiclass" => (TaskType::MulticlassClassification, 4),
        _ => {
            eprintln!("Unknown task '{}', defaulting to regression", s);
            (TaskType::Regression, 1)
        }
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let dev = device();

    println!("Backend: {}", backend::NAME);

    // Build config from file or CLI args
    let config = if let Some(config_path) = &args.config {
        let json = std::fs::read_to_string(config_path)?;
        serde_json::from_str::<TrainConfig>(&json)?
    } else {
        // Build from CLI args
        let weights = args.weights.as_deref()
            .expect("--weights required when not using --config");
        let task_str = args.task.as_deref()
            .expect("--task required when not using --config");
        let train_csv = args.train_csv.as_deref()
            .expect("--train-csv required when not using --config");
        let val_csv = args.val_csv.as_deref()
            .expect("--val-csv required when not using --config");
        let root_dir = args.root_dir.as_deref()
            .expect("--root-dir required when not using --config");

        let (task, num_classes) = parse_task(task_str);

        TrainConfig {
            task,
            num_classes,
            max_epochs: args.epochs.unwrap_or(200),
            batch_size: 16,
            lr: args.lr.unwrap_or(1e-3),
            weight_decay: args.weight_decay.unwrap_or(1e-4),
            freeze_backbone: args.freeze,
            backbone_weights: weights.to_string(),
            train_csv: train_csv.to_string(),
            val_csv: val_csv.to_string(),
            root_dir: root_dir.to_string(),
            save_dir: args.save_dir.unwrap_or_else(|| "./checkpoints".to_string()),
            model: ModelConfig::default(),
            log_interval: args.log_interval.unwrap_or(10),
            cosine_t0: 50,
            num_workers: 4,
        }
    };

    // Dispatch to training
    match config.task {
        TaskType::Regression
        | TaskType::BinaryClassification
        | TaskType::MulticlassClassification => {
            let state = training::train_single_scan::<B>(&config, dev)?;
            println!("\nFinal: {} epochs, best metric: {:.6}", state.epoch, state.best_metric);
        }
        TaskType::DualBinaryClassification => {
            let state = training::train_dual_scan::<B>(
                &config, ["t2f", "t1ce"], dev,
            )?;
            println!("\nFinal: {} epochs, best AUC: {:.6}", state.epoch, state.best_metric);
        }
        TaskType::QuadBinaryClassification => {
            let state = training::train_quad_scan::<B>(
                &config, ["t1ce", "t1n", "t2w", "t2f"], dev,
            )?;
            println!("\nFinal: {} epochs, best AUC: {:.6}", state.epoch, state.best_metric);
        }
        TaskType::FeatureExtraction => {
            anyhow::bail!("Cannot train with FeatureExtraction task type. Choose a downstream task.");
        }
    }

    Ok(())
}
