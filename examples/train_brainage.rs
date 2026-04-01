/// Example: Fine-tune BrainIAC for brain age prediction.
///
/// Usage:
///   cargo run --release --example train_brainage -- \
///     --weights brainiac_backbone.safetensors \
///     --train-csv data/csvs/brainage_train.csv \
///     --val-csv data/csvs/brainage_val.csv \
///     --root-dir data/images \
///     --epochs 50 \
///     --lr 0.001

use brainiac::config::{TrainConfig, TaskType, ModelConfig};
use brainiac::training;

type InnerB = burn::backend::NdArray;
type B = burn::backend::Autodiff<InnerB>;
type Device = burn::backend::ndarray::NdArrayDevice;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();

    let get_arg = |name: &str| -> Option<String> {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };

    let weights = get_arg("--weights").expect("--weights <path> required");
    let train_csv = get_arg("--train-csv").expect("--train-csv <path> required");
    let val_csv = get_arg("--val-csv").expect("--val-csv <path> required");
    let root_dir = get_arg("--root-dir").expect("--root-dir <path> required");
    let epochs: usize = get_arg("--epochs").and_then(|s| s.parse().ok()).unwrap_or(50);
    let lr: f64 = get_arg("--lr").and_then(|s| s.parse().ok()).unwrap_or(1e-3);
    let freeze = args.iter().any(|a| a == "--freeze");

    let config = TrainConfig {
        task: TaskType::Regression,
        num_classes: 1,
        max_epochs: epochs,
        batch_size: 16,
        lr,
        weight_decay: 1e-4,
        freeze_backbone: freeze,
        backbone_weights: weights,
        train_csv,
        val_csv,
        root_dir,
        save_dir: "./checkpoints/brainage".to_string(),
        model: ModelConfig::default(),
        log_interval: 10,
        cosine_t0: 50,
        num_workers: 4,
    };

    println!("Training brain age prediction model");
    if freeze {
        println!("  Mode: Linear probing (backbone frozen)");
    } else {
        println!("  Mode: End-to-end fine-tuning");
    }

    let state = training::train_single_scan::<B>(&config, Device::Cpu)?;
    println!("\nDone! Best validation MAE: {:.4}", state.best_metric);

    Ok(())
}
