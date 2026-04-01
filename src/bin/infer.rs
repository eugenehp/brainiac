/// BrainIAC inference CLI.
///
/// Build — CPU (default):
///   cargo build --release
///
/// Build — GPU (Metal on macOS):
///   cargo build --release --no-default-features --features metal
///
/// Usage:
///   infer --weights backbone.safetensors --input brain.nii.gz
///   infer --weights backbone.safetensors --head head.safetensors --task brain_age --input brain.nii.gz
///   infer --weights backbone.safetensors --task idh --input flair.nii.gz --input t1ce.nii.gz

use std::path::Path;
use std::time::Instant;
use clap::Parser;
use brainiac::{BrainiacEncoder, TaskType};

// ── Backend ───────────────────────────────────────────────────────────────────
#[cfg(all(feature = "wgpu", not(feature = "ndarray")))]
mod backend {
    pub use burn::backend::{Wgpu as B, wgpu::WgpuDevice as Device};
    pub fn device() -> Device { Device::DefaultDevice }
    #[cfg(feature = "metal")]
    pub const NAME: &str = "GPU (wgpu — Metal)";
    #[cfg(feature = "vulkan")]
    pub const NAME: &str = "GPU (wgpu — Vulkan)";
    #[cfg(not(any(feature = "metal", feature = "vulkan")))]
    pub const NAME: &str = "GPU (wgpu — WGSL)";
}

#[cfg(feature = "ndarray")]
mod backend {
    pub use burn::backend::NdArray as B;
    pub type Device = burn::backend::ndarray::NdArrayDevice;
    pub fn device() -> Device { Device::Cpu }
    #[cfg(feature = "blas-accelerate")]
    pub const NAME: &str = "CPU (NdArray + Apple Accelerate)";
    #[cfg(feature = "openblas-system")]
    pub const NAME: &str = "CPU (NdArray + OpenBLAS)";
    #[cfg(not(any(feature = "blas-accelerate", feature = "openblas-system")))]
    pub const NAME: &str = "CPU (NdArray + Rayon)";
}

use backend::{B, device};

// ── CLI ───────────────────────────────────────────────────────────────────────
#[derive(Parser, Debug)]
#[command(name = "brainiac-infer")]
#[command(about = "BrainIAC Brain MRI foundation model inference")]
struct Args {
    /// Path to backbone .safetensors weights.
    #[arg(long)]
    weights: String,

    /// Path to downstream task head .safetensors weights (optional).
    #[arg(long)]
    head: Option<String>,

    /// Task type: features, brain_age, mci, idh, survival, sequence, stroke
    #[arg(long, default_value = "features")]
    task: String,

    /// Input NIfTI file(s). Use multiple --input for dual/quad tasks.
    #[arg(long, required = true)]
    input: Vec<String>,

    /// Print verbose output.
    #[arg(long, short = 'v')]
    verbose: bool,
}

fn parse_task(s: &str) -> (TaskType, usize) {
    match s {
        "features" | "feature_extraction" => (TaskType::FeatureExtraction, 1),
        "brain_age" | "brainage" => (TaskType::Regression, 1),
        "stroke" | "timetostroke" => (TaskType::Regression, 1),
        "mci" => (TaskType::BinaryClassification, 1),
        "idh" => (TaskType::DualBinaryClassification, 1),
        "survival" | "os" => (TaskType::QuadBinaryClassification, 1),
        "sequence" | "multiclass" => (TaskType::MulticlassClassification, 4),
        _ => {
            eprintln!("Unknown task '{}', defaulting to feature extraction", s);
            (TaskType::FeatureExtraction, 1)
        }
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let t0 = Instant::now();
    let dev = device();

    let (task, num_classes) = parse_task(&args.task);

    println!("BrainIAC-RS Inference");
    println!("═══════════════════════════════════════════════════════════════");
    println!("Backend : {}", backend::NAME);
    println!("Task    : {:?}", task);

    // Load model
    let (encoder, timings) = BrainiacEncoder::<B>::load(
        &args.weights,
        args.head.as_deref(),
        task,
        num_classes,
        dev,
    )?;

    println!("Model   : {}", encoder.describe());
    println!("Weights : {:.0} ms", timings.weights_ms);

    // Validate input count
    let expected_inputs = match task {
        TaskType::DualBinaryClassification => 2,
        TaskType::QuadBinaryClassification => 4,
        _ => 1,
    };
    if args.input.len() != expected_inputs {
        anyhow::bail!(
            "Task {:?} requires {} input file(s), got {}",
            task, expected_inputs, args.input.len()
        );
    }

    // Run inference
    let t_infer = Instant::now();

    let output = if args.input.len() == 1 {
        encoder.infer_nifti(Path::new(&args.input[0]))?
    } else {
        let paths: Vec<&Path> = args.input.iter().map(|s| Path::new(s.as_str())).collect();
        encoder.infer_multi_nifti(&paths)?
    };

    let ms_infer = t_infer.elapsed().as_secs_f64() * 1000.0;

    println!("───────────────────────────────────────────────────────────────");
    println!("Output  : {}", output);

    if args.verbose {
        if let brainiac::InferenceOutput::Features(ref feats) = output {
            let mean: f64 = feats.iter().map(|&v| v as f64).sum::<f64>() / feats.len() as f64;
            let std: f64 = (feats.iter().map(|&v| {
                let d = v as f64 - mean;
                d * d
            }).sum::<f64>() / feats.len() as f64).sqrt();
            println!("  mean={mean:+.6}  std={std:.6}  dim={}", feats.len());
        }
    }

    let ms_total = t0.elapsed().as_secs_f64() * 1000.0;
    println!("───────────────────────────────────────────────────────────────");
    println!("Timing:");
    println!("  Model init : {:.0} ms", timings.model_init_ms);
    println!("  Weights    : {:.0} ms", timings.weights_ms);
    println!("  Inference  : {:.0} ms", ms_infer);
    println!("  Total      : {:.0} ms", ms_total);

    Ok(())
}
