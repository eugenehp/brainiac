/// Example: Extract 768-dim features from a brain MRI.
///
/// Usage:
///   cargo run --release --example extract_features -- \
///     --weights brainiac_backbone.safetensors \
///     --input brain_t1.nii.gz

use std::path::Path;
use std::time::Instant;
use brainiac::{BrainiacEncoder, TaskType};

#[cfg(feature = "ndarray")]
type B = burn::backend::NdArray;
#[cfg(feature = "ndarray")]
fn device() -> burn::backend::ndarray::NdArrayDevice {
    burn::backend::ndarray::NdArrayDevice::Cpu
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();

    let weights = args.iter()
        .position(|a| a == "--weights")
        .and_then(|i| args.get(i + 1))
        .expect("Usage: --weights <path> --input <nifti>");

    let input = args.iter()
        .position(|a| a == "--input")
        .and_then(|i| args.get(i + 1))
        .expect("Usage: --weights <path> --input <nifti>");

    let dev = device();

    println!("Loading BrainIAC backbone...");
    let (encoder, timings) = BrainiacEncoder::<B>::load(
        weights,
        None,
        TaskType::FeatureExtraction,
        1,
        dev,
    )?;
    println!("Model: {} ({:.0} ms)", encoder.describe(), timings.total_ms);

    println!("Encoding: {}", input);
    let t = Instant::now();
    let features = encoder.encode_nifti(Path::new(input))?;
    let ms = t.elapsed().as_secs_f64() * 1000.0;

    let mean: f64 = features.iter().map(|&v| v as f64).sum::<f64>() / features.len() as f64;
    let std: f64 = (features.iter().map(|&v| {
        let d = v as f64 - mean; d * d
    }).sum::<f64>() / features.len() as f64).sqrt();

    println!("Features: dim={}, mean={:+.6}, std={:.6} ({:.0} ms)", features.len(), mean, std, ms);

    // Print first 10 values
    print!("First 10: [");
    for (i, v) in features.iter().take(10).enumerate() {
        if i > 0 { print!(", "); }
        print!("{:.4}", v);
    }
    println!(", ...]");

    Ok(())
}
