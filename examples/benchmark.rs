/// Benchmark: Measure inference throughput with synthetic data.
///
/// Usage:
///   cargo run --release --example benchmark -- --weights brainiac_backbone.safetensors

use std::time::Instant;
use brainiac::{BrainiacEncoder, TaskType, ModelConfig};

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
        .expect("Usage: --weights <path>");

    let n_iters: usize = args.iter()
        .position(|a| a == "--iters")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);

    let dev = device();
    let cfg = ModelConfig::default();
    let vol_size = cfg.in_channels * cfg.img_size * cfg.img_size * cfg.img_size;

    println!("Loading model...");
    let (encoder, timings) = BrainiacEncoder::<B>::load(
        weights,
        None,
        TaskType::FeatureExtraction,
        1,
        dev,
    )?;
    println!("Model: {} ({:.0} ms)", encoder.describe(), timings.total_ms);

    // Synthetic input (zeros — just measuring compute time)
    let dummy_data = vec![0.1f32; vol_size];

    println!("\nBenchmarking {} iterations...", n_iters);

    // Warmup
    let _ = encoder.model.extract_features(&dummy_data, &encoder.device);

    let t0 = Instant::now();
    for _ in 0..n_iters {
        let _ = encoder.model.extract_features(&dummy_data, &encoder.device);
    }
    let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let per_iter = total_ms / n_iters as f64;

    println!("Results:");
    println!("  Total    : {:.0} ms", total_ms);
    println!("  Per scan : {:.1} ms", per_iter);
    println!("  Throughput: {:.2} scans/sec", 1000.0 / per_iter);

    Ok(())
}
