/// Comprehensive benchmark for BrainIAC-RS.
///
/// Benchmarks every component with timing:
/// 1. Weight loading
/// 2. Preprocessing (resize + normalize)
/// 3. Patch embedding
/// 4. Single transformer block
/// 5. Full backbone forward (12 layers)
/// 6. Full forward with attention (saliency)
/// 7. Classifier (single/dual/quad)
///
/// Usage:
///   # CPU (NdArray)
///   cargo run --release --example bench_all
///
///   # GPU (Metal on macOS)
///   cargo run --release --no-default-features --features metal --example bench_all
///
///   # Parity + benchmark with test vectors
///   cargo run --release --example bench_all -- --parity

use std::time::Instant;

#[cfg(all(feature = "wgpu", not(feature = "ndarray")))]
mod backend {
    pub use burn::backend::Wgpu as B;
    pub type Device = burn::backend::wgpu::WgpuDevice;
    pub fn device() -> Device { Device::DefaultDevice }
    #[cfg(feature = "metal")]
    pub const NAME: &str = "GPU (Metal)";
    #[cfg(feature = "vulkan")]
    pub const NAME: &str = "GPU (Vulkan)";
    #[cfg(not(any(feature = "metal", feature = "vulkan")))]
    pub const NAME: &str = "GPU (wgpu/WGSL)";
}

#[cfg(feature = "ndarray")]
mod backend {
    pub use burn::backend::NdArray as B;
    pub type Device = burn::backend::ndarray::NdArrayDevice;
    pub fn device() -> Device { Device::Cpu }
    #[cfg(feature = "blas-accelerate")]
    pub const NAME: &str = "CPU (Accelerate)";
    #[cfg(not(feature = "blas-accelerate"))]
    pub const NAME: &str = "CPU (NdArray)";
}

use backend::{B, device};
use burn::prelude::*;
use brainiac::ModelConfig;

fn bench<F: FnMut()>(name: &str, mut f: F, iters: usize) -> f64 {
    // Warmup
    f();

    let t0 = Instant::now();
    for _ in 0..iters {
        f();
    }
    let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let per_ms = total_ms / iters as f64;
    println!("  {:40} {:7.1} ms  ({} iters, {:.0} ms total)",
        name, per_ms, iters, total_ms);
    per_ms
}

fn main() {
    let dev = device();
    let cfg = ModelConfig::default();
    let vol_size = cfg.in_channels * cfg.img_size.pow(3);

    println!("═══════════════════════════════════════════════════════════════");
    println!("BrainIAC-RS Benchmark");
    println!("═══════════════════════════════════════════════════════════════");
    println!("Backend   : {}", backend::NAME);
    println!("Model     : {}", brainiac::model::backbone::describe_model(&cfg));
    println!("Volume    : {}×{}×{} = {} voxels", cfg.img_size, cfg.img_size, cfg.img_size, cfg.img_size.pow(3));
    println!("───────────────────────────────────────────────────────────────");

    // Synthetic input
    let input = vec![0.1f32; vol_size];

    // 1. Model init (zero-init)
    let t0 = Instant::now();
    let backbone = brainiac::model::backbone::ViTBackbone::<B>::new(&cfg, &dev);
    let init_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("  {:40} {:7.1} ms", "Model init (zero-init)", init_ms);

    // 2. Full load cycle (init + weight loading) if test vectors available
    let weights_path = "tests/vectors/weights.safetensors";
    if std::path::Path::new(weights_path).exists() {
        // Method A: zero-init + load_backbone_weights
        let t0 = Instant::now();
        let mut bb_a = brainiac::model::backbone::ViTBackbone::<B>::new(&cfg, &dev);
        let mut wm = brainiac::weights::WeightMap::from_file(weights_path).unwrap();
        brainiac::weights::load_backbone_weights(&mut wm, &mut bb_a, &dev).unwrap();
        let ms_a = t0.elapsed().as_secs_f64() * 1000.0;
        println!("  {:40} {:7.1} ms", "Full load (init + load_weights)", ms_a);

        // Method B: from_weights (direct build) — with breakdown
        let t0 = Instant::now();
        let mut wm = brainiac::weights::WeightMap::from_file(weights_path).unwrap();
        let disk_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let t1 = Instant::now();
        let _bb_b = brainiac::model::backbone::ViTBackbone::<B>::from_weights(
            &cfg, &mut wm, &dev
        ).unwrap();
        let build_ms = t1.elapsed().as_secs_f64() * 1000.0;
        println!("  {:40} {:7.1} ms  (disk={:.1} build={:.1})",
            "Full load (from_weights)", disk_ms + build_ms, disk_ms, build_ms);

        // Method C: fast_load (single bulk GPU transfer)
        let t0 = Instant::now();
        let _bb_c = brainiac::fast_load::load_backbone_fast::<B>(
            &cfg, weights_path, &dev
        ).unwrap();
        let ms_c = t0.elapsed().as_secs_f64() * 1000.0;
        println!("  {:40} {:7.1} ms", "Full load (fast_load bulk)", ms_c);
    }

    // 2. Preprocessing
    let raw_vol = vec![42.0f32; 128 * 128 * 128];
    bench("Preprocess (128³ → 96³ + normalize)", || {
        let vol = brainiac::NiftiVolume {
            data: raw_vol.clone(),
            dims: [128, 128, 128],
            pixdim: [1.0, 1.0, 1.0],
        };
        let _ = brainiac::preprocess(&vol, 96);
    }, 5);

    // 3. Patch embedding only
    bench("Patch embedding (96³ → [216, 768])", || {
        let _ = backbone.patch_embed.forward_from_flat(&input, 1, &dev);
    }, 10);

    // 4. Single block
    let dummy_tokens = Tensor::<B, 3>::zeros([1, 216, 768], &dev);
    bench("Single ViT block (216 tokens)", || {
        let _ = backbone.blocks[0].forward(dummy_tokens.clone());
    }, 10);

    // 5. Full backbone forward
    bench("Full backbone (12 layers)", || {
        let _ = backbone.forward(&input, 1, &dev);
    }, 5);

    // 6. Forward with attention
    bench("Backbone + attention maps", || {
        let _ = backbone.forward_with_attn(&input, 1, &dev);
    }, 3);

    // 7. Classifier heads
    let head_1 = burn::nn::LinearConfig::new(768, 1).with_bias(true).init::<B>(&dev);
    let features = Tensor::<B, 2>::zeros([1, 768], &dev);
    bench("Linear head (768 → 1)", || {
        let _ = head_1.forward(features.clone());
    }, 100);

    let head_4 = burn::nn::LinearConfig::new(768, 4).with_bias(true).init::<B>(&dev);
    bench("Linear head (768 → 4)", || {
        let _ = head_4.forward(features.clone());
    }, 100);

    // 8. Augmentation
    let aug_cfg = brainiac::AugConfig::default();
    let mut aug_data = vec![0.5f32; 96 * 96 * 96];
    bench("Data augmentation (96³)", || {
        let mut rng = brainiac::augmentation::Rng::new(42);
        brainiac::augmentation::augment_volume(&mut aug_data, 96, &mut rng, &aug_cfg);
    }, 5);

    // 9. Saliency (if forward_with_attn is available)
    // The saliency computation itself (excluding forward pass)
    bench("Saliency upsample (6³ → 96³)", || {
        let attn_data = vec![0.01f32; 216];
        let _ = brainiac::saliency::generate_saliency::<B>(
            &backbone, &input, -1, &cfg, &dev,
        );
    }, 3);

    println!("───────────────────────────────────────────────────────────────");

    // End-to-end
    let e2e_ms = bench("End-to-end (preprocess + forward)", || {
        let vol = brainiac::NiftiVolume {
            data: raw_vol.clone(),
            dims: [128, 128, 128],
            pixdim: [1.0, 1.0, 1.0],
        };
        let preprocessed = brainiac::preprocess(&vol, 96);
        let _ = backbone.forward(&preprocessed, 1, &dev);
    }, 3);

    println!("═══════════════════════════════════════════════════════════════");
    println!("Throughput: {:.2} scans/sec (end-to-end)", 1000.0 / e2e_ms);
}
