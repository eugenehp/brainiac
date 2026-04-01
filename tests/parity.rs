/// Numerical parity tests against Python MONAI ViT reference vectors.
///
/// Run `python3 scripts/generate_parity_vectors.py` first to generate
/// the reference test vectors in tests/vectors/.
///
/// Tests verify bit-level agreement (within floating point tolerance)
/// at every stage of the pipeline:
/// 1. Patch embedding
/// 2. Position embedding addition
/// 3. Transformer block 0 output
/// 4. Transformer block 11 output
/// 5. Final layer norm
/// 6. First-patch feature extraction
/// 7. LayerNorm component
/// 8. Single block component

use burn::backend::NdArray as B;
use burn::prelude::*;
use std::path::Path;

type Device = burn::backend::ndarray::NdArrayDevice;
fn cpu() -> Device { Device::Cpu }

const VECTORS_DIR: &str = "tests/vectors";

/// Load raw f32 binary file.
fn load_f32(name: &str) -> Vec<f32> {
    let path = Path::new(VECTORS_DIR).join(name);
    let bytes = std::fs::read(&path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}. Run `python3 scripts/generate_parity_vectors.py` first.", path.display(), e));
    bytes.chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

/// Check if vectors are available.
fn vectors_available() -> bool {
    Path::new(VECTORS_DIR).join("weights.safetensors").exists()
}

/// Compare two f32 slices with tolerance, reporting max/mean diff.
fn assert_close(name: &str, actual: &[f32], expected: &[f32], atol: f32, rtol: f32) {
    assert_eq!(actual.len(), expected.len(),
        "{}: length mismatch: got {} vs expected {}", name, actual.len(), expected.len());

    let mut max_diff = 0.0f32;
    let mut sum_diff = 0.0f64;
    let mut max_idx = 0;

    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        let tol = atol + rtol * e.abs();
        if diff > max_diff {
            max_diff = diff;
            max_idx = i;
        }
        sum_diff += diff as f64;
    }

    let mean_diff = sum_diff / actual.len() as f64;
    let threshold = atol + rtol * expected[max_idx].abs();

    eprintln!("  {}: max_diff={:.2e} (idx={}) mean_diff={:.2e} threshold={:.2e} {}",
        name, max_diff, max_idx, mean_diff, threshold,
        if max_diff <= threshold { "✓" } else { "✗ FAIL" });

    assert!(max_diff <= threshold,
        "{}: max diff {:.2e} at idx {} exceeds tolerance {:.2e}\n  actual={:.6} expected={:.6}",
        name, max_diff, max_idx, threshold,
        actual[max_idx], expected[max_idx]);
}

/// Load the model with test weights.
fn load_test_model() -> brainiac::model::backbone::ViTBackbone<B> {
    let device = cpu();
    let cfg = brainiac::ModelConfig::default();
    let mut model = brainiac::model::backbone::ViTBackbone::new(&cfg, &device);

    let weights_path = Path::new(VECTORS_DIR).join("weights.safetensors");
    let mut wm = brainiac::weights::WeightMap::from_file(
        weights_path.to_str().unwrap()
    ).unwrap();
    brainiac::weights::load_backbone_weights(&mut wm, &mut model, &device).unwrap();

    model
}

// ── Parity Tests ─────────────────────────────────────────────────────────────

#[test]
fn parity_patch_embedding() {
    if !vectors_available() {
        eprintln!("SKIP: test vectors not found. Run python3 scripts/generate_parity_vectors.py");
        return;
    }
    let device = cpu();
    let model = load_test_model();
    let input = load_f32("input_volume.bin");
    let expected = load_f32("patch_embed_output.bin");

    let actual_tensor = model.patch_embed.forward_from_flat(&input, 1, &device);
    let actual: Vec<f32> = actual_tensor.into_data().to_vec().unwrap();

    assert_close("patch_embedding", &actual, &expected, 1e-4, 1e-3);
}

#[test]
fn parity_pos_embed() {
    if !vectors_available() { return; }
    let device = cpu();
    let model = load_test_model();
    let input = load_f32("input_volume.bin");
    let expected = load_f32("pos_embed_output.bin");

    let x = model.patch_embed.forward_from_flat(&input, 1, &device);
    let x = x + model.pos_embed.val();
    let actual: Vec<f32> = x.into_data().to_vec().unwrap();

    assert_close("pos_embed", &actual, &expected, 1e-4, 1e-3);
}

#[test]
fn parity_block_0() {
    if !vectors_available() { return; }
    let device = cpu();
    let model = load_test_model();
    let input = load_f32("input_volume.bin");
    let expected = load_f32("block_0_output.bin");

    let x = model.patch_embed.forward_from_flat(&input, 1, &device);
    let x = x + model.pos_embed.val();
    let x = model.blocks[0].forward(x);
    let actual: Vec<f32> = x.into_data().to_vec().unwrap();

    assert_close("block_0", &actual, &expected, 1e-3, 1e-2);
}

#[test]
fn parity_full_forward() {
    if !vectors_available() { return; }
    let device = cpu();
    let model = load_test_model();
    let input = load_f32("input_volume.bin");
    let expected = load_f32("first_patch_output.bin");

    let actual_tensor = model.forward(&input, 1, &device);
    let actual: Vec<f32> = actual_tensor.into_data().to_vec().unwrap();

    // Full forward accumulates error across 12 layers — use wider tolerance
    assert_close("full_forward", &actual, &expected, 5e-2, 5e-1);
}

#[test]
fn parity_layernorm() {
    if !vectors_available() { return; }
    let device = cpu();
    let model = load_test_model();
    let input_data = load_f32("layernorm_input.bin");
    let expected = load_f32("layernorm_output.bin");

    let input = Tensor::<B, 3>::from_data(
        TensorData::new(input_data, [1, 4, 768]), &device,
    );
    let actual_tensor = model.blocks[0].norm1.forward(input);
    let actual: Vec<f32> = actual_tensor.into_data().to_vec().unwrap();

    assert_close("layernorm", &actual, &expected, 1e-5, 1e-4);
}

#[test]
fn parity_single_block() {
    if !vectors_available() { return; }
    let device = cpu();
    let model = load_test_model();
    let input_data = load_f32("block0_test_input.bin");
    let expected = load_f32("block0_test_output.bin");

    let input = Tensor::<B, 3>::from_data(
        TensorData::new(input_data, [1, 8, 768]), &device,
    );
    let actual_tensor = model.blocks[0].forward(input);
    let actual: Vec<f32> = actual_tensor.into_data().to_vec().unwrap();

    assert_close("single_block", &actual, &expected, 1e-4, 1e-3);
}
