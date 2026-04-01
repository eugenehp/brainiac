/// Integration tests for BrainIAC model components.

use burn::backend::NdArray as B;
use burn::prelude::*;

type Device = burn::backend::ndarray::NdArrayDevice;

fn cpu() -> Device {
    Device::Cpu
}

#[test]
fn test_patch_embed_shape() {
    use brainiac::ModelConfig;

    let cfg = ModelConfig::default();
    let device = cpu();
    let pe = brainiac::model::patch_embed_3d::PatchEmbed3d::<B>::new(
        cfg.in_channels,
        cfg.hidden_size,
        cfg.img_size,
        cfg.patch_size,
        &device,
    );

    assert_eq!(pe.num_patches(), 216); // 6×6×6
    assert_eq!(pe.patches_per_axis(), 6);

    // Forward pass with dummy data
    let vol_size = cfg.in_channels * cfg.img_size * cfg.img_size * cfg.img_size;
    let dummy = vec![0.1f32; vol_size];
    let output = pe.forward_from_flat(&dummy, 1, &device);
    let dims = output.dims();
    assert_eq!(dims, [1, 216, 768]);
}

#[test]
fn test_vit_block_shape() {
    let device = cpu();
    let block = brainiac::model::vit::ViTBlock::<B>::new(768, 12, 3072, 1e-6, &device);

    // Input: [B=1, N=217, D=768]
    let x = Tensor::<B, 3>::zeros([1, 217, 768], &device);
    let out = block.forward(x);
    assert_eq!(out.dims(), [1, 217, 768]);
}

#[test]
fn test_vit_block_with_attn() {
    let device = cpu();
    let block = brainiac::model::vit::ViTBlock::<B>::new(768, 12, 3072, 1e-6, &device);

    let x = Tensor::<B, 3>::zeros([1, 217, 768], &device);
    let (out, attn) = block.forward_with_attn(x);
    assert_eq!(out.dims(), [1, 217, 768]);
    assert_eq!(attn.dims(), [1, 12, 217, 217]); // [B, H, N, N]
}

#[test]
fn test_layernorm() {
    let device = cpu();
    let ln = brainiac::model::vit::LayerNorm::<B>::new(4, 1e-5, &device);

    let x = Tensor::<B, 3>::from_data(
        TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [1, 1, 4]),
        &device,
    );
    let out = ln.forward(x);
    let vals: Vec<f32> = out.to_data().to_vec().unwrap();
    // Should be normalized: mean≈0, std≈1
    let mean: f32 = vals.iter().sum::<f32>() / vals.len() as f32;
    assert!(mean.abs() < 1e-5, "LayerNorm output mean should be ~0, got {}", mean);
}

#[test]
fn test_backbone_shape() {
    let device = cpu();
    let cfg = brainiac::ModelConfig {
        num_layers: 1, // Use 1 layer for speed
        ..Default::default()
    };
    let backbone = brainiac::model::backbone::ViTBackbone::<B>::new(&cfg, &device);

    let vol_size = cfg.in_channels * cfg.img_size * cfg.img_size * cfg.img_size;
    let dummy = vec![0.01f32; vol_size];
    let features = backbone.forward(&dummy, 1, &device);
    assert_eq!(features.dims(), [1, 768]);
}

#[test]
fn test_classifier_single_scan() {
    use brainiac::config::TaskType;

    let device = cpu();
    let cfg = brainiac::ModelConfig {
        num_layers: 1,
        ..Default::default()
    };
    let model = brainiac::BrainiacModel::<B>::new(&cfg, TaskType::Regression, 1, &device);

    let vol_size = cfg.in_channels * cfg.img_size * cfg.img_size * cfg.img_size;
    let dummy = vec![0.01f32; vol_size];
    let output = model.predict_single(&dummy, &device);
    assert_eq!(output.dims(), [1, 1]);
}

#[test]
fn test_classifier_multi_scan() {
    use brainiac::config::TaskType;

    let device = cpu();
    let cfg = brainiac::ModelConfig {
        num_layers: 1,
        ..Default::default()
    };
    let model = brainiac::BrainiacModel::<B>::new(
        &cfg, TaskType::DualBinaryClassification, 1, &device,
    );

    let vol_size = cfg.in_channels * cfg.img_size * cfg.img_size * cfg.img_size;
    let scan1 = vec![0.01f32; vol_size];
    let scan2 = vec![0.02f32; vol_size];
    let output = model.predict_multi(&[&scan1, &scan2], &device);
    assert_eq!(output.dims(), [1, 1]);
}

#[test]
fn test_describe_model() {
    let cfg = brainiac::ModelConfig::default();
    let desc = brainiac::model::backbone::describe_model(&cfg);
    assert!(desc.contains("ViT-B/16"));
    assert!(desc.contains("216 patches"));
    assert!(desc.contains("12 layers"));
    assert!(desc.contains("12 heads"));
}
