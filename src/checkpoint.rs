/// Checkpoint saving and loading for trained models.
///
/// Saves model weights as safetensors files that can be loaded back
/// for inference or further training. Compatible with the Python
/// export_safetensors.py format.

use std::collections::HashMap;
use std::path::Path;
use anyhow::Result;
use burn::prelude::*;

use crate::training::TrainableModel;
use crate::model::backbone::ViTBackbone;

/// Save a trained model's weights to safetensors format.
///
/// Saves both backbone and head weights with MONAI-compatible key names.
pub fn save_model_safetensors<B: Backend>(
    model: &TrainableModel<B>,
    path: &Path,
) -> Result<()> {
    let mut tensors: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();

    // Save backbone weights
    save_backbone_weights(&model.backbone, &mut tensors)?;

    // Save head weights
    let head_w: Vec<f32> = model.head.weight.val().to_data().to_vec().unwrap();
    let head_w_shape: Vec<usize> = model.head.weight.val().dims().to_vec();
    // Burn stores as [in, out], PyTorch expects [out, in]
    let [in_dim, out_dim] = [head_w_shape[0], head_w_shape[1]];
    let mut transposed = vec![0.0f32; in_dim * out_dim];
    for i in 0..in_dim {
        for j in 0..out_dim {
            transposed[j * in_dim + i] = head_w[i * out_dim + j];
        }
    }
    tensors.insert("fc.weight".to_string(), (transposed, vec![out_dim, in_dim]));

    if let Some(ref bias) = model.head.bias {
        let b: Vec<f32> = bias.val().to_data().to_vec().unwrap();
        let shape = vec![b.len()];
        tensors.insert("fc.bias".to_string(), (b, shape));
    }

    write_safetensors(path, &tensors)
}

/// Save only the backbone weights to safetensors.
pub fn save_backbone_safetensors<B: Backend>(
    backbone: &ViTBackbone<B>,
    path: &Path,
) -> Result<()> {
    let mut tensors: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    save_backbone_weights(backbone, &mut tensors)?;
    write_safetensors(path, &tensors)
}

fn save_backbone_weights<B: Backend>(
    backbone: &ViTBackbone<B>,
    tensors: &mut HashMap<String, (Vec<f32>, Vec<usize>)>,
) -> Result<()> {
    // Position embeddings [1, num_patches, hidden_dim]
    let pe: Vec<f32> = backbone.pos_embed.val().to_data().to_vec().unwrap();
    let pe_len = pe.len();
    tensors.insert("patch_embedding.position_embeddings".into(),
        (pe, vec![1, pe_len / backbone.hidden_dim, backbone.hidden_dim]));

    // Patch embedding conv
    let pw: Vec<f32> = backbone.patch_embed.proj_weight.val().to_data().to_vec().unwrap();
    let pw_shape = backbone.patch_embed.proj_weight.val().dims().to_vec();
    tensors.insert("patch_embedding.patch_embeddings.0.weight".into(), (pw, pw_shape));

    let pb: Vec<f32> = backbone.patch_embed.proj_bias.val().to_data().to_vec().unwrap();
    tensors.insert("patch_embedding.patch_embeddings.0.bias".into(), (pb, vec![backbone.hidden_dim]));

    // Transformer blocks
    for (i, block) in backbone.blocks.iter().enumerate() {
        let p = format!("blocks.{i}");

        // norm1
        save_layernorm(tensors, &block.norm1, &format!("{p}.norm1"));
        // attn.qkv
        save_linear(tensors, &block.attn.qkv, &format!("{p}.attn.qkv"));
        // attn.out_proj
        save_linear(tensors, &block.attn.proj, &format!("{p}.attn.out_proj"));
        // norm2
        save_layernorm(tensors, &block.norm2, &format!("{p}.norm2"));
        // mlp
        save_linear(tensors, &block.mlp.fc1, &format!("{p}.mlp.linear1"));
        save_linear(tensors, &block.mlp.fc2, &format!("{p}.mlp.linear2"));
    }

    // Final norm
    save_layernorm(tensors, &backbone.norm, "norm");

    Ok(())
}

fn save_linear<B: Backend>(
    tensors: &mut HashMap<String, (Vec<f32>, Vec<usize>)>,
    linear: &burn::nn::Linear<B>,
    prefix: &str,
) {
    // Burn: [in, out] → PyTorch: [out, in]
    let w: Vec<f32> = linear.weight.val().to_data().to_vec().unwrap();
    let shape = linear.weight.val().dims();
    let [in_d, out_d] = [shape[0], shape[1]];
    let mut transposed = vec![0.0f32; in_d * out_d];
    for i in 0..in_d {
        for j in 0..out_d {
            transposed[j * in_d + i] = w[i * out_d + j];
        }
    }
    tensors.insert(format!("{prefix}.weight"), (transposed, vec![out_d, in_d]));

    if let Some(ref bias) = linear.bias {
        let b: Vec<f32> = bias.val().to_data().to_vec().unwrap();
        tensors.insert(format!("{prefix}.bias"), (b, vec![out_d]));
    }
}

fn save_layernorm<B: Backend>(
    tensors: &mut HashMap<String, (Vec<f32>, Vec<usize>)>,
    norm: &crate::model::vit::LayerNorm<B>,
    prefix: &str,
) {
    let w: Vec<f32> = norm.weight.val().to_data().to_vec().unwrap();
    let dim = w.len();
    tensors.insert(format!("{prefix}.weight"), (w, vec![dim]));
    let b: Vec<f32> = norm.bias.val().to_data().to_vec().unwrap();
    tensors.insert(format!("{prefix}.bias"), (b, vec![dim]));
}

/// Write tensors to a safetensors file.
fn write_safetensors(
    path: &Path,
    tensors: &HashMap<String, (Vec<f32>, Vec<usize>)>,
) -> Result<()> {
    use std::collections::BTreeMap;

    // Build safetensors metadata
    let mut st_tensors: BTreeMap<String, safetensors::tensor::TensorView<'_>> = BTreeMap::new();

    // Store byte data so references remain valid
    let byte_data: HashMap<String, Vec<u8>> = tensors.iter()
        .map(|(k, (data, _))| {
            let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
            (k.clone(), bytes)
        })
        .collect();

    for (key, (_, shape)) in tensors {
        let bytes = &byte_data[key];
        let view = safetensors::tensor::TensorView::new(
            safetensors::Dtype::F32,
            shape.clone(),
            bytes,
        )?;
        st_tensors.insert(key.clone(), view);
    }

    let serialized = safetensors::tensor::serialize(&st_tensors, None)?;
    std::fs::write(path, serialized)?;

    eprintln!("Saved {} tensors to {}", tensors.len(), path.display());
    Ok(())
}
