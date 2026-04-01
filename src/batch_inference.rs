/// Batch inference: CSV input → predictions CSV output with metrics.
///
/// Matches the Python `test_inference_finetune.py`:
/// - Reads test CSV with (pat_id, label) columns
/// - Runs inference on all samples
/// - Computes metrics (MAE/AUC/accuracy)
/// - Saves predictions to output CSV
/// - Saves metrics to JSON

use std::path::Path;
use std::time::Instant;
use anyhow::Result;
use burn::prelude::*;

use crate::config::TaskType;
use crate::data::{SingleScanDataset, DualScanDataset, QuadScanDataset};
use crate::metrics;
use crate::model::classifier::BrainiacModel;

/// Batch inference result.
#[derive(Debug, Clone)]
pub struct BatchResult {
    pub pat_ids: Vec<String>,
    pub labels: Vec<f32>,
    pub predictions: Vec<f32>,
    pub raw_logits: Vec<f32>,
    pub metrics: InferenceMetrics,
    pub elapsed_ms: f64,
}

/// Computed metrics.
#[derive(Debug, Clone)]
pub struct InferenceMetrics {
    pub task: TaskType,
    pub n_samples: usize,
    pub primary_metric_name: String,
    pub primary_metric_value: f64,
    pub additional: Vec<(String, f64)>,
}

/// Run batch inference on a single-scan dataset.
pub fn batch_infer_single<B: Backend>(
    model: &BrainiacModel<B>,
    dataset: &mut SingleScanDataset,
    task: TaskType,
    num_classes: usize,
    device: &B::Device,
) -> Result<BatchResult> {
    let t0 = Instant::now();
    let n = dataset.len();

    let mut pat_ids = Vec::with_capacity(n);
    let mut labels = Vec::with_capacity(n);
    let mut raw_logits = Vec::new();
    let mut predictions = Vec::new();

    for idx in 0..n {
        let sample = dataset.get(idx)?;
        let logits = model.predict_single(&sample.data, device);
        let logits_vec: Vec<f32> = logits.to_data().to_vec().unwrap();

        pat_ids.push(sample.pat_id);
        labels.push(sample.label);
        raw_logits.extend_from_slice(&logits_vec);

        match task {
            TaskType::Regression => {
                predictions.push(logits_vec[0]);
            }
            TaskType::BinaryClassification => {
                predictions.push(sigmoid(logits_vec[0]));
            }
            TaskType::MulticlassClassification => {
                let probs = softmax_vec(&logits_vec);
                predictions.extend_from_slice(&probs);
            }
            _ => predictions.push(logits_vec[0]),
        }

        if (idx + 1) % 50 == 0 || idx == n - 1 {
            eprintln!("  Inference: {}/{}", idx + 1, n);
        }
    }

    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Compute metrics
    let metrics_result = compute_metrics(&predictions, &labels, task, num_classes);

    Ok(BatchResult {
        pat_ids,
        labels,
        predictions,
        raw_logits,
        metrics: metrics_result,
        elapsed_ms,
    })
}

/// Run batch inference on a dual-scan dataset.
pub fn batch_infer_dual<B: Backend>(
    model: &BrainiacModel<B>,
    dataset: &DualScanDataset,
    task: TaskType,
    device: &B::Device,
) -> Result<BatchResult> {
    let t0 = Instant::now();
    let n = dataset.len();

    let mut pat_ids = Vec::with_capacity(n);
    let mut labels = Vec::with_capacity(n);
    let mut raw_logits = Vec::new();
    let mut predictions = Vec::new();

    for idx in 0..n {
        let sample = dataset.get(idx)?;
        let scan_refs: Vec<&[f32]> = sample.scans.iter().map(|s| s.as_slice()).collect();
        let logits = model.predict_multi(&scan_refs, device);
        let logits_vec: Vec<f32> = logits.to_data().to_vec().unwrap();

        pat_ids.push(sample.pat_id);
        labels.push(sample.label);
        raw_logits.extend_from_slice(&logits_vec);
        predictions.push(sigmoid(logits_vec[0]));

        if (idx + 1) % 50 == 0 || idx == n - 1 {
            eprintln!("  Inference: {}/{}", idx + 1, n);
        }
    }

    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let met = compute_metrics(&predictions, &labels, task, 1);

    Ok(BatchResult { pat_ids, labels, predictions, raw_logits, metrics: met, elapsed_ms })
}

/// Run batch inference on a quad-scan dataset.
pub fn batch_infer_quad<B: Backend>(
    model: &BrainiacModel<B>,
    dataset: &QuadScanDataset,
    task: TaskType,
    device: &B::Device,
) -> Result<BatchResult> {
    let t0 = Instant::now();
    let n = dataset.len();

    let mut pat_ids = Vec::with_capacity(n);
    let mut labels = Vec::with_capacity(n);
    let mut raw_logits = Vec::new();
    let mut predictions = Vec::new();

    for idx in 0..n {
        let sample = dataset.get(idx)?;
        let scan_refs: Vec<&[f32]> = sample.scans.iter().map(|s| s.as_slice()).collect();
        let logits = model.predict_multi(&scan_refs, device);
        let logits_vec: Vec<f32> = logits.to_data().to_vec().unwrap();

        pat_ids.push(sample.pat_id);
        labels.push(sample.label);
        raw_logits.extend_from_slice(&logits_vec);
        predictions.push(sigmoid(logits_vec[0]));

        if (idx + 1) % 50 == 0 || idx == n - 1 {
            eprintln!("  Inference: {}/{}", idx + 1, n);
        }
    }

    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let met = compute_metrics(&predictions, &labels, task, 1);

    Ok(BatchResult { pat_ids, labels, predictions, raw_logits, metrics: met, elapsed_ms })
}

fn compute_metrics(predictions: &[f32], labels: &[f32], task: TaskType, num_classes: usize) -> InferenceMetrics {
    let n = labels.len();
    match task {
        TaskType::Regression => {
            let mae_val = metrics::mae(predictions, labels);
            // RMSE
            let mse: f32 = predictions.iter().zip(labels).map(|(p, t)| (p - t).powi(2)).sum::<f32>() / n as f32;
            let rmse = mse.sqrt();
            InferenceMetrics {
                task, n_samples: n,
                primary_metric_name: "MAE".into(),
                primary_metric_value: mae_val as f64,
                additional: vec![
                    ("RMSE".into(), rmse as f64),
                ],
            }
        }
        TaskType::BinaryClassification
        | TaskType::DualBinaryClassification
        | TaskType::QuadBinaryClassification => {
            let auc = metrics::auc_roc(predictions, labels);
            let acc = metrics::binary_accuracy(predictions, labels);
            InferenceMetrics {
                task, n_samples: n,
                primary_metric_name: "AUC".into(),
                primary_metric_value: auc as f64,
                additional: vec![
                    ("Accuracy".into(), acc as f64),
                ],
            }
        }
        TaskType::MulticlassClassification => {
            let acc = metrics::multiclass_accuracy(predictions, labels, num_classes);
            InferenceMetrics {
                task, n_samples: n,
                primary_metric_name: "Accuracy".into(),
                primary_metric_value: acc as f64,
                additional: vec![],
            }
        }
        TaskType::FeatureExtraction => {
            InferenceMetrics {
                task, n_samples: n,
                primary_metric_name: "N/A".into(),
                primary_metric_value: 0.0,
                additional: vec![],
            }
        }
    }
}

/// Save batch results to CSV.
pub fn save_predictions_csv(result: &BatchResult, path: &Path, task: TaskType) -> Result<()> {
    let mut lines = Vec::new();

    match task {
        TaskType::Regression => {
            lines.push("pat_id,label,predicted_value".to_string());
            for i in 0..result.pat_ids.len() {
                lines.push(format!("{},{},{}", result.pat_ids[i], result.labels[i], result.predictions[i]));
            }
        }
        TaskType::BinaryClassification
        | TaskType::DualBinaryClassification
        | TaskType::QuadBinaryClassification => {
            lines.push("pat_id,label,probability,predicted_class".to_string());
            for i in 0..result.pat_ids.len() {
                let cls = if result.predictions[i] > 0.5 { 1 } else { 0 };
                lines.push(format!("{},{},{:.6},{}", result.pat_ids[i], result.labels[i], result.predictions[i], cls));
            }
        }
        TaskType::MulticlassClassification => {
            lines.push("pat_id,label,predicted_class".to_string());
            // predictions are flattened probabilities
            let nc = result.predictions.len() / result.pat_ids.len();
            for i in 0..result.pat_ids.len() {
                let offset = i * nc;
                let pred_class = (0..nc)
                    .max_by(|&a, &b| result.predictions[offset + a]
                        .partial_cmp(&result.predictions[offset + b]).unwrap())
                    .unwrap_or(0);
                lines.push(format!("{},{},{}", result.pat_ids[i], result.labels[i], pred_class));
            }
        }
        _ => {}
    }

    std::fs::write(path, lines.join("\n"))?;
    Ok(())
}

/// Save metrics to JSON.
pub fn save_metrics_json(result: &BatchResult, path: &Path) -> Result<()> {
    let mut extra = serde_json::Map::new();
    for (k, v) in &result.metrics.additional {
        extra.insert(k.clone(), serde_json::Value::from(*v));
    }
    let json = serde_json::json!({
        "task": format!("{:?}", result.metrics.task),
        "n_samples": result.metrics.n_samples,
        "primary_metric": result.metrics.primary_metric_name,
        "primary_value": result.metrics.primary_metric_value,
        "additional_metrics": extra,
        "elapsed_ms": result.elapsed_ms,
    });
    std::fs::write(path, serde_json::to_string_pretty(&json)?)?;
    Ok(())
}

fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

fn softmax_vec(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}
