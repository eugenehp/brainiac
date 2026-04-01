/// Evaluation metrics for BrainIAC training.
///
/// Matches the Python training metrics:
/// - MAE for regression
/// - Accuracy, AUC for binary classification
/// - Accuracy for multi-class classification

/// Mean Absolute Error.
pub fn mae(predictions: &[f32], targets: &[f32]) -> f32 {
    assert_eq!(predictions.len(), targets.len());
    if predictions.is_empty() { return 0.0; }
    let sum: f32 = predictions.iter().zip(targets).map(|(p, t)| (p - t).abs()).sum();
    sum / predictions.len() as f32
}

/// Binary accuracy (threshold = 0.5 on probabilities).
pub fn binary_accuracy(probabilities: &[f32], targets: &[f32]) -> f32 {
    assert_eq!(probabilities.len(), targets.len());
    if probabilities.is_empty() { return 0.0; }
    let correct: usize = probabilities.iter().zip(targets)
        .filter(|(&p, &t)| {
            let pred = if p > 0.5 { 1.0 } else { 0.0 };
            (pred - t).abs() < 0.5
        })
        .count();
    correct as f32 / probabilities.len() as f32
}

/// Multi-class accuracy.
pub fn multiclass_accuracy(logits_flat: &[f32], targets: &[f32], num_classes: usize) -> f32 {
    assert_eq!(logits_flat.len(), targets.len() * num_classes);
    if targets.is_empty() { return 0.0; }
    let batch_size = targets.len();
    let mut correct = 0usize;
    for b in 0..batch_size {
        let offset = b * num_classes;
        let mut best_class = 0;
        let mut best_val = f32::NEG_INFINITY;
        for c in 0..num_classes {
            if logits_flat[offset + c] > best_val {
                best_val = logits_flat[offset + c];
                best_class = c;
            }
        }
        if best_class == targets[b] as usize {
            correct += 1;
        }
    }
    correct as f32 / batch_size as f32
}

/// Area Under ROC Curve (simple trapezoidal approximation).
///
/// Scores and labels must have the same length.
/// Labels are binary (0.0 or 1.0).
pub fn auc_roc(scores: &[f32], labels: &[f32]) -> f32 {
    assert_eq!(scores.len(), labels.len());
    let n = scores.len();
    if n == 0 { return 0.0; }

    // Sort by score descending
    let mut pairs: Vec<(f32, f32)> = scores.iter().zip(labels).map(|(&s, &l)| (s, l)).collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let n_pos: f32 = labels.iter().filter(|&&l| l > 0.5).count() as f32;
    let n_neg: f32 = n as f32 - n_pos;

    if n_pos == 0.0 || n_neg == 0.0 {
        return 0.5; // Undefined, return 0.5
    }

    let mut tp = 0.0f32;
    let mut fp = 0.0f32;
    let mut auc = 0.0f32;
    let mut prev_fp = 0.0f32;
    let mut prev_tp = 0.0f32;

    let mut prev_score = f32::NAN;

    for &(score, label) in &pairs {
        if score != prev_score && !prev_score.is_nan() {
            // Trapezoid
            auc += (fp - prev_fp) * (tp + prev_tp) / 2.0;
            prev_fp = fp;
            prev_tp = tp;
        }
        if label > 0.5 {
            tp += 1.0;
        } else {
            fp += 1.0;
        }
        prev_score = score;
    }
    auc += (fp - prev_fp) * (tp + prev_tp) / 2.0;

    auc / (n_pos * n_neg)
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Convert logits to probabilities via sigmoid.
pub fn sigmoid_vec(logits: &[f32]) -> Vec<f32> {
    logits.iter().map(|&x| sigmoid(x)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mae() {
        let preds = vec![1.0, 2.0, 3.0];
        let targets = vec![1.5, 2.5, 3.5];
        assert!((mae(&preds, &targets) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_binary_accuracy() {
        let probs = vec![0.9, 0.1, 0.8, 0.3];
        let targets = vec![1.0, 0.0, 1.0, 0.0];
        assert!((binary_accuracy(&probs, &targets) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_auc_perfect() {
        let scores = vec![0.9, 0.8, 0.3, 0.1];
        let labels = vec![1.0, 1.0, 0.0, 0.0];
        let auc = auc_roc(&scores, &labels);
        assert!((auc - 1.0).abs() < 1e-6, "Perfect separation should give AUC=1.0, got {}", auc);
    }

    #[test]
    fn test_auc_random() {
        // Interleaved scores → AUC ≈ 0.5
        let scores = vec![0.9, 0.7, 0.5, 0.3];
        let labels = vec![1.0, 0.0, 1.0, 0.0];
        let auc = auc_roc(&scores, &labels);
        assert!(auc > 0.4 && auc < 0.9, "Mixed separation AUC should be between 0.4-0.9, got {}", auc);
    }

    #[test]
    fn test_multiclass_accuracy() {
        // 2 samples, 3 classes
        let logits = vec![
            0.1, 0.9, 0.0, // sample 0 → class 1
            0.0, 0.1, 0.9, // sample 1 → class 2
        ];
        let targets = vec![1.0, 2.0];
        assert!((multiclass_accuracy(&logits, &targets, 3) - 1.0).abs() < 1e-6);
    }
}
