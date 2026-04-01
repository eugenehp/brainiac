/// Loss functions for BrainIAC training.
///
/// Matches the Python training scripts:
/// - MSELoss for regression (brain age, time-to-stroke)
/// - BCEWithLogitsLoss for binary classification (MCI, IDH, survival)
/// - CrossEntropyLoss for multi-class classification (MR sequence)

use burn::prelude::*;

/// Mean Squared Error loss for regression tasks.
///
/// loss = mean((predictions - targets)²)
pub fn mse_loss<B: Backend>(
    predictions: Tensor<B, 2>, // [B, 1]
    targets: Tensor<B, 2>,     // [B, 1]
) -> Tensor<B, 1> {
    let diff = predictions - targets;
    let sq = diff.clone() * diff;
    sq.mean().reshape([1])
}

/// Binary Cross-Entropy with Logits loss for binary classification.
///
/// Numerically stable: loss = max(x, 0) - x*y + log(1 + exp(-|x|))
/// where x = logits, y = targets ∈ {0, 1}.
pub fn bce_with_logits_loss<B: Backend>(
    logits: Tensor<B, 2>,  // [B, 1]
    targets: Tensor<B, 2>, // [B, 1]
) -> Tensor<B, 1> {
    // Stable computation: max(x,0) - x*y + log(1 + exp(-|x|))
    let zeros = Tensor::zeros_like(&logits);
    let pos_part = logits.clone().max_pair(zeros); // max(x, 0)
    let neg_abs = logits.clone().abs().neg();       // -|x|
    let log_term = neg_abs.exp().add_scalar(1.0).log(); // log(1 + exp(-|x|))
    let xy = logits * targets;                      // x * y
    let loss = pos_part - xy + log_term;
    loss.mean().reshape([1])
}

/// Cross-Entropy loss for multi-class classification.
///
/// Expects raw logits [B, C] and integer class targets as [B, 1] float.
/// Implements: -log(softmax(logits)[target_class])
pub fn cross_entropy_loss<B: Backend>(
    logits: Tensor<B, 2>,  // [B, num_classes]
    targets: Tensor<B, 2>, // [B, 1] with class indices as float
) -> Tensor<B, 1> {
    let [batch_size, num_classes] = logits.dims();

    // Log-softmax: log_softmax = logits - log(sum(exp(logits)))
    let max_logits = logits.clone().max_dim(1); // [B, 1]
    let shifted = logits - max_logits.clone();
    let exp_shifted = shifted.clone().exp();
    let log_sum_exp = exp_shifted.sum_dim(1).log(); // [B, 1]
    let log_softmax = shifted - log_sum_exp; // [B, C]

    // Gather the log-probability at the target class
    // We need to select log_softmax[b, target[b]] for each b
    let targets_data: Vec<f32> = targets.to_data().to_vec().unwrap();
    let log_softmax_data: Vec<f32> = log_softmax.to_data().to_vec().unwrap();

    let mut loss_sum = 0.0f32;
    for b in 0..batch_size {
        let target_class = targets_data[b] as usize;
        let target_class = target_class.min(num_classes - 1);
        loss_sum -= log_softmax_data[b * num_classes + target_class];
    }

    let device = max_logits.device();
    Tensor::<B, 1>::from_data(
        TensorData::new(vec![loss_sum / batch_size as f32], [1]),
        &device,
    )
}
