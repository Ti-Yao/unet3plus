def focal_tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-6):
    """
    Focal Tversky loss for multi-class 3D segmentation.

    Args:
    y_true: tensor of shape [B, D, H, W, C]
    y_pred: tensor of shape [B, D, H, W, C]
    alpha: controls the penalty for false positives
    beta: controls the penalty for false negatives
    gamma: focal parameter to down-weight easy examples
    smooth: smoothing constant to avoid division by zero

    Returns:
    loss: computed Focal Tversky loss
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, smooth, 1.0 - smooth)  # Clipping to avoid log(0)
    
    num_classes = 6
    loss = 0.0
    
    for c in range(num_classes):
        y_true_c = y_true[..., c]
        y_pred_c = y_pred[..., c]
        
        true_pos = tf.reduce_sum(y_true_c * y_pred_c)
        false_neg = tf.reduce_sum(y_true_c * (1 - y_pred_c))
        false_pos = tf.reduce_sum((1 - y_true_c) * y_pred_c)
        
        tversky_index = (true_pos + smooth) / (true_pos + alpha * false_neg + beta * false_pos + smooth)
        loss_c = tf.pow((1 - tversky_index), gamma)
        loss += loss_c
    
    loss /= tf.cast(num_classes, tf.float32)  # Averaging over all classes
    return loss
