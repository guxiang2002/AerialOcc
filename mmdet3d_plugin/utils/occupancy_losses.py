import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def KL_sep(p, target):
    """
    KL divergence on nonzeros classes
    """
    nonzeros = target != 0
    nonzero_p = p[nonzeros]
    kl_term = F.kl_div(torch.log(nonzero_p), target[nonzeros], reduction="sum")
    return kl_term


def _scaling_bce_to_one(probability):
    # Independent FP32 reductions can round a mathematical probability slightly
    # outside [0, 1] (for example 1.000000119). CUDA BCE asserts on that value.
    # Preserve the original BCE for every valid probability, including 0 and 1.
    probability = probability.clamp(min=0.0, max=1.0)
    return F.binary_cross_entropy(probability, torch.ones_like(probability))


def geo_scal_loss(
    pred, ssc_target, ignore_index=255, non_empty_idx=0, pred_is_prob=False,
    reduction_chunk_size=65536,
):

    # Get softmax probabilities
    if not pred_is_prob:
        pred = F.softmax(pred, dim=1)

    empty_probs = pred[:, non_empty_idx].reshape(-1)
    targets = ssc_target.reshape(-1)
    intersection = pred.new_zeros((), dtype=torch.float32)
    nonempty_prob_sum = pred.new_zeros((), dtype=torch.float32)
    specificity_sum = pred.new_zeros((), dtype=torch.float32)
    nonempty_target_sum = 0
    empty_target_sum = 0

    for start in range(0, targets.numel(), reduction_chunk_size):
        end = min(start + reduction_chunk_size, targets.numel())
        target_chunk = targets[start:end]
        valid = target_chunk != ignore_index
        if not torch.any(valid):
            continue
        empty_chunk = empty_probs[start:end].float()[valid]
        nonempty_chunk = 1.0 - empty_chunk
        nonempty_target = target_chunk[valid] != non_empty_idx
        nonempty_target_float = nonempty_target.float()
        intersection = intersection + (nonempty_target_float * nonempty_chunk).sum()
        nonempty_prob_sum = nonempty_prob_sum + nonempty_chunk.sum()
        specificity_sum = specificity_sum + (
            (~nonempty_target).float() * empty_chunk
        ).sum()
        nonempty_target_sum += int(nonempty_target.sum())
        empty_target_sum += int((~nonempty_target).sum())

    if nonempty_target_sum + empty_target_sum == 0:
        return pred.reshape(-1)[0] * 0.0

    eps = 1e-5
    precision = intersection / (nonempty_prob_sum + eps)
    recall = intersection / (nonempty_target_sum + eps)
    spec = specificity_sum / (empty_target_sum + eps)
    return (
        _scaling_bce_to_one(precision)
        + _scaling_bce_to_one(recall)
        + _scaling_bce_to_one(spec)
    )


def sem_scal_loss(
    pred, ssc_target, ignore_index=255, pred_is_prob=False,
    reduction_chunk_size=65536,
):
    # Get softmax probabilities
    if not pred_is_prob:
        pred = F.softmax(pred, dim=1)
    loss = 0
    count = 0
    n_classes = pred.shape[1]
    targets = ssc_target.reshape(-1)
    for i in range(0, n_classes):
        probabilities = pred[:, i].reshape(-1)
        probability_sum = pred.new_zeros((), dtype=torch.float32)
        numerator = pred.new_zeros((), dtype=torch.float32)
        positive_count = 0
        negative_count = 0

        for start in range(0, targets.numel(), reduction_chunk_size):
            end = min(start + reduction_chunk_size, targets.numel())
            target_chunk = targets[start:end]
            valid = target_chunk != ignore_index
            if not torch.any(valid):
                continue
            probability_chunk = probabilities[start:end].float()[valid]
            positive = target_chunk[valid] == i
            probability_sum = probability_sum + probability_chunk.sum()
            numerator = numerator + probability_chunk[positive].sum()
            positive_count += int(positive.sum())
            negative_count += int((~positive).sum())

        if positive_count > 0:
            count += 1.0
            loss_class = 0
            if probability_sum.detach().item() > 0:
                precision = numerator / probability_sum
                loss_precision = _scaling_bce_to_one(precision)
                loss_class += loss_precision
            if positive_count > 0:
                recall = numerator / positive_count
                loss_recall = _scaling_bce_to_one(recall)
                loss_class += loss_recall
            if negative_count > 0:
                negative_probability_sum = probability_sum - numerator
                specificity = (
                    negative_count - negative_probability_sum
                ) / negative_count
                loss_specificity = _scaling_bce_to_one(specificity)
                loss_class += loss_specificity
            loss += loss_class
    if count == 0:
        return pred.reshape(-1)[0] * 0.0
    return loss / count


def CE_ssc_loss(pred, target, class_weights=None, ignore_index=255):
    """
    :param: prediction: the predicted tensor, must be [BS, C, ...]
    """
    criterion = nn.CrossEntropyLoss(
        weight=class_weights, ignore_index=ignore_index, reduction="mean"
    )
    loss = criterion(pred, target.long())

    return loss


def focal_ssc_loss(pred, target, class_weights=None, ignore_index=255, gamma=2.0):
    target = target.long()
    valid_mask = target != ignore_index
    if not torch.any(valid_mask):
        return pred.sum() * 0.0

    target_clamped = target.clamp(min=0, max=pred.shape[1] - 1)
    log_probs = F.log_softmax(pred, dim=1)
    probs = log_probs.exp()

    gathered_log_probs = torch.gather(log_probs, 1, target_clamped.unsqueeze(1)).squeeze(1)
    gathered_probs = torch.gather(probs, 1, target_clamped.unsqueeze(1)).squeeze(1)

    pt = gathered_probs[valid_mask]
    log_pt = gathered_log_probs[valid_mask]
    loss = -((1.0 - pt) ** gamma) * log_pt

    if class_weights is not None:
        class_weights = class_weights.to(pred.device, dtype=pred.dtype)
        loss = loss * class_weights[target_clamped[valid_mask]]

    return loss.mean()

def vel_loss(pred, gt):
    return F.l1_loss(pred, gt)
