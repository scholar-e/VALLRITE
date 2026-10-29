"""Auxiliary losses that preserve the distinction between CTC blank and silence."""
import torch
from torch import nn


def expected_phone_count(logits, lengths):
    """Exact expected collapsed length for independent CTC alignment emissions.

    Inputs are T,B,C. A nonblank emits at t when the preceding frame differs.
    Repeats separated by blank count twice; sustained repeats count once.
    """
    probabilities = logits.softmax(-1)[..., 1:]
    previous = torch.cat((torch.zeros_like(probabilities[:1]), probabilities[:-1]))
    emissions = (probabilities * (1 - previous)).sum(-1)
    valid = torch.arange(len(logits), device=logits.device)[:, None] < lengths.to(logits.device)[None]
    return (emissions * valid).sum(0)


def phone_count_loss(logits, lengths, target_lengths):
    expected = expected_phone_count(logits, lengths)
    targets = target_lengths.to(logits.device).float().clamp_min(1)
    return nn.functional.smooth_l1_loss(expected / targets, torch.ones_like(targets))


def masked_teacher_loss(logits, targets, mask):
    """T,B,C soft targets; unlabeled frames contribute no loss or blank target."""
    if not mask.any():
        return logits.sum() * 0
    return -(targets[mask] * logits.log_softmax(-1)[mask]).sum(-1).mean()
