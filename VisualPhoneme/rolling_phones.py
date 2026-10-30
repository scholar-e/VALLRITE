"""Causal rolling phone posteriors; blank remains an alignment state, not silence."""
from collections import deque
import math
import torch


class RollingPhoneAverage:
    """Chunk-invariant trailing arithmetic mean over original video frames.

    Construct a fresh instance for each clip/stream. No future frames or labels
    are used. The upstream visual encoder may still be noncausal.
    """
    def __init__(self, window_frames=3):
        if type(window_frames) is not int or window_frames < 1:
            raise ValueError('positive integer window required')
        self.window_frames = window_frames
        self.history = deque(maxlen=window_frames)
        self.classes = None

    def update(self, probabilities):
        if probabilities.ndim != 2 or probabilities.shape[1] < 2:
            raise ValueError('expected [frames, blank plus phones]')
        if (not torch.isfinite(probabilities).all() or (probabilities < 0).any() or
                not torch.allclose(probabilities.sum(-1), torch.ones_like(probabilities[:, 0]), atol=1e-5)):
            raise ValueError('finite normalized probabilities required')
        if self.classes is not None and self.classes != probabilities.shape[1]:
            raise ValueError('phone inventory changed within stream')
        self.classes = probabilities.shape[1]
        result = []
        for row in probabilities:
            self.history.append(row.clone())
            result.append(torch.stack(list(self.history)).mean(0))
        return torch.stack(result) if result else probabilities.clone()


def smooth_repeated_logits(logits, window_frames, upsample_factor=2, blank_bias=-1., preserve_blank=False):
    """Undo exact CTC repeats, smooth probabilities, then restore alignment states."""
    if (type(upsample_factor) is not int or upsample_factor < 1 or
            logits.ndim != 2 or not len(logits) or len(logits) % upsample_factor or
            not math.isfinite(blank_bias) or not torch.isfinite(logits).all()):
        raise ValueError('invalid cached logits or upsample factor')
    grouped = logits.reshape(-1, upsample_factor, logits.shape[-1])
    if not torch.equal(grouped, grouped[:, :1].expand_as(grouped)):
        raise ValueError('cache is not exact repeated original-frame emissions')
    original = grouped[:, 0].clone()
    original[:, 0] += blank_bias
    probabilities = original.softmax(-1)
    rolling = RollingPhoneAverage(window_frames).update(probabilities)
    if preserve_blank and window_frames > 1:
        # Keep instantaneous CTC emission mass; average relative phone identity
        # weighted by the nonblank mass in the trailing window.
        nonblank = rolling[:, 1:]
        mass = nonblank.sum(-1, keepdim=True)
        current_mass = probabilities[:, 1:].sum(-1, keepdim=True)
        corrected = nonblank / mass.clamp_min(torch.finfo(rolling.dtype).tiny) * current_mass
        rolling = torch.cat((probabilities[:, :1], corrected), dim=-1)
    if window_frames == 1:
        # Preserve the precise baseline beam scores, avoiding exp/log roundoff.
        beam_logp = original.log_softmax(-1).repeat_interleave(upsample_factor, 0)
    else:
        beam_logp = rolling.clamp_min(torch.finfo(rolling.dtype).tiny).log().repeat_interleave(upsample_factor, 0)
    return rolling, beam_logp


def phone_segments(probabilities, fps, labels):
    """Greedy runs with frame intervals and full mean posterior, including blank.

    These runs describe the rolling frame argmax, NOT the separate CTC beam
    hypothesis. A blank-separated repeated phone remains two distinct runs.
    Posterior mass is model confidence, not calibrated probability of correctness.
    """
    if not math.isfinite(fps) or fps <= 0 or probabilities.ndim != 2 or probabilities.shape[1] != len(labels)+1:
        raise ValueError('invalid fps or inventory')
    if not len(probabilities):return []
    ids = probabilities.argmax(-1).tolist()
    result = []; start = 0
    for end in range(1, len(ids)+1):
        if end < len(ids) and ids[end] == ids[start]:continue
        mean = probabilities[start:end].mean(0)
        token = ids[start]
        result.append(dict(phone=labels[token-1] if token else None,
                           state='phone' if token else 'ctc_blank',
                           start_frame=start, end_frame_exclusive=end,
                           start_seconds=start/fps, end_seconds=end/fps,
                           posterior_mass=float(mean[token]),
                           mean_distribution=mean.tolist()))
        start = end
    return result
