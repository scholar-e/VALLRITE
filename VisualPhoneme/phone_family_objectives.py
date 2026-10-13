"""Experimental partitions for auxiliary probabilistic family CTC.

These are hand-specified hypotheses, not claimed anatomical equivalences.
All models retain the same 39-phone output vocabulary.
"""
import torch
from VisualPhoneme.data import PHONEMES
from VisualPhoneme.visemes import VISUAL_PHONE_GROUPS


def family_partition(name):
    groups = list(VISUAL_PHONE_GROUPS.values())
    if name == 'fine':
        # Preserve only a small set of plausible ambiguous consonant pairs.
        pairs = [('B','P'), ('F','V'), ('DH','TH'), ('D','T'),
                 ('S','Z'), ('CH','JH'), ('SH','ZH'), ('G','K')]
        grouped = {p for pair in pairs for p in pair}
        groups = pairs + [(p,) for p in PHONEMES if p not in grouped]
    elif name == 'coarse':
        # Merge several tongue-place distinctions and vowel subdivisions.
        groups = [tuple('B M P'.split()), tuple('F V'.split()),
                  tuple('DH TH D N T S Z'.split()), tuple('CH JH SH ZH'.split()),
                  tuple('G K NG'.split()), tuple('L R ER'.split()),
                  tuple('UH UW W AW OW OY'.split()), tuple('IH IY Y AY EY'.split()),
                  tuple('AA AH AO AE EH'.split()), ('HH',)]
    elif name != 'original':
        raise ValueError('unknown family partition')
    flattened = [p for group in groups for p in group]
    if len(flattened) != len(PHONEMES) or set(flattened) != set(PHONEMES):
        raise ValueError('families must partition all phones exactly once')
    mapping = [0] + [next(i+1 for i,g in enumerate(groups) if p in g) for p in PHONEMES]
    return groups, mapping


def family_log_probs(phone_logits, mapping):
    """Sum normalized phone probabilities, keeping blank in its own class."""
    log_probs = phone_logits.log_softmax(-1)
    mapping = torch.as_tensor(mapping, device=phone_logits.device)
    return torch.stack([torch.logsumexp(log_probs[..., mapping == i], dim=-1)
                        for i in range(int(mapping.max())+1)], dim=-1)
