"""Compare frozen greedy VALLR predictions to explicit reference phonemes.

Run from the root using .venv-qwen/bin/python -m
VisualPhoneticAlphabet.evaluate. Prepare crops with the crop CLI first.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


def score(reference, hypothesis):
    """Levenshtein alignment; ties favor match/substitution, then deletion."""
    if not reference:
        raise ValueError('reference cannot be empty')
    table = [[0]*(len(hypothesis)+1) for _ in range(len(reference)+1)]
    for i in range(len(reference)+1):
        table[i][0] = i
    for j in range(len(hypothesis)+1):
        table[0][j] = j
    for i, ref in enumerate(reference, 1):
        for j, hyp in enumerate(hypothesis, 1):
            table[i][j] = min(table[i-1][j-1]+(ref != hyp), table[i-1][j]+1, table[i][j-1]+1)
    i, j = len(reference), len(hypothesis)
    alignment = []
    while i or j:
        if i and j and table[i][j] == table[i-1][j-1]+(reference[i-1] != hypothesis[j-1]):
            alignment.append({'operation': 'match' if reference[i-1] == hypothesis[j-1] else 'substitution', 'reference': reference[i-1], 'hypothesis': hypothesis[j-1]})
            i, j = i-1, j-1
        elif i and table[i][j] == table[i-1][j]+1:
            alignment.append({'operation': 'deletion', 'reference': reference[i-1], 'hypothesis': None})
            i -= 1
        else:
            alignment.append({'operation': 'insertion', 'reference': None, 'hypothesis': hypothesis[j-1]})
            j -= 1
    return {'reference_length': len(reference), 'errors': table[-1][-1],
            'per': table[-1][-1]/len(reference),
            **{kind+'s': sum(a['operation'] == kind for a in alignment) for kind in ('substitution', 'deletion', 'insertion')},
            'alignment': list(reversed(alignment))}


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('datasets/grid-smoke'))
    parser.add_argument('--manifest', type=Path, default=Path(__file__).with_name('evaluation')/'grid-smoke.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    # The upstream CLI uses imports relative to the original VALLR directory.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch
    import transformers
    from vallrite import load_visual_model, load_video, ctc_decode, DEFAULT_VISUAL_CHECKPOINT, PHONEMES
    torch.set_num_threads(4)
    manifest = json.loads(args.manifest.read_text())
    model = load_visual_model(DEFAULT_VISUAL_CHECKPOINT, torch.device('cpu'))
    rows = []
    for clip in manifest['clips']:
        for mode, suffix in [('original', '.mpg'), ('face_neck', '-face-neck.mp4')]:
            path = args.data_dir/(clip['id']+suffix)
            inputs = load_video(path)
            start = time.perf_counter()
            with torch.inference_mode():
                logits, _ = model(inputs)
            elapsed = time.perf_counter()-start
            probs = logits.softmax(-1)[0]
            hypothesis = ctc_decode(logits)
            metrics = score(clip['reference'].split(), hypothesis)
            variant_scores = [metrics['per']] + [score(ref.split(), hypothesis)['per'] for ref in clip['reference_alternatives']]
            row = {'clip_id': clip['id'], 'mode': mode, 'source_sha256': digest(path),
                   'transcript': clip['transcript'], 'reference': clip['reference'].split(),
                   'hypothesis': hypothesis, 'metrics': metrics,
                   'pronunciation_variant_per_range': [min(variant_scores), max(variant_scores)],
                   'ctc_steps': len(probs), 'mean_blank_probability': float(probs[:, 0].mean()),
                   'blank_argmax_steps': int((probs.argmax(-1) == 0).sum()),
                   'forward_seconds': elapsed,
                   'raw_probabilities': probs.tolist()}
            if mode == 'face_neck':
                sidecar = path.with_suffix('.jsonl')
                crop_rows = [json.loads(line) for line in sidecar.read_text().splitlines()]
                row['crop_observed_fraction'] = sum(r['crop']['status']=='observed' for r in crop_rows)/len(crop_rows)
            rows.append(row)
            print(f"{clip['id']} {mode}: {' '.join(hypothesis) or '(empty)'}; PER={metrics['per']:.1%}", flush=True)
    report = {'scope': manifest['description'], 'manifest': manifest,
              'checkpoint_sha256': digest(DEFAULT_VISUAL_CHECKPOINT),
              'torch_version': torch.__version__, 'transformers_version': transformers.__version__,
              'device': 'cpu', 'decoder': 'greedy CTC; no LM or Qwen',
              'preprocessing': 'Existing CLI unchanged: 16 uniformly sampled RGB frames, 224 square, float32 0..255; crop condition uses 256 letterboxed face/neck MP4.',
              'vpa_phoneme_head': 'not implemented; only existing VALLR checkpoint tested',
              'probability_type': 'raw softmax, uncalibrated', 'vocabulary': list(PHONEMES),
              'rows': rows, 'aggregate': {}}
    for mode in ('original', 'face_neck'):
        selected = [r['metrics'] for r in rows if r['mode']==mode]
        errors, length = sum(r['errors'] for r in selected), sum(r['reference_length'] for r in selected)
        report['aggregate'][mode] = {'errors': errors, 'reference_phonemes': length, 'per': errors/length}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
