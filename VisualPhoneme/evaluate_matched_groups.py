"""Compare greedy family errors on an exact, previously exported validation set."""
import argparse
import hashlib
import json
from pathlib import Path
import signal

import torch
from torch.utils.data import DataLoader

from VisualPhoneme.data import PHONEMES, collate_clips, collate_fusion_clips, greedy_decode
from VisualPhoneme.evaluate_nbest import configure_logging, load_model, LOGGER
from VisualPhoneme.lrs3_data import Lrs3Clips
from VisualPhoneme.train import edit_totals, upsample_ctc_logits
from VisualPhoneme.visemes import PHONE_TO_VISUAL_GROUP, VISUAL_GROUPS, VISUAL_GROUPS_WITH_REST
from VisualPhoneme.thermal import ThermalGuard


def group_sequence(sequence, labels):
    # Collapse CTC first; retain repeated families from distinct phone emissions.
    return [PHONE_TO_VISUAL_GROUP.get(labels[t - 1], labels[t - 1])
            for t in sequence if labels[t - 1] != 'rest']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, action='append', required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=Path('datasets/lrs3'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        parser.error('invalid batch size or worker count')
    args.output.mkdir(parents=True, exist_ok=False)
    configure_logging(args.output/'evaluation.log')
    source = [json.loads(line) for line in args.source.read_text().splitlines() if line.strip()]
    ids = [r['clip_id'] for r in source]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('empty or duplicate source clips')
    references = {r['clip_id']: [PHONE_TO_VISUAL_GROUP[p] for p in r['reference_phones']]
                  for r in source}
    stopped = False
    def stop(*unused):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; use host GPU launcher')
    reports = []
    with ThermalGuard(args.output, gpu_required=True):
        for path in args.checkpoint:
            checkpoint = torch.load(path, map_location='cpu', weights_only=True)
            labels = tuple(checkpoint['phones'])
            if labels not in (PHONEMES, VISUAL_GROUPS, VISUAL_GROUPS_WITH_REST):
                raise ValueError('unexpected label inventory')
            architecture = checkpoint['architecture']
            if architecture not in ('large-image', 'gated-fusion'):
                raise ValueError('unsupported architecture for matched evaluation')
            fusion = architecture == 'gated-fusion'
            data = Lrs3Clips(args.data_root, 'validation', int(checkpoint['image_size']),
                            checkpoint['crop'], include_landmarks=fusion,
                            horizontal_flip=False,
                            coordinate_mode=checkpoint.get('coordinate_mode', 'eye-normalized'),
                            coordinate_features=checkpoint.get('coordinate_features', 'position'))
            rows = {r['clip_id']: r for r in data.rows}
            if any(i not in rows for i in ids):
                raise ValueError('matched source clips missing; refusing subset evaluation')
            data.rows = [rows[i] for i in ids]
            for original, row in zip(source, data.rows, strict=True):
                if original['reference_phones'] != row['phonemes']:
                    raise ValueError('reference phones differ from original export')
            _, model = load_model(checkpoint, torch.device('cuda'))
            loader = DataLoader(data, batch_size=args.batch_size, num_workers=args.workers,
                                collate_fn=collate_fusion_clips if fusion else collate_clips,
                                pin_memory=True)
            details = []
            with torch.inference_mode():
                for batch in loader:
                    if stopped or Path('PAUSE').exists() or (args.output/'PAUSE').exists():
                        LOGGER.info('Paused; completed model reports retained')
                        return
                    if fusion:
                        video, landmarks, mask, _, lengths, _, clip_ids = batch
                        logits = model(video.cuda(), landmarks.cuda(), mask.cuda())
                    else:
                        video, _, lengths, _, clip_ids = batch
                        logits = model(video.cuda())
                    logits, lengths = upsample_ctc_logits(
                        logits, lengths, int(checkpoint.get('ctc_upsample_factor', 1)))
                    for clip_id, sequence in zip(clip_ids, greedy_decode(logits, lengths), strict=True):
                        prediction = group_sequence(sequence, labels)
                        errors, count = edit_totals(references[clip_id], prediction)
                        details.append(dict(clip_id=clip_id, errors=errors, reference_tokens=count,
                                            prediction=prediction, reference=references[clip_id]))
                    if len(details) % 80 == 0:
                        LOGGER.info('%s processed=%d/%d', path.parent.name, len(details), len(data))
            assert [d['clip_id'] for d in details] == ids
            errors = sum(d['errors'] for d in details)
            count = sum(d['reference_tokens'] for d in details)
            with path.open('rb') as handle:
                checkpoint_hash = hashlib.file_digest(handle, 'sha256').hexdigest()
            report = dict(checkpoint=str(path.resolve()), checkpoint_sha256=checkpoint_hash,
                          epoch=checkpoint.get('epoch'), architecture=architecture,
                          parameters=sum(p.numel() for p in model.parameters()),
                          inventory=checkpoint.get('target_inventory', 'phones'),
                          clips=len(details), errors=errors, reference_tokens=count,
                          group_per=errors/count, details=details)
            reports.append(report)
            (args.output/f'{path.parent.name}.json').write_text(json.dumps(report, indent=2)+'\n')
            LOGGER.info('%s complete group_PER=%.2f%%', path.parent.name, 100*errors/count)
            del model, checkpoint, logits, loader
            torch.cuda.empty_cache()
    summary = dict(source=str(args.source.resolve()),
                   source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
                   scoring='greedy CTC; map phones to families; strip REST; no repeat-family merging',
                   scope='matched full validation clips, not aligned training chunks; not exact-phone PER or WER',
                   models=[{k:v for k,v in r.items() if k != 'details'} for r in reports])
    (args.output/'metrics.json').write_text(json.dumps(summary, indent=2)+'\n')


if __name__ == '__main__':
    main()
