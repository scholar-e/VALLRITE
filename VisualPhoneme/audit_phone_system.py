"""Audit padding, blank emissions and train-side blank calibration without training."""
import argparse
import hashlib
import json
import logging
from pathlib import Path
import random

import torch
from VisualPhoneme.data import PHONEMES, collate_clips, greedy_decode
from VisualPhoneme.lrs3_data import Lrs3Clips
from VisualPhoneme.train import edit_totals, upsample_ctc_logits
from VisualPhoneme.train_phone_head import load_phone_export
from VisualPhoneme.thermal import ThermalGuard

LOG = logging.getLogger(__name__)


def metrics(records, bias=0):
    errors = tokens = predicted = blanks = frames = 0
    for row in records:
        scores = row['logits'].clone()
        scores[:, 0] += bias
        hyp = greedy_decode(scores[:, None, :], torch.tensor([len(scores)]))[0]
        errors += edit_totals(row['target'], hyp)[0]
        tokens += len(row['target'])
        predicted += len(hyp)
        blanks += int((scores.argmax(-1) == 0).sum())
        frames += len(scores)
    return dict(errors=errors, phones=tokens, per=errors/tokens,
                predicted_phones=predicted, blank_argmax_fraction=blanks/frames)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO,
        format='%(asctime)s %(filename)s:%(lineno)d %(message)s',
        handlers=[logging.FileHandler(args.output/'audit.log'), logging.StreamHandler()])
    torch.set_num_threads(4)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    data_args = dict(root=Path('datasets/lrs3'), size=checkpoint['image_size'],
                     crop=checkpoint['crop'], horizontal_flip=False)
    train = Lrs3Clips(split='train', **data_args)
    dev = Lrs3Clips(split='validation', **data_args)
    manifest = json.loads((args.checkpoint.parent/'manifest.json').read_text())
    seen = set(manifest['train_ids'])
    eligible = [r for r in train.rows if r['clip_id'] not in seen]
    train.rows = random.Random(20261028).sample(eligible, 256)
    refs = [json.loads(s) for s in Path('checkpoints/context-word-eval-20261014/phone-hypotheses.jsonl').read_text().splitlines()]
    by_id = {r['clip_id']:r for r in dev.rows}
    dev.rows = [by_id[r['clip_id']] for r in refs]
    assert all(r['phonemes']==ref['reference_phones'] for r,ref in zip(dev.rows,refs,strict=True))
    assert not {r['clip_id'] for r in train.rows} & set(by_id)
    report = dict(checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  calibration_ids=[r['clip_id'] for r in train.rows],
                  caveat='Calibration clips exclude this phone-head training subset, but may be seen by the pretrained encoder. Development is repeatedly inspected.',
                  bias_grid=[0,-0.5,-1,-2,-3], splits={})
    caches = {}
    with ThermalGuard(args.output, gpu_required=True), torch.inference_mode():
        model = load_phone_export(checkpoint).cuda().eval()
        for name, dataset in [('calibration',train),('development',dev)]:
            padded, solo = [], []
            frame_counts = []
            for start in range(0,len(dataset),8):
                if Path('PAUSE').exists() or (args.output/'PAUSE').exists():
                    raise InterruptedError('PAUSE requested; no completed metrics')
                items = [dataset[i] for i in range(start,min(start+8,len(dataset)))]
                video,_,lengths,_,_ = collate_clips(items)
                logits, out = upsample_ctc_logits(model(video.cuda()),lengths,2)
                logits = logits.cpu()
                for i,(clip,target,clip_id) in enumerate(items):
                    frame_counts.append(len(clip))
                    row = dict(clip_id=clip_id,target=target.tolist(),logits=logits[:int(out[i]),i].clone())
                    padded.append(row)
                    single,_ = upsample_ctc_logits(model(clip[None].cuda()),torch.tensor([len(clip)]),2)
                    solo.append(dict(clip_id=clip_id,target=target.tolist(),logits=single[:,0].cpu()))
                if start%160==0:
                    LOG.info('%s encoded=%d/%d',name,len(padded),len(dataset))
            caches[name] = dict(padded=padded, solo=solo)
            torch.save(caches[name],args.output/f'{name}-logits.pt')
            report['splits'][name] = dict(clips=len(dataset),
                frame_min=min(frame_counts),frame_median=sorted(frame_counts)[len(frame_counts)//2],frame_max=max(frame_counts),
                padded=metrics(padded),solo=metrics(solo))
    # Choose exclusively on calibration labels; development never chooses the bias.
    grid = {str(bias):metrics(caches['calibration']['padded'],bias) for bias in report['bias_grid']}
    chosen = min(report['bias_grid'],key=lambda b:grid[str(b)]['per'])
    report.update(calibration_grid=grid,selected_blank_bias=chosen,
                  development_selected_bias=metrics(caches['development']['padded'],chosen))
    (args.output/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    export_greedy(caches['development']['padded'], refs, chosen, args.output/'calibrated-greedy.jsonl')
    LOG.info('Complete splits=%s selected_bias=%s development=%s',report['splits'],chosen,report['development_selected_bias'])


def export_greedy(rows, references, bias, path):
    with path.open('w') as handle:
        for row, ref in zip(rows, references, strict=True):
            if row['clip_id'] != ref['clip_id']:
                raise ValueError('reference/export ID mismatch')
            logits = row['logits'].clone()
            logits[:,0] += bias
            ids = greedy_decode(logits[:,None],torch.tensor([len(logits)]))[0]
            record = {k:ref[k] for k in ('clip_id','reference_words','reference_phones')}
            record.update(format='lrs3-phoneme-nbest-0.1', blank_logit_bias=bias,
                decoding='greedy', score_type='greedy alignment log probability, not sequence marginal',
                phone_hypotheses=[dict(rank=1,phones=[PHONEMES[i-1] for i in ids],
                    ctc_log_score=float(logits.log_softmax(-1).max(-1).values.sum()))])
            handle.write(json.dumps(record)+'\n')


if __name__ == '__main__':
    main()
