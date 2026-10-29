"""Freeze a broad full-clip decoder split before generating visual errors."""
import argparse
from collections import Counter
import hashlib
import json
import logging
from pathlib import Path

from PhonemeDecoder.evaluate_qwen_direct import words


def video_id(clip_id):
    parts = clip_id.split('/')
    if len(parts) < 2:
        raise ValueError('expected LRS3 source path')
    return parts[-2]


def select(rows, excluded_videos, excluded_texts, counts=(750, 100, 100), seed=20261029):
    """Hash-sample one clip per video; discard duplicate normalized transcripts."""
    key = lambda value: hashlib.sha256(f'{seed}:{value}'.encode()).hexdigest()
    by_video = {}
    for row in rows:
        text = ' '.join(words(row['transcript']))
        video = video_id(row['clip_id'])
        if (video in excluded_videos or text in excluded_texts or not text or
                row.get('pretrain_chunk_index') is not None or
                not 60 <= row['frames'] <= 250 or not row.get('phonemes') or
                row.get('pronunciation_coverage', 0) < .95):
            continue
        if video not in by_video or key(row['clip_id']) < key(by_video[video]['clip_id']):
            by_video[video] = row
    eligible = []; seen = set()
    for video in sorted(by_video, key=key):
        row = by_video[video]; text = ' '.join(words(row['transcript']))
        if text not in seen:
            eligible.append(row); seen.add(text)
    if len(eligible) < sum(counts):
        raise ValueError(f'need {sum(counts)} distinct videos/transcripts, found {len(eligible)}')
    result = {}; start = 0
    for split, count in zip(('train', 'validation', 'evaluation'), counts):
        result[split] = eligible[start:start+count]; start += count
    return result, len(eligible)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(filename)s:%(lineno)d %(message)s',
                        handlers=[logging.FileHandler(args.output/'run.log'), logging.StreamHandler()])
    root = Path('checkpoints'); paths = [Path('datasets/lrs3/manifests/train.jsonl')]
    excluded_videos = set(); excluded_texts = set()
    for rel in ('context-word-eval-20261014/phone-hypotheses.jsonl',
                'context-train-export-20261026/phone-hypotheses.jsonl'):
        path = root/rel; paths.append(path)
        for line in path.read_text().splitlines():
            row = json.loads(line); excluded_videos.add(video_id(row['clip_id']))
            excluded_texts.add(' '.join(words(' '.join(row['reference_words']))))
    for rel, keys in [('fullclip-phone-context-20261013/manifest.json', ('train_ids',)),
                      ('blank-strategies-20261028/manifest.json', ('long_ids',)),
                      ('fixed-crop-followup-20261028/manifest.json', ('calibration_ids',))]:
        path = root/rel; paths.append(path); manifest = json.loads(path.read_text())
        for name in keys:
            excluded_videos.update(map(video_id, manifest[name]))
    with paths[0].open() as handle:
        splits, eligible = select((json.loads(line) for line in handle), excluded_videos, excluded_texts)
    for split, rows in splits.items():
        (args.output/f'{split}.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    summary = dict(seed=20261029, eligible_unique_videos=eligible, excluded_source_videos=len(excluded_videos),
                   selection='one full clip per video, 60-250 frames, SHA256 deterministic ordering, transcript deduplication',
                   sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}, splits={})
    for split, rows in splits.items():
        frames = sorted(r['frames'] for r in rows)
        summary['splits'][split] = dict(clips=len(rows), videos=len({video_id(r['clip_id']) for r in rows}),
            median_frames=frames[len(frames)//2], reference_phones=sum(len(r['phonemes']) for r in rows),
            label_sources=dict(Counter(r.get('label_source', 'unknown') for r in rows)),
            sha256=hashlib.sha256((args.output/f'{split}.jsonl').read_bytes()).hexdigest())
    summary['caveat'] = ('Held out from the listed recent phone-head subsets and current decoder export only; '
                         'pretrained encoder and older adapters may have seen these videos. Not an untouched end-to-end test.')
    (args.output/'manifest.json').write_text(json.dumps(summary, indent=2)+'\n')
    logging.info('Prepared %s', summary['splits'])


if __name__ == '__main__':
    main()
