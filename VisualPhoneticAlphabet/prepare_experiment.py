"""Prepare the fixed GRID pilot for independent audio alignment and visual probes."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'datasets/grid-pilot'
RUN = ROOT / 'landmark-experiment'


def prepare_corpus():
    rows = [json.loads(line) for line in (ROOT/'clips.jsonl').read_text().splitlines()]
    vocabulary = set()
    for row in rows:
        words = [line.split()[2].lower() for line in (ROOT/row['alignment']).read_text().splitlines()
                 if line.split()[2].lower() not in ('sil', 'sp')]
        if len(words) != 6 or len(words[3]) != 1:
            raise ValueError(f'Unexpected GRID sentence: {row["clip_id"]}: {words}')
        words[3] = 'letter'+words[3]
        vocabulary.update(words)
        target = RUN/'corpus'/('s'+row['speaker_id'])/Path(row['audio']).name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.symlink_to((ROOT/row['audio']).resolve())
        target.with_suffix('.lab').write_text(' '.join(words)+'\n')
    (RUN/'vocabulary.txt').write_text('\n'.join(sorted(vocabulary))+'\n')
    print(f'Prepared {len(rows)} full-length audio/transcript pairs; {len(vocabulary)} tokens.', flush=True)


def extract_one(row, model):
    import numpy as np
    from .__main__ import extract
    from .core import POINT_IDS
    out = RUN/'landmarks'/('s'+row['speaker_id'])/(Path(row['video']).stem+'.npz')
    if out.exists():
        return row['clip_id'], 'cached'
    record = extract(ROOT/row['video'], Path(model))
    coords = np.full((len(record['frames']), len(POINT_IDS), 2), np.nan, dtype=np.float32)
    for f, frame in enumerate(record['frames']):
        if frame['landmarks'] is not None:
            coords[f] = [frame['landmarks'][str(p)] for p in POINT_IDS]
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix('.partial.npz')
    np.savez_compressed(temporary, coordinates=coords, point_ids=np.array(POINT_IDS),
                        time_ms=np.array([f['time_ms'] for f in record['frames']]),
                        source_fps=record['source_fps'], provenance=json.dumps(record['provenance']))
    temporary.replace(out)
    return row['clip_id'], record['summary']['observed_fraction']


def extract_all(model, jobs, limit):
    rows = [json.loads(line) for line in (ROOT/'clips.jsonl').read_text().splitlines()]
    if limit:
        rows = rows[:limit]
    errors=[]
    RUN.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures={pool.submit(extract_one, row, model):row['clip_id'] for row in rows}
        for n, future in enumerate(as_completed(futures), 1):
            try:
                clip, observed=future.result()
            except Exception as e:
                errors.append({'clip_id':futures[future],'error':str(e)})
            if n%100==0 or n==len(rows):
                print(f'Landmarks: {n}/{len(rows)} clips processed; {len(errors)} errors', flush=True)
    (RUN/'extraction-errors.json').write_text(json.dumps(errors,indent=2)+'\n')
    if errors:
        raise SystemExit(f'{len(errors)} clips failed; see extraction-errors.json')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['corpus','landmarks'])
    parser.add_argument('--model', default='/tmp/vpa-overlay-face.task')
    parser.add_argument('--jobs', type=int, default=6)
    parser.add_argument('--limit', type=int)
    args=parser.parse_args()
    if args.stage=='corpus': prepare_corpus()
    else: extract_all(args.model,args.jobs,args.limit)

if __name__=='__main__': main()
