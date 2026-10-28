"""Measure saved phone candidate errors without fitting decoding parameters."""
import argparse
from collections import Counter
import hashlib
import json
import logging
from pathlib import Path

from CompetitiveWordDecoder.decoder_tolerance import edits
from CompetitiveWordDecoder.train_vallr_clean import edit_distance


def summarize(rows):
    counts = Counter()
    for row in rows:
        reference = row['reference_phones']
        paths = row['phone_hypotheses']
        hypothesis = paths[0]['phones']
        operations = Counter(op for op, _, _ in edits(reference, hypothesis))
        assert sum(operations.values()) == edit_distance(reference, hypothesis)
        assert len(hypothesis) - len(reference) == operations['ins'] - operations['del']
        counts.update(operations)
        counts.update(clips=1, reference_phones=len(reference), predicted_phones=len(hypothesis),
                      empty=int(not hypothesis), below_half_length=int(len(hypothesis)*2 < len(reference)),
                      oracle_errors=min(edit_distance(reference, p['phones']) for p in paths),
                      longest_candidate_phones=max(len(p['phones']) for p in paths))
    denominator = counts['reference_phones']
    return dict(counts, per=sum(counts[k] for k in ('sub','del','ins'))/denominator,
                oracle_per=counts['oracle_errors']/denominator,
                predicted_reference_length_ratio=counts['predicted_phones']/denominator)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO,
        format='%(asctime)s %(filename)s:%(lineno)d %(message)s',
        handlers=[logging.FileHandler(args.output/'audit.log'), logging.StreamHandler()])
    rows = [json.loads(line) for line in args.source.read_text().splitlines() if line.strip()]
    if not rows or len({r['clip_id'] for r in rows}) != len(rows):
        raise ValueError('empty or duplicate source IDs')
    if any(not r['reference_phones'] or [p['rank'] for p in r['phone_hypotheses']] != list(range(1,6)) for r in rows):
        raise ValueError('expected nonempty references and five ordered candidates')
    report = dict(source=str(args.source.resolve()),
                  source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
                  alignment='minimum edit distance; ties prefer diagonal then deletion',
                  overall=summarize(rows), by_reference_length={})
    for low, high in ((1,20),(21,40),(41,80),(81,100000)):
        subset = [r for r in rows if low <= len(r['reference_phones']) <= high]
        if subset:
            report['by_reference_length'][f'{low}-{high}'] = summarize(subset)
    (args.output/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    logging.info('Completed %s', report['overall'])


if __name__ == '__main__':
    main()
