"""Compare saved phone and ranked-word errors without fitting to references."""
import argparse
from collections import Counter
import hashlib
import json
import logging
from pathlib import Path

from CompetitiveWordDecoder.audit_phone_candidates import summarize
from CompetitiveWordDecoder.decoder_tolerance import edits
from PhonemeDecoder.evaluate_qwen_direct import words


def pair(phone_a, phone_b, word_a, word_b):
    sources = [phone_a, phone_b, word_a, word_b]
    indices = [{r['clip_id']: r for r in rows} for rows in sources]
    if any(len(index) != len(rows) for index, rows in zip(indices, sources)):
        raise ValueError('duplicate clip IDs')
    if not indices[0] or any(set(index) != set(indices[0]) for index in indices):
        raise ValueError('mismatched clip IDs')
    summaries = [Counter(), Counter()]
    joint = Counter()
    details = []
    for clip_id in indices[0]:
        a, b, wa, wb = [index[clip_id] for index in indices]
        reference = words(' '.join(a['reference_words']))
        if (a['reference_phones'] != b['reference_phones'] or
                reference != words(' '.join(b['reference_words'])) or
                any(reference != w['reference_words'] for w in (wa, wb))):
            raise ValueError('mismatched references')
        phone_errors, word_errors = [], []
        for p, w, counts in zip((a, b), (wa, wb), summaries):
            pe = len(edits(p['reference_phones'], p['phone_hypotheses'][0]['phones']))
            selected = w['candidates'][w['selected_index']]
            prediction = words(selected['text'])
            ops = Counter(op for op, _, _ in edits(reference, prediction))
            we = sum(ops.values())
            if we != w['errors'][w['selected_index']]:
                raise ValueError('saved selected error count differs')
            counts.update(ops)
            counts.update(clips=1, reference_words=len(reference), predicted_words=len(prediction),
                          selected_caps=int(selected['hit_generation_cap']),
                          any_caps=int(any(c['hit_generation_cap'] for c in w['candidates'])),
                          oracle_errors=min(w['errors']), word_errors=we, phone_errors=pe)
            phone_errors.append(pe)
            word_errors.append(we)
        pd = phone_errors[1] - phone_errors[0]
        wd = word_errors[1] - word_errors[0]
        direction = lambda x: 'better' if x < 0 else 'worse' if x > 0 else 'tie'
        joint[f'phone_{direction(pd)}__word_{direction(wd)}'] += 1
        details.append(dict(clip_id=clip_id, phone_error_delta=pd, word_error_delta=wd))
    return dict(baseline=dict(summaries[0]), challenger=dict(summaries[1]),
                paired_clip_counts=dict(joint), details=details)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline-phones', 'challenger-phones', 'baseline-ranked', 'challenger-ranked', 'training-phones', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(filename)s:%(lineno)d %(message)s',
                        handlers=[logging.FileHandler(args.output/'run.log'), logging.StreamHandler()])
    read = lambda p: [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    a, b, train = map(read, (args.baseline_phones, args.challenger_phones, args.training_phones))
    wa, wb = [json.loads(p.read_text()) for p in (args.baseline_ranked, args.challenger_ranked)]
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    for report, source in ((wa, args.baseline_phones), (wb, args.challenger_phones)):
        if report['source_sha256'] != sha(source):
            raise ValueError('ranked report source hash mismatch')
    if wa['adapter_sha256'] != wb['adapter_sha256']:
        raise ValueError('different adapters')
    result = pair(a, b, wa['details'], wb['details'])
    for name, rows in [('baseline_phones', a), ('challenger_phones', b), ('training_phones', train)]:
        result[name] = summarize(rows)
    result['sources'] = {k: dict(path=str(v), sha256=sha(v)) for k, v in vars(args).items() if k != 'output'}
    result['caveats'] = ['Repeatedly inspected development; descriptive comparison, not causal isolation.',
                         'Minimum-edit alignment ties prefer diagonal then deletion.',
                         'Phone metric uses rank one; word metric uses probability-selected candidate.',
                         'Training-source summary includes full export, not only selected adaptation IDs.']
    (args.output/'metrics.json').write_text(json.dumps(result, indent=2)+'\n')
    logging.info('Paired comparison: %s', {k: v for k, v in result.items() if k not in ('details', 'sources')})


if __name__ == '__main__':
    main()
