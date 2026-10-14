"""Deterministic, split-checked actual-error decoder continuation examples."""
import hashlib
import json
import random
from pathlib import Path
from CompetitiveWordDecoder.noisy_teacher import load_nbest
from CompetitiveWordDecoder.evaluate_vallr_qwen import vallr_prompt
from PhonemeDecoder.evaluate_qwen_direct import words


def build_rows(source, heldout, train_manifest):
    records=load_nbest(source)
    excluded=load_nbest(heldout)
    heldout_ids={r['clip_id'] for r in excluded}
    texts={' '.join(words(' '.join(r['reference_words']))) for r in excluded}
    allowed={r['clip_id'] for r in map(json.loads,Path(train_manifest).read_text().splitlines())}
    if any(r['clip_id'] in heldout_ids or r['clip_id'] not in allowed for r in records):
        raise ValueError('visual-error source contains non-training or held-out IDs')
    unique=[]; seen=set()
    for row in records:
        text=' '.join(words(' '.join(row['reference_words'])))
        if not text or text in texts or text in seen:
            continue
        seen.add(text); unique.append(row)
    random.Random(42).shuffle(unique)
    if len(unique)<1100:
        raise ValueError('need 1100 unique eligible training clips')
    selected={'train':unique[:1000],'validation':unique[1000:1100]}
    result={}
    for split,rows in selected.items():
        result[split]=[]
        for row in rows:
            for kind,phones in [('clean',row['reference_phones']),('visual',row['phone_hypotheses'][0]['phones'])]:
                result[split].append(dict(prompt=vallr_prompt(phones),target=' '.join(row['reference_words']),
                                         unknown_words=0,clip_id=row['clip_id'],input_kind=kind))
    metadata=dict(source_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                  heldout_sha256=hashlib.sha256(Path(heldout).read_bytes()).hexdigest(),
                  train_manifest_sha256=hashlib.sha256(Path(train_manifest).read_bytes()).hexdigest(),
                  selected_ids={k:[r['clip_id'] for r in v] for k,v in selected.items()},
                  caveat='older visual model, in-sample training predictions; not out-of-fold')
    return result,metadata
