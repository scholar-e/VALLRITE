"""Deterministic, split-checked actual-error decoder continuation examples."""
import hashlib
import json
import random
from pathlib import Path
from CompetitiveWordDecoder.noisy_teacher import load_nbest
from CompetitiveWordDecoder.evaluate_vallr_qwen import vallr_prompt
from PhonemeDecoder.evaluate_qwen_direct import words


def build_rows(source, heldout, train_manifest, mode='mixed'):
    if mode not in ('mixed','clean'):
        raise ValueError('unknown visual error mixture')
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
                if mode=='clean':
                    kind,phones='clean',row['reference_phones']
                result[split].append(dict(prompt=vallr_prompt(phones),target=' '.join(row['reference_words']),
                                         unknown_words=0,clip_id=row['clip_id'],input_kind=kind))
    metadata=dict(source_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                  heldout_sha256=hashlib.sha256(Path(heldout).read_bytes()).hexdigest(),
                  train_manifest_sha256=hashlib.sha256(Path(train_manifest).read_bytes()).hexdigest(),
                  selected_ids={k:[r['clip_id'] for r in v] for k,v in selected.items()},
                  caveat='older visual model, in-sample training predictions; not out-of-fold')
    provenance=Path(source).parent/'phone-eval.json'
    if provenance.exists():
        report=json.loads(provenance.read_text())
        if report.get('architecture')=='phone-family-context':
            if report.get('split')!='train':
                raise ValueError('current-model source must be a training export')
            metadata.update(visual_source=report,
                caveat='current context model; pretrained backbone saw training corpus; not out-of-fold')
    if mode!='mixed': metadata['mode']=mode
    return result,metadata


def build_frozen_rows(source, heldout, split_manifests, mode='mixed'):
    """Use exact predeclared splits; reject video, transcript and reference leakage."""
    from CompetitiveWordDecoder.prepare_source_disjoint import video_id
    if mode not in ('mixed', 'clean'):
        raise ValueError('unknown visual error mixture')
    source=Path(source); split_manifests=Path(split_manifests)
    paths={'train':source,'validation':source.parent.parent/'validation/phone-hypotheses.jsonl','evaluation':Path(heldout)}
    result={}; seen_videos=set(); seen_texts=set(); hashes={}
    for split,path in paths.items():
        records=load_nbest(path)
        manifest=split_manifests/f'{split}.jsonl'
        expected=[json.loads(l) for l in manifest.read_text().splitlines()]
        if len(records)!=len(expected) or not records:raise ValueError('frozen split count mismatch')
        converted=[]
        for row,ref in zip(records,expected):
            text=' '.join(words(' '.join(row['reference_words'])))
            if row['clip_id']!=ref['clip_id'] or row['reference_phones']!=ref['phonemes'] or text!=' '.join(words(ref['transcript'])):
                raise ValueError('frozen reference mismatch')
            video=video_id(row['clip_id'])
            if video in seen_videos or text in seen_texts:raise ValueError('source-video or transcript leakage')
            seen_videos.add(video);seen_texts.add(text)
            if split=='evaluation':continue
            for kind,phones in [('clean',row['reference_phones']),('visual',row['phone_hypotheses'][0]['phones'])]:
                if mode=='clean':kind,phones='clean',row['reference_phones']
                converted.append(dict(prompt=vallr_prompt(phones),target=' '.join(row['reference_words']),unknown_words=0,clip_id=row['clip_id'],input_kind=kind))
        if split!='evaluation':result[split]=converted
        hashes[split]=dict(source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
    return result,dict(mode=mode,frozen_splits=hashes,caveat='Source-video-disjoint recent subsets; historical pretraining exposure not fully audited')
