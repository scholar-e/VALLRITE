"""Validation-selected rolling-posterior pilot using saved visual logits."""
import argparse
import hashlib
import json
import logging
from pathlib import Path
import signal
import time
import torch
from VisualPhoneme.data import PHONEMES, ctc_prefix_beam_search
from VisualPhoneme.rolling_phones import smooth_repeated_logits, phone_segments
from CompetitiveWordDecoder.audit_phone_candidates import summarize


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('checkpoints/matched-phones-20261029'))
    p.add_argument('--manifests',type=Path,default=Path('datasets/decoder-training/source-disjoint-950-20261029'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-minutes',type=float,default=15)
    p.add_argument('--preserve-blank',action='store_true',help='smooth relative phone probabilities while keeping instantaneous blank mass')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(filename)s:%(lineno)d %(message)s',handlers=[logging.FileHandler(args.output/'run.log'),logging.StreamHandler()])
    torch.set_num_threads(2)
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    plan=dict(windows_original_frames=[1,3,5],preserve_blank=args.preserve_blank,averaging='causal trailing probability mean with shorter startup denominator',blank_bias=-1,ctc_upsample_factor=2,beam=64,top_k=12,top_n=5,lm_weight=0,
              selection='minimum validation rank-one PER; ties prefer smaller window; freeze before evaluation',
              hashes={str(path):sha(path) for split in ('validation','evaluation') for path in (args.source/split/'logits.pt',args.source/split/'phone-hypotheses.jsonl',args.manifests/f'{split}.jsonl')})
    manifest=args.output/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())!=plan:raise ValueError('resume mismatch')
    manifest.write_text(json.dumps(plan,indent=2)+'\n')
    stopping=False
    def stop(*_):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    started=time.monotonic()
    def paused():return stopping or Path('PAUSE').exists() or (args.output/'PAUSE').exists() or time.monotonic()-started>args.max_minutes*60
    def evaluate(split,window):
        cache=torch.load(args.source/split/'logits.pt',map_location='cpu',weights_only=True)
        refs=[json.loads(l) for l in (args.manifests/f'{split}.jsonl').read_text().splitlines()]
        baseline=[json.loads(l) for l in (args.source/split/'phone-hypotheses.jsonl').read_text().splitlines()]
        assert len(cache)==len(refs)==len(baseline)
        output=args.output/f'{split}-w{window}';output.mkdir(exist_ok=True)
        dest=output/'phone-hypotheses.jsonl'
        saved=[json.loads(l) for l in dest.read_text().splitlines()] if dest.exists() else []
        if len(saved)>len(cache) or [r['clip_id'] for r in saved]!=[r['clip_id'] for r in cache[:len(saved)]]:raise ValueError('partial mismatch')
        distributions=[]
        with dest.open('a',buffering=1) as handle:
            for i,(cached,ref,original) in enumerate(zip(cache,refs,baseline)):
                if paused():return None
                assert cached['clip_id']==ref['clip_id']==original['clip_id']
                posterior,logp=smooth_repeated_logits(cached['logits'],window,preserve_blank=args.preserve_blank)
                assert len(posterior)==ref['frames']
                distributions.append(dict(clip_id=ref['clip_id'],fps=ref['fps'],probabilities=posterior))
                if i<len(saved):continue
                candidates=ctc_prefix_beam_search(logp,64,5,12)
                hypotheses=[dict(rank=j+1,phones=[PHONEMES[k-1] for k in hyp],ctc_log_score=score) for j,(hyp,score) in enumerate(candidates)]
                if window==1:
                    assert hypotheses==original['phone_hypotheses'],'identity baseline changed'
                segments=phone_segments(posterior,ref['fps'],PHONEMES)
                record=dict(clip_id=ref['clip_id'],reference_words=original['reference_words'],reference_phones=original['reference_phones'],phone_hypotheses=hypotheses,
                            rolling_window_frames=window,preserve_blank=args.preserve_blank,greedy_rolling_string=[s['phone'] for s in segments if s['phone'] is not None],rolling_greedy_segments=segments,
                            timing_note='Frame-argmax run boundaries, not alignments of the separate beam strings; blank is not silence; posterior mass is uncalibrated')
                saved.append(record);handle.write(json.dumps(record)+'\n')
                if (i+1)%25==0:logging.info('%s window=%d clips=%d/%d',split,window,i+1,len(cache))
        torch.save(dict(labels=['CTC_BLANK',*PHONEMES],rows=distributions),output/'rolling-posteriors.pt')
        report=summarize(saved)
        (output/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
        logging.info('%s window=%d PER=%.4f oracle=%.4f',split,window,report['per'],report['oracle_per'])
        return report
    calibration={}
    for window in plan['windows_original_frames']:
        result=evaluate('validation',window)
        if result is None:return
        calibration[str(window)]=result
    selected=min(plan['windows_original_frames'],key=lambda w:calibration[str(w)]['per'])
    (args.output/'selection.json').write_text(json.dumps(dict(validation=calibration,selected_window=selected),indent=2)+'\n')
    evaluation={}
    for window in sorted({1,selected}):
        result=evaluate('evaluation',window)
        if result is None:return
        evaluation[str(window)]=result
    (args.output/'metrics.json').write_text(json.dumps(dict(validation=calibration,selected_window=selected,evaluation=evaluation),indent=2)+'\n')
    (args.output/'COMPLETE').write_text('complete\n')
    logging.info('Complete selected window=%d',selected)

if __name__=='__main__':main()
