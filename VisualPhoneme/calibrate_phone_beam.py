"""Select a beam blank bias on saved training-side logits, then evaluate once."""
import argparse
import hashlib
import json
import logging
from pathlib import Path
import signal
import time
import torch
from VisualPhoneme.data import PHONEMES, ctc_prefix_beam_search
from VisualPhoneme.train import edit_totals

LOG=logging.getLogger(__name__)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source',type=Path,default=Path('checkpoints/phone-system-audit-20261028'),
                        help='directory with calibration-logits.pt and development-logits.pt')
    parser.add_argument('--max-minutes',type=float,default=15)
    parser.add_argument('--calibration-only',action='store_true',
                        help='stop after selecting bias; do not decode development')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(filename)s:%(lineno)d %(message)s',
        handlers=[logging.FileHandler(args.output/'run.log'),logging.StreamHandler()])
    torch.set_num_threads(1)
    stopped=False
    def stop(*unused):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    started=time.monotonic()
    def paused():return stopped or Path('PAUSE').exists() or (args.output/'PAUSE').exists() or time.monotonic()-started>args.max_minutes*60
    source=args.source
    paths=[source/'calibration-logits.pt',source/'development-logits.pt']
    manifest=dict(biases=[0,-.5,-1],beam_width=64,token_top_k=12,top_n=5,lm_weight=0,
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        selection='minimum calibration rank-one phone errors; ties favor earlier bias',
        caveat='Calibration excludes exact head-training IDs, not all encoder or source-video exposure; development is repeatedly inspected.')
    manifest_path=args.output/'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text())!=manifest:raise ValueError('resume source mismatch')
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    calibration=torch.load(paths[0],map_location='cpu',weights_only=True)['padded']
    development=torch.load(paths[1],map_location='cpu',weights_only=True)['padded']
    refs=[json.loads(s) for s in Path('checkpoints/context-word-eval-20261014/phone-hypotheses.jsonl').read_text().splitlines()]
    def decode(rows,bias,path,references=None):
        saved=[json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
        for a,b in zip(saved,rows):
            if a['clip_id']!=b['clip_id'] or a['blank_logit_bias']!=bias:raise ValueError('partial output mismatch')
        with path.open('a',buffering=1) as handle:
            for index in range(len(saved),len(rows)):
                if paused():return None
                row=rows[index]; logits=row['logits'].clone(); logits[:,0]+=bias
                candidates=ctc_prefix_beam_search(logits.log_softmax(-1),64,5,12)
                errors=[edit_totals(row['target'],hyp)[0] for hyp,_ in candidates]
                record=dict(clip_id=row['clip_id'],blank_logit_bias=bias,errors=errors,
                    reference_phones=[PHONEMES[i-1] for i in row['target']],lm_weight=0,
                    phone_hypotheses=[dict(rank=j+1,phones=[PHONEMES[i-1] for i in hyp],ctc_log_score=score) for j,(hyp,score) in enumerate(candidates)])
                if references is not None:
                    ref=references[index]
                    assert ref['clip_id']==row['clip_id'] and ref['reference_phones']==record['reference_phones']
                    record['reference_words']=ref['reference_words']
                saved.append(record);handle.write(json.dumps(record)+'\n')
                if (index+1)%100==0:LOG.info('%s bias=%s clips=%d/%d',path.stem,bias,index+1,len(rows))
        count=sum(len(r['reference_phones']) for r in saved)
        return dict(clips=len(saved),phones=count,errors=sum(r['errors'][0] for r in saved),
                    per=sum(r['errors'][0] for r in saved)/count,
                    oracle_per=sum(min(r['errors']) for r in saved)/count)
    results={}
    for bias in manifest['biases']:
        report=decode(calibration,bias,args.output/f'calibration-{bias}.jsonl')
        if report is None:return
        results[str(bias)]=report;LOG.info('Calibration bias=%s result=%s',bias,report)
    chosen=min(manifest['biases'],key=lambda b:results[str(b)]['errors'])
    (args.output/'selection.json').write_text(json.dumps(dict(calibration=results,chosen_bias=chosen),indent=2)+'\n')
    if args.calibration_only:
        (args.output/'CALIBRATED').write_text('complete\n')
        return
    report=decode(development,chosen,args.output/'phone-hypotheses.jsonl',refs)
    if report is None:return
    final=dict(manifest,calibration=results,chosen_bias=chosen,development=report,seconds=time.monotonic()-started)
    (args.output/'metrics.json').write_text(json.dumps(final,indent=2)+'\n')
    (args.output/'COMPLETE').write_text('complete\n')
    LOG.info('Complete %s',final['development'])


if __name__=='__main__':main()
