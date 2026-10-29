"""Resumable fixed-crop export for a frozen source-disjoint manifest."""
import argparse
import hashlib
import json
import logging
from pathlib import Path
import re
import signal
import time
import torch
from VisualPhoneme.data import PHONEMES, collate_clips, ctc_prefix_beam_search
from VisualPhoneme.lrs3_data import Lrs3Clips
from VisualPhoneme.train import upsample_ctc_logits
from VisualPhoneme.train_phone_head import load_phone_export
from VisualPhoneme.thermal import ThermalGuard


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifests',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-minutes',type=float,default=20)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(filename)s:%(lineno)d %(message)s',handlers=[logging.FileHandler(args.output/'run.log'),logging.StreamHandler()])
    torch.set_num_threads(2)
    sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
    plan=dict(checkpoint_sha256=sha(args.checkpoint),manifests={s:sha(args.manifests/f'{s}.jsonl') for s in ('train','validation','evaluation')},batch_size=8,crop='checkpoint crop with landmarks disabled',blank_bias=-1,beam=64,top_k=12,top_n=5,lm_weight=0)
    path=args.output/'manifest.json'
    if path.exists() and json.loads(path.read_text())!=plan:raise ValueError('resume mismatch')
    path.write_text(json.dumps(plan,indent=2)+'\n')
    stopping=False
    def stop(*_):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    start=time.monotonic()
    def paused():return stopping or Path('PAUSE').exists() or (args.output/'PAUSE').exists() or time.monotonic()-start>args.max_minutes*60
    with ThermalGuard(args.output,gpu_required=True):
        checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
        model=load_phone_export(checkpoint).cuda().eval()
        data=Lrs3Clips(root=Path('datasets/lrs3'),split='train',size=checkpoint['image_size'],crop=checkpoint['crop'],limit=1,horizontal_flip=False)
        for split in ('train','validation','evaluation'):
            rows=[json.loads(l) for l in (args.manifests/f'{split}.jsonl').read_text().splitlines()]
            data.rows=[dict(r,landmarks=None) for r in rows]
            directory=args.output/split;directory.mkdir(exist_ok=True)
            logits_path=directory/'logits.pt'
            cached=torch.load(logits_path,weights_only=True) if logits_path.exists() else []
            if [r['clip_id'] for r in cached]!=[r['clip_id'] for r in rows[:len(cached)]]:raise ValueError('cached IDs mismatch')
            with torch.inference_mode():
                for offset in range(len(cached),len(rows),8):
                    if paused():return
                    video,targets,lengths,target_lengths,ids=collate_clips([data[i] for i in range(offset,min(offset+8,len(rows)))])
                    logits,out_lengths=upsample_ctc_logits(model(video.cuda()),lengths,int(checkpoint.get('ctc_upsample_factor',1)))
                    assert logits.shape[1] == len(ids)
                    for j,clip_id in enumerate(ids):
                        emissions=logits[:int(out_lengths[j]),j].cpu()
                        assert emissions.shape == (int(out_lengths[j]),len(PHONEMES)+1)
                        cached.append(dict(clip_id=clip_id,logits=emissions))
                    tmp=logits_path.with_suffix('.tmp');torch.save(cached,tmp);tmp.replace(logits_path)
                    if len(cached)%80==0:logging.info('%s encoded %d/%d',split,len(cached),len(rows))
            dest=directory/'phone-hypotheses.jsonl'
            saved=[json.loads(l) for l in dest.read_text().splitlines()] if dest.exists() else []
            if [r['clip_id'] for r in saved]!=[r['clip_id'] for r in rows[:len(saved)]]:raise ValueError('partial IDs mismatch')
            with dest.open('a',buffering=1) as f:
                for i in range(len(saved),len(rows)):
                    if paused():return
                    logits=cached[i]['logits'].clone();logits[:,0]-=1
                    candidates=ctc_prefix_beam_search(logits.log_softmax(-1),64,5,12)
                    row=rows[i]
                    record=dict(clip_id=row['clip_id'],reference_phones=row['phonemes'],reference_words=re.findall(r"[a-z]+(?:'[a-z]+)*",row['transcript'].lower()),phone_hypotheses=[dict(rank=j+1,phones=[PHONEMES[k-1] for k in hyp],ctc_log_score=score) for j,(hyp,score) in enumerate(candidates)])
                    f.write(json.dumps(record)+'\n')
                    if (i+1)%100==0:logging.info('%s decoded %d/%d',split,i+1,len(rows))
            (directory/'COMPLETE').write_text('complete\n')
        (args.output/'COMPLETE').write_text('complete\n')
        logging.info('All exports complete')

if __name__=='__main__':main()
