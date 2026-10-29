"""Bounded matched head-continuation pilots for blank mitigation strategies."""
import argparse
from collections import Counter
import copy
import hashlib
import json
import logging
from pathlib import Path
import random
import re
import signal
import time

import torch
from torch import nn
from CompetitiveWordDecoder.decoder_tolerance import edits
from VisualPhoneme.blank_losses import phone_count_loss, masked_teacher_loss
from VisualPhoneme.data import PHONEMES, PHONE_TO_ID, collate_clips, greedy_decode
from VisualPhoneme.lrs3_data import Lrs3Clips, _sequence_agreement, _words
from VisualPhoneme.train import upsample_ctc_logits
from VisualPhoneme.train_phone_head import load_phone_export
from VisualPhoneme.thermal import ThermalGuard

LOG = logging.getLogger(__name__)
ROOT = Path('datasets/lrs3')
CHECKPOINT = Path('checkpoints/fullclip-phone-context-20261013/phones-step-0750.pt')
REFERENCE = Path('checkpoints/context-word-eval-20261014/phone-hypotheses.jsonl')
TEACHERS = ROOT/'audio-teacher-full-forced/labels.jsonl'
ARMS = ('legacy', 'long_fixed', 'long_landmark', 'count', 'teacher')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def talk(row):
    return Path(row['video']).parent.name


def teacher_targets(row, document, frames):
    """80 ms windows; retain >=20 ms covered and >=70% one-phone mass.

    Clip confidence is checked during selection. These are interval-derived
    soft targets, not claimed acoustic posteriors. Uncovered windows stay masked.
    """
    targets = torch.zeros(frames, len(PHONEMES)+1)
    mask = torch.zeros(frames, dtype=torch.bool)
    entries = document['tiers']['phones']['entries']
    centers = (torch.arange(frames,dtype=torch.float64)+.5)/float(row['fps'])
    for start,end,phone in entries:
        phone = re.sub(r'\d','',str(phone)).upper()
        if phone not in PHONE_TO_ID:
            continue
        overlap = (torch.minimum(centers+.04,torch.tensor(float(end))) -
                   torch.maximum(centers-.04,torch.tensor(float(start)))).clamp_min(0)
        targets[:,PHONE_TO_ID[phone]] += overlap.float()
    coverage = targets.sum(-1)
    mask = (coverage>=.02) & (targets.max(-1).values/coverage.clamp_min(1e-8)>=.7)
    targets = targets/coverage[:,None].clamp_min(1e-8)
    targets[~mask]=0
    return targets, mask


def prepare(output, paused):
    checkpoint = torch.load(CHECKPOINT,map_location='cpu',weights_only=True)
    kwargs = dict(root=ROOT,size=checkpoint['image_size'],crop=checkpoint['crop'],horizontal_flip=False)
    train = Lrs3Clips(split='train',**kwargs)
    dev = Lrs3Clips(split='validation',**kwargs)
    refs = [json.loads(s) for s in REFERENCE.read_text().splitlines()]
    by_id = {r['clip_id']:r for r in dev.rows}
    dev.rows = [by_id[r['clip_id']] for r in refs]
    assert all(r['phonemes']==ref['reference_phones'] for r,ref in zip(dev.rows,refs,strict=True))
    forbidden_talks = {talk(r) for r in dev.rows}
    original = set(json.loads((CHECKPOINT.parent/'manifest.json').read_text())['train_ids'])
    legacy_pool = [r for r in train.rows if r['clip_id'] in original and talk(r) not in forbidden_talks]
    legacy = random.Random(20261028).sample(legacy_pool,512)
    teachers = {r['source']:r for r in map(json.loads,TEACHERS.read_text().splitlines())}
    candidates = []
    rejected = Counter()
    documents = {}
    for row in train.rows:
        if paused(): raise InterruptedError('PAUSE during teacher selection')
        if not row.get('landmarks') or row['frames']<60 or row.get('pretrain_chunk_index') is not None or talk(row) in forbidden_talks:
            continue
        record = teachers.get(str((ROOT/row['video']).resolve()))
        if not record or record.get('status')!='accepted':
            rejected['missing_or_rejected']+=1; continue
        document = json.loads(Path(record['alignment']).read_text())
        quality = record.get('quality') or document.get('quality',{})
        if quality.get('score',0)<.95 or quality.get('forced_alignment_score',0)<.8 or quality.get('teacher_agreement',0)<.95:
            rejected['clip_confidence']+=1; continue
        teacher_words = document.get('provenance',{}).get('teacher_transcripts',[])
        teacher_words = teacher_words[0] if teacher_words else [str(e[2]) for e in document['tiers']['words']['entries']]
        if _sequence_agreement(_words(row['transcript']),teacher_words)<.95:
            rejected['transcript_mismatch']+=1; continue
        entries = document['tiers']['phones']['entries']
        duration = row['frames']/float(row['fps'])
        if any(float(a)<0 or float(b)<=float(a) or float(b)>duration+.05 for a,b,_ in entries):
            rejected['interval_bounds']+=1; continue
        target,mask = teacher_targets(row,document,row['frames'])
        if float(mask.float().mean())<.1:
            rejected['insufficient_supervision']+=1; continue
        candidates.append(row)
        documents[row['clip_id']] = (record['alignment'],target,mask)
    if len(candidates)<512:
        raise ValueError(f'Only {len(candidates)} eligible long/teacher clips; need 512. Rejections={rejected}')
    longer = random.Random(20261028).sample(candidates,512)
    plan = dict(checkpoint_sha256=sha(CHECKPOINT),reference_sha256=sha(REFERENCE),teacher_manifest_sha256=sha(TEACHERS),
        seed=20261028,steps=300,batch_size=8,learning_rate=3e-5,weight_decay=.01,
        count_weight=.1,teacher_weight=.1,legacy_ids=[r['clip_id'] for r in legacy],
        long_ids=[r['clip_id'] for r in longer],development_ids=[r['clip_id'] for r in dev.rows],
        teacher_files={documents[r['clip_id']][0]:sha(Path(documents[r['clip_id']][0])) for r in longer},
        eligible_teacher_clips=len(candidates),rejected=dict(rejected),
        caveat='Single-seed development pilot; encoder may have prior training exposure. Long-vs-legacy changes source composition as well as length.',
        data_summary={name:dict(clips=len(rows),median_frames=sorted(r['frames'] for r in rows)[len(rows)//2],
                               landmarks=sum(bool(r.get('landmarks')) for r in rows)) for name,rows in [('legacy',legacy),('long',longer),('development',dev.rows)]})
    (output/'manifest.json').write_text(json.dumps(plan,indent=2)+'\n')
    model = load_phone_export(checkpoint).cuda().eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    caches = {}
    with torch.inference_mode():
        for name,rows in [('legacy',legacy),('long_fixed',[dict(r,landmarks=None) for r in longer]),('long_landmark',longer),('development',dev.rows)]:
            dataset = copy.copy(train if name!='development' else dev)
            dataset.rows = rows
            batches = []
            for start in range(0,len(rows),8):
                if paused(): raise InterruptedError('PAUSE during feature preparation')
                samples = [dataset[i] for i in range(start,min(start+8,len(rows)))]
                video,targets,lengths,target_lengths,ids = collate_clips(samples)
                features = model.forward_features(video.cuda()).cpu()
                batch = dict(features=features,targets=targets,lengths=lengths,target_lengths=target_lengths,ids=list(ids))
                if name=='long_landmark':
                    soft = torch.zeros(video.shape[1],len(samples),len(PHONEMES)+1)
                    valid = torch.zeros(video.shape[1],len(samples),dtype=torch.bool)
                    for i,clip_id in enumerate(ids):
                        _,target,mask = documents[clip_id]
                        if len(target)!=lengths[i]: raise ValueError('teacher frame count mismatch')
                        soft[:len(target),i]=target; valid[:len(target),i]=mask
                    batch.update(teacher=soft,teacher_mask=valid)
                batches.append(batch)
            caches[name]=batches
            LOG.info('Cached %s clips=%d batches=%d',name,len(rows),len(batches))
    torch.save(dict(caches=caches,checkpoint=checkpoint,refs=refs),output/'features.pt')
    plan['teacher_labelled_frames']=sum(int(b['teacher_mask'].sum()) for b in caches['long_landmark'])
    plan['teacher_total_frames']=sum(int(b['lengths'].sum()) for b in caches['long_landmark'])
    (output/'manifest.json').write_text(json.dumps(plan,indent=2)+'\n')
    (output/'PREPARED').write_text('complete\n')
    LOG.info('Prepared: %s',plan['data_summary'])


def evaluate(head, batches, refs, output=None):
    counts=Counter(); predictions=[]
    head.eval()
    with torch.inference_mode():
        for batch in batches:
            logits,lengths=upsample_ctc_logits(head(batch['features'].cuda()).transpose(0,1),batch['lengths'],2)
            sequences=greedy_decode(logits,lengths)
            offset=0
            for i,(clip_id,hyp) in enumerate(zip(batch['ids'],sequences)):
                n=int(batch['target_lengths'][i]); target=batch['targets'][offset:offset+n].tolist(); offset+=n
                counts.update(op for op,_,_ in edits(target,hyp))
                counts.update(phones=n,predicted=len(hyp),frames=int(lengths[i]),
                              blanks=int((logits[:int(lengths[i]),i].argmax(-1)==0).sum()))
                predictions.append((clip_id,hyp,float(logits[:int(lengths[i]),i].log_softmax(-1).max(-1).values.sum())))
    result=dict(counts,per=sum(counts[k] for k in ('sub','del','ins'))/counts['phones'],blank_fraction=counts['blanks']/counts['frames'])
    if output:
        with (output/'phone-hypotheses.jsonl').open('w') as handle:
            for (clip_id,hyp,score),ref in zip(predictions,refs,strict=True):
                assert clip_id==ref['clip_id']
                row={k:ref[k] for k in ('clip_id','reference_words','reference_phones')}
                row.update(decoding='greedy',score_type='greedy alignment log probability, not sequence marginal',
                    phone_hypotheses=[dict(rank=1,phones=[PHONEMES[i-1] for i in hyp],ctc_log_score=score)])
                handle.write(json.dumps(row)+'\n')
    return result


def train_arm(args, data, paused):
    arm=args.arm; output=args.output/arm
    output.mkdir(exist_ok=args.resume)
    checkpoint=data['checkpoint']
    model=load_phone_export(checkpoint)
    head=copy.deepcopy(model.phone_head).cuda()
    optimizer=torch.optim.AdamW(head.parameters(),lr=3e-5,weight_decay=.01)
    batches=data['caches'][arm if arm in ('legacy','long_fixed') else 'long_landmark']
    development=data['caches']['development']
    first=0; history=[]
    if args.resume and (output/'resume.pt').exists():
        state=torch.load(output/'resume.pt',map_location='cpu',weights_only=True)
        if state['manifest_sha256']!=sha(args.output/'manifest.json'): raise ValueError('resume source mismatch')
        head.load_state_dict(state['head']); optimizer.load_state_dict(state['optimizer'])
        first=state['step']; history=state['history']
    elif args.resume:
        raise ValueError('missing resume checkpoint')
    ctc=nn.CTCLoss(blank=0,zero_infinity=False)
    if not history:
        initial=evaluate(head,development,data['refs'])
        if initial['sub']+initial['del']+initial['ins']!=36860: raise ValueError('cached baseline differs from original')
        history.append(dict(step=0,**initial))
    completed=first
    def save():
        state=dict(head=head.state_dict(),optimizer=optimizer.state_dict(),step=completed,history=history,
                   manifest_sha256=sha(args.output/'manifest.json'))
        temporary=output/'resume.tmp';torch.save(state,temporary);temporary.replace(output/'resume.pt')
        export=dict(checkpoint); export['model']=dict(checkpoint['model'])
        export['model'].update({'phone_head.'+k:v.detach().cpu() for k,v in head.state_dict().items()})
        export.update(blank_strategy=arm,continuation_steps=completed)
        torch.save(export,output/f'phones-step-{completed:04d}.pt')
    started=time.monotonic()
    for step in range(first,300):
        if paused() or time.monotonic()-started>=args.max_minutes*60:
            save(); LOG.info('Stopped %s step=%d; resume saved',arm,completed); return
        order=list(range(len(batches)));random.Random(20261028+step//len(batches)).shuffle(order)
        batch=batches[order[step%len(batches)]]
        head.train()
        original=head(batch['features'].cuda()).transpose(0,1)
        logits,lengths=upsample_ctc_logits(original,batch['lengths'],2)
        base=ctc(logits.log_softmax(-1),batch['targets'].cuda(),lengths,batch['target_lengths'])
        count=phone_count_loss(logits,lengths,batch['target_lengths']) if arm=='count' else base*0
        teacher=masked_teacher_loss(original,batch['teacher'].cuda(),batch['teacher_mask'].cuda()) if arm=='teacher' else base*0
        loss=base+.1*count+.1*teacher
        if not torch.isfinite(loss): raise ValueError('nonfinite loss')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        nn.utils.clip_grad_norm_(head.parameters(),5,error_if_nonfinite=True);optimizer.step();completed=step+1
        if completed%50==0:
            LOG.info('%s step=%d ctc=%.4f count=%.4f teacher=%.4f',arm,completed,float(base.detach()),float(count.detach()),float(teacher.detach()))
        if completed%100==0:
            history.append(dict(step=completed,**evaluate(head,development,data['refs'])));save()
            LOG.info('%s evaluation=%s',arm,history[-1])
    final=evaluate(head,development,data['refs'],output)
    report=dict(arm=arm,completed_steps=completed,history=history,final=final,seconds=time.monotonic()-started,
                manifest_sha256=sha(args.output/'manifest.json'),frozen_encoder=True)
    (output/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    (output/'COMPLETE').write_text('complete\n')
    LOG.info('Complete %s %s',arm,final)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--arm',choices=ARMS)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--max-minutes',type=float,default=5)
    args=parser.parse_args()
    if args.prepare==bool(args.arm) or args.max_minutes<=0: parser.error('choose preparation or one arm; positive cap required')
    args.output.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(filename)s:%(lineno)d %(message)s',
        handlers=[logging.FileHandler(args.output/'run.log'),logging.StreamHandler()])
    stopped=False
    def stop(*unused):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
    def paused():return stopped or Path('PAUSE').exists() or (args.output/'PAUSE').exists() or bool(args.arm and (args.output/args.arm/'PAUSE').exists())
    torch.set_num_threads(4); torch.manual_seed(20261028)
    if paused():return
    with ThermalGuard(args.output,gpu_required=True):
        if args.prepare:
            if (args.output/'PREPARED').exists():raise ValueError('already prepared')
            prepare(args.output,paused)
        else:
            if not (args.output/'PREPARED').exists():raise ValueError('missing completed preparation')
            data=torch.load(args.output/'features.pt',map_location='cpu',weights_only=True)
            train_arm(args,data,paused)


if __name__=='__main__':main()
