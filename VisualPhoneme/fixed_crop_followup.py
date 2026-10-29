"""Matched fixed-crop continuation with source-video-disjoint calibration."""
import argparse
import copy
import json
import logging
import math
from pathlib import Path
import random
import re
import signal
import time

import torch
from torch import nn
from VisualPhoneme.audit_phone_system import metrics, export_greedy
from VisualPhoneme.data import collate_clips
from VisualPhoneme.lrs3_data import Lrs3Clips
from VisualPhoneme.test_blank_strategies import evaluate, sha, talk
from VisualPhoneme.train import upsample_ctc_logits
from VisualPhoneme.train_phone_head import load_phone_export
from VisualPhoneme.thermal import ThermalGuard

LOG=logging.getLogger(__name__)
SOURCE=Path('checkpoints/blank-strategies-20261028')
START=SOURCE/'long_fixed/phones-step-0300.pt'
ORIGINAL=Path('checkpoints/fullclip-phone-context-20261013/manifest.json')
ROOT=Path('datasets/lrs3')
SEED=20261029


def prepare(output,paused):
    data=torch.load(SOURCE/'features.pt',map_location='cpu',weights_only=True)
    checkpoint=torch.load(START,map_location='cpu',weights_only=True)
    prior=json.loads((SOURCE/'manifest.json').read_text())
    kwargs=dict(root=ROOT,size=checkpoint['image_size'],crop=checkpoint['crop'],horizontal_flip=False)
    train=Lrs3Clips(split='train',**kwargs)
    dev=Lrs3Clips(split='validation',**kwargs)
    by_id={r['clip_id']:r for r in dev.rows}
    dev.rows=[by_id[r['clip_id']] for r in data['refs']]
    original=json.loads(ORIGINAL.read_text())
    excluded={i.split('/')[-2] for i in original['train_ids']+prior['long_ids']}
    excluded.update(talk(r) for r in dev.rows)
    eligible=[r for r in train.rows if r.get('landmarks') and r['frames']>=60
              and r.get('pretrain_chunk_index') is None and talk(r) not in excluded]
    random.Random(SEED).shuffle(eligible)
    selected=[]; used=set()
    for row in eligible:
        if talk(row) not in used:
            selected.append(row);used.add(talk(row))
        if len(selected)==128:break
    if len(selected)!=128:raise ValueError(f'Only {len(selected)} source-disjoint calibration videos available')
    calibration=copy.copy(train);calibration.rows=selected
    calibration_refs=[dict(clip_id=r['clip_id'],reference_phones=r['phonemes'],
        reference_words=re.findall(r"[a-z]+(?:'[a-z]+)*",r['transcript'].lower())) for r in selected]
    plan=dict(start_sha256=sha(START),source_manifest_sha256=sha(SOURCE/'manifest.json'),
        source_features_sha256=sha(SOURCE/'features.pt'),seed=SEED,steps=600,batch_size=8,
        calibration_ids=[r['clip_id'] for r in selected],calibration_videos=sorted(used),
        training_ids=prior['long_ids'],development_ids=prior['development_ids'],
        arms={'constant':dict(lr=3e-5),'low_cosine':dict(lr=1e-5,minimum_lr=1e-6)},
        optimizer='fresh AdamW in both arms, weight decay .01',
        calibration_median_frames=sorted(r['frames'] for r in selected)[64],
        caveat='Calibration excludes source videos of both phone-head training subsets and development; pretrained encoder exposure remains possible.')
    (output/'manifest.json').write_text(json.dumps(plan,indent=2)+'\n')
    model=load_phone_export(checkpoint).cuda().eval()
    caches={}
    with torch.inference_mode():
        for name,dataset in [('calibration',calibration),('development',dev)]:
            dataset=copy.copy(dataset);dataset.rows=[dict(r,landmarks=None) for r in dataset.rows]
            batches=[]
            for start in range(0,len(dataset),8):
                if paused():raise InterruptedError('Paused during preparation')
                video,targets,lengths,target_lengths,ids=collate_clips([dataset[i] for i in range(start,min(start+8,len(dataset)))])
                batches.append(dict(features=model.forward_features(video.cuda()).cpu(),targets=targets,
                    lengths=lengths,target_lengths=target_lengths,ids=list(ids)))
            caches[name]=batches
            LOG.info('Cached %s fixed crops: %d clips',name,len(dataset))
    payload=dict(checkpoint=checkpoint,train=data['caches']['long_fixed'],
        development_landmark=data['caches']['development'],development=caches['development'],
        calibration=caches['calibration'],refs=data['refs'],calibration_refs=calibration_refs)
    torch.save(payload,output/'features.pt')
    (output/'PREPARED').write_text('complete\n')


def ctc_loss(head,batches):
    total=0.;count=0
    with torch.inference_mode():
        for batch in batches:
            logits,lengths=upsample_ctc_logits(head(batch['features'].cuda()).transpose(0,1),batch['lengths'],2)
            loss=nn.functional.ctc_loss(logits.log_softmax(-1),batch['targets'].cuda(),lengths,batch['target_lengths'])
            total+=float(loss)*len(batch['ids']);count+=len(batch['ids'])
    return total/count


def run_arm(args,data,paused):
    path=args.output/args.arm;path.mkdir(exist_ok=args.resume)
    head=copy.deepcopy(load_phone_export(data['checkpoint']).phone_head).cuda()
    optimizer=torch.optim.AdamW(head.parameters(),lr=3e-5 if args.arm=='constant' else 1e-5,weight_decay=.01)
    first=0;history=[];best_error=None;best_step=0
    if args.resume:
        state=torch.load(path/'resume.pt',map_location='cpu',weights_only=True)
        if state['manifest_sha256']!=sha(args.output/'manifest.json'):raise ValueError('resume source mismatch')
        head.load_state_dict(state['head']);optimizer.load_state_dict(state['optimizer'])
        first=state['step'];history=state['history'];best_error=state['best_error'];best_step=state['best_step']
    def export(step):
        result=dict(data['checkpoint']);result['model']=dict(result['model'])
        result['model'].update({'phone_head.'+k:v.detach().cpu() for k,v in head.state_dict().items()})
        result.update(fixed_crop_arm=args.arm,followup_steps=step)
        torch.save(result,path/f'phones-step-{step:04d}.pt')
    def measure(step):
        nonlocal best_error,best_step
        result=evaluate(head,data['calibration'],data['calibration_refs'])
        result.update(step=step,train_ctc=ctc_loss(head,data['train']),calibration_ctc=ctc_loss(head,data['calibration']))
        history.append(result)
        if best_error is None or result['per']<best_error:best_error=result['per'];best_step=step
        export(step)
        LOG.info('%s calibration=%s',args.arm,result)
    if not history:measure(0)
    completed=first
    def save():
        temporary=path/'resume.tmp'
        torch.save(dict(head=head.state_dict(),optimizer=optimizer.state_dict(),step=completed,
            history=history,best_error=best_error,best_step=best_step,manifest_sha256=sha(args.output/'manifest.json')),temporary)
        temporary.replace(path/'resume.pt')
    started=time.monotonic();batches=data['train']
    for step in range(first,600):
        if paused() or time.monotonic()-started>args.max_minutes*60:
            save();export(completed);return
        if args.arm=='low_cosine':
            optimizer.param_groups[0]['lr']=1e-6+9e-6*(1+math.cos(math.pi*step/599))/2
        order=list(range(len(batches)));random.Random(SEED+step//len(batches)).shuffle(order)
        batch=batches[order[step%len(batches)]]
        head.train()
        logits,lengths=upsample_ctc_logits(head(batch['features'].cuda()).transpose(0,1),batch['lengths'],2)
        loss=nn.functional.ctc_loss(logits.log_softmax(-1),batch['targets'].cuda(),lengths,batch['target_lengths'])
        if not torch.isfinite(loss):raise ValueError('nonfinite CTC loss')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        nn.utils.clip_grad_norm_(head.parameters(),5,error_if_nonfinite=True);optimizer.step();completed=step+1
        if completed%100==0:measure(completed);save()
    (path/'metrics.json').write_text(json.dumps(dict(steps=completed,best_step=best_step,
        best_calibration_per=best_error,history=history,seconds=time.monotonic()-started),indent=2)+'\n')
    (path/'COMPLETE').write_text('complete\n')


def cache_logits(head,batches,paused):
    rows=[]
    with torch.inference_mode():
        for batch in batches:
            if paused():raise InterruptedError('Paused during final encoding')
            logits,lengths=upsample_ctc_logits(head(batch['features'].cuda()).transpose(0,1),batch['lengths'],2)
            offset=0
            for i,clip_id in enumerate(batch['ids']):
                n=int(batch['target_lengths'][i]);target=batch['targets'][offset:offset+n].tolist();offset+=n
                rows.append(dict(clip_id=clip_id,target=target,logits=logits[:int(lengths[i]),i].cpu()))
    return rows


def finish(args,data,paused):
    reports={arm:json.loads((args.output/arm/'metrics.json').read_text()) for arm in ('constant','low_cosine')}
    chosen=min(reports,key=lambda a:(reports[a]['best_calibration_per'],reports[a]['best_step'],a))
    step=reports[chosen]['best_step'];selected=args.output/chosen/f'phones-step-{step:04d}.pt'
    selection=dict(arm=chosen,step=step,checkpoint=str(selected),checkpoint_sha256=sha(selected),
        selection_metric='calibration greedy PER only',calibration_per=reports[chosen]['best_calibration_per'])
    (args.output/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    result={}
    choices=[('starting',START),('constant_final',args.output/'constant/phones-step-0600.pt'),
             ('low_cosine_final',args.output/'low_cosine/phones-step-0600.pt'),('selected',selected)]
    for name,path in choices:
        if paused():raise InterruptedError('Paused during final evaluation')
        checkpoint=torch.load(path,map_location='cpu',weights_only=True)
        head=load_phone_export(checkpoint).phone_head.cuda().eval()
        result[name]=dict(fixed=evaluate(head,data['development'],data['refs']),
                          landmark=evaluate(head,data['development_landmark'],data['refs']))
        if name=='starting':
            rows=cache_logits(head,data['development'],paused)
            export_greedy(rows,data['refs'],0,args.output/'starting-greedy.jsonl')
        if name=='selected':
            calibration=cache_logits(head,data['calibration'],paused)
            development=cache_logits(head,data['development'],paused)
            torch.save(dict(padded=calibration),args.output/'calibration-logits.pt')
            torch.save(dict(padded=development),args.output/'development-logits.pt')
            grid={str(b):metrics(calibration,b) for b in (0,-.5,-1,-2)}
            bias=min((0,-.5,-1,-2),key=lambda b:grid[str(b)]['per'])
            result['greedy_calibration']=dict(grid=grid,selected_bias=bias,development=metrics(development,bias))
            export_greedy(development,data['refs'],bias,args.output/'selected-greedy.jsonl')
        LOG.info('%s development=%s',name,result[name])
    (args.output/'metrics.json').write_text(json.dumps(dict(selection=selection,results=result),indent=2)+'\n')
    (args.output/'COMPLETE').write_text('complete\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare',action='store_true');group.add_argument('--arm',choices=('constant','low_cosine'))
    group.add_argument('--finish',action='store_true')
    group.add_argument('--cache-final-beams',action='store_true')
    parser.add_argument('--max-minutes',type=float,default=5)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.max_minutes<=0:parser.error('positive time cap required')
    args.output.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(filename)s:%(lineno)d %(message)s',
        handlers=[logging.FileHandler(args.output/'run.log'),logging.StreamHandler()])
    stopped=False
    def stop(*unused):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    def paused():return stopped or Path('PAUSE').exists() or (args.output/'PAUSE').exists() or bool(args.arm and (args.output/args.arm/'PAUSE').exists())
    torch.set_num_threads(4);torch.manual_seed(SEED)
    if paused():return
    with ThermalGuard(args.output,gpu_required=True):
        if args.prepare:
            if (args.output/'PREPARED').exists():raise ValueError('already prepared')
            prepare(args.output,paused)
        else:
            if not (args.output/'PREPARED').exists():raise ValueError('preparation incomplete')
            data=torch.load(args.output/'features.pt',map_location='cpu',weights_only=True)
            if args.cache_final_beams:
                for arm in ('constant','low_cosine'):
                    source=args.output/arm/'phones-step-0600.pt'
                    path=args.output/'beam-caches'/arm;path.mkdir(parents=True,exist_ok=True)
                    checkpoint=torch.load(source,map_location='cpu',weights_only=True)
                    head=load_phone_export(checkpoint).phone_head.cuda().eval()
                    for name in ('calibration','development'):
                        torch.save(dict(padded=cache_logits(head,data[name],paused)),path/f'{name}-logits.pt')
                    (path/'manifest.json').write_text(json.dumps(dict(checkpoint=str(source),checkpoint_sha256=sha(source)),indent=2)+'\n')
            elif args.finish:
                if not all((args.output/a/'COMPLETE').exists() for a in ('constant','low_cosine')):raise ValueError('training incomplete')
                finish(args,data,paused)
            else:run_arm(args,data,paused)


if __name__=='__main__':main()
