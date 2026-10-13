"""Audit frame timing and overfit a training-only sample with frozen visual features."""
import argparse
import copy
import hashlib
import json
import logging
from pathlib import Path
import re
import signal
import shutil
import subprocess
import time

import torch
from torch import nn
from VisualPhoneme.data import PHONEMES, PHONE_TO_ID, collate_aligned_clips, greedy_decode
from VisualPhoneme.lrs3_data import Lrs3Clips, decode_lrs3_video
from VisualPhoneme.model import LargeVisualPhoneme
from VisualPhoneme.train import edit_totals, upsample_ctc_logits
from VisualPhoneme.train_phone_head import aligned_phone_loss
from VisualPhoneme.thermal import ThermalGuard

LOG=logging.getLogger(__name__)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--max-minutes',type=float,default=15)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if args.steps<1 or args.max_minutes<=0:
        p.error('positive limits required')
    args.output.mkdir(parents=True,exist_ok=args.resume)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(filename)s:%(lineno)d %(message)s',
        handlers=[logging.FileHandler(args.output/'diagnostic.log'),logging.StreamHandler()])
    torch.set_num_threads(4); torch.manual_seed(42)
    stopped=False
    def stop(*unused):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGINT,stop); signal.signal(signal.SIGTERM,stop)
    def paused():
        return stopped or Path('PAUSE').exists() or (args.output/'PAUSE').exists()
    checkpoint_path=Path('checkpoints/individual-phone-head-20261013/phones-step-0100.pt')
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    source_manifest=json.loads(Path('checkpoints/frozen-phone-aligned-20261013/manifest.json').read_text())
    root=Path('datasets/lrs3')
    teacher=root/'audio-teacher-full-forced'
    data=Lrs3Clips(root,'train',checkpoint['image_size'],checkpoint['crop'],
        horizontal_flip=False,teacher_labels_dir=teacher,max_chunk_phones=40,include_frame_targets=True)
    rows={r['clip_id']:r for r in data.rows}
    data.rows=[rows[i] for i in source_manifest['train_ids']]
    if any(r['source_clip_id'] in source_manifest['validation_ids'] for r in data.rows):
        raise ValueError('validation overlap')
    labels={r['source']:r['alignment'] for r in map(json.loads,(teacher/'labels.jsonl').read_text().splitlines())}
    frame_total=sum(len(r['frame_phone_ids']) for r in data.rows)
    labelled=sum(sum(t>=0 for t in r['frame_phone_ids']) for r in data.rows)
    audit=dict(chunks=len(data),frames=frame_total,labelled_frames=labelled,labelled_fraction=labelled/frame_total,
               independent_label_mismatches=0,masked_margin_labels=0,intervals=0,intervals_without_frame_center=0,
               intervals_shorter_than_one_video_frame=0,out_of_bounds_intervals=0,
               checked_chunks=128,checked_files={},video_checks=[])
    durations=[]
    for row in data.rows[:128]:
        path=Path(labels[str((root/row['video']).resolve())])
        document=json.loads(path.read_text())
        audit['checked_files'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        entries=document['tiers']['phones']['entries']
        fps=row['fps']
        centers=[(frame+.5)/fps for frame in range(row['frame_start'],row['frame_end'])]
        expected=[]
        for center in centers:
            overlaps=[e for e in entries if float(e[0])<=center<float(e[1])]
            expected.append(PHONE_TO_ID.get(re.sub(r'\d','',str(overlaps[0][2])).upper(),-100) if overlaps else -100)
        # The loader intentionally excludes neighbouring-word phones from padded
        # chunk margins. Report these separately; never excuse a wrong known label
        # or an unlabelled frame in the chunk interior.
        for index,(a,b) in enumerate(zip(expected,row['frame_phone_ids'],strict=True)):
            if a==b:
                continue
            edge=index<3 or index>=len(expected)-3
            if b==-100 and a>=0 and edge:
                audit['masked_margin_labels']+=1
            else:
                audit['independent_label_mismatches']+=1
        for start,end,phone in entries:
            start,end=float(start),float(end)
            if end<=row['frame_start']/fps or start>=row['frame_end']/fps:
                continue
            if re.sub(r'\d','',str(phone)).upper() not in PHONE_TO_ID:
                continue
            durations.append(end-start); audit['intervals']+=1
            audit['intervals_without_frame_center']+=not any(start<=c<end for c in centers)
            audit['intervals_shorter_than_one_video_frame']+=end-start<1/fps
            audit['out_of_bounds_intervals']+=start<0 or end<=start or end>float(document['end'])+.05
    audit['median_phone_interval_seconds']=sorted(durations)[len(durations)//2]
    samples=[]
    for index,row in enumerate(data.rows[:16]):
        if paused():
            LOG.info('Paused during preparation'); return
        sample=data[index]; samples.append(sample)
        raw,_=data._landmarks(row)
        direct=decode_lrs3_video(root/row['video'],raw,checkpoint['crop'],checkpoint['image_size'],row['frame_start'],row['frame_end'])
        matches=direct.shape==sample[0].shape and torch.equal(direct,sample[0])
        streams=None
        if shutil.which('ffprobe'):
            probe=subprocess.run(['ffprobe','-v','error','-show_entries','stream=codec_type,start_time,duration,r_frame_rate','-of','json',str(root/row['video'])],capture_output=True,text=True,check=True)
            streams=json.loads(probe.stdout)['streams']
        audit['video_checks'].append(dict(clip_id=row['clip_id'],slice_matches_direct_decode=matches,
            frames=len(sample[0]),streams=streams,
            stream_timing_status='checked' if streams is not None else 'ffprobe unavailable; audio/video stream offsets not verified'))
    (args.output/'alignment-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    LOG.info('Audit labelled_fraction=%.3f intervals_without_centers=%d/%d mismatches=%d',
             audit['labelled_fraction'],audit['intervals_without_frame_center'],audit['intervals'],audit['independent_label_mismatches'])
    if audit['independent_label_mismatches'] or not all(r['slice_matches_direct_decode'] for r in audit['video_checks']):
        raise ValueError('timing/slicing audit failed; not training')
    manifest=dict(steps=args.steps,seed=42,learning_rate=.003,sample_ids=[r['clip_id'] for r in data.rows[:16]],
        checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        sample_sha256=hashlib.sha256(json.dumps(data.rows[:16],sort_keys=True).encode()).hexdigest(),
        scope='in-sample diagnostic only, not validation accuracy')
    if args.resume and json.loads((args.output/'manifest.json').read_text())!=manifest:
        raise ValueError('resume manifest mismatch')
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    with ThermalGuard(args.output,gpu_required=True):
        model=LargeVisualPhoneme().cuda().eval(); model.load_state_dict(checkpoint['model'])
        for param in model.parameters(): param.requires_grad_(False)
        video,targets,lengths,target_lengths,frames,ids=collate_aligned_clips(samples)
        with torch.no_grad(): features=model.forward_features(video.cuda()).detach()
        frames=frames.cuda(); targets=targets.cuda()
        valid=frames>=0
        ctc=nn.CTCLoss(blank=0,zero_infinity=False)
        report=dict(manifest=manifest,arms={})
        deadline=time.monotonic()+args.max_minutes*60
        for arm in ('ctc_aligned','frame_only'):
            head=copy.deepcopy(model.classifier)
            for param in head.parameters(): param.requires_grad_(True)
            optimizer=torch.optim.AdamW(head.parameters(),lr=.003,weight_decay=0)
            history=[]; first=0
            state_path=args.output/(arm+'-resume.pt')
            if args.resume and state_path.exists():
                state=torch.load(state_path,map_location='cuda',weights_only=True)
                head.load_state_dict(state['head']); optimizer.load_state_dict(state['optimizer'])
                first=state['step']; history=state['history']
            def measure():
                with torch.no_grad():
                    logits=head(features).transpose(0,1)
                    up,out=upsample_ctc_logits(logits,lengths,2)
                    ce=aligned_phone_loss(logits,frames).item()
                    accuracy=((logits.argmax(-1).T==frames)&valid).sum().item()/valid.sum().item()
                    predictions=greedy_decode(up,out)
                    offset=errors=tokens=0
                    for hypothesis,n in zip(predictions,target_lengths.tolist()):
                        reference=targets[offset:offset+n].tolist(); offset+=n
                        e,c=edit_totals(reference,hypothesis); errors+=e; tokens+=c
                    return dict(frame_ce=ce,labelled_frame_accuracy=accuracy,per=errors/tokens,
                                errors=errors,phones=tokens,labelled_frames=int(valid.sum()))
            if not history: history.append(dict(step=0,**measure()))
            completed=first
            def save():
                torch.save(dict(head=head.state_dict(),optimizer=optimizer.state_dict(),step=completed,history=history),state_path)
            for step in range(first,args.steps):
                if paused() or time.monotonic()>=deadline:
                    save(); LOG.info('Stopped %s at %d; resume saved',arm,completed); return
                logits=head(features).transpose(0,1)
                loss=aligned_phone_loss(logits,frames)
                if arm=='ctc_aligned':
                    up,out=upsample_ctc_logits(logits,lengths,2)
                    loss=.1*loss+ctc(up.log_softmax(-1),targets,out,target_lengths)
                if not torch.isfinite(loss): raise ValueError('nonfinite loss')
                optimizer.zero_grad(set_to_none=True); loss.backward()
                nn.utils.clip_grad_norm_(head.parameters(),5,error_if_nonfinite=True); optimizer.step()
                completed=step+1
                if completed%100==0 or completed==args.steps:
                    metrics=measure(); history.append(dict(step=completed,**metrics)); save()
                    LOG.info('%s step=%d %s',arm,completed,metrics)
            report['arms'][arm]=dict(history=history,final=measure())
        unchanged=all(torch.equal(v,model.state_dict()[k].cpu()) for k,v in checkpoint['model'].items())
        if not unchanged: raise ValueError('frozen model changed')
        report['frozen_model_unchanged']=True
        (args.output/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
        LOG.info('Completed both in-sample diagnostic arms')


if __name__=='__main__':
    main()
