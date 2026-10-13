"""Bounded individual-phone head adaptation with auxiliary family CTC."""
import argparse
import hashlib
import json
import logging
import math
from pathlib import Path
import signal
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from VisualPhoneme.data import PHONEMES, collate_clips, collate_aligned_clips, greedy_decode
from VisualPhoneme.lrs3_data import Lrs3Clips
from VisualPhoneme.model import LargeVisualPhoneme
from VisualPhoneme.train import edit_totals, upsample_ctc_logits
from VisualPhoneme.visemes import VISUAL_GROUPS, PHONE_TO_VISUAL_GROUP, VISUAL_PHONE_GROUPS, PHONE_ID_TO_VISUAL_GROUP_ID
from VisualPhoneme.thermal import ThermalGuard
from VisualPhoneme.phone_family_objectives import family_partition, family_log_probs

LOG = logging.getLogger(__name__)


class ContextPhoneHead(nn.Module):
    """Residual nine-frame correction to the original linear phone logits.

    Symmetric context is offline, not causal. Zero output initialization makes
    the starting predictions exactly equal to the existing linear head.
    """
    def __init__(self, linear):
        super().__init__()
        self.linear = linear
        self.context = nn.Sequential(
            nn.LayerNorm(linear.in_features), nn.Linear(linear.in_features, 128), nn.GELU())
        self.temporal = nn.Conv1d(128, 128, kernel_size=9, padding=4)
        self.output = nn.Linear(128, linear.out_features)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features):
        context = self.temporal(self.context(features).transpose(1, 2)).transpose(1, 2)
        return self.linear(features) + self.output(nn.functional.gelu(context))


def load_phone_export(checkpoint):
    """Load either standard phone exports or explicitly tagged context exports."""
    if checkpoint['architecture'] == 'phone-family-context':
        model = PhoneFamilyModel()
        model.phone_head = ContextPhoneHead(model.phone_head)
    elif checkpoint['architecture'] == 'large-image':
        model = LargeVisualPhoneme()
    else:
        raise ValueError('unsupported phone export architecture')
    model.load_state_dict(checkpoint['model'])
    return model


def aligned_phone_loss(logits, frame_labels):
    valid = frame_labels >= 0
    if not valid.any():
        return logits.sum()*0
    return nn.functional.cross_entropy(logits.permute(1,2,0),frame_labels,ignore_index=-100)


class PhoneFamilyModel(LargeVisualPhoneme):
    def __init__(self):
        super().__init__(classes=len(VISUAL_GROUPS)+1)
        self.phone_head = nn.Linear(self.classifier.in_features, len(PHONEMES)+1)

    def initialize_phones(self):
        # Uniform within-family prior preserves total family probability mass.
        mapping = torch.tensor(PHONE_ID_TO_VISUAL_GROUP_ID)
        sizes = [1] + [len(VISUAL_PHONE_GROUPS[PHONE_TO_VISUAL_GROUP[p]]) for p in PHONEMES]
        with torch.no_grad():
            self.phone_head.weight.copy_(self.classifier.weight[mapping])
            self.phone_head.bias.copy_(self.classifier.bias[mapping] - torch.tensor(sizes).log())

    def both(self, video):
        features = self.forward_features(video)
        return self.phone_head(features).transpose(0,1), self.classifier(features).transpose(0,1)

    def forward(self, video):
        return self.phone_head(self.forward_features(video)).transpose(0,1)

    def export_phone_state(self):
        state = {k:v for k,v in self.state_dict().items() if not k.startswith('phone_head.')}
        state['classifier.weight'] = self.phone_head.weight.detach()
        state['classifier.bias'] = self.phone_head.bias.detach()
        return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=Path('datasets/lrs3'))
    parser.add_argument('--steps', type=int, default=400)
    parser.add_argument('--warmup-steps', type=int, default=100)
    parser.add_argument('--train-limit', type=int, default=4000)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-minutes', type=float, default=30)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--teacher-labels', type=Path)
    parser.add_argument('--aligned-weight', type=float, default=0)
    parser.add_argument('--freeze-encoder', action='store_true')
    parser.add_argument('--initial-phone-checkpoint', type=Path)
    parser.add_argument('--eval-every', type=int, default=100)
    parser.add_argument('--context-head', action='store_true')
    parser.add_argument('--lr-schedule', choices=('constant','cosine'), default='constant')
    parser.add_argument('--phone-family', choices=('none','original','fine','coarse'), default='none')
    parser.add_argument('--phone-family-weight', type=float, default=0.1)
    args = parser.parse_args()
    if args.eval_every<1 or not math.isfinite(args.aligned_weight) or args.aligned_weight<0:
        parser.error('invalid evaluation interval or alignment weight')
    if args.aligned_weight and not args.teacher_labels:
        parser.error('alignment loss requires teacher labels')
    if not math.isfinite(args.phone_family_weight) or args.phone_family_weight<0:
        parser.error('invalid phone family weight')
    if min(args.steps,args.train_limit,args.batch_size,args.max_minutes)<=0 or args.warmup_steps<0:
        parser.error('invalid limits')
    args.output.mkdir(parents=True,exist_ok=args.resume)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s',
                        handlers=[logging.FileHandler(args.output/'train.log'),logging.StreamHandler()])
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    stopped = False
    def stop(*unused):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGINT,stop)
    signal.signal(signal.SIGTERM,stop)
    def paused():
        return stopped or Path('PAUSE').exists() or (args.output/'PAUSE').exists()
    if paused():
        LOG.info('Pause requested; not starting')
        return
    torch.manual_seed(42)
    torch.set_num_threads(4)
    checkpoint = torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if checkpoint['architecture']!='large-image' or tuple(checkpoint['phones'])!=VISUAL_GROUPS:
        raise ValueError('expected large-image family checkpoint without REST')
    data_args = dict(root=args.data_root,size=checkpoint['image_size'],crop=checkpoint['crop'],horizontal_flip=False)
    aligned_args = dict(teacher_labels_dir=args.teacher_labels,max_chunk_phones=40,
                        include_frame_targets=True) if args.teacher_labels else {}
    train = Lrs3Clips(split='train',**data_args,**aligned_args)
    if not len(train):
        raise ValueError('no usable training examples')
    generator = torch.Generator().manual_seed(42)
    subset = torch.randperm(len(train),generator=generator)[:args.train_limit].tolist()
    train.rows = [train.rows[i] for i in subset]
    validation = Lrs3Clips(split='validation',**data_args)
    source = [json.loads(s) for s in args.source.read_text().splitlines() if s.strip()]
    by_id = {r['clip_id']:r for r in validation.rows}
    validation.rows = [by_id[r['clip_id']] for r in source]
    if len({r['clip_id'] for r in source})!=len(source):
        raise ValueError('duplicate evaluation IDs')
    if {r.get('source_clip_id',r['clip_id']) for r in train.rows} & set(by_id):
        raise ValueError('split overlap')
    for row, reference in zip(validation.rows,source,strict=True):
        if row['phonemes']!=reference['reference_phones']:
            raise ValueError('reference mismatch')
    manifest = dict(checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        train_ids=[r['clip_id'] for r in train.rows],validation_ids=[r['clip_id'] for r in validation.rows],
        steps=args.steps,warmup_steps=args.warmup_steps,batch_size=args.batch_size,seed=42,
        auxiliary_weight=.3,head_lr=3e-4,encoder_lr=1e-5,ctc_upsample_factor=2,
        labels=list(PHONEMES),scope='transcript-derived phones; full development-validation clips')
    if args.teacher_labels or args.freeze_encoder or args.initial_phone_checkpoint:
        manifest.update(aligned_weight=args.aligned_weight,freeze_encoder=args.freeze_encoder,
            eval_every=args.eval_every,
            teacher_labels=str(args.teacher_labels.resolve()) if args.teacher_labels else None,
            train_rows_sha256=hashlib.sha256(json.dumps(train.rows,sort_keys=True).encode()).hexdigest(),
            initial_phone_sha256=hashlib.sha256(args.initial_phone_checkpoint.read_bytes()).hexdigest()
                if args.initial_phone_checkpoint else None)
    manifest_path=args.output/'manifest.json'
    if args.context_head:
        manifest.update(context_head=dict(kernel=9,width=128,causal=False,zero_initialized_output=True))
    auxiliary_mapping=None
    if args.phone_family != 'none':
        groups,auxiliary_mapping=family_partition(args.phone_family)
        manifest.update(phone_family=args.phone_family,phone_family_groups=groups,
                        phone_family_weight=args.phone_family_weight)
        # JSON canonical form is also the resume comparison form.
        manifest=json.loads(json.dumps(manifest))
    if args.lr_schedule != 'constant':
        manifest.update(lr_schedule=args.lr_schedule,minimum_lr_fraction=0.1)
    if args.resume:
        if json.loads(manifest_path.read_text())!=manifest:
            raise ValueError('resume manifest mismatch')
    else:
        manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    model = PhoneFamilyModel()
    missing,unexpected=model.load_state_dict(checkpoint['model'],strict=False)
    if set(missing)!={'phone_head.weight','phone_head.bias'} or unexpected:
        raise ValueError('incompatible initial state')
    model.initialize_phones()
    if args.initial_phone_checkpoint:
        initial=torch.load(args.initial_phone_checkpoint,map_location='cpu',weights_only=True)
        if tuple(initial['phones'])!=PHONEMES or initial['architecture']!='large-image':
            raise ValueError('initial phone checkpoint incompatible')
        for name,value in checkpoint['model'].items():
            if not name.startswith('classifier.') and not torch.equal(value,initial['model'][name]):
                raise ValueError('initial phone encoder differs from frozen family encoder')
        model.phone_head.load_state_dict({k:initial['model']['classifier.'+k] for k in ('weight','bias')})
    if args.context_head:
        model.phone_head=ContextPhoneHead(model.phone_head)
    frozen_state={k:v.clone() for k,v in model.state_dict().items() if not k.startswith('phone_head.')}
    model.cuda()
    shared = [p for name,p in model.named_parameters() if not name.startswith('phone_head.')]
    optimizer=torch.optim.AdamW([dict(params=shared,lr=1e-5),dict(params=model.phone_head.parameters(),lr=3e-4)],weight_decay=.01)
    mapping=torch.tensor(PHONE_ID_TO_VISUAL_GROUP_ID,device='cuda')
    ctc=nn.CTCLoss(blank=0,zero_infinity=False)
    loader=DataLoader(validation,batch_size=args.batch_size,num_workers=2,collate_fn=collate_clips)
    def evaluate():
        model.eval()
        totals=dict(phone_errors=0,phone_group_errors=0,family_head_errors=0,tokens=0)
        predictions=[]
        with torch.inference_mode():
            for video,packed,lengths,target_lengths,ids in loader:
                if paused():
                    raise InterruptedError('evaluation paused')
                phones,families=model.both(video.cuda())
                phones,out_lengths=upsample_ctc_logits(phones,lengths,2)
                families,_=upsample_ctc_logits(families,lengths,2)
                phone_sequences=greedy_decode(phones,out_lengths)
                group_sequences=greedy_decode(families,out_lengths)
                offset=0
                for i,n in enumerate(target_lengths.tolist()):
                    reference=packed[offset:offset+n].tolist(); offset+=n
                    groups=[PHONE_ID_TO_VISUAL_GROUP_ID[t] for t in reference]
                    guessed=[PHONE_ID_TO_VISUAL_GROUP_ID[t] for t in phone_sequences[i]]
                    totals['phone_errors']+=edit_totals(reference,phone_sequences[i])[0]
                    totals['phone_group_errors']+=edit_totals(groups,guessed)[0]
                    totals['family_head_errors']+=edit_totals(groups,group_sequences[i])[0]
                    totals['tokens']+=n
                    predictions.append(dict(clip_id=ids[i],phones=[PHONEMES[t-1] for t in phone_sequences[i]]))
        return dict(**totals,per=totals['phone_errors']/totals['tokens'],
                    phone_group_per=totals['phone_group_errors']/totals['tokens'],
                    family_head_per=totals['family_head_errors']/totals['tokens'],clips=len(predictions)),predictions
    first=0
    history=[]
    if args.resume:
        state=torch.load(args.output/'resume.pt',map_location='cpu',weights_only=True)
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        first=state['step']; history=state['history']
        torch.set_rng_state(state['rng']); torch.cuda.set_rng_state(state['cuda_rng'])
    def save(step):
        state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,history=history,
                   rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state())
        temporary=args.output/'resume.tmp'
        torch.save(state,temporary); temporary.replace(args.output/'resume.pt')
        exported=dict(checkpoint,model=model.state_dict() if args.context_head else model.export_phone_state(),phones=PHONEMES,target_inventory='phones',
                      ctc_upsample_factor=2,phone_head_step=step)
        if args.context_head:
            exported.update(architecture='phone-family-context',context_head=manifest['context_head'])
        torch.save(exported,args.output/f'phones-step-{step:04d}.pt')
    with ThermalGuard(args.output,gpu_required=True):
        if not history:
            metrics,_=evaluate(); history.append(dict(step=0,**metrics))
            LOG.info('Initial %s',metrics)
        started=time.monotonic(); completed=first
        try:
            for step in range(first,args.steps):
                if paused() or time.monotonic()-started>=args.max_minutes*60:
                    break
                warm=args.freeze_encoder or step<args.warmup_steps
                if args.lr_schedule == 'cosine':
                    factor=0.1+0.9*0.5*(1+math.cos(math.pi*step/max(1,args.steps-1)))
                    for group,base_lr in zip(optimizer.param_groups,(1e-5,3e-4)):
                        group['lr']=base_lr*factor
                for p in shared:
                    p.requires_grad_(not warm)
                model.train()
                if warm:
                    model.eval(); model.phone_head.train()
                # Deterministic reshuffle each traversal; resume depends only on step.
                epoch=(step*args.batch_size)//len(train)
                order=torch.randperm(len(train),generator=torch.Generator().manual_seed(42+epoch)).tolist()
                offset=(step*args.batch_size)%len(train)
                indices=order[offset:offset+args.batch_size]
                frame_labels=None
                if args.teacher_labels:
                    video,packed,lengths,target_lengths,frame_labels,_=collate_aligned_clips([train[i] for i in indices])
                else:
                    video,packed,lengths,target_lengths,_=collate_clips([train[i] for i in indices])
                phones,families=model.both(video.cuda())
                frame_loss=aligned_phone_loss(phones,frame_labels.cuda()) if frame_labels is not None else phones.sum()*0
                phones,out_lengths=upsample_ctc_logits(phones,lengths,2)
                families,_=upsample_ctc_logits(families,lengths,2)
                targets=packed.cuda()
                phone_loss=ctc(phones.log_softmax(-1),targets,out_lengths,target_lengths)
                family_loss=ctc(families.log_softmax(-1),mapping[targets],out_lengths,target_lengths)
                loss=phone_loss+.3*family_loss+args.aligned_weight*frame_loss
                if auxiliary_mapping is not None:
                    auxiliary_targets=torch.tensor(auxiliary_mapping,device=targets.device)[targets]
                    loss=loss+args.phone_family_weight*ctc(
                        family_log_probs(phones,auxiliary_mapping),auxiliary_targets,out_lengths,target_lengths)
                if not torch.isfinite(loss):
                    raise ValueError('nonfinite loss')
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True)
                optimizer.step(); completed=step+1
                if completed%10==0:
                    LOG.info('step=%d/%d warmup=%s phone_loss=%.4f family_loss=%.4f frame_loss=%.4f',completed,args.steps,warm,phone_loss.item(),family_loss.item(),frame_loss.item())
                if completed%args.eval_every==0 or completed==args.steps:
                    save(completed)
                    metrics,_=evaluate(); history.append(dict(step=completed,**metrics)); save(completed)
                    LOG.info('Validation %s',history[-1])
            save(completed)
            if paused():
                LOG.info('Paused at step=%d; resume saved',completed); return
            metrics,predictions=evaluate()
            if args.freeze_encoder:
                unchanged=all(torch.equal(v,model.state_dict()[k].cpu()) for k,v in frozen_state.items())
                if not unchanged:
                    raise ValueError('frozen encoder or family head changed')
                metrics['frozen_encoder_and_family_head_unchanged']=True
            (args.output/'metrics.json').write_text(json.dumps(dict(step=completed,planned_steps=args.steps,
                completed=completed==args.steps,history=history,final=metrics,elapsed_seconds=time.monotonic()-started),indent=2)+'\n')
            (args.output/'predictions.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in predictions))
            LOG.info('Complete step=%d %s',completed,metrics)
        except InterruptedError:
            save(completed); LOG.info('Paused during evaluation; resume saved')


if __name__=='__main__':
    main()
