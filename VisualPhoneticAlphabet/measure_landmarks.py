"""Speaker-disjoint GRID landmark ablations using audio-aligned phoneme targets.

Stages: assemble writes frame samples; measure fits GPU ridge and neural probes.
Labels are machine alignments, not human-verified phoneme boundaries.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np
from .core import ARPABET_IPA, POINT_IDS
from .prepare_experiment import ROOT, RUN

LIP_IDS = [p for p in POINT_IDS if p != 152]
BASE_IDS = [61, 291, 13, 14, 0, 17]
RESULTS = Path(__file__).with_name('evaluation')/'grid-landmark-results.json'


def read_intervals(path, tier):
    data=json.loads(path.read_text())
    return data['tiers'][tier]['entries']


def aligned_labels(stamps, intervals, vocabulary):
    """Map timestamps to ARPAbet centers, dropping silence and boundary margins."""
    mapping={p:i for i,p in enumerate(vocabulary)}
    label=np.full(len(stamps),-1,dtype=np.int16)
    for a,b,raw in intervals:
        phone=re.sub(r'\d','',raw).upper()
        if phone in mapping:
            mask=(stamps>=a*1000+20)&(stamps<b*1000-20)
            label[mask]=mapping[phone]
    return label


def assemble():
    rows=[json.loads(line) for line in (ROOT/'clips.jsonl').read_text().splitlines()]
    xs=[]; ys=[]; speakers=[]; clips=[]; times=[]; counters=Counter(); failures=[]; offsets=[]
    phone_vocab=list(ARPABET_IPA)
    phones_path=RUN/'phoneme-alignments.jsonl'
    with phones_path.open('w') as output:
        for clip_index,row in enumerate(rows):
            s='s'+row['speaker_id']; name=Path(row['video']).stem
            alignment=RUN/'aligned'/s/(name+'.json'); points=RUN/'landmarks'/s/(name+'.npz')
            if not alignment.exists() or not points.exists():
                failures.append({'clip_id':row['clip_id'],'reason':'missing alignment or landmarks'});continue
            intervals=read_intervals(alignment,'phones')
            output.write(json.dumps({'clip_id':row['clip_id'],'speaker_id':row['speaker_id'],
                'split':row['split'],'method':'MFA 3.4.2 english_us_arpa; machine-aligned',
                'intervals':[{'start_ms':a*1000,'end_ms':b*1000,'phone':p} for a,b,p in intervals]})+'\n')
            word_intervals=read_intervals(alignment,'words')
            official=[line.split() for line in (ROOT/row['alignment']).read_text().splitlines()]
            official=[(int(a)/25000,int(b)/25000,w) for a,b,w in official if w not in ('sil','sp')]
            aligned_words=[(a,b,w) for a,b,w in word_intervals if w not in ('','sil','sp')]
            # Compare names with the spoken-letter normalization reversed.
            word_match=len(aligned_words)==len(official) and all(w.removeprefix('letter')==ref[2] for (_,_,w),ref in zip(aligned_words,official))
            if not word_match:
                failures.append({'clip_id':row['clip_id'],'reason':'aligned word sequence differs from GRID'});continue
            error=max(abs(t-r) for (a,b,_),(c,d,_) in zip(aligned_words,official) for t,r in [(a,c),(b,d)])
            offsets.append(error)
            # Predeclared gross synchronization filter, not selected from model accuracy.
            if error>.25:
                failures.append({'clip_id':row['clip_id'],'reason':'word boundary disagreement >250ms','max_error_s':error});continue
            data=np.load(points)
            coords=data['coordinates'][:,[list(data['point_ids']).index(p) for p in LIP_IDS],:]
            stamps=data['time_ms']; counters['frames_total']+=len(stamps)
            counters['frames_observed']+=int(np.isfinite(coords).all(axis=(1,2)).sum())
            label=aligned_labels(stamps,intervals,phone_vocab)
            counters['labeled_centers']+=int((label>=0).sum())
            # 3-frame window: one video frame of look-ahead, no imputation across gaps.
            for i in range(1,len(stamps)-1):
                if label[i]<0:continue
                window=coords[i-1:i+2]
                if not np.isfinite(window).all() or np.max(np.diff(stamps[i-1:i+2]))>60:continue
                xs.append(window.transpose(1,0,2).reshape(-1));ys.append(label[i]);speakers.append(int(row['speaker_id']));clips.append(clip_index);times.append(stamps[i])
            counters['clips_included']+=1
    if not xs:raise ValueError('No valid aligned samples')
    np.savez_compressed(RUN/'samples.npz',x=np.asarray(xs,dtype=np.float32),y=np.array(ys),
        speakers=np.array(speakers),clips=np.array(clips),time_ms=np.array(times),
        point_ids=np.array(LIP_IDS),phone_vocab=np.array(phone_vocab))
    report={'counts':dict(counters),'samples':len(xs),'excluded_clips':failures,
        'word_boundary_max_error_percentiles_ms':{str(q):float(np.percentile(offsets,q)*1000) for q in [50,90,95,99]},
        'phoneme_margin_ms':20,'word_boundary_limit_ms':250,'temporal_window':'previous/current/next video frame',
        'class_counts':{p:int(sum(y==i for y in ys)) for i,p in enumerate(phone_vocab)}}
    (RUN/'sample-quality.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'samples':len(xs),'counts':dict(counters),'excluded_clips':len(failures)}),flush=True)


def measure():
    import torch
    from torch import nn
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for this experiment; run outside the device-restricted sandbox')
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    device='cuda'; start=time.time()
    data=np.load(RUN/'samples.npz'); x=data['x'];y=data['y'];speaker=data['speakers'];clip=data['clips'];vocab=data['phone_vocab'].tolist()
    masks={'train':speaker<=8,'validation':(speaker>=9)&(speaker<=10),'test':speaker>=11}
    # Feature normalization uses training speakers only.
    mean=x[masks['train']].mean(0);sd=x[masks['train']].std(0);sd=np.maximum(sd,1e-4)
    x=(x-mean)/sd
    train_classes=np.unique(y[masks['train']]);valid_classes=train_classes.tolist()
    counts=np.bincount(y[masks['train']],minlength=len(vocab))
    X={k:torch.tensor(x[m],device=device) for k,m in masks.items()}
    Y={k:torch.tensor(y[m],device=device,dtype=torch.long) for k,m in masks.items()}
    def stats(pred,labels):
        conf=torch.bincount(labels*len(vocab)+pred,minlength=len(vocab)**2).reshape(len(vocab),len(vocab)).cpu().numpy()
        support=conf.sum(1);recall=np.divide(conf.diagonal(),support,out=np.zeros(len(vocab),float),where=support>0)
        included=(support>0)&(counts>0)
        return {'accuracy':float(conf.trace()/conf.sum()),'balanced_accuracy':float(recall[included].mean()),
                'per_phone_recall':{p:float(recall[i]) for i,p in enumerate(vocab) if support[i]},
                'support':{p:int(support[i]) for i,p in enumerate(vocab) if support[i]},'confusion':conf.tolist()}
    def columns(ids):return [6*LIP_IDS.index(p)+j for p in ids for j in range(6)]
    # Class-balanced least-squares probe. Cached sufficient statistics permit true retraining ablations.
    XT=torch.cat([X['train'].double(),torch.ones((len(X['train']),1),device=device,dtype=torch.float64)],1)
    weights=torch.tensor(np.where(counts>0,1/np.maximum(counts,1),0),device=device,dtype=torch.float64)[Y['train']]
    weights/=weights.sum()
    target=nn.functional.one_hot(Y['train'],len(vocab)).double()
    gram=XT.T@(XT*weights[:,None]);cross=XT.T@(target*weights[:,None]);del XT,target
    def ridge(ids,alpha):
        inds=columns(ids)+[x.shape[1]]
        g=gram[inds][:,inds];penalty=torch.eye(len(inds),device=device,dtype=torch.float64)*alpha;penalty[-1,-1]=0
        return torch.linalg.solve(g+penalty,cross[inds]).float()
    def predict_ridge(ids,w,split):
        z=X[split][:,columns(ids)];scores=z@w[:-1]+w[-1];scores[:,counts==0]=-torch.inf
        return scores.argmax(1)
    def tune(ids):
        candidates=[]
        for alpha in [1e-4,1e-3,1e-2,1e-1]:
            w=ridge(ids,alpha);score=stats(predict_ridge(ids,w,'validation'),Y['validation'])['balanced_accuracy'];candidates.append((score,alpha,w))
        score,alpha,w=max(candidates,key=lambda t:t[0]);return score,alpha,w
    full_score,full_alpha,full_w=tune(LIP_IDS)
    ranking=[]
    for point in LIP_IDS:
        ids=[p for p in LIP_IDS if p!=point]
        score,alpha,_=tune(ids)
        ranking.append({'point':point,'validation_drop':full_score-score,'without_score':score,'alpha':alpha})
    ranking.sort(key=lambda r:(-r['validation_drop'],r['point']))
    ordered=[r['point'] for r in ranking]
    candidates={f'ranked_{k}':ordered[:k] for k in [6,10,16,24,32]}
    candidates['original_6']=BASE_IDS;candidates['full_40']=LIP_IDS
    ridge_results={};ridge_models={}
    for name,ids in candidates.items():
        score,alpha,w=tune(ids);ridge_models[name]=w
        ridge_results[name]={'points':ids,'alpha':alpha,'validation_balanced_accuracy':score}
        print(f'Ridge {name}: validation balanced accuracy {score:.4f}',flush=True)
    acceptable=[name for name in candidates if name.startswith('ranked_') and ridge_results[name]['validation_balanced_accuracy']>=full_score-.01]
    selected=min(acceptable,key=lambda name:len(candidates[name])) if acceptable else 'full_40'
    # Freeze point selection before any test predictions are computed.
    selection={'method':'remove-one-point retrained class-balanced ridge; validation-only subset curve',
        'tolerance_absolute':.01,'selected':selected,'point_ranking':ranking,'ridge_validation':ridge_results}
    (RUN/'selection-frozen.json').write_text(json.dumps(selection,indent=2)+'\n')
    def fit_neural(ids,seed):
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        cols=columns(ids);train=X['train'][:,cols];val=X['validation'][:,cols]
        model=nn.Sequential(nn.Linear(len(cols),128),nn.ReLU(),nn.Dropout(.15),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,len(vocab))).to(device)
        opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
        # Inverse class frequency, normalized; balanced accuracy is the selection criterion.
        class_weights=torch.tensor(np.where(counts>0,counts.sum()/np.maximum(counts,1),0),device=device,dtype=torch.float32)
        class_weights/=class_weights[class_weights>0].mean()
        best=-1;state=None;best_epoch=0;stale=0
        for epoch in range(1,41):
            model.train();order=torch.randperm(len(train),device=device)
            for batch in order.split(4096):
                opt.zero_grad(set_to_none=True);loss=nn.functional.cross_entropy(model(train[batch]),Y['train'][batch],weight=class_weights);loss.backward();opt.step()
            model.eval()
            with torch.no_grad():
                logits=torch.cat([model(z) for z in val.split(8192)]);logits[:,counts==0]=-torch.inf
                score=stats(logits.argmax(1),Y['validation'])['balanced_accuracy']
            if score>best+1e-5:
                best=score;best_epoch=epoch;state={k:v.detach().clone() for k,v in model.state_dict().items()};stale=0
            else:stale+=1
            if stale>=6:break
        model.load_state_dict(state);model.eval()
        return model,best,best_epoch
    neural={};predictions={}
    # A nonlinear cross-check of the ridge-selected subset; no test-driven changes.
    for name in dict.fromkeys(['original_6',selected,'full_40']):
        ids=candidates[name];neural[name]=[]
        for seed in [17,29,43]:
            model,val,epoch=fit_neural(ids,seed)
            with torch.no_grad():
                z=X['test'][:,columns(ids)];logits=torch.cat([model(b) for b in z.split(8192)]);logits[:,counts==0]=-torch.inf;pred=logits.argmax(1)
            test=stats(pred,Y['test']);by_speaker={}
            for s in [11,12]:
                m=torch.tensor(speaker[masks['test']]==s,device=device);by_speaker[str(s)]=stats(pred[m],Y['test'][m])
            neural[name].append({'seed':seed,'best_epoch':epoch,'validation_balanced_accuracy':val,'test':test,'test_by_speaker':by_speaker})
            predictions[f'{name}_seed{seed}']=pred.cpu().numpy()
            checkpoint=RUN/'models'/f'{name}_seed{seed}.pt';checkpoint.parent.mkdir(exist_ok=True)
            torch.save({'state_dict':model.state_dict(),'points':ids,'normalization_mean':mean.tolist(),'normalization_sd':sd.tolist(),'vocabulary':vocab},checkpoint)
            print(f'MLP {name} seed {seed}: validation {val:.4f}; test {test["balanced_accuracy"]:.4f}',flush=True)
    for name,ids in candidates.items():
        pred=predict_ridge(ids,ridge_models[name],'test');ridge_results[name]['test']=stats(pred,Y['test'])
    np.savez_compressed(RUN/'test-predictions.npz',labels=y[masks['test']],speakers=speaker[masks['test']],clips=clip[masks['test']],**predictions)
    # Paired, clip-level bootstrap, stratified by each of the two test speakers.
    # This interval quantifies within-speaker clip variation, not new-speaker uncertainty.
    test_y=y[masks['test']];test_clip=clip[masks['test']];test_speaker=speaker[masks['test']]
    unique_clips=np.unique(test_clip);loc={c:i for i,c in enumerate(unique_clips)}
    support=np.zeros((len(unique_clips),len(vocab)))
    clip_speaker=np.zeros(len(unique_clips),int)
    for c in unique_clips:
        m=test_clip==c;support[loc[c]]=np.bincount(test_y[m],minlength=len(vocab));clip_speaker[loc[c]]=test_speaker[m][0]
    def correctness(name):
        result=np.zeros_like(support)
        for seed in [17,29,43]:
            pred=predictions[f'{name}_seed{seed}']
            for c in unique_clips:
                m=(test_clip==c)&(pred==test_y);result[loc[c]]+=np.bincount(test_y[m],minlength=len(vocab))/3
        return result
    base_correct=correctness('full_40');bootstrap={};rng=np.random.default_rng(20260910)
    groups=[np.where(clip_speaker==s)[0] for s in [11,12]]
    for name in dict.fromkeys(['original_6',selected]):
        difference=correctness(name)-base_correct;values=[]
        for _ in range(1000):
            draws=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups]);den=support[draws].sum(0);ok=(den>0)&(counts>0)
            values.append(float((difference[draws].sum(0)[ok]/den[ok]).mean()))
        bootstrap[name]={'difference_vs_full_mean':float(np.mean(values)),'within_test_speakers_clip_bootstrap_95':np.percentile(values,[2.5,97.5]).tolist()}
    report={'scope':'12-speaker GRID pilot; audio-generated phoneme labels, not manually verified ground truth',
        'splits':{'train':list(range(1,9)),'validation':[9,10],'test':[11,12]},
        'counts':{k:int(m.sum()) for k,m in masks.items()},'vocabulary':vocab,
        'gpu':torch.cuda.get_device_name(),'torch_version':torch.__version__,'cuda_version':torch.version.cuda,
        'elapsed_seconds':time.time()-start,'selection':selection,'ridge':ridge_results,'neural':neural,'bootstrap':bootstrap,
        'protocol':json.loads((RUN/'protocol.json').read_text()),
        'quality':json.loads((RUN/'sample-quality.json').read_text()),
        'limits':['Frame classification, not end-to-end phoneme error rate or transcript WER.',
                  'Three-frame window includes one frame of look-ahead.',
                  '40 ms boundary exclusion underrepresents short phones.',
                  'Only two validation and two test speakers; not a population-level necessity claim.',
                  'GRID scripted vocabulary and 360x288 video; fine contours can be noisy.',
                  'Point ranking is conditional on a ridge model and correlated landmarks; nonlinear subset check is reported separately.']}
    RESULTS.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f'Results written to {RESULTS}',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['assemble','measure']);args=p.parse_args()
    if args.stage=='assemble':assemble()
    else:measure()

if __name__=='__main__':main()
