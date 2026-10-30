"""Reference-free CTC alignment and timed posterior features for phone prompts."""
import numpy as np
import torch
from torch import nn
from VisualPhoneme.data import PHONEMES
from CompetitiveWordDecoder.evaluate_vallr_qwen import vallr_prompt

FEATURE_DIM=44
PREFIX='<S2S>\n<PHONEMES>\n'


def align_phone_evidence(logits, phones, fps, factor=2):
    """Viterbi align a predicted phone path; never accept transcript targets here.

    40 posterior means, start seconds/10, duration seconds, preceding blank gap
    seconds, normalized entropy. Times are model alignments, not ground truth.
    """
    if logits.ndim!=2 or logits.shape[1]!=40 or len(logits)%factor or fps<=0:
        raise ValueError('invalid emissions/fps')
    if not phones:return torch.zeros(0,FEATURE_DIM)
    ids=[PHONEMES.index(p)+1 for p in phones]
    z=logits.float().clone();z[:,0]-=1
    logp=z.log_softmax(-1).numpy();posterior=np.exp(logp)
    states=np.zeros(2*len(ids)+1,dtype=np.int64);states[1::2]=ids
    n,s=len(logp),len(states)
    previous=np.full(s,-np.inf);previous[0]=logp[0,0];previous[1]=logp[0,ids[0]]
    back=np.zeros((n,s),dtype=np.int64)
    skip=np.zeros(s,dtype=bool);skip[2:]=(states[2:]!=0)&(states[2:]!=states[:-2])
    for t in range(1,n):
        options=np.stack((previous,np.r_[-np.inf,previous[:-1]],np.r_[-np.inf,-np.inf,previous[:-2]]))
        options[2,~skip]=-np.inf
        choice=np.argmax(options,axis=0)
        back[t]=np.arange(s)-choice
        previous=options[choice,np.arange(s)]+logp[t,states]
    state=s-1 if previous[-1]>=previous[-2] else s-2
    if not np.isfinite(previous[state]):raise ValueError('predicted phone path cannot align')
    path=np.empty(n,dtype=np.int64);path[-1]=state
    for t in range(n-1,0,-1):path[t-1]=back[t,path[t]]
    features=[];prior_end=0
    for j in range(len(ids)):
        frames=np.flatnonzero(path==2*j+1)
        if not len(frames):raise ValueError('phone missing in alignment')
        start,end=int(frames[0]),int(frames[-1])+1
        mass=posterior[frames].mean(0)
        entropy=float(-(mass*np.log(np.maximum(mass,1e-30))).sum()/np.log(40))
        features.append(np.r_[mass,start/(factor*fps*10),(end-start)/(factor*fps),(start-prior_end)/(factor*fps),entropy])
        prior_end=end
    return torch.tensor(np.asarray(features),dtype=torch.float32)


def prompt_features(phones, evidence, tokenizer):
    if evidence.shape!=(len(phones),FEATURE_DIM):raise ValueError('evidence/phone count mismatch')
    prompt=vallr_prompt(phones)
    encoded=tokenizer(prompt,add_special_tokens=False,return_offsets_mapping=True)
    ids=encoded['input_ids'];features=torch.zeros(len(ids),FEATURE_DIM)
    intervals=[];start=len(PREFIX)
    for phone in phones:intervals.append((start,start+len(phone)));start+=len(phone)+1
    for i,(left,right) in enumerate(encoded['offset_mapping']):
        overlaps=[j for j,(a,b) in enumerate(intervals) if max(a,left)<min(b,right)]
        if overlaps:features[i]=evidence[overlaps].mean(0)
    return ids,features


class TimedPhoneAdapter(nn.Module):
    """Zero-start bounded residual on phone token embeddings only."""
    def __init__(self,width,hidden=128,scale=.1):
        super().__init__();self.scale=scale
        self.input=nn.Linear(FEATURE_DIM,hidden)
        self.output=nn.Linear(hidden,width,bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self,features):
        mask=(features[...,:40].sum(-1,keepdim=True)>0).to(features.dtype)
        return self.scale*torch.tanh(self.output(torch.nn.functional.gelu(self.input(features))))*mask


def bound_embedding_delta(delta, embeddings, max_ratio):
    """Bound the whole residual vector relative to its original token embedding."""
    if not 0 < max_ratio <= 1 or delta.shape != embeddings.shape:
        raise ValueError('invalid relative residual bound')
    limit=embeddings.float().norm(dim=-1,keepdim=True)*max_ratio
    factor=(limit/delta.float().norm(dim=-1,keepdim=True).clamp_min(1e-12)).clamp(max=1)
    return delta*factor
