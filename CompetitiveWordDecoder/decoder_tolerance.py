"""Matched decoder tolerance curves using training-derived phone edit patterns."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import logging
from pathlib import Path
import random
import signal

from CompetitiveWordDecoder.noisy_teacher import load_nbest
from CompetitiveWordDecoder.evaluate_vallr_qwen import vallr_prompt
from CompetitiveWordDecoder.train_vallr_clean import MODEL, REVISION, edit_distance
from PhonemeDecoder.evaluate_qwen_direct import words
from VisualPhoneme.thermal import ThermalGuard

LOG=logging.getLogger(__name__)


def edits(reference, hypothesis):
    """Deterministic minimum-edit alignment; ties prefer diagonal, then deletion."""
    n,m=len(reference),len(hypothesis)
    d=[[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): d[i][0]=i
    for j in range(m+1): d[0][j]=j
    for i in range(1,n+1):
        for j in range(1,m+1):
            d[i][j]=min(d[i-1][j]+1,d[i][j-1]+1,d[i-1][j-1]+(reference[i-1]!=hypothesis[j-1]))
    result=[]; i,j=n,m
    while i or j:
        if i and j and d[i][j]==d[i-1][j-1]+(reference[i-1]!=hypothesis[j-1]):
            if reference[i-1]!=hypothesis[j-1]: result.append(('sub',reference[i-1],hypothesis[j-1]))
            i-=1; j-=1
        elif i and d[i][j]==d[i-1][j]+1:
            result.append(('del',reference[i-1],None)); i-=1
        else:
            result.append(('ins',None,hypothesis[j-1])); j-=1
    return result


def draw(counter,rng):
    return rng.choices(list(counter),weights=list(counter.values()),k=1)[0]


def corrupt(phones,rate,rng,operations,substitutions,insertions):
    selected=set(rng.sample(range(len(phones)),round(len(phones)*rate)))
    output=[]
    global_sub=Counter()
    for counts in substitutions.values(): global_sub.update(counts)
    for i,p in enumerate(phones):
        if i not in selected:
            output.append(p); continue
        op=draw(operations,rng)
        if op=='sub':
            counts=substitutions.get(p) or Counter({q:c for q,c in global_sub.items() if q!=p})
            output.append(draw(counts,rng))
        elif op=='ins': output.extend((p,draw(insertions,rng)))
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=False)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(filename)s:%(lineno)d %(message)s',
                        handlers=[logging.FileHandler(args.output/'run.log'),logging.StreamHandler()])
    stopping=False
    def stop(*unused):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
    def paused(): return stopping or Path('PAUSE').exists() or (args.output/'PAUSE').exists()
    train_path=Path('datasets/decoder-training/lrs3-full-forced-direct-train-5k/phone-hypotheses.jsonl')
    eval_path=Path('checkpoints/context-word-eval-20261014/phone-hypotheses.jsonl')
    train=load_nbest(train_path); evaluation=load_nbest(eval_path)
    assert not {r['clip_id'] for r in train}&{r['clip_id'] for r in evaluation}
    allowed={json.loads(line)['clip_id'] for line in Path('datasets/lrs3/manifests/train.jsonl').read_text().splitlines()}
    assert all(r['clip_id'] in allowed for r in train)
    operations=Counter(); substitutions=defaultdict(Counter); insertions=Counter()
    for row in train:
        for op,a,b in edits(row['reference_phones'],row['phone_hypotheses'][0]['phones']):
            operations[op]+=1
            if op=='sub': substitutions[a][b]+=1
            if op=='ins': insertions[b]+=1
    selected=random.Random(20261026).sample(evaluation,100)
    conditions={}
    for rate in (0,.2,.4,.6,.8):
        conditions[f'noise{round(rate*100)}']=[corrupt(r['reference_phones'],rate,
            random.Random(f"20261026:{r['clip_id']}:{rate}"),operations,substitutions,insertions) for r in selected]
    conditions['actual']=[r['phone_hypotheses'][0]['phones'] for r in selected]
    record=dict(seed=20261026,selected_ids=[r['clip_id'] for r in selected],conditions=conditions,
                operations=operations,substitutions=substitutions,insertions=insertions,
                source_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (train_path,eval_path)},
                caveat='training-derived independent edit model; not full visual-error simulation; one seed and 100 development clips')
    (args.output/'inputs.json').write_text(json.dumps(record,indent=2)+'\n')
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,set_seed
    from peft import PeftModel
    torch.set_num_threads(4)
    reports={}
    with ThermalGuard(args.output,gpu_required=True):
        for name,adapter in [('clean','checkpoints/vallr-eos-20261010/adapter'),
                             ('adapted','checkpoints/vallr-actual-errors-20261014/adapter')]:
            if paused(): return
            set_seed(42)
            tokenizer=AutoTokenizer.from_pretrained(adapter,local_files_only=True)
            tokenizer.padding_side='left'; tokenizer.pad_token=tokenizer.eos_token
            model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,cache_dir='.cache/huggingface/hub',
                local_files_only=True,dtype=torch.float16,attn_implementation='sdpa')
            model.resize_token_embeddings(len(tokenizer))
            model=PeftModel.from_pretrained(model,adapter,local_files_only=True).cuda().eval()
            model.config.use_cache=True
            reports[name]={'adapter_sha256':hashlib.sha256((Path(adapter)/'adapter_model.safetensors').read_bytes()).hexdigest(),'conditions':{}}
            for condition,sequences in conditions.items():
                details=[]; errors=tokens=phone_errors=phone_tokens=capped=0
                with torch.inference_mode():
                    for start in range(0,len(selected),16):
                        if paused():
                            LOG.info('Paused; partial condition results preserved'); return
                        prompts=[vallr_prompt(s) for s in sequences[start:start+16]]
                        if any(len(tokenizer.encode(s,add_special_tokens=False))>512 for s in prompts):
                            raise ValueError('prompt exceeds context')
                        inputs=tokenizer(prompts,add_special_tokens=False,padding=True,return_tensors='pt').to('cuda')
                        output=model.generate(**inputs,do_sample=False,max_new_tokens=256,
                            pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
                        for k,ids in enumerate(output[:,inputs['input_ids'].shape[1]:].tolist()):
                            row=selected[start+k]; phones=sequences[start+k]
                            ended=tokenizer.eos_token_id in ids
                            if ended: ids=ids[:ids.index(tokenizer.eos_token_id)]
                            capped+=not ended and len(ids)>=256
                            text=tokenizer.decode(ids,skip_special_tokens=True)
                            reference=words(' '.join(row['reference_words']))
                            e=edit_distance(reference,words(text)); errors+=e; tokens+=len(reference)
                            phone_errors+=edit_distance(row['reference_phones'],phones); phone_tokens+=len(row['reference_phones'])
                            details.append(dict(clip_id=row['clip_id'],prediction=text,word_errors=e))
                result=dict(wer=errors/tokens,word_errors=errors,words=tokens,realized_per=phone_errors/phone_tokens,
                            capped=capped,details=details)
                reports[name]['conditions'][condition]=result
                (args.output/f'{name}-{condition}.json').write_text(json.dumps(result,indent=2)+'\n')
                LOG.info('%s %s PER=%.2f%% WER=%.2f%%',name,condition,result['realized_per']*100,result['wer']*100)
            del model
            torch.cuda.empty_cache()
    (args.output/'metrics.json').write_text(json.dumps(reports,indent=2)+'\n')
    LOG.info('Completed both decoders on all conditions')


if __name__=='__main__': main()
