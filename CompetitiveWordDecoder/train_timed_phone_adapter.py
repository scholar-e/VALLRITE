"""Freeze Llama and train a zero-start timed-posterior input residual."""
import argparse
import hashlib
import json
import logging
import math
from pathlib import Path
import random
import signal
import time
import torch
from CompetitiveWordDecoder.timed_phone_features import TimedPhoneAdapter,FEATURE_DIM,bound_embedding_delta
from CompetitiveWordDecoder.generated_tokens import generation_record,scoring_tokens
from CompetitiveWordDecoder.evaluate_probability_ranker import summarize
from CompetitiveWordDecoder.train_vallr_clean import MODEL,REVISION
from VisualPhoneme.thermal import ThermalGuard


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data',type=Path,default=Path('datasets/decoder-training/timed-phone-features-20261029'))
    p.add_argument('--adapter',type=Path,default=Path('checkpoints/decoder-current-mixed-20261027/adapter'))
    p.add_argument('--max-minutes',type=float,default=15)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--diagnostic-checkpoint',type=Path,help='validation-only residual-bound diagnostic; no new training')
    p.add_argument('--relative-norm-cap',type=float,default=.05)
    p.add_argument('--legacy-coordinate-bound',action='store_true',help='only for reproducing/resuming the original uncapped pilot')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=a.resume)
    if not 0<a.relative_norm_cap<=1 or (a.legacy_coordinate_bound and a.diagnostic_checkpoint):
        p.error('relative norm cap must be in (0,1]; legacy reproduction cannot be combined with diagnostic mode')
    if a.legacy_coordinate_bound:a.relative_norm_cap=None
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(filename)s:%(lineno)d %(message)s',handlers=[logging.FileHandler(a.output/'run.log'),logging.StreamHandler()]);torch.set_num_threads(2)
    from transformers import AutoModelForCausalLM,AutoTokenizer,set_seed
    from peft import PeftModel
    set_seed(42)
    sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
    data_manifest=json.loads((a.data/'manifest.json').read_text())
    assert sha(a.data/'data.pt')==data_manifest['data_sha256']
    assert sha(a.adapter/'adapter_model.safetensors')==data_manifest['adapter_sha256']
    plan=dict(data_manifest_sha256=sha(a.data/'manifest.json'),adapter_sha256=data_manifest['adapter_sha256'],model=MODEL,revision=REVISION,precision='bfloat16 for frozen model, float32 residual',steps=188,batch_size=4,lr=.001,weight_decay=.01,seed=42,hidden=128,residual_scale=.1,feature_dim=44,validation_steps=[0,94,188],selection='minimum validation probability-ranked word errors; tie baseline then earlier step',max_new_tokens=256)
    if not a.legacy_coordinate_bound:plan['relative_norm_cap']=a.relative_norm_cap
    if a.diagnostic_checkpoint:
        plan.update(diagnostic_checkpoint_sha256=sha(a.diagnostic_checkpoint),relative_norm_cap=a.relative_norm_cap,steps=0,validation_steps=[0,188])
    manifest=a.output/'manifest.json'
    if a.resume and json.loads(manifest.read_text())!=plan:raise ValueError('resume mismatch')
    manifest.write_text(json.dumps(plan,indent=2)+'\n')
    data=torch.load(a.data/'data.pt',map_location='cpu',weights_only=True)
    stopping=False
    def stop(*_):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    def paused():return stopping or Path('PAUSE').exists() or (a.output/'PAUSE').exists()
    with ThermalGuard(a.output,gpu_required=True):
        tokenizer=AutoTokenizer.from_pretrained(a.adapter,local_files_only=True)
        base=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,cache_dir='.cache/huggingface/hub',local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa')
        base.resize_token_embeddings(len(tokenizer));base=PeftModel.from_pretrained(base,a.adapter,local_files_only=True).cuda().eval()
        for parameter in base.parameters():parameter.requires_grad_(False)
        base.config.use_cache=False
        residual=TimedPhoneAdapter(base.config.hidden_size).cuda()
        optimizer=torch.optim.AdamW(residual.parameters(),lr=.001,weight_decay=.01)
        logging.info('Trainable input parameters=%d frozen model trainable=%d',sum(p.numel() for p in residual.parameters()),sum(p.numel() for p in base.parameters() if p.requires_grad))
        embeddings=base.get_input_embeddings()
        def batch_inputs(items,with_target=False,left=False):
            lengths=[len(x['ids'])+(len(x['target']) if with_target else 0) for x in items];width=max(lengths)
            ids=torch.full((len(items),width),tokenizer.pad_token_id,dtype=torch.long,device='cuda')
            mask=torch.zeros_like(ids);labels=torch.full_like(ids,-100)
            features=torch.zeros(len(items),width,FEATURE_DIM,device='cuda')
            for i,(item,length) in enumerate(zip(items,lengths)):
                offset=width-length if left else 0;prompt=len(item['ids'])
                tokens=item['ids']+(item['target'] if with_target else [])
                ids[i,offset:offset+length]=torch.tensor(tokens,device='cuda');mask[i,offset:offset+length]=1
                features[i,offset:offset+prompt]=item['features'].to('cuda')
                if with_target:labels[i,offset+prompt:offset+length]=torch.tensor(item['target'],device='cuda')
            original=embeddings(ids);delta=residual(features)
            if a.relative_norm_cap is not None:delta=bound_embedding_delta(delta,original,a.relative_norm_cap)
            value=original+delta.to(original.dtype)
            return dict(inputs_embeds=value,attention_mask=mask),labels,ids
        def save(step):
            payload=dict(step=step,residual=residual.state_dict(),optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
            tmp=a.output/'resume.tmp';torch.save(payload,tmp);tmp.replace(a.output/'resume.pt')
            torch.save(dict(step=step,residual=residual.state_dict()),a.output/f'adapter-step-{step:04d}.pt')
        def evaluate(split,step):
            output=a.output/f'{split}-step-{step:04d}';output.mkdir(exist_ok=True)
            if (output/'metrics.json').exists():return json.loads((output/'metrics.json').read_text())
            residual.eval();records=data[split];items=[];generated=[[None]*5 for _ in records]
            for i,row in enumerate(records):
                for j,candidate in enumerate(row['candidates']):items.append((i,j,candidate))
            with torch.inference_mode(),(output/'generations.jsonl').open('w') as handle:
                for start in range(0,len(items),16):
                    if paused():return None
                    batch=items[start:start+16];inputs,_,ids=batch_inputs([x[2] for x in batch],left=True)
                    output_ids=base.generate(**inputs,do_sample=False,max_new_tokens=256,eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.pad_token_id,use_cache=True)
                    if step==0 and start==0:
                        # Establish that zero residual and inputs_embeds preserve hard-prompt generation.
                        plain=base.generate(input_ids=ids,attention_mask=inputs['attention_mask'],do_sample=False,max_new_tokens=256,eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.pad_token_id,use_cache=True)[:,ids.shape[1]:]
                        assert torch.equal(output_ids,plain),'zero-input adapter does not reproduce baseline'
                        logging.info('Zero-residual generation identity passed')
                    assert output_ids.shape[1]<=256,'unexpected prompt tokens in inputs_embeds generation'
                    for (i,j,c),tokens in zip(batch,output_ids.tolist(),strict=True):
                        result=dict(generation_record(tokens,tokenizer),phone_path_log_score=c['phone_path_log_score'])
                        generated[i][j]=result;handle.write(json.dumps(dict(clip_id=records[i]['clip_id'],rank=j+1,**result))+'\n')
                    handle.flush()
                    if start%160==0:logging.info('%s step=%d generated=%d/%d',split,step,min(start+16,len(items)),len(items))
                for start in range(0,len(items),4):
                    if paused():return None
                    batch=items[start:start+4];scored=[]
                    for i,j,c in batch:
                        tokens,_=scoring_tokens(generated[i][j],tokenizer,require_exact=True)
                        scored.append(dict(c,target=tokens))
                    inputs,labels,_=batch_inputs(scored,with_target=True)
                    logits=base(**inputs,use_cache=False).logits
                    for k,((i,j,c),item) in enumerate(zip(batch,scored)):
                        n=len(c['ids']);m=len(item['target'])
                        logp=-torch.nn.functional.cross_entropy(logits[k,n-1:n+m-1].float(),labels[k,n:n+m],reduction='sum').item()
                        if not math.isfinite(logp):raise ValueError('nonfinite score')
                        generated[i][j].update(decoder_log_probability=logp,scored_tokens=m)
            report=dict(split=split,step=step,precision='bfloat16',adapter_sha256=plan['adapter_sha256'],data_manifest_sha256=plan['data_manifest_sha256'],scoring_token_modes=['exact_generated_tokens'],capped_generations=sum(c['hit_generation_cap'] for rows in generated for c in rows),**summarize(records,generated))
            (output/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
            logging.info('%s step=%d metrics=%s',split,step,report['metrics']);return report
        if not a.resume:save(0)
        zero=torch.load(a.output/'adapter-step-0000.pt',map_location='cuda',weights_only=True);residual.load_state_dict(zero['residual'])
        baseline=evaluate('validation',0)
        if baseline is None:return
        if a.diagnostic_checkpoint:
            state=torch.load(a.diagnostic_checkpoint,map_location='cuda',weights_only=True)
            if state['step']!=188:raise ValueError('expected final diagnostic checkpoint')
            residual.load_state_dict(state['residual'])
            challenger=evaluate('validation',188)
            if challenger is None:return
            winner=188 if challenger['metrics']['probability_ranked']['word_errors']<baseline['metrics']['probability_ranked']['word_errors'] else 0
            (a.output/'selection.json').write_text(json.dumps(dict(selected_step=winner,diagnostic=True,relative_norm_cap=a.relative_norm_cap),indent=2)+'\n')
            if winner:
                for step,weights in ((0,zero['residual']),(188,state['residual'])):
                    residual.load_state_dict(weights)
                    if evaluate('evaluation',step) is None:return
            (a.output/'COMPLETE').write_text('diagnostic complete\n');logging.info('Diagnostic selected step=%d',winner);return
        first=0
        if a.resume:
            state=torch.load(a.output/'resume.pt',map_location='cuda',weights_only=True);residual.load_state_dict(state['residual']);optimizer.load_state_dict(state['optimizer']);first=state['step'];torch.set_rng_state(state['torch_rng'].cpu());torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda_rng']])
            if first in (94,188) and evaluate('validation',first) is None:return
        order=list(range(len(data['train'])));random.Random(42).shuffle(order);started=time.monotonic()
        for step in range(first,188):
            if paused() or time.monotonic()-started>=a.max_minutes*60:save(step);logging.info('Paused at step=%d',step);return
            residual.train();batch=[]
            for index in order[step*4:min(step*4+4,len(order))]:
                row=data['train'][index];batch.append(dict(row['candidates'][0],target=row['target']))
            if not batch:raise ValueError('training step exceeds frozen sample count')
            inputs,labels,_=batch_inputs(batch,with_target=True)
            optimizer.zero_grad(set_to_none=True)
            loss=base(**inputs,labels=labels,use_cache=False).loss
            if not torch.isfinite(loss):raise ValueError('nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(residual.parameters(),1,error_if_nonfinite=True);optimizer.step()
            if step==0:assert all(p.grad is None for p in base.parameters()),'frozen model got gradients'
            if (step+1)%10==0:logging.info('step=%d/188 loss=%.4f',step+1,loss.item())
            if (step+1)%47==0:save(step+1)
            if step+1 in (94,188):
                report=evaluate('validation',step+1)
                if report is None:return
        reports={step:json.loads((a.output/f'validation-step-{step:04d}/metrics.json').read_text()) for step in (0,94,188)}
        winner=min(reports,key=lambda s:(reports[s]['metrics']['probability_ranked']['word_errors'],s))
        (a.output/'selection.json').write_text(json.dumps(dict(selected_step=winner,validation={s:r['metrics'] for s,r in reports.items()}),indent=2)+'\n')
        logging.info('Selected validation checkpoint=%d',winner)
        # Evaluate a frozen winner only when it improves over the baseline; no extra test search.
        if winner:
            for step in (0,winner):
                state=torch.load(a.output/f'adapter-step-{step:04d}.pt',map_location='cuda',weights_only=True);residual.load_state_dict(state['residual'])
                if evaluate('evaluation',step) is None:return
        (a.output/'COMPLETE').write_text('training and planned evaluation complete\n')
        logging.info('Complete')
if __name__=='__main__':main()
