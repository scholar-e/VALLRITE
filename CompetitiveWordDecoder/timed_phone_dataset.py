"""Build exact split-checked numeric phone evidence; references only supply labels."""
import argparse
import hashlib
import json
import logging
from pathlib import Path
import torch
from transformers import AutoTokenizer
from CompetitiveWordDecoder.visual_error_data import build_frozen_rows
from CompetitiveWordDecoder.timed_phone_features import align_phone_evidence,prompt_features


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(filename)s:%(lineno)d %(message)s',handlers=[logging.FileHandler(a.output/'run.log'),logging.StreamHandler()]);torch.set_num_threads(2)
    source=Path('checkpoints/matched-phones-20261029');manifests=Path('datasets/decoder-training/source-disjoint-950-20261029');adapter=Path('checkpoints/decoder-current-mixed-20261027/adapter')
    build_frozen_rows(source/'train/phone-hypotheses.jsonl',source/'evaluation/phone-hypotheses.jsonl',manifests)
    tokenizer=AutoTokenizer.from_pretrained(adapter,local_files_only=True);data={};hashes={}
    sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
    for split in ('train','validation','evaluation'):
        cache=torch.load(source/split/'logits.pt',map_location='cpu',weights_only=True)
        refs=[json.loads(l) for l in (manifests/f'{split}.jsonl').read_text().splitlines()]
        paths=[json.loads(l) for l in (source/split/'phone-hypotheses.jsonl').read_text().splitlines()]
        data[split]=[]
        for cached,ref,record in zip(cache,refs,paths,strict=True):
            if Path('PAUSE').exists():raise InterruptedError('paused during preparation')
            assert cached['clip_id']==ref['clip_id']==record['clip_id']
            candidates=record['phone_hypotheses'][:1] if split=='train' else record['phone_hypotheses']
            entries=[]
            target=tokenizer.encode(' '.join(record['reference_words']),add_special_tokens=False)+[tokenizer.eos_token_id]
            for candidate in candidates:
                evidence=align_phone_evidence(cached['logits'],candidate['phones'],ref['fps'])
                ids,features=prompt_features(candidate['phones'],evidence,tokenizer)
                if len(ids)+len(target)>512:raise ValueError('complete pair overlength')
                entries.append(dict(ids=ids,features=features,phone_path_log_score=candidate['ctc_log_score']))
            data[split].append(dict(clip_id=record['clip_id'],reference_words=record['reference_words'],target=target,candidates=entries))
        for f in (source/split/'logits.pt',source/split/'phone-hypotheses.jsonl',manifests/f'{split}.jsonl'):hashes[str(f)]=sha(f)
        logging.info('%s clips=%d',split,len(data[split]))
    torch.save(data,a.output/'data.pt')
    (a.output/'manifest.json').write_text(json.dumps(dict(sources=hashes,adapter_sha256=sha(adapter/'adapter_model.safetensors'),data_sha256=sha(a.output/'data.pt'),feature_dim=44,tokenizer_source=str(adapter),selection='frozen 750/100/100 video-disjoint pilot splits; no new filtering'),indent=2)+'\n')
    logging.info('Complete')
if __name__=='__main__':main()
