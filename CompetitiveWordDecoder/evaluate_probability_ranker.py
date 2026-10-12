"""Frozen scored-phone-path -> sentence ranking, without reference-based selection."""
import argparse
import hashlib
import json
import logging
import math
from pathlib import Path
import signal
import time

from CompetitiveWordDecoder.noisy_teacher import load_nbest
from CompetitiveWordDecoder.evaluate_vallr_qwen import vallr_prompt
from CompetitiveWordDecoder.train_vallr_clean import MODEL, REVISION, edit_distance
from PhonemeDecoder.evaluate_qwen_direct import words

LOG = logging.getLogger(__name__)


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def logsumexp(values):
    largest = max(values)
    return largest + math.log(sum(math.exp(v-largest) for v in values))


def choose(candidates, length_normalized=False):
    """Aggregate duplicate normalized sentences across paths; never access references.

    The summed variant multiplies retained-path mass by conditional text likelihood.
    Length normalization is a predeclared diagnostic heuristic, not a posterior.
    """
    groups = {}
    for index, item in enumerate(candidates):
        decoder = item['decoder_log_probability']
        if length_normalized:
            decoder /= item['scored_tokens']
        value = item['phone_path_log_score'] + decoder
        if not math.isfinite(value):
            raise ValueError('nonfinite ranking score')
        key = tuple(words(item['text']))
        groups.setdefault(key, []).append((index, value))
    scores = [(entries[0][0], logsumexp([v for _, v in entries])) for entries in groups.values()]
    winner = max(scores, key=lambda pair: (pair[1], -pair[0]))[0]
    normalizer = logsumexp([value for _, value in scores])
    return winner, [{'representative_index': i, 'relative_mass': math.exp(v-normalizer)} for i, v in scores]


def summarize(records, candidates):
    totals = dict(baseline=0, probability_ranked=0, length_normalized=0, oracle_at_5=0)
    changes = dict(probability_ranked=0, length_normalized=0)
    count = 0
    details = []
    for row, choices in zip(records, candidates, strict=True):
        # Make both choices BEFORE references enter scoring.
        selected, mass = choose(choices)
        normalized, normalized_mass = choose(choices, True)
        reference = words(' '.join(row['reference_words']))
        if not reference:
            raise ValueError('empty reference')
        errors = [edit_distance(reference, words(c['text'])) for c in choices]
        count += len(reference)
        totals['baseline'] += errors[0]
        totals['probability_ranked'] += errors[selected]
        totals['length_normalized'] += errors[normalized]
        totals['oracle_at_5'] += min(errors)
        changes['probability_ranked'] += words(choices[selected]['text']) != words(choices[0]['text'])
        changes['length_normalized'] += words(choices[normalized]['text']) != words(choices[0]['text'])
        details.append(dict(clip_id=row['clip_id'], reference_words=reference,
                            selected_index=selected, normalized_index=normalized,
                            relative_mass=mass, normalized_mass=normalized_mass,
                            errors=errors, candidates=choices))
    return dict(records=len(records), reference_words=count,
                metrics={k: dict(word_errors=v, wer=v/count) for k,v in totals.items()},
                changed_sentences=changes, details=details)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hypotheses', type=Path, required=True)
    parser.add_argument('--decoder-report', type=Path, required=True)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error('positive batch size and limit required')
    records = load_nbest(args.hypotheses)
    old = json.loads(args.decoder_report.read_text())
    if old['source_sha256'] != sha(args.hypotheses):
        raise ValueError('cached decoder source mismatch')
    if old['adapter_sha256'] != sha(args.adapter/'adapter_model.safetensors'):
        raise ValueError('cached decoder adapter mismatch')
    if old['model'] != MODEL or old['revision'] != REVISION:
        raise ValueError('cached base model mismatch')
    if len(records) != len(old['records_detail']):
        raise ValueError('cached record count mismatch')
    cached = []
    for row, decoded in zip(records, old['records_detail'], strict=True):
        if row['clip_id'] != decoded['clip_id'] or words(' '.join(row['reference_words'])) != decoded['reference_words']:
            raise ValueError('cached record identity mismatch')
        if [h['rank'] for h in row['phone_hypotheses']] != list(range(1,6)):
            raise ValueError('expected five ordered phone paths')
        if [h['phone_rank'] for h in decoded['hypotheses']] != list(range(1,6)):
            raise ValueError('expected five ordered decoded sentences')
        cached.append([dict(text=h['text'], hit_generation_cap=h['hit_generation_cap'],
                            phone_path_log_score=p['ctc_log_score'])
                       for p,h in zip(row['phone_hypotheses'], decoded['hypotheses'], strict=True)])
    if args.limit:
        records, cached = records[:args.limit], cached[:args.limit]
    args.output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO,
        format='%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s',
        handlers=[logging.FileHandler(args.output/'eval.log'), logging.StreamHandler()])
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from VisualPhoneme.thermal import ThermalGuard
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; use host launcher')
    stopping = False
    def stop(*unused):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    tokenizer = AutoTokenizer.from_pretrained(args.adapter, local_files_only=True)
    items = []
    for i, row in enumerate(records):
        for j, path in enumerate(row['phone_hypotheses']):
            prefix = tokenizer.encode(vallr_prompt(path['phones']), add_special_tokens=False)
            # Retokenize the exact saved sentence. EOS is scored only for terminated output.
            target = tokenizer.encode(cached[i][j]['text'], add_special_tokens=False)
            if not cached[i][j]['hit_generation_cap']:
                target.append(tokenizer.eos_token_id)
            if not prefix or not target or len(prefix)+len(target) > 1024:
                raise ValueError('invalid or oversized scoring sequence; no truncation')
            items.append((i,j,prefix,target))
    manifest = dict(model=MODEL, revision=REVISION, adapter_sha256=old['adapter_sha256'],
        source_sha256=old['source_sha256'], decoder_report_sha256=sha(args.decoder_report),
        source=str(args.hypotheses.resolve()), decoder_report=str(args.decoder_report.resolve()),
        records=len(records), limit=args.limit, score_weight=1.0,
        selection='logsumexp(path beam score + decoder token log-prob sum) over identical normalized sentences',
        diagnostic='same with decoder mean token log probability; no validation-based weight tuning',
        caveats=['path scores include phone bigram prior weight 0.5, not pure visual CTC',
                 'relative masses only within retained candidates; not calibrated confidence',
                 'cached generated text retokenized for conditional scoring',
                 'capped outputs score observed prefix without EOS',
                 'one frozen sentence per phone path; oracle pool unchanged',
                 'development validation, not official test results'])
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    started = time.monotonic()
    with ThermalGuard(args.output, gpu_required=True):
        model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION,
            cache_dir='.cache/huggingface/hub', local_files_only=True,
            dtype=torch.float16, attn_implementation='sdpa')
        model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(model, args.adapter, local_files_only=True).cuda().eval()
        model.config.use_cache = False
        with torch.inference_mode(), (args.output/'scores.jsonl').open('w') as handle:
            for start in range(0,len(items),args.batch_size):
                if stopping or Path('PAUSE').exists() or (args.output/'PAUSE').exists():
                    LOG.info('Paused; no completed metrics produced')
                    return
                batch = items[start:start+args.batch_size]
                width = max(len(p)+len(t) for _,_,p,t in batch)
                ids = torch.full((len(batch),width), tokenizer.pad_token_id, device='cuda', dtype=torch.long)
                mask = torch.zeros_like(ids)
                for b,(_,_,prefix,target) in enumerate(batch):
                    tokens = prefix+target
                    ids[b,:len(tokens)] = torch.tensor(tokens,device='cuda')
                    mask[b,:len(tokens)] = 1
                logits = model(input_ids=ids,attention_mask=mask,use_cache=False).logits
                for b,(i,j,prefix,target) in enumerate(batch):
                    prediction = logits[b,len(prefix)-1:len(prefix)+len(target)-1].float()
                    targets = ids[b,len(prefix):len(prefix)+len(target)]
                    logp = -torch.nn.functional.cross_entropy(prediction,targets,reduction='sum').item()
                    if not math.isfinite(logp):
                        raise ValueError('nonfinite decoder score')
                    cached[i][j].update(decoder_log_probability=logp,scored_tokens=len(target))
                    handle.write(json.dumps(dict(clip_id=records[i]['clip_id'],phone_rank=j+1,**cached[i][j]))+'\n')
                handle.flush()
                del logits, prediction
                if start == 0 or (start//args.batch_size) % 100 == 0:
                    LOG.info('scored=%d/%d elapsed=%.1fs',min(start+len(batch),len(items)),len(items),time.monotonic()-started)
    report = dict(manifest, **summarize(records,cached), elapsed_seconds=time.monotonic()-started)
    (args.output/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    LOG.info('Complete %s',report['metrics'])


if __name__ == '__main__':
    main()
