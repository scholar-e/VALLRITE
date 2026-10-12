"""Frozen Llama decoder evaluation on stored visual ARPAbet N-best predictions."""
import argparse
import hashlib
import json
import logging
from pathlib import Path
import signal
import time

from CompetitiveWordDecoder.noisy_teacher import load_nbest
from CompetitiveWordDecoder.evaluate_vallr_qwen import vallr_prompt
from CompetitiveWordDecoder.train_vallr_clean import MODEL, REVISION, edit_distance
from PhonemeDecoder.evaluate_qwen_direct import words

LOG = logging.getLogger(__name__)


def score_records(records, generated, reference_mode=False):
    """One decoded output per ranked visual hypothesis; oracle uses references only for scoring."""
    total = top = oracle = exact = capped = 0
    details = []
    if len(records) != len(generated):
        raise ValueError('record/output count mismatch')
    for row, outputs in zip(records, generated, strict=True):
        if len(outputs) != (1 if reference_mode else 5) or any(item is None for item in outputs):
            raise ValueError('unexpected number of complete generations')
        reference = words(' '.join(row['reference_words']))
        if not reference:
            raise ValueError('empty normalized reference')
        errors = [edit_distance(reference, words(item['text'])) for item in outputs]
        total += len(reference)
        top += errors[0]
        oracle += min(errors)
        exact += int(min(errors) == 0)
        capped += sum(item['hit_generation_cap'] for item in outputs)
        details.append({'clip_id': row['clip_id'], 'reference_words': reference,
                        'hypotheses': [dict(item, word_errors=error, phone_rank=index+1)
                                       for index, (item, error) in enumerate(zip(outputs, errors))]})
    if not total:
        raise ValueError('empty evaluation')
    report = {'records': len(records), 'reference_words': total,
            'top_1': {'word_errors': top, 'wer': top/total},
            'oracle_at_5': {'word_errors': oracle, 'wer': oracle/total,
                            'exact_hits': exact, 'exact_accuracy': exact/len(records)},
            'capped_generations': capped, 'records_detail': details}
    if reference_mode:
        report.pop('oracle_at_5')
        report['exact_hits'] = exact
        report['exact_accuracy'] = exact/len(records)
    return report


def input_items(records, reference_mode=False, visual_families=False, family_format='codes'):
    if reference_mode:
        for row in records:
            phones = row.get('reference_phones')
            if not isinstance(phones, list) or not phones or any(
                    not isinstance(phone, str) or not phone for phone in phones):
                raise ValueError('Missing or invalid reference phones')
        from CompetitiveWordDecoder.family_input import family_codes
        return [(i, 0, vallr_prompt(family_codes(row['reference_phones'], family_format) if visual_families
                                    else row['reference_phones'])) for i, row in enumerate(records)]
    return [(i, j, vallr_prompt(h['phones'])) for i, row in enumerate(records)
            for j, h in enumerate(row['phone_hypotheses'])]


def _main(guard_stack):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hypotheses', type=Path, required=True)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--limit', type=int, help='Smoke-test first N records only')
    parser.add_argument('--reference-phones', action='store_true',
                        help='Diagnostic: decode one reference phone sequence per clip, not visual predictions')
    parser.add_argument('--visual-families', action='store_true')
    parser.add_argument('--family-format', choices=('codes', 'alternatives'), default='codes')
    parser.add_argument('--max-prompt-tokens', type=int, default=512)
    args = parser.parse_args()
    if args.max_prompt_tokens < 1 or (args.family_format != 'codes' and not args.visual_families):
        parser.error('invalid context limit or family format')
    if args.visual_families and not args.reference_phones:
        parser.error('--visual-families currently requires --reference-phones')
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error('batch size and limit must be positive')
    if not args.hypotheses.is_file() or not (args.adapter/'adapter_config.json').is_file():
        parser.error('hypothesis file and adapter must exist')
    args.output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s',
                        handlers=[logging.FileHandler(args.output/'eval.log'), logging.StreamHandler()])
    logging.getLogger('httpx').setLevel(logging.WARNING)
    if args.visual_families:
        from VisualPhoneme.thermal import ThermalGuard
        guard_stack.enter_context(ThermalGuard(args.output, gpu_required=True))
    records = load_nbest(args.hypotheses)
    if args.limit:
        records = records[:args.limit]
    for row in records:
        if [h['rank'] for h in row['phone_hypotheses']] != [1, 2, 3, 4, 5]:
            raise ValueError('Expected five hypotheses in visual rank order')
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from peft import PeftModel
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; use host GPU launcher')
    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(args.adapter, local_files_only=True)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    items = input_items(records, args.reference_phones, args.visual_families, args.family_format)
    lengths = [len(tokenizer.encode(item[2], add_special_tokens=False)) for item in items]
    if max(lengths) > args.max_prompt_tokens:
        raise ValueError('Input exceeds training context; no silent truncation permitted')
    source_metadata = args.hypotheses.parent/'phone-eval.json'
    with (args.adapter/'adapter_model.safetensors').open('rb') as handle:
        adapter_hash = hashlib.file_digest(handle, 'sha256').hexdigest()
    provenance = {'model': MODEL, 'revision': REVISION, 'adapter': str(args.adapter.resolve()),
                  'adapter_sha256': adapter_hash, 'source': str(args.hypotheses.resolve()),
                  'source_sha256': hashlib.sha256(args.hypotheses.read_bytes()).hexdigest(),
                  'records': len(records), 'max_prompt_tokens': max(lengths), 'batch_size': args.batch_size,
                  'max_new_tokens': 256, 'do_sample': False, 'seed': 42, 'limit': args.limit,
                  'add_special_tokens': False, 'prompt': 'VALLR tag format, no chat template',
                  'normalization': "lowercase [a-z]+(?:'[a-z]+)*, same as prior direct decoder evaluator",
                  'scope': 'stored LRS3 development-validation visual predictions; not official test WER',
                  'oracle_definition': 'minimum word edit count among five separately decoded visual hypotheses'}
    provenance['input_mode'] = 'reference_phones' if args.reference_phones else 'visual_nbest'
    provenance['prompt_token_limit'] = args.max_prompt_tokens
    training_metrics = args.adapter.parent/'training-metrics.json'
    if training_metrics.exists():
        provenance['adapter_training_metrics'] = json.loads(training_metrics.read_text())
    if args.visual_families:
        from CompetitiveWordDecoder.family_input import VERSION, FAMILY_CODES
        provenance.update(input_mode='reference_families', family_version=VERSION, family_codes=FAMILY_CODES)
        if args.family_format == 'alternatives':
            from VisualPhoneme.visemes import VISUAL_PHONE_GROUPS
            provenance.update(family_version='visual-families-alternatives-v1',
                              family_alternatives=VISUAL_PHONE_GROUPS)
            provenance.pop('family_codes')
    if args.reference_phones:
        provenance['scope'] = 'LRS3 reference-phone reconstruction diagnostic; not visual or official test WER'
        provenance['oracle_definition'] = None
    if source_metadata.exists():
        source = json.loads(source_metadata.read_text())
        provenance['visual_source'] = {key: source.get(key) for key in
            ('checkpoint', 'checkpoint_epoch', 'architecture', 'validation_clips', 'greedy_per', 'results')}
    (args.output/'manifest.json').write_text(json.dumps(provenance, indent=2)+'\n')
    stopping = False
    def pause(*unused):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGINT, pause)
    signal.signal(signal.SIGTERM, pause)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION,
                cache_dir='.cache/huggingface/hub', local_files_only=True,
                dtype=torch.float16, attn_implementation='sdpa')
    model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model, args.adapter, local_files_only=True).cuda().eval()
    model.config.use_cache = True
    generated = [[None]*(1 if args.reference_phones else 5) for _ in records]
    started = time.monotonic()
    LOG.info('Starting records=%d generations=%d', len(records), len(items))
    with (args.output/'generations.jsonl').open('w') as handle, torch.inference_mode():
        for start in range(0, len(items), args.batch_size):
            if stopping or (args.output/'PAUSE').exists() or Path('PAUSE').exists():
                LOG.info('Paused; partial generations saved, no completed WER report')
                return
            batch = items[start:start+args.batch_size]
            inputs = tokenizer([item[2] for item in batch], add_special_tokens=False,
                               padding=True, return_tensors='pt').to(model.device)
            result = model.generate(**inputs, do_sample=False, max_new_tokens=256,
                                    pad_token_id=tokenizer.pad_token_id,
                                    eos_token_id=tokenizer.eos_token_id)
            continuations = result[:, inputs['input_ids'].shape[1]:].tolist()
            for (i, j, _), ids in zip(batch, continuations, strict=True):
                terminated = tokenizer.eos_token_id in ids
                if terminated:
                    ids = ids[:ids.index(tokenizer.eos_token_id)]
                output = {'text': tokenizer.decode(ids, skip_special_tokens=True),
                          'hit_generation_cap': not terminated and len(ids) >= 256}
                generated[i][j] = output
                handle.write(json.dumps({'clip_id': records[i]['clip_id'], 'phone_rank': j+1,
                                         **output})+'\n')
            handle.flush()
            if start == 0 or (start//args.batch_size) % 10 == 0:
                LOG.info('generated=%d/%d elapsed=%.1fs', min(start+len(batch), len(items)),
                         len(items), time.monotonic()-started)
    report = dict(provenance, **score_records(records, generated, args.reference_phones))
    report['elapsed_seconds'] = time.monotonic()-started
    (args.output/'metrics.json').write_text(json.dumps(report, indent=2)+'\n')
    LOG.info('Complete mode=%s top1_WER=%.2f%% capped=%d',
             provenance['input_mode'], report['top_1']['wer']*100, report['capped_generations'])


def main():
    from contextlib import ExitStack
    with ExitStack() as stack:
        _main(stack)


if __name__ == '__main__':
    main()
