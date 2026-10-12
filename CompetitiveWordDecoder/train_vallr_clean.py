"""Code-faithful VALLR Llama/WikiText baseline, with pause/resume and clean WER."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from pathlib import Path
import re
import signal
import time

LOG = logging.getLogger(__name__)
MODEL = "meta-llama/Llama-3.2-3B-Instruct"
REVISION = "0cb88a4f764b7a12671c53f0838cd831a0843b95"
TAGS = ["<S2S>", "<PHONEMES>", "</PHONEMES>", "<TEXT>"]
WORD = re.compile(r"[A-Za-z']+")


def example(text):
    import pronouncing
    text = text.strip()
    words = WORD.findall(text)
    phones, unknown = [], 0
    for word in words:
        entries = pronouncing.phones_for_word(word.lower())
        if entries:
            phones.append(re.sub(r"\d", "", entries[0]))
        else:
            phones.append(f"UNK({word.lower()})")
            unknown += 1
    return {"prompt": f"<S2S>\n<PHONEMES>\n{' '.join(phones)}\n</PHONEMES>\n<TEXT>\n",
            "target": text, "unknown_words": unknown}


def encode(row, tokenizer):
    # Deliberately preserve upstream truncation and omission of target EOS.
    prefix = tokenizer.encode(row["prompt"], add_special_tokens=False)[:508]
    target = tokenizer.encode(row["target"], add_special_tokens=False)[:512-len(prefix)]
    ids = prefix + target
    return {"input_ids": ids + [tokenizer.pad_token_id] * (512-len(ids)),
            "attention_mask": [1] * len(ids) + [0] * (512-len(ids)),
            "labels": [-100] * len(prefix) + target + [-100] * (512-len(ids))}

def encode_eos(row, tokenizer, max_length=512):
    """Encode with EOS continuation from PEFT adapter.
    
    Appends EOS token ID to complete target, masks prompt labels as -100.
    Returns None if pair + EOS exceeds 512 tokens (no truncation).
    Requires tokenizer.eos_token_id to be set.
    """
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("tokenizer.eos_token_id must be set")
    
    prefix = tokenizer.encode(row["prompt"], add_special_tokens=False)
    target = tokenizer.encode(row["target"], add_special_tokens=False)
    
    # Check total + EOS exceeds 512 (never truncate transcript)
    if len(prefix) + len(target) + 1 > max_length:
        return None
    
    ids = prefix + target + [eos_id]
    labels = [-100] * len(prefix) + target + [eos_id]
    
    return {
        "input_ids": ids + [tokenizer.pad_token_id] * (max_length - len(ids)),
        "attention_mask": [1] * len(ids) + [0] * (max_length - len(ids)),
        "labels": labels + [-100] * (max_length - len(ids))
    }




def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, word in enumerate(reference, 1):
        current = [i]
        for j, other in enumerate(hypothesis, 1):
            current.append(min(current[-1]+1, previous[j]+1,
                               previous[j-1]+(word != other)))
        previous = current
    return previous[-1]


def evaluate_clean(model, tokenizer, rows, output, paused, max_new_tokens=256, eos_token_id=None,
                   split="test", max_length=512):
    # Explicitly set eos_token_id for EOS continuation mode
    import torch
    model.eval()
    model.config.use_cache = True
    totals = {name: {"examples": 0, "words": 0, "edits": 0, "exact": 0}
              for name in ("all", "dictionary_only")}
    skipped = capped = 0
    with (output / "clean-test-predictions.jsonl").open("w") as handle:
        for index, row in enumerate(rows):
            if paused():
                LOG.info("Evaluation paused; rerun with --evaluate-only to restart evaluation")
                return
            prefix = tokenizer.encode(row["prompt"], add_special_tokens=False)
            target = tokenizer.encode(row["target"], add_special_tokens=False)
            # Fixed clean benchmark: no missing prompt/target caused by truncation.
            if len(prefix)+len(target) > max_length or len(target) >= max_new_tokens:
                skipped += 1
                continue
            inputs = torch.tensor([prefix], device=model.device)
            with torch.inference_mode():
                stop_args = {"eos_token_id": eos_token_id} if eos_token_id is not None else {}
                generated = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                           max_new_tokens=max_new_tokens, do_sample=False,
                                           pad_token_id=tokenizer.pad_token_id, **stop_args)
            ids = generated[0, len(prefix):].tolist()
            prediction = tokenizer.decode(ids, skip_special_tokens=True)  # EOS mode retains skip_special_tokens
            capped += int(len(ids) == max_new_tokens)
            reference = WORD.findall(row["target"].lower())
            hypothesis = WORD.findall(prediction.lower())
            errors = edit_distance(reference, hypothesis)
            for name in ("all", "dictionary_only"):
                if name == "dictionary_only" and row["unknown_words"]:
                    continue
                group = totals[name]
                group["examples"] += 1
                group["words"] += len(reference)
                group["edits"] += errors
                group["exact"] += int(reference == hypothesis)
            handle.write(json.dumps({"index": index, **row, "prediction": prediction,
                                     "edits": errors, "hit_generation_cap": len(ids) == max_new_tokens})+"\n")
            handle.flush()
            if totals["all"]["examples"] % 25 == 0:
                LOG.info("Clean test progress %s / %s", index+1, len(rows))
    for group in totals.values():
        group["wer"] = group["edits"] / group["words"] if group["words"] else None
    result = {"split": split, "normalization": "lowercase ASCII words/apostrophes",
              "skipped_length": skipped, "generation_cap": max_new_tokens,
              "capped_generations": capped, "metrics": totals}
    (output / "clean-test-metrics.json").write_text(json.dumps(result, indent=2)+"\n")
    LOG.info("Clean held-out test results: %s", result)


def _main(guard_stack):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--max-minutes", type=float)
    parser.add_argument("--smoke", action="store_true", help="Separate two-step integration test")
    parser.add_argument("--end-of-answer", action="store_true")
    parser.add_argument("--initial-adapter", type=Path)
    parser.add_argument("--epochs", type=float, default=10)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--test-limit", type=int, default=0, help="Seeded clean test sample; 0 means all")
    parser.add_argument("--synthetic-noise", action="store_true",
                        help="Mixed clean/synthetic errors with fixed before/after validation WER")
    parser.add_argument("--clean-fraction", type=float, default=.5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument('--visual-families', action='store_true')
    parser.add_argument('--family-format', choices=('codes', 'alternatives'), default='codes')
    parser.add_argument('--max-length', type=int, default=512)
    parser.add_argument('--family-reference', type=Path,
                        help='Held-out LRS3 JSONL: exclude normalized transcript duplicates from training')
    args = parser.parse_args()
    if args.max_length < 32 or (args.max_length != 512 and not args.end_of_answer):
        parser.error('custom --max-length requires EOS and must be at least 32')
    if args.family_format != 'codes' and not args.visual_families:
        parser.error('--family-format requires --visual-families')
    if args.visual_families and (not args.end_of_answer or args.synthetic_noise or not args.family_reference):
        parser.error('--visual-families requires EOS, --family-reference, and no synthetic noise')
    if args.family_reference and not args.visual_families:
        parser.error('--family-reference requires --visual-families')
    if args.max_minutes is not None and args.max_minutes <= 0:
        parser.error("--max-minutes must be positive")
    if not math.isfinite(args.epochs) or args.epochs <= 0:
        parser.error("--epochs must be finite and positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be finite and positive")
    if args.test_limit < 0:
        parser.error("--test-limit must be nonnegative")
    if min(args.batch_size, args.gradient_accumulation) < 1:
        parser.error("batch size and gradient accumulation must be positive")
    if args.synthetic_noise and (not args.end_of_answer or args.initial_adapter is None):
        parser.error("--synthetic-noise requires --end-of-answer and --initial-adapter")
    if not 0 <= args.clean_fraction <= 1:
        parser.error("--clean-fraction must be between zero and one")
    if not args.synthetic_noise and args.clean_fraction != .5:
        parser.error("--clean-fraction requires --synthetic-noise")
    if args.initial_adapter is not None:
        if not args.end_of_answer or not (args.initial_adapter / "adapter_config.json").is_file():
            parser.error("--initial-adapter requires --end-of-answer and a valid adapter directory")
        args.initial_adapter = args.initial_adapter.resolve()
    args.output.mkdir(parents=True, exist_ok=args.resume or args.evaluate_only)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s",
                        handlers=[logging.FileHandler(args.output / "train.log"), logging.StreamHandler()])
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if args.visual_families:
        from VisualPhoneme.thermal import ThermalGuard
        guard_stack.enter_context(ThermalGuard(args.output, gpu_required=True))
    import torch
    from datasets import Dataset, load_dataset
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback,
                              TrainingArguments, default_data_collator, set_seed)
    from transformers.trainer_utils import get_last_checkpoint
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; use tools/with-vpa-gpu with host execution")
    stopping = False

    def request_pause(*unused):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, request_pause)
    signal.signal(signal.SIGTERM, request_pause)

    def paused():
        return stopping or Path("PAUSE").exists() or (args.output / "PAUSE").exists()

    if paused():
        LOG.info("PAUSE marker present; not starting")
        return
    set_seed(42)
    cache = ".cache/huggingface/hub"
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, cache_dir=cache)
    tokenizer.add_special_tokens({"additional_special_tokens": TAGS})
    tokenizer.pad_token = tokenizer.eos_token
    if args.initial_adapter:
        saved_tokenizer = AutoTokenizer.from_pretrained(args.initial_adapter, local_files_only=True)
        if saved_tokenizer.get_vocab() != tokenizer.get_vocab():
            raise ValueError("Initial adapter tokenizer vocabulary differs from recipe")
    corpus = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",
                          revision="b08601e04326c79dfdd32d625aee71d232d685c3",
                          cache_dir=".cache/huggingface/datasets")
    rows = {split: [example(row["text"]) for row in corpus[split] if row["text"].strip()]
            for split in ("train", "validation", "test")}
    seen = set()
    duplicates = {}
    for split in rows:
        original = len(rows[split])
        unique = []
        for row in rows[split]:
            key = " ".join(WORD.findall(row["target"].lower()))
            if key and key not in seen:
                seen.add(key)
                unique.append(row)
        rows[split] = unique
        duplicates[split] = original-len(unique)
    family_stats = None
    if args.visual_families:
        from CompetitiveWordDecoder.family_input import family_example, FAMILY_CODES, VERSION
        from PhonemeDecoder.evaluate_qwen_direct import words as normalized_words
        heldout = [json.loads(line) for line in args.family_reference.read_text().splitlines() if line.strip()]
        excluded_texts = {' '.join(normalized_words(' '.join(r['reference_words']))) for r in heldout}
        family_stats = {}
        for split, records in rows.items():
            converted = []
            for row in records:
                if len(normalized_words(row['target'])) < 2:
                    continue
                if ' '.join(normalized_words(row['target'])) in excluded_texts:
                    continue
                item = family_example(row, args.family_format)
                if item is not None:
                    converted.append(item)
            family_stats[split] = {'original': len(records), 'kept': len(converted)}
            rows[split] = converted
    manifest = {"model": MODEL, "revision": REVISION, "seed": 42, "smoke": args.smoke,
                "upstream_commit": "db932ca83ff8b2b858eb6a4c78d02b5e3805c299",
                "split_counts": {k: len(v) for k, v in rows.items()}, "duplicates_removed": duplicates,
                "split_hashes": {k: hashlib.sha256(json.dumps(v).encode()).hexdigest() for k, v in rows.items()},
                "epochs": args.epochs, "learning_rate": args.learning_rate,
                "batch_size": args.batch_size, "gradient_accumulation": args.gradient_accumulation,
                "lora_rank": 16, "lora_alpha": 32, "lora_dropout": .05,
                "caveats": ["upstream prompt-first truncation retained", "no target EOS upstream",
                            "UNK(word) spelling leakage reported separately",
                            "exact normalized-text deduplication added", "gradient checkpointing added"],
                "versions": {"torch": torch.__version__}}
    if args.end_of_answer:
        manifest.update(end_of_answer=True,
                        initial_adapter=str(args.initial_adapter) if args.initial_adapter else None,
                        test_limit=args.test_limit, eos_token_id=tokenizer.eos_token_id,
                        encoding="complete-pair-plus-supervised-eos-v1")
        manifest["caveats"] = ["oversized complete pairs skipped; never truncated",
                               "UNK(word) spelling leakage reported separately",
                               "exact normalized-text deduplication added", "gradient checkpointing added"]
        if args.initial_adapter:
            weights = args.initial_adapter / "adapter_model.safetensors"
            with weights.open("rb") as handle:
                manifest["initial_adapter_sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
    elif args.test_limit:
        manifest["test_limit"] = args.test_limit
    if args.synthetic_noise:
        from CompetitiveWordDecoder.synthetic_noise import VERSION
        manifest["synthetic_noise"] = {"version": VERSION, "clean_fraction": args.clean_fraction,
                                      "rates": [.05, .10, .20], "rate_weights": [.5, .4, .1],
                                      "operation_weights": [.6, .25, .15],
                                      "evaluation_split": "validation"}
    manifest_path = args.output / "manifest.json"
    if args.max_length != 512:
        manifest['max_length'] = args.max_length
    if args.visual_families:
        manifest['visual_families'] = dict(version=VERSION, codes=FAMILY_CODES, filtering=family_stats,
            precision='bfloat16',
            reference_sha256=hashlib.sha256(args.family_reference.read_bytes()).hexdigest(),
            evaluation='WikiText validation only; separate matched LRS3 reference-family diagnostic')
        manifest['caveats'] = ['dictionary-only multi-word rows; no UNK spelling leakage',
                             'no word boundaries; repeated families retained',
                             'complete pair plus EOS; oversized rows excluded',
                             'continuation differs from baseline in training data and representation']
        if args.family_format != 'codes':
            from VisualPhoneme.visemes import VISUAL_PHONE_GROUPS
            manifest['visual_families'].update(version='visual-families-alternatives-v1',
                format=args.family_format, alternatives=VISUAL_PHONE_GROUPS)
            manifest['visual_families'].pop('codes')
            # JSON round-trip converts tuple inventories to lists for stable resume comparison.
            manifest = json.loads(json.dumps(manifest))
    if args.resume or args.evaluate_only:
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Resume manifest mismatch")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
    LOG.info("Dataset manifest: %s", manifest)
    if args.smoke:
        rows = {k: v[:8] for k, v in rows.items()}
    from functools import partial
    encoder = partial(encode_eos, max_length=args.max_length) if args.end_of_answer else encode
    noise_evaluation = None
    if args.synthetic_noise:
        from CompetitiveWordDecoder.synthetic_noise import mixed_training, fixed_evaluation
        rows["train"] = mixed_training(rows["train"], tokenizer, encoder, args.clean_fraction)
        noise_evaluation = fixed_evaluation(rows["validation"], tokenizer, encoder, args.test_limit or 100)
        if not rows["train"] or not noise_evaluation["clean"]:
            raise ValueError("Synthetic noise filtering left an empty split")
        stats = {"rows": len(rows["train"]),
                 "noisy_rows": sum("corruption" in r for r in rows["train"]),
                 "validation_examples_per_condition": len(noise_evaluation["clean"])}
        for name, records in [("training", rows["train"]), *noise_evaluation.items()]:
            noise = [r["corruption"] for r in records if "corruption" in r]
            phones = sum(r["reference_phones"] for r in noise)
            stats[name + "_realized_phone_error_rate"] = (
                sum(r["phone_edits"] for r in noise) / phones if phones else 0)
        LOG.info("Synthetic noise data: %s", stats)
        data_record = {"stats": stats, "train_sha256": hashlib.sha256(
            json.dumps(rows["train"]).encode()).hexdigest(), "evaluation": noise_evaluation}
        data_path = args.output / "synthetic-data.json"
        if args.resume or args.evaluate_only:
            if json.loads(data_path.read_text()) != data_record:
                raise ValueError("Synthetic data mismatch on resume")
        else:
            data_path.write_text(json.dumps(data_record, indent=2)+"\n")
    encoded = {}
    for split in ("train", "validation"):
        items = [encoder(row, tokenizer) for row in rows[split]]
        encoded[split] = [item for item in items if item is not None]
        LOG.info("Encoded %s: kept=%d skipped=%d", split, len(encoded[split]),
                 len(items)-len(encoded[split]))
        if not encoded[split]:
            raise ValueError(f"No complete examples remain in {split}")
    if paused():
        LOG.info("Paused before model loading")
        return
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, cache_dir=cache,
                                               dtype=torch.bfloat16 if args.visual_families else torch.float16,
                                               attn_implementation="sdpa")
    model.resize_token_embeddings(len(tokenizer))
    if args.evaluate_only:
        model = PeftModel.from_pretrained(model, args.output / "adapter").cuda()
    else:
        if args.initial_adapter:
            model = PeftModel.from_pretrained(model, args.initial_adapter,
                                             is_trainable=True, local_files_only=True)
            LOG.info("Loaded initial adapter %s; fresh optimizer unless --resume", args.initial_adapter)
        else:
            model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=16, lora_alpha=32,
                                  lora_dropout=.05, bias="none", target_modules=["q_proj", "k_proj",
                                  "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
        model.config.use_cache = False
        model.enable_input_require_grads()
        if args.synthetic_noise and not args.resume:
            model.cuda()
            for condition, records in noise_evaluation.items():
                if paused():
                    LOG.info("Paused during baseline evaluation; no training started")
                    return
                directory = args.output / ("baseline-" + condition)
                directory.mkdir(exist_ok=True)
                evaluate_clean(model, tokenizer, records, directory, paused,
                               eos_token_id=tokenizer.eos_token_id, split="validation")
            model.train()
            model.config.use_cache = False
        start = time.monotonic()

        class Progress(TrainerCallback):
            def on_step_end(self, training_args, state, control, **kwargs):
                expired = args.max_minutes and time.monotonic()-start >= args.max_minutes*60
                if paused() or expired:
                    control.should_save = True
                    control.should_training_stop = True
                    # Avoid a long epoch-loss evaluation when the user requests a pause.
                    control.should_evaluate = False
                    LOG.info("Saving resumable checkpoint at step %d", state.global_step)
                return control

            def on_epoch_end(self, training_args, state, control, **kwargs):
                expired = args.max_minutes and time.monotonic()-start >= args.max_minutes*60
                if paused() or expired:
                    control.should_evaluate = False
                return control

            def on_log(self, training_args, state, control, logs=None, **kwargs):
                LOG.info("step=%d/%d epoch=%s %s", state.global_step, state.max_steps, state.epoch, logs)

        effective_batch = args.batch_size * args.gradient_accumulation
        total_steps = 2 if args.smoke else math.ceil(math.ceil(len(encoded["train"])/effective_batch)*args.epochs)
        trainer = Trainer(model=model, args=TrainingArguments(
            output_dir=str(args.output), num_train_epochs=args.epochs, max_steps=2 if args.smoke else -1,
            per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=1,
            gradient_accumulation_steps=args.gradient_accumulation, learning_rate=args.learning_rate, weight_decay=.01,
            warmup_steps=math.ceil(.03 * total_steps),
            fp16=not args.visual_families, bf16=args.visual_families, gradient_checkpointing=True,
            eval_strategy="epoch", save_strategy="steps", save_steps=250, save_total_limit=2,
            logging_steps=1 if args.smoke else 10, report_to="none", disable_tqdm=True, seed=42),
            train_dataset=Dataset.from_list(encoded["train"]),
            eval_dataset=Dataset.from_list(encoded["validation"]),
            data_collator=default_data_collator, processing_class=tokenizer, callbacks=[Progress()])
        checkpoint = get_last_checkpoint(str(args.output)) if args.resume else None
        if args.resume and checkpoint is None:
            raise ValueError("No resumable checkpoint found")
        if paused():
            LOG.info("Paused before training")
            return
        result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model(str(args.output / "adapter"))
        tokenizer.save_pretrained(args.output / "adapter")
        (args.output / "training-metrics.json").write_text(json.dumps(result.metrics, indent=2)+"\n")
        if trainer.state.global_step < trainer.state.max_steps or paused():
            LOG.info("Training paused; evaluation deferred until completion")
            return
    if args.synthetic_noise:
        report = {"split": "validation", "scope": "synthetic errors; not visual-input WER",
                  "conditions": {}}
        for condition, records in noise_evaluation.items():
            if paused():
                return
            directory = args.output / ("final-" + condition)
            directory.mkdir(exist_ok=True)
            evaluate_clean(model, tokenizer, records, directory, paused,
                           eos_token_id=tokenizer.eos_token_id, split="validation")
            if paused():
                return
            entry = {}
            for phase in ("baseline", "final"):
                path = args.output / (phase + "-" + condition) / "clean-test-metrics.json"
                entry[phase] = json.loads(path.read_text())
                entry[phase]["split"] = "validation"
            report["conditions"][condition] = entry
        (args.output / "synthetic-results.json").write_text(json.dumps(report, indent=2)+"\n")
        LOG.info("Synthetic evaluation complete: %s", args.output / "synthetic-results.json")
        return
    test_rows = rows["validation"] if args.visual_families else rows["test"]
    if args.test_limit:
        import random
        eligible = [r for r in test_rows
                    if len(tokenizer.encode(r["prompt"], add_special_tokens=False)) +
                    len(tokenizer.encode(r["target"], add_special_tokens=False)) <= args.max_length
                    and len(tokenizer.encode(r["target"], add_special_tokens=False)) < 256]
        test_rows = random.Random(42).sample(eligible, min(args.test_limit, len(eligible)))
        (args.output / "test-sample.json").write_text(json.dumps({
            "seed": 42, "eligible_rows": len(eligible),
            "selected_targets": [r["target"] for r in test_rows]}, indent=2)+"\n")
    evaluate_clean(model, tokenizer, test_rows, args.output, paused,
                   eos_token_id=tokenizer.eos_token_id if args.end_of_answer else None,
                   split='validation' if args.visual_families else 'test', max_length=args.max_length)


def main():
    from contextlib import ExitStack
    with ExitStack() as stack:
        _main(stack)


if __name__ == "__main__":
    main()
