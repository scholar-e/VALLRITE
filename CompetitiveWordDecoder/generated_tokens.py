"""Preserve the generated token path, including its actual termination token."""


def generation_record(ids, tokenizer, max_new_tokens=256):
    ids = list(ids)
    terminated = tokenizer.eos_token_id in ids
    if terminated:
        ids = ids[:ids.index(tokenizer.eos_token_id)+1]
    return dict(text=tokenizer.decode(ids, skip_special_tokens=True),
                generated_token_ids=ids,
                hit_generation_cap=not terminated and len(ids) >= max_new_tokens)


def scoring_tokens(item, tokenizer, require_exact=False, retokenize=False):
    if 'generated_token_ids' in item and not retokenize:
        ids = item['generated_token_ids']
        if (not isinstance(ids, list) or not ids or
                any(type(i) is not int or not 0 <= i < len(tokenizer) for i in ids)):
            raise ValueError('invalid generated tokens')
        terminated = ids[-1] == tokenizer.eos_token_id
        if tokenizer.eos_token_id in ids[:-1] or terminated == bool(item['hit_generation_cap']):
            raise ValueError('inconsistent termination')
        if tokenizer.decode(ids, skip_special_tokens=True) != item['text']:
            raise ValueError('generated tokens differ from saved text')
        return list(ids), 'exact_generated_tokens'
    if require_exact:
        raise ValueError('exact generated tokens required')
    ids = tokenizer.encode(item['text'], add_special_tokens=False)
    if not item['hit_generation_cap']:
        ids.append(tokenizer.eos_token_id)
    return ids, 'legacy_retokenized_text'
