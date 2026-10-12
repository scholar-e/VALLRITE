"""Deterministic synthetic ARPAbet edits; never corrupt tags, targets or UNK spelling."""
import hashlib
import random
from CompetitiveWordDecoder.train_vallr_clean import edit_distance

PHONES = tuple('AA AE AH AO AW AY B CH D DH EH ER EY F G HH IH IY JH K L M N NG OW OY P R S SH T TH UH UW V W Y Z ZH'.split())
VERSION = 'arpabet-edits-v1'


def corrupt(row, rate, seed):
    if not 0 <= rate <= 1:
        raise ValueError('rate must be between zero and one')
    prefix, body = row['prompt'].split('<PHONEMES>\n', 1)
    body, suffix = body.split('\n</PHONEMES>', 1)
    tokens = body.split()
    positions = [i for i, token in enumerate(tokens) if token in PHONES]
    rng = random.Random(int.from_bytes(hashlib.sha256(
        f'{VERSION}:{seed}:{rate}:{row["prompt"]}'.encode()).digest()[:8], 'big'))
    count = min(len(positions), max(1, round(len(positions)*rate))) if positions and rate else 0
    selected = set(rng.sample(positions, count))
    output = []
    operations = dict(substitution=0, deletion=0, insertion=0)
    for index, token in enumerate(tokens):
        if index not in selected:
            output.append(token)
            continue
        draw = rng.random()
        if draw < .6:
            output.append(rng.choice([p for p in PHONES if p != token]))
            operations['substitution'] += 1
        elif draw < .85:
            operations['deletion'] += 1
        else:
            output.extend((token, rng.choice(PHONES)))
            operations['insertion'] += 1
    result = dict(row)
    result['prompt'] = prefix + '<PHONEMES>\n' + ' '.join(output) + '\n</PHONEMES>' + suffix
    reference = [p for p in tokens if p in PHONES]
    changed = [p for p in output if p in PHONES]
    result['corruption'] = {'rate': rate, 'seed': seed, 'operations': operations,
                            'reference_phones': len(reference),
                            'phone_edits': edit_distance(reference, changed)}
    return result


def mixed_training(rows, tokenizer, encoder, clean_fraction=.5):
    """Keep the same eligible rows; change only the fraction receiving errors."""
    if not 0 <= clean_fraction <= 1:
        raise ValueError('clean_fraction must be between zero and one')
    paired = []
    for row in rows:
        if encoder(row, tokenizer) is None:
            continue
        selector = int(hashlib.sha256(row['prompt'].encode()).hexdigest()[:8], 16) % 10
        rate = .05 if selector < 5 else .10 if selector < 9 else .20
        noisy = corrupt(row, rate, 314159)
        if encoder(noisy, tokenizer) is not None and noisy['corruption']['phone_edits']:
            paired.append((row, noisy))
    indices = list(range(len(paired)))
    random.Random(42).shuffle(indices)
    noise_indices = set(indices[:int(len(indices)*(1-clean_fraction))])
    return [pair[int(i in noise_indices)] for i, pair in enumerate(paired)]


def fixed_evaluation(rows, tokenizer, encoder, limit=100):
    """Common length-eligible held-out rows, independently seeded evaluation errors."""
    paired = []
    for row in rows:
        if len(tokenizer.encode(row['target'], add_special_tokens=False)) >= 256:
            continue
        variants = [row, corrupt(row, .10, 271828), corrupt(row, .20, 271828)]
        if all(encoder(item, tokenizer) is not None for item in variants):
            paired.append(variants)
    selected = random.Random(42).sample(paired, min(limit, len(paired)))
    return {name: [items[i] for items in selected]
            for i, name in enumerate(('clean', 'noise10', 'noise20'))}
