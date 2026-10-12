"""Fixed, boundary-free family codes shared by sentence training and evaluation."""
from VisualPhoneme.visemes import PHONE_TO_VISUAL_GROUP, VISUAL_GROUPS, VISUAL_PHONE_GROUPS

VERSION = 'visual-families-A-Q-v1'
FAMILY_CODES = {family: chr(ord('A') + i) for i, family in enumerate(VISUAL_GROUPS)}


def family_codes(phones, format='codes'):
    # Preserve repetitions: two phones in one family remain two target positions.
    if format == 'alternatives':
        return ['[' + '/'.join(VISUAL_PHONE_GROUPS[PHONE_TO_VISUAL_GROUP[phone]]) + ']'
                for phone in phones]
    if format != 'codes':
        raise ValueError('unknown family format')
    return [FAMILY_CODES[PHONE_TO_VISUAL_GROUP[phone]] for phone in phones]


def family_example(row, format='codes'):
    if row['unknown_words']:
        return None
    before, remainder = row['prompt'].split('<PHONEMES>\n', 1)
    payload, after = remainder.split('\n</PHONEMES>', 1)
    phones = payload.split()
    if any(phone not in PHONE_TO_VISUAL_GROUP for phone in phones):
        return None
    codes = family_codes(phones, format)
    if not codes:
        return None
    return dict(row, prompt=before + '<PHONEMES>\n' + ' '.join(codes) + '\n</PHONEMES>' + after)
