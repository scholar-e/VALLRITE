"""Dependency-free, bounded incremental CTC search with lexical constraints."""
from __future__ import annotations
import math

PHONES = tuple('AA AE AH AO AW AY B CH D DH EH ER EY F G HH IH IY JH K L M N NG OW OY P R S SH T TH UH UW V W Y Z ZH'.split())
VOCABULARY = ('<blank>',) + PHONES
NEG = -math.inf


def add(a, b):
    if a == NEG:
        return b
    if b == NEG:
        return a
    return max(a, b) + math.log1p(math.exp(-abs(a-b)))


class Lexicon:
    """Trie retaining homophones and unique pronunciations from the design format."""
    def __init__(self, document):
        self.nodes = [{}]
        self.ends = [[]]
        self.words = {}
        self.inventory = tuple(document['phone_inventory'])
        if not self.inventory or len(set(self.inventory)) != len(self.inventory):
            raise ValueError('empty or duplicate lexicon inventory')
        self.vocabulary = ('<blank>',) + self.inventory
        self.boundary_token = (self.vocabulary.index('rest')
                               if 'rest' in self.vocabulary else None)
        for word in document['words']:
            wid = word['id']
            if wid in self.words or not word['text']:
                raise ValueError('duplicate word ID or empty word')
            self.words[wid] = word['text']
            seen, ids, total = set(), set(), 0.0
            for pron in word['pronunciations']:
                phones = tuple(pron['phones'])
                prior = pron['prior']
                if not phones or len(phones) > 64 or phones in seen or pron['id'] in ids:
                    raise ValueError('empty, oversized, or duplicate pronunciation')
                if not math.isfinite(prior) or not 0 < prior <= 1:
                    raise ValueError('invalid pronunciation prior')
                seen.add(phones); ids.add(pron['id']); total += prior
                node = 0
                for phone in phones:
                    token = self.vocabulary.index(phone)
                    if token == 0:
                        raise ValueError('blank in pronunciation')
                    if token not in self.nodes[node]:
                        self.nodes[node][token] = len(self.nodes)
                        self.nodes.append({}); self.ends.append([])
                    node = self.nodes[node][token]
                self.ends[node].append((wid, math.log(prior)))
            if abs(total-1) > 1e-5:
                raise ValueError('pronunciation priors must sum to one')

    def advance(self, frontier, token):
        if token == self.boundary_token:
            return frozenset({0}) if 0 in frontier else frozenset()
        result = set()
        for node in frontier:
            child = self.nodes[node].get(token)
            if child is not None:
                result.add(child)
                if self.ends[child]:
                    result.add(0)
        return frozenset(result)

    def segment(self, phones, beam, max_words):
        states = {0: {(): 0.0}}
        for offset in range(len(phones)+1):
            current = states.pop(offset, {})
            current = dict(sorted(current.items(), key=lambda x: (-x[1], x[0]))[:beam])
            if offset == len(phones):
                return current
            if phones[offset] == self.boundary_token:
                target = states.setdefault(offset + 1, {})
                for words, score in current.items():
                    target[words] = add(target.get(words, NEG), score)
                continue
            node = 0
            for end in range(offset, len(phones)):
                node = self.nodes[node].get(phones[end])
                if node is None:
                    break
                for wid, prior in self.ends[node]:
                    target = states.setdefault(end+1, {})
                    for words, score in current.items():
                        if len(words) < max_words:
                            key = words + (wid,)
                            target[key] = add(target.get(key, NEG), score+prior)
                    # Bound pending states too, not only states being expanded.
                    if len(target) > beam:
                        states[end+1] = dict(sorted(target.items(), key=lambda x: (-x[1], x[0]))[:beam])
        return {}


class StreamingDecoder:
    """Incremental probabilities in; provisional complete words out.

    Search state is bounded by beam and utterance caps. This Python reference
    is not a mobile latency claim. Call reset() at utterance boundaries.
    """
    def __init__(self, lexicon, beam_width=16, lexical_beam=32, max_steps=1000,
                 max_phones=256, max_words=64, soft_rest_boundaries=False,
                 rest_boundary_bonus=0.0):
        for value in (beam_width, lexical_beam, max_steps, max_phones, max_words):
            if not isinstance(value, int) or value < 1:
                raise ValueError('limits must be positive integers')
        self.lexicon = lexicon
        self.beam_width, self.lexical_beam = beam_width, lexical_beam
        self.max_steps, self.max_phones, self.max_words = max_steps, max_phones, max_words
        if not math.isfinite(rest_boundary_bonus):
            raise ValueError('REST boundary bonus must be finite')
        self.soft_rest_boundaries = soft_rest_boundaries
        self.rest_boundary_bonus = rest_boundary_bonus
        self.reset()

    def reset(self):
        self.beam = {(): (0.0, NEG)}
        self.frontiers = {(): frozenset({0})}
        self.steps = 0

    def accept(self, rows):
        for row in rows:
            row = tuple(float(x) for x in row)
            if (len(row) != len(self.lexicon.vocabulary)
                    or any(not math.isfinite(x) or x < 0 or x > 1 for x in row)
                    or abs(sum(row)-1) > 1e-5):
                raise ValueError('expected normalized finite checkpoint-vocabulary probability row')
            if self.steps >= self.max_steps:
                raise RuntimeError('utterance step budget exceeded; reset required')
            logs = [(i, math.log(p)) for i, p in enumerate(row) if p > 0]
            next_beam, frontiers = {}, {}
            def accumulate(q, slot, score, frontier):
                masses = next_beam.setdefault(q, [NEG, NEG])
                masses[slot] = add(masses[slot], score)
                frontiers[q] = frontier
            for q, (blank, nonblank) in self.beam.items():
                total = add(blank, nonblank)
                frontier = self.frontiers[q]
                for token, emission in logs:
                    if token == 0:
                        accumulate(q, 0, total+emission, frontier)
                        continue
                    if token == self.lexicon.boundary_token and self.soft_rest_boundaries:
                        # Preserve a no-boundary interpretation so uncertain REST evidence
                        # cannot prune an otherwise valid pronunciation path.
                        accumulate(q, 0, total + emission, frontier)
                    repeated = bool(q) and token == q[-1]
                    if repeated and nonblank != NEG:
                        accumulate(q, 1, nonblank+emission, frontier)
                    source = blank if repeated else total
                    if source == NEG:
                        continue
                    child = self.lexicon.advance(frontier, token)
                    if child:
                        if len(q) >= self.max_phones:
                            raise RuntimeError('phone budget exceeded; reset required')
                        bonus = (self.rest_boundary_bonus
                                 if token == self.lexicon.boundary_token else 0.0)
                        accumulate(q+(token,), 1, source+emission+bonus, child)
            ranked = sorted(next_beam, key=lambda q: (-add(*next_beam[q]), q))[:self.beam_width]
            self.beam = {q: tuple(next_beam[q]) for q in ranked}
            self.frontiers = {q: frontiers[q] for q in ranked}
            self.steps += 1

    def result(self, nbest=5):
        if not isinstance(nbest, int) or nbest < 1:
            raise ValueError('nbest must be positive')
        scores = {}
        for phones, masses in self.beam.items():
            for words, prior in self.lexicon.segment(phones, self.lexical_beam, self.max_words).items():
                scores[words] = add(scores.get(words, NEG), add(*masses)+prior)
        # Do not let a tiny empty path hide a nonempty incomplete best prefix.
        if scores and not any(scores) and next(iter(self.beam), ()):
            scores = {}
        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:nbest]
        return {'format': 'phoneme-decoder-runtime-0.1',
                'status': 'no_evidence' if not self.steps else ('ok' if ranked else 'no_lexical_path_in_beam'),
                'steps': self.steps, 'approximate_search': True,
                'candidates': [{'words': list(words), 'text': ' '.join(self.lexicon.words[w] for w in words),
                                'log_score': score} for words, score in ranked] if self.steps else []}


def probabilities_from_logits(logits, phones, valid_steps=None, batch_index=0):
    """Adapt VisualPhoneme [T,B,C] logits; torch stays producer-side."""
    from VisualPhoneme.visemes import VISUAL_GROUPS, VISUAL_GROUPS_WITH_REST
    if (tuple(phones) not in {PHONES, VISUAL_GROUPS, VISUAL_GROUPS_WITH_REST}
            or logits.ndim != 3
            or logits.shape[2] != len(phones) + 1):
        raise ValueError('expected checkpoint vocabulary and matching [T,B,C] logits')
    length = logits.shape[0] if valid_steps is None else valid_steps
    if not isinstance(length, int) or not 0 <= length <= logits.shape[0] or not 0 <= batch_index < logits.shape[1]:
        raise ValueError('invalid valid length or batch index')
    return logits[:length, batch_index].detach().float().softmax(-1).cpu().tolist()
