"""Runtime correctness tests against exhaustive CTC fixtures."""
import json
import math
from pathlib import Path
import unittest
from PhonemeDecoder.decoder import Lexicon, StreamingDecoder, PHONES, VOCABULARY

ROOT = Path(__file__).resolve().parents[1]


def lexicon(entries):
    return Lexicon({'phone_inventory': PHONES, 'words': [
        {'id': word, 'text': word, 'pronunciations': [
            {'id': word, 'phones': phones, 'prior': 1}]} for word, phones in entries]})


def row(phone):
    return [float(p == phone) for p in VOCABULARY]


class DecoderTests(unittest.TestCase):
    def test_oracle_and_chunk_equivalence(self):
        lex = Lexicon(json.loads((ROOT/'examples/lexicon.json').read_text()))
        rows = json.loads((ROOT/'examples/request.json').read_text())['probabilities']
        whole = StreamingDecoder(lex, beam_width=512)
        whole.accept(rows)
        chunked = StreamingDecoder(lex, beam_width=512)
        for r in rows:
            chunked.accept([r])
        self.assertEqual(whole.result(), chunked.result())
        expected = json.loads((ROOT/'examples/response.json').read_text())['candidates']
        for actual, oracle in zip(whole.result()['candidates'], expected, strict=True):
            self.assertEqual(actual['text'], ' '.join(w['text'] for w in oracle['words']))
            self.assertAlmostEqual(actual['log_score'], oracle['base_score'])

    def test_repeated_phone_word_boundary(self):
        decoder = StreamingDecoder(lexicon([('p', ['P'])]))
        decoder.accept([row('P'), row('P')])
        self.assertEqual(decoder.result()['candidates'][0]['text'], 'p')
        decoder.accept([row('<blank>'), row('P')])
        self.assertEqual(decoder.result()['candidates'][0]['text'], 'p p')

    def test_homophones_and_shared_prefix(self):
        decoder = StreamingDecoder(lexicon([('a', ['P']), ('b', ['P']), ('c', ['P','AE'])]))
        decoder.accept([row('P')])
        self.assertEqual({c['text'] for c in decoder.result()['candidates']}, {'a','b'})
        decoder.accept([row('AE')])
        self.assertEqual(decoder.result()['candidates'][0]['text'], 'c')

    def test_empty_invalid_oov_reset_and_limits(self):
        decoder = StreamingDecoder(lexicon([('p', ['P'])]), max_steps=2)
        self.assertEqual(decoder.result()['status'], 'no_evidence')
        with self.assertRaises(ValueError):
            decoder.accept([[math.nan]*40])
        decoder.accept([row('<blank>')])
        self.assertEqual(decoder.result()['candidates'][0]['text'], '')
        decoder.accept([row('B')])
        self.assertEqual(decoder.result()['status'], 'no_lexical_path_in_beam')
        with self.assertRaises(RuntimeError):
            decoder.accept([row('P')])
        decoder.reset()
        decoder.accept([row('P')])
        self.assertEqual(decoder.result()['candidates'][0]['text'], 'p')

    def test_adapter_layout_and_valid_length(self):
        try:
            import torch
        except ImportError:
            self.skipTest('torch producer dependency unavailable')
        from PhonemeDecoder import probabilities_from_logits
        logits = torch.zeros(3, 2, 40)
        logits[:, 1, 27] = 5
        rows = probabilities_from_logits(logits, PHONES, 2, 1)
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(sum(rows[0]), 1, places=6)
        self.assertEqual(max(range(40), key=rows[0].__getitem__), 27)
        with self.assertRaises(ValueError):
            probabilities_from_logits(logits, PHONES[::-1])
