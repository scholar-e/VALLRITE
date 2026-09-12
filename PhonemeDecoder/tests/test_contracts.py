"""Check design fixtures, not a production decoder implementation."""
from copy import deepcopy
import hashlib
import itertools
import json
import math
from pathlib import Path
import unittest

import jsonschema

ROOT = Path(__file__).resolve().parents[1]


def read(relative):
    return json.loads((ROOT / relative).read_text())


def collapse(path):
    return [phone for phone, _ in itertools.groupby(path) if phone != '<blank>']


class ContractFixtureTests(unittest.TestCase):
    def setUp(self):
        self.request = read('examples/request.json')
        self.response = read('examples/response.json')
        self.lexicon = read('examples/lexicon.json')

    def test_schemas_and_examples(self):
        for name in ('request', 'response', 'lexicon'):
            schema = read(f'schemas/{name}.schema.json')
            jsonschema.Draft202012Validator.check_schema(schema)
            jsonschema.validate(read(f'examples/{name}.json'), schema)

    def test_full_ctc_contract_rejects_other_formats(self):
        schema = read('schemas/request.schema.json')
        sparse = deepcopy(self.request)
        sparse['probabilities'][0] = sparse['probabilities'][0][:5]
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(sparse, schema)
        classifier = deepcopy(self.request)
        classifier['kind'] = 'frame_classifier'
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(classifier, schema)

    def test_failure_cannot_contain_word_candidates(self):
        wrong = deepcopy(self.response)
        wrong['status'] = 'no_evidence'
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(wrong, read('schemas/response.schema.json'))

    def test_input_semantics_and_hashes(self):
        request = self.request
        rows = request['probabilities']
        self.assertEqual(request['producer']['valid_steps'], len(rows))
        self.assertEqual(len(request['step_times_ms']), len(rows))
        self.assertEqual(len(request['observed_steps']), len(rows))
        self.assertTrue(all(b > a for a, b in zip(request['step_times_ms'], request['step_times_ms'][1:])))
        for row in rows:
            self.assertTrue(all(math.isfinite(p) for p in row))
            self.assertAlmostEqual(sum(row), 1.0)
        self.assertEqual(request['resources']['lexicon']['sha256'], hashlib.sha256((ROOT/'examples/lexicon.json').read_bytes()).hexdigest())
        self.assertEqual(self.response['request_sha256'], hashlib.sha256((ROOT/'examples/request.json').read_bytes()).hexdigest())
        for word in self.lexicon['words']:
            self.assertAlmostEqual(sum(p['prior'] for p in word['pronunciations']), 1.0)
            self.assertEqual(len({tuple(p['phones']) for p in word['pronunciations']}), len(word['pronunciations']))

    def test_inconsistent_configuration_rejected(self):
        schema = read('schemas/request.schema.json')
        for field in ('lm_weight', 'reranker_weight'):
            request = deepcopy(self.request)
            request['scoring'][field] = 1
            with self.assertRaises(jsonschema.ValidationError):
                jsonschema.validate(request, schema)
        for change in ({'temperature': 2}, {'status': 'temperature_scaled'}):
            request = deepcopy(self.request)
            request['calibration'].update(change)
            with self.assertRaises(jsonschema.ValidationError):
                jsonschema.validate(request, schema)

    def test_pre_search_error_envelope(self):
        response = deepcopy(self.response)
        response.update(status='invalid_input', request_id=None,
                        request_sha256=None, candidates=[])
        response['diagnostics'].update(search_mode='not_applicable',
                                      retained_prefix_log_mass=None,
                                      empty_prefix_log_mass=None,
                                      error_code='malformed_json',
                                      failure_reason='Request could not be parsed.')
        jsonschema.validate(response, read('schemas/response.schema.json'))

    def test_repeat_semantics(self):
        self.assertEqual(collapse(['P', 'P']), ['P'])
        self.assertEqual(collapse(['P', '<blank>', 'P']), ['P', 'P'])
        self.assertEqual(collapse(['<blank>', '<blank>']), [])

    def test_exhaustive_ctc_arithmetic(self):
        choices = [[(phone, p) for phone, p in zip(self.request['vocabulary'], row) if p > 0]
                   for row in self.request['probabilities']]
        masses = {}
        count = 0
        for path in itertools.product(*choices):
            count += 1
            key = tuple(collapse([phone for phone, _ in path]))
            masses[key] = masses.get(key, 0.0) + math.prod(p for _, p in path)
        self.assertEqual(count, 216)
        self.assertAlmostEqual(sum(masses.values()), 1.0)
        by_word = {w['id']: w for w in self.lexicon['words']}
        candidates = self.response['candidates']
        self.assertEqual([c['id'] for c in candidates], ['pat', 'bat', 'mat', 'empty'])
        for candidate in candidates:
            path = candidate['representative_path']
            expected = math.log(masses[tuple(path['phones'])])
            self.assertAlmostEqual(path['ctc_log_mass'], expected)
            self.assertAlmostEqual(path['weighted_total'], expected)
            self.assertAlmostEqual(candidate['base_score'], expected)
            self.assertAlmostEqual(candidate['final_score'], expected)
            self.assertEqual(path['word_count'], len(candidate['words']))
            phones = []
            for word, pronunciation_id in zip(candidate['words'], path['pronunciation_ids'], strict=True):
                entry = by_word[word['word_id']]
                self.assertEqual(word['text'], entry['text'])
                pronunciation = next(p for p in entry['pronunciations'] if p['id'] == pronunciation_id)
                phones.extend(pronunciation['phones'])
            self.assertEqual(phones, path['phones'])
