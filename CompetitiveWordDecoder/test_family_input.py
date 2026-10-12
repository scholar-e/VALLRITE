"""Contract tests for boundary-free family sentence inputs."""
import unittest
from CompetitiveWordDecoder.family_input import family_codes, family_example, FAMILY_CODES
from CompetitiveWordDecoder.evaluate_llama_visual import input_items
from VisualPhoneme.data import PHONEMES


class FamilyInputs(unittest.TestCase):
    def test_explicit_alternatives_are_identical_within_family(self):
        self.assertEqual(family_codes(['B', 'M', 'P'], 'alternatives'), ['[B/M/P]'] * 3)
        self.assertEqual(family_codes(['ER'], 'alternatives'), ['[ER]'])
        with self.assertRaises(ValueError):
            family_codes(['B'], 'invalid')

    def test_alternative_training_evaluation_agree(self):
        row = dict(prompt='<S2S>\n<PHONEMES>\nB M P\n</PHONEMES>\n<TEXT>\n',
                   target='private target', unknown_words=0)
        expected = family_example(row, 'alternatives')['prompt']
        records = [dict(reference_phones=['B', 'M', 'P'], reference_words=['private', 'target'])]
        self.assertEqual(input_items(records, True, True, 'alternatives')[0][2], expected)
        self.assertNotIn('private', expected)

    def test_inventory(self):
        self.assertEqual(len(PHONEMES), 39)
        self.assertEqual(set(family_codes(PHONEMES)), set('ABCDEFGHIJKLMNOPQ'))
        self.assertEqual(len(FAMILY_CODES), 17)

    def test_repeated_families(self):
        self.assertEqual(family_codes(['B', 'M', 'P']), ['A', 'A', 'A'])

    def test_invalid_phone(self):
        with self.assertRaises(KeyError):
            family_codes(['BOGUS'])

    def test_training_example(self):
        row = dict(prompt='<S2S>\n<PHONEMES>\nB M P\n</PHONEMES>\n<TEXT>\n',
                   target='secret target words', unknown_words=0)
        converted = family_example(row)
        self.assertEqual(converted['target'], row['target'])
        self.assertEqual(converted['prompt'], row['prompt'].replace('B M P', 'A A A'))
        self.assertNotIn('secret', converted['prompt'])
        self.assertIsNone(family_example(dict(row, unknown_words=1)))
        self.assertIsNone(family_example(dict(row, prompt=row['prompt'].replace('B M P', '#'))))

    def test_evaluation_matches_training(self):
        rows = [dict(clip_id='a', reference_words=['secret'], reference_phones=['B', 'M', 'P'])]
        items = input_items(rows, True, True)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0][:2], (0, 0))
        self.assertIn('\nA A A\n', items[0][2])
        self.assertNotIn('secret', items[0][2])


if __name__ == '__main__':
    unittest.main()
