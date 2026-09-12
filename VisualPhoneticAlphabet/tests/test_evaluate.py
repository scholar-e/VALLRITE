import unittest

from VisualPhoneticAlphabet.evaluate import score


class ScoreTests(unittest.TestCase):
    def test_exact(self):
        self.assertEqual(score(['B', 'IY'], ['B', 'IY'])['per'], 0)

    def test_empty_prediction_is_all_deletions(self):
        result = score(['B', 'IY'], [])
        self.assertEqual((result['deletions'], result['per']), (2, 1))

    def test_substitution_and_insertion(self):
        result = score(['B', 'IY'], ['B', 'AA', 'T'])
        self.assertEqual(result['errors'], 2)
        self.assertEqual(result['substitutions'], 1)
        self.assertEqual(result['insertions'], 1)

    def test_per_can_exceed_one(self):
        self.assertEqual(score(['B'], ['AA', 'T', 'K'])['per'], 3)

    def test_empty_reference_rejected(self):
        with self.assertRaises(ValueError):
            score([], ['B'])
