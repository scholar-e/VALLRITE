import unittest
from CompetitiveWordDecoder.test_vallr_clean import FakeTokenizer, encode_eos
from CompetitiveWordDecoder.synthetic_noise import PHONES, mixed_training


class NoiseMixtureTests(unittest.TestCase):
    def _row(self, phones=" ".join(PHONES[:10])):
        return {"prompt": f"<PHONEMES>\n{phones}\n</PHONEMES>", "target": "test", "unknown_words": 0}

    def test_default_yields_four(self):
        rows = [self._row() for _ in range(8)]
        r = mixed_training(rows, FakeTokenizer(), encode_eos, .5)
        self.assertEqual(sum(1 for x in r if "corruption" in x), 4)

    def test_75_yields_two(self):
        rows = [self._row() for _ in range(8)]
        r = mixed_training(rows, FakeTokenizer(), encode_eos, .75)
        self.assertEqual(sum(1 for x in r if "corruption" in x), 2)

    def test_zero_yields_all(self):
        rows = [self._row() for _ in range(8)]
        r = mixed_training(rows, FakeTokenizer(), encode_eos, .0)
        self.assertEqual(sum(1 for x in r if "corruption" in x), 8)

    def test_one_yields_none(self):
        rows = [self._row() for _ in range(8)]
        r = mixed_training(rows, FakeTokenizer(), encode_eos, 1.0)
        self.assertEqual(sum(1 for x in r if "corruption" in x), 0)

    def test_targets_unchanged(self):
        base = self._row()
        for frac in [.5, .75, .0, 1.0]:
            rows = [base for _ in range(8)]
            r = mixed_training(rows, FakeTokenizer(), encode_eos, frac)
            for x in r:
                self.assertEqual(x["target"], base["target"])

    def test_row_count_unchanged(self):
        counts = {frac: len(mixed_training([self._row() for _ in range(8)], FakeTokenizer(), encode_eos, frac))
                  for frac in [.5, .75, .0]}
        self.assertEqual(counts[.5], counts[.75])
        self.assertEqual(counts[.5], counts[.0])

    def test_noisy_subset_identical(self):
        rows = [self._row(' '.join(PHONES[:10]) + f' UNK(word{i})') for i in range(8)]
        f5 = mixed_training(rows, FakeTokenizer(), encode_eos, .5)
        f75 = mixed_training(rows, FakeTokenizer(), encode_eos, .75)
        n5 = [x for x in f5 if "corruption" in x]
        n75 = [x for x in f75 if "corruption" in x]
        for r in n75:
            self.assertIn(r["prompt"], [x["prompt"] for x in n5])
            c1 = r.get("corruption", {})
            c2 = next(x.get("corruption", {}) for x in n5 if x["prompt"] == r["prompt"])
            self.assertEqual(c1, c2)

    def test_invalid_fraction(self):
        rows = [self._row() for _ in range(8)]
        with self.assertRaises(ValueError):
            mixed_training(rows, FakeTokenizer(), encode_eos, -0.1)
        with self.assertRaises(ValueError):
            mixed_training(rows, FakeTokenizer(), encode_eos, 1.5)
        with self.assertRaises(ValueError):
            mixed_training(rows, FakeTokenizer(), encode_eos, float('nan'))


