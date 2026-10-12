"""Unit tests for synthetic_noise.corrupt()."""
import unittest
from CompetitiveWordDecoder.synthetic_noise import corrupt, PHONES, mixed_training, fixed_evaluation
from CompetitiveWordDecoder.test_vallr_clean import FakeTokenizer
from CompetitiveWordDecoder.train_vallr_clean import encode_eos


class SyntheticNoiseTests(unittest.TestCase):
    def test_zero_rate_returns_unchanged(self):
        """rate=0 returns identical row."""
        row = {"prompt": "hello <PHONEMES>\nAA AE AH\n</PHONEMES> world"}
        result = corrupt(row, 0.0, 42)
        self.assertEqual(result["prompt"], row["prompt"])

    def test_deterministic_seed(self):
        """Same seed produces identical output."""
        row = {"prompt": "test <PHONEMES>\nAA AE\n</PHONEMES>"}
        r1 = corrupt(row, 0.2, 12345)
        r2 = corrupt(row, 0.2, 12345)
        self.assertEqual(r1["prompt"], r2["prompt"])

    def test_deterministic_seed_differs_from_none(self):
        """Different seeds produce different outputs."""
        row = {"prompt": "test <PHONEMES>\nAA AE\n</PHONEMES>"}
        r1 = corrupt(row, 0.2, 1)
        r2 = corrupt(row, 0.2, 2)
        self.assertNotEqual(r1["prompt"], r2["prompt"])

    def test_tags_targets_unk_preserved(self):
        """Tags, targets, and UNK(fake) are never corrupted."""
        row = {
            "prompt": "<S2S>\n<PHONEMES>\nAA AE AH\n</PHONEMES>\n<TEXT>\ntest\n</TEXT>",
            "target": "test",
            "unknown_words": 0,
        }
        result = corrupt(row, 0.3, 99)
        self.assertEqual(result["target"], row["target"])
        self.assertEqual(result["unknown_words"], row["unknown_words"])
        self.assertEqual(result["prompt"].split('<PHONEMES>')[0], row["prompt"].split('<PHONEMES>')[0])
        self.assertEqual(result["prompt"].split('</PHONEMES>')[1], row["prompt"].split('</PHONEMES>')[1])

    def test_unknown_spelling_preserved(self):
        """UNK(fake) spelling is never altered."""
        row = {"prompt": "<PHONEMES>\nAA UNK(zzzunknown)\n</PHONEMES>"}
        result = corrupt(row, 0.5, 42)
        self.assertIn("UNK(zzzunknown)", result["prompt"])

    def test_positive_edit_count_at_02_rate(self):
        """At rate=0.2 with 40 known phones, expect ~8 edits."""
        row = {"prompt": "<PHONEMES>\n" + " ".join(PHONES[:40]) + "\n</PHONEMES>"}
        result = corrupt(row, 0.2, 1)
        ops = result["corruption"]["operations"]
        self.assertEqual(sum(ops.values()), round(len(PHONES)*.2))
        self.assertGreater(result["corruption"]["phone_edits"], 0)

    def test_balanced_training_and_fixed_evaluation(self):
        rows = [{"prompt": '<S2S>\n<PHONEMES>\n' + ' '.join(PHONES) +
                 f' UNK(word{i})\n</PHONEMES>\n<TEXT>\n', 'target': f'word{i}',
                 'unknown_words': 1} for i in range(10)]
        tokenizer = FakeTokenizer()
        mixed = mixed_training(rows, tokenizer, encode_eos)
        self.assertEqual(len(mixed), 10)
        self.assertEqual(sum('corruption' in row for row in mixed), 5)
        self.assertTrue(all(encode_eos(row, tokenizer) is not None for row in mixed))
        evaluation = fixed_evaluation(rows, tokenizer, encode_eos, 4)
        self.assertEqual(evaluation, fixed_evaluation(rows, tokenizer, encode_eos, 4))
        targets = [r['target'] for r in evaluation['clean']]
        self.assertEqual(targets, [r['target'] for r in evaluation['noise10']])
        self.assertEqual(targets, [r['target'] for r in evaluation['noise20']])

    def test_invalid_rate_raises_valueerror(self):
        """Rates outside [0,1] raise ValueError."""
        row = {"prompt": "test <PHONEMES>\nAA\n</PHONEMES>"}
        with self.assertRaises(ValueError):
            corrupt(row, -0.1, 1)
        with self.assertRaises(ValueError):
            corrupt(row, 1.5, 1)

    def test_input_row_unmodified(self):
        """Original row dict is not mutated."""
        original = {"prompt": "test <PHONEMES>\nAA AE\n</PHONEMES>"}
        original_copy = original.copy()
        result = corrupt(original, 0.2, 42)
        self.assertEqual(original, original_copy)

    def test_corruption_metadata_present(self):
        """Result includes corruption metadata."""
        row = {"prompt": "test <PHONEMES>\nAA AE\n</PHONEMES>"}
        result = corrupt(row, 0.15, 7)
        self.assertIn("corruption", result)
        self.assertIn("rate", result["corruption"])
        self.assertIn("seed", result["corruption"])
        self.assertIn("operations", result["corruption"])


if __name__ == "__main__":
    unittest.main()
