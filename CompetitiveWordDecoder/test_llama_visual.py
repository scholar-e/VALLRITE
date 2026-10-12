"""Unit tests for score_records in evaluate_llama_visual.py."""
import unittest
from CompetitiveWordDecoder.evaluate_llama_visual import score_records, input_items


class TestScoreRecords(unittest.TestCase):
    """Tests for score_records function."""

    def test_reference_mode_uses_only_reference_phones(self):
        rows = [{'clip_id': 'r', 'reference_words': ['secret'],
                 'reference_phones': ['K', 'AE', 'T'],
                 'phone_hypotheses': [{'phones': ['WRONG']}]}]
        items = input_items(rows, True)
        self.assertEqual(len(items), 1)
        self.assertIn('K AE T', items[0][2])
        self.assertNotIn('secret', items[0][2])
        self.assertNotIn('WRONG', items[0][2])
        result = score_records(rows, [[{'text': 'secret', 'hit_generation_cap': False}]], True)
        self.assertEqual(result['top_1']['wer'], 0)
        self.assertEqual(result['exact_hits'], 1)
        self.assertNotIn('oracle_at_5', result)

    def test_reference_mode_rejects_missing_phones(self):
        with self.assertRaises(ValueError):
            input_items([{'reference_words': ['cat']}], True)

    def test_top1_vs_oracle_minimum(self):
        """Oracle WER must be <= top-1 WER (oracle uses min of 5)."""
        records = [
            {
                "clip_id": "c1",
                "reference_words": ["cat", "dog"],
                "phone_hypotheses": [[{"text": "cat dog"}, {"text": "cat dog"}]]
            }
        ]
        generated = [
            [{"text": "wrong answer", "hit_generation_cap": 0}, {"text": "cat dog", "hit_generation_cap": 0},
             {"text": "cat dog", "hit_generation_cap": 0}, {"text": "cat dog", "hit_generation_cap": 0},
             {"text": "cat dog", "hit_generation_cap": 0}]
        ]
        result = score_records(records, generated)
        self.assertLessEqual(result["oracle_at_5"]["wer"], result["top_1"]["wer"])
        self.assertEqual(result["top_1"]["wer"], 1.0)
        self.assertEqual(result["oracle_at_5"]["wer"], 0.0)

    def test_empty_text_counts_deletions_insertions(self):
        """Empty output counts as deletions; insertions permit WER > 1."""
        records = [
            {
                "clip_id": "c2",
                "reference_words": ["hello world"],
                "phone_hypotheses": [[{"text": ""}, {"text": "xyz"}]]
            }
        ]
        generated = [
            [{"text": "", "hit_generation_cap": 0}, {"text": "xyz", "hit_generation_cap": 1},
             {"text": "xyz", "hit_generation_cap": 1}, {"text": "xyz", "hit_generation_cap": 1},
             {"text": "xyz", "hit_generation_cap": 1}]
        ]
        result = score_records(records, generated)
        # Empty string vs "hello world" gives exactly WER=1.0 (all deletions)
        self.assertEqual(result["top_1"]["wer"], 1.0)
        generated[0][0]['text'] = 'hello world plus several extra words'
        self.assertGreater(score_records(records, generated)['top_1']['wer'], 1.0)

    def test_incomplete_generations_raise_valueerror(self):
        """Exactly 5 generations required; incomplete raises ValueError."""
        records = [
            {
                "clip_id": "c3",
                "reference_words": ["test"],
                "phone_hypotheses": [[{"text": "test"}]]
            }
        ]
        generated = [
            [{"text": "test", "hit_generation_cap": 0}, {"text": "test", "hit_generation_cap": 1},
             {"text": "test", "hit_generation_cap": 1}, {"text": "test", "hit_generation_cap": 1}]
        ]
        with self.assertRaises(ValueError):
            score_records(records, generated)

    def test_clip_id_and_reference_words_in_detail(self):
        """Records detail includes clip_id and reference_words."""
        records = [
            {
                "clip_id": "c4",
                "reference_words": ["foo bar"],
                "phone_hypotheses": [[{"text": "foo bar"}]]
            }
        ]
        generated = [
            [{"text": "foo bar", "hit_generation_cap": 0}, {"text": "foo bar", "hit_generation_cap": 1},
             {"text": "foo bar", "hit_generation_cap": 1}, {"text": "foo bar", "hit_generation_cap": 1},
             {"text": "foo bar", "hit_generation_cap": 1}]
        ]
        result = score_records(records, generated)
        detail = result["records_detail"][0]
        self.assertEqual(detail["clip_id"], "c4")
        self.assertIn("reference_words", detail)


if __name__ == "__main__":
    unittest.main()
