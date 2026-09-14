import unittest

from PhonemeDecoder.evaluate_candidate_scores import evaluate


class CandidateScoreEvaluationTests(unittest.TestCase):
    def test_external_scores_rerank_and_report_oracle(self):
        records = [{"clip_id": "s9:x", "reference_words": ["bat"],
                    "candidates": [
                        {"text": "pat", "words": ["pat"], "log_score": -1.0},
                        {"text": "bat", "words": ["bat"], "log_score": -1.2}]}]
        scores = {("s9:x", 0): ("pat", -2.0),
                  ("s9:x", 1): ("bat", -0.1)}
        result = evaluate(records, scores, [0.0, 1.0], 5.0)
        self.assertEqual(result["baseline"]["word_errors"], 1)
        self.assertEqual(result["oracle_at_n"]["word_errors"], 0)
        self.assertEqual(result["results_by_reranker_weight"]["1.0"]["word_errors"], 0)

    def test_candidate_identity_is_validated(self):
        records = [{"clip_id": "s9:x", "reference_words": ["bat"],
                    "candidates": [{"text": "bat", "words": ["bat"],
                                    "log_score": -1.0}]}]
        with self.assertRaises(ValueError):
            evaluate(records, {("s9:x", 0): ("wrong", 0.0)}, [1.0], 5.0)
