import unittest
from PhonemeDecoder.rerank import rerank


class RerankTests(unittest.TestCase):
    def setUp(self):
        self.result = {"status": "ok", "candidates": [
            {"words": ["pat"], "text": "pat", "log_score": -1},
            {"words": ["bat"], "text": "bat", "log_score": -2},
            {"words": ["mat"], "text": "mat", "log_score": -20}]}

    def test_only_eligible_candidates_can_win(self):
        def score(record, indices):
            self.assertEqual(indices, [0, 1])
            self.assertNotIn("reference_words", record)
            return [-10, 0]
        out = rerank(self.result, ["B", "AE", "T"], score)
        self.assertEqual(out["candidates"][0]["text"], "bat")
        self.assertEqual(out["reranker"]["filtered"], 1)
        self.assertEqual(self.result["candidates"][0]["text"], "pat")

    def test_failure_preserves_ranking(self):
        out = rerank(self.result, [], lambda *_: [float("nan"), 0])
        self.assertEqual(out["candidates"], self.result["candidates"])
        self.assertEqual(out["reranker"]["status"], "fallback")

    def test_zero_weight_does_not_load_model(self):
        out = rerank(self.result, [], None, weight=0)
        self.assertEqual(out["candidates"], self.result["candidates"])
