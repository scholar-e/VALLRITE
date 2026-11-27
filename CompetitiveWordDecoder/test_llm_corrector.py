import math
import unittest

from CompetitiveWordDecoder.llm_corrector import final_choice


class CorrectorTest(unittest.TestCase):
    def test_replaces_only_when_evidence_beats_margin(self):
        weights = {"ctc": 1.0, "ranker": 1.0, "words": 0.0, "unknown": 0.0, "lm": 1.0}
        chosen = {"words": ["a", "nice", "day"], "ctc": -10.0, "ranker": 0.0, "words_count": 3,
                  "unknown": 0, "lm": -20.0, "first_index": 0}
        clip = {"groups": [chosen]}
        better = {"words": ["a", "nice", "date"], "ctc": -9.0, "words_count": 3, "unknown": 0, "lm": -18.0}
        self.assertEqual(final_choice(clip, weights, 0.0, better), better["words"])      # +3 > 0
        self.assertEqual(final_choice(clip, weights, 5.0, better), chosen["words"])      # +3 < 5
        self.assertEqual(final_choice(clip, weights, math.inf, better), chosen["words"]) # never
        self.assertEqual(final_choice(clip, weights, -math.inf, None), chosen["words"])  # no correction


if __name__ == "__main__":
    unittest.main()
