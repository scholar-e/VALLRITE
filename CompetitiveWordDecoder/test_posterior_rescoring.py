import math
import unittest

import torch

from CompetitiveWordDecoder.posterior_fused_decoding import (
    extend_many, initial_ctc_state, sequence_log_probability)
from CompetitiveWordDecoder.posterior_rescoring import ctc_log_likelihood, select


class PosteriorRescoringTest(unittest.TestCase):
    def test_ctc_likelihood_matches_enumerated_alignments(self):
        torch.manual_seed(0)
        log_probabilities = torch.randn(4, 3).log_softmax(-1)
        total = []
        for path in torch.cartesian_prod(*[torch.arange(3)] * 4).tolist():
            collapsed, previous = [], None
            for label in path:
                if label != previous and label != 0:
                    collapsed.append(label)
                previous = label
            if collapsed == [1, 2]:
                total.append(sum(float(log_probabilities[t, label]) for t, label in enumerate(path)))
        expected = max(total) + math.log(sum(math.exp(v - max(total)) for v in total))
        self.assertAlmostEqual(ctc_log_likelihood(log_probabilities, [1, 2]), expected, places=6)
        self.assertEqual(ctc_log_likelihood(log_probabilities[:1], [1, 2]), -math.inf)

    def test_select_uses_weights_and_breaks_ties_by_rank(self):
        groups = [{"ctc": -5.0, "ranker": -1.0, "words_count": 3, "unknown": 0, "first_index": 0},
                  {"ctc": -2.0, "ranker": -3.0, "words_count": 3, "unknown": 0, "first_index": 1}]
        self.assertIs(select(groups, {"ctc": 1, "ranker": 0, "words": 0, "unknown": 0}), groups[1])
        self.assertIs(select(groups, {"ctc": 0, "ranker": 1, "words": 0, "unknown": 0}), groups[0])
        self.assertIs(select(groups, {"ctc": 1, "ranker": 1.5, "words": 0, "unknown": 0}), groups[0])
        self.assertIs(select(groups, {"ctc": 0, "ranker": 0, "words": 1, "unknown": 0}), groups[0])

    def test_incremental_extension_matches_whole_sequence_likelihood(self):
        torch.manual_seed(1)
        log_probabilities = torch.randn(12, 5).log_softmax(-1).double()
        words = [[1, 2], [2, 2, 3], [], [4]]
        start = initial_ctc_state(log_probabilities)[None].expand(2, -1, -1)
        states, last = start, torch.tensor([-1, -1])
        prefix = torch.zeros(2, dtype=torch.double)
        sequences = [[], []]
        for word in words:  # second hypothesis receives words in a different grouping
            states, last, prefix = extend_many(log_probabilities, states, last, prefix,
                                               [word, word[:1]])
            sequences[0] += word
            sequences[1] += word[:1]
        for row, sequence in enumerate(sequences):
            self.assertAlmostEqual(float(sequence_log_probability(states)[row:row + 1][0]),
                                   ctc_log_likelihood(log_probabilities.float(), sequence), places=4)
        self.assertGreaterEqual(float(prefix[0]), float(sequence_log_probability(states)[0]))


if __name__ == "__main__":
    unittest.main()
