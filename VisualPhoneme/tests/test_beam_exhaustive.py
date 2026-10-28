"""Check beam scores against all alignments, without duplicating the recurrence."""
from collections import defaultdict
from itertools import product, groupby
import math
import unittest
import torch
from VisualPhoneme.data import ctc_prefix_beam_search


class ExhaustiveBeamTests(unittest.TestCase):
    def test_exhaustive(self):
        logp = torch.tensor([[.3,.5,.2],[.6,.1,.3],[.2,.3,.5]]).log()
        lm = torch.tensor([[.1,.6,.3],[.1,.2,.7],[.1,.8,.1]]).log()
        for weight in (0.,.5):
            with self.subTest(lm_weight=weight):
                mass = defaultdict(float)
                for alignment in product(range(3),repeat=3):
                    collapsed = tuple(k for k,_ in groupby(alignment) if k != 0)
                    score = sum(float(logp[t,k]) for t,k in enumerate(alignment))
                    previous = 0
                    for token in collapsed:
                        score += weight*float(lm[previous,token])
                        previous = token
                    mass[collapsed] += math.exp(score)
                actual = ctc_prefix_beam_search(logp,beam_width=100,top_n=len(mass),
                                               transition_log_probs=lm,lm_weight=weight)
                self.assertEqual({tuple(x) for x,_ in actual},set(mass))
                for sequence,score in actual:
                    self.assertAlmostEqual(score,math.log(mass[tuple(sequence)]),places=5)
                self.assertIn((1,1),mass)


if __name__ == '__main__':
    unittest.main()
