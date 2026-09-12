import unittest
import numpy as np
from VisualPhoneticAlphabet.measure_landmarks import aligned_labels, LIP_IDS, BASE_IDS


class AlignmentLabelsTests(unittest.TestCase):
    def test_stress_and_boundaries(self):
        times=np.array([0,19,20,40,79,80,100,120,159,160,180])
        labels=aligned_labels(times,[(0,.1,'AA1'),(.1,.18,'B')],['AA','B'])
        np.testing.assert_array_equal(labels,[-1,-1,0,0,0,-1,-1,1,1,-1,-1])

    def test_unknown_and_short_intervals_are_excluded(self):
        times=np.array([0,20,40,60,80])
        labels=aligned_labels(times,[(0,.02,'AA1'),(.02,.06,'sil'),(.06,.1,'spn')],['AA'])
        self.assertTrue(np.all(labels==-1))

    def test_point_groups(self):
        self.assertEqual(len(LIP_IDS),40)
        self.assertEqual(len(set(LIP_IDS)),40)
        self.assertTrue(set(BASE_IDS)<set(LIP_IDS))
        self.assertNotIn(152,LIP_IDS)
