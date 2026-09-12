import json
from pathlib import Path
import tempfile
import unittest

import torch

from VisualPhoneme.data import (GridClips, ctc_prefix_beam_search,
                                      fit_bigram_log_probs, greedy_decode,
                                      load_video, normalize_phone, phoneme_target,
                                      transform_landmarks)
from VisualPhoneme.model import (CompactFusionVisualPhoneme,
                                       CompactGatedFusionVisualPhoneme,
                                       CompactLandmarkPhoneme,
                                       CompactTongueGatedFusionVisualPhoneme,
                                       CompactVisualPhoneme)
from VisualPhoneme.train import edit_totals


class VisualPhonemeTests(unittest.TestCase):
    def test_model_preserves_time_and_is_small(self):
        model = CompactVisualPhoneme()
        output = model(torch.zeros(2, 9, 1, 64, 64))
        self.assertEqual(output.shape, (9, 2, 40))
        self.assertLess(model.parameter_count, 500_000)

    def test_fusion_model_handles_missing_landmarks(self):
        model = CompactFusionVisualPhoneme()
        video = torch.zeros(2, 9, 1, 64, 64)
        landmarks = torch.randn(2, 9, 41, 2)
        landmarks[0, 3] = torch.nan
        mask = torch.isfinite(landmarks).all(dim=(2, 3))
        output = model(video, landmarks, mask)
        self.assertEqual(output.shape, (9, 2, 40))
        self.assertTrue(torch.isfinite(output).all())

    def test_coordinate_model_handles_missing_landmarks(self):
        model = CompactLandmarkPhoneme()
        landmarks = torch.randn(2, 9, 41, 2)
        landmarks[0, 3] = torch.nan
        mask = torch.isfinite(landmarks).all(dim=(2, 3))
        output = model(landmarks, mask)
        self.assertEqual(output.shape, (9, 2, 40))
        self.assertTrue(torch.isfinite(output).all())
        self.assertLess(model.parameter_count, 150_000)

    def test_motion_features_and_gated_fusion(self):
        points = torch.zeros(6, 41, 2)
        points[:, :, 0] = torch.arange(6).reshape(-1, 1)
        features, mask = transform_landmarks(points, "clip-centered", "motion")
        self.assertEqual(features.shape, (6, 41, 10))
        self.assertTrue(mask.all())
        self.assertTrue(torch.equal(features[1, :, 2], torch.ones(41)))
        model = CompactGatedFusionVisualPhoneme(coordinate_dimensions=10)
        output = model(torch.zeros(2, 6, 1, 64, 64), features.repeat(2, 1, 1, 1),
                       mask.repeat(2, 1))
        self.assertEqual(output.shape, (6, 2, 40))
        self.assertLess(model.image_gate, 0.05)

    def test_inner_mouth_branch_is_observability_gated(self):
        model = CompactTongueGatedFusionVisualPhoneme(
            coordinate_dimensions=10, image_gate_probability=0.05,
            inner_mouth_gate_probability=0.05,
        )
        video = torch.zeros(2, 6, 1, 64, 64)
        landmarks = torch.zeros(2, 6, 41, 10)
        landmarks[:, :, 0, 0] = -0.5
        landmarks[:, :, 1, 0] = 0.5
        landmarks[1, :, 2, 1] = -0.1
        landmarks[1, :, 3, 1] = 0.1
        mask = torch.ones(2, 6, dtype=torch.bool)
        observability = model.oral_observability(landmarks, mask)
        self.assertTrue(torch.equal(observability[0], torch.zeros(6)))
        self.assertTrue(torch.all(observability[1] > 0))
        self.assertEqual(model.inner_mouth_crop(video).shape, (2, 6, 1, 32, 40))
        output = model(video, landmarks, mask)
        self.assertEqual(output.shape, (6, 2, 40))
        self.assertTrue(torch.isfinite(output).all())
        self.assertLess(model.parameter_count, 350_000)

    def test_bigram_model_is_smoothed(self):
        transitions = fit_bigram_log_probs([[1, 2], [1, 2], [2, 1]], 3)
        self.assertEqual(transitions.shape, (3, 3))
        self.assertTrue(torch.isfinite(transitions[:, 1:]).all())
        self.assertGreater(transitions[1, 2], transitions[1, 1])

    def test_coordinate_modes_are_validated(self):
        with self.assertRaisesRegex(ValueError, "invalid coordinate mode"):
            GridClips(Path("unused"), "train", coordinate_mode="unknown")

    def test_frame_cache_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            cached = Path(directory) / "frames.npy"
            expected = torch.arange(18, dtype=torch.uint8).reshape(2, 3, 3).numpy()
            with cached.open("wb") as output:
                import numpy as np
                np.save(output, expected, allow_pickle=False)
            actual = load_video(Path("not-read.mpg"), (0, 0, 1, 1), 3, cached)
            self.assertEqual(actual.shape, (2, 1, 3, 3))
            self.assertTrue(torch.equal(actual.mul(255).byte().squeeze(1),
                                        torch.from_numpy(expected)))

    def test_phone_normalization_and_alignment_target(self):
        self.assertEqual(normalize_phone("AH0"), "AH")
        self.assertIsNone(normalize_phone("sil"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "alignment.json"
            path.write_text(json.dumps({"tiers": {"phones": {"entries": [
                [0, 0.1, "sil"], [0.1, 0.2, "B"], [0.2, 0.3, "IY1"]
            ]}}}))
            self.assertEqual(len(phoneme_target(str(path))), 2)

    def test_ctc_greedy_collapse(self):
        ids = torch.tensor([[0, 2], [1, 2], [1, 0], [0, 3], [1, 3]])
        logits = torch.full((5, 2, 40), -10.0)
        logits.scatter_(2, ids.unsqueeze(-1), 10.0)
        self.assertEqual(greedy_decode(logits, torch.tensor([5, 4])), [[1, 1], [2, 3]])

    def test_ctc_prefix_beam_returns_ranked_sequences(self):
        logits = torch.tensor([[0.0, 5.0, 0.0],
                               [5.0, 0.0, 0.0],
                               [0.0, 0.0, 5.0]])
        candidates = ctc_prefix_beam_search(logits.log_softmax(-1),
                                            beam_width=6, top_n=3)
        self.assertEqual(candidates[0][0], [1, 2])
        self.assertGreaterEqual(candidates[0][1], candidates[1][1])

    def test_edit_distance(self):
        self.assertEqual(edit_totals([1, 2, 3], [1, 4, 3]), (1, 3))
        self.assertEqual(edit_totals([1, 2], [1, 2, 3]), (1, 2))


if __name__ == "__main__":
    unittest.main()
