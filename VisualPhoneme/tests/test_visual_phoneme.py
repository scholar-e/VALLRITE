import json
from pathlib import Path
import pickle
import tempfile
import unittest

import numpy as np
import torch

from VisualPhoneme.data import (GridClips, collate_aligned_landmark_clips,
                                      ctc_prefix_beam_search,
                                      fit_bigram_log_probs, fit_trigram_log_probs,
                                      greedy_decode,
                                      load_video, normalize_phone, phoneme_target,
                                      transform_landmarks)
from VisualPhoneme.model import (AutoAvsrFusionVisualPhoneme,
                                       CompactFusionVisualPhoneme,
                                       CompactGatedFusionVisualPhoneme,
                                       CompactLandmarkPhoneme,
                                       CompactTongueGatedFusionVisualPhoneme,
                                       CompactVisualPhoneme,
                                       LargeGatedFusionVisualPhoneme,
                                       LargeVisualPhoneme,
                                       LargeTransformerFusionVisualPhoneme)
from VisualPhoneme.lrs3_data import (LRS3_LANDMARK_POINTS, Lrs3Clips,
                                     eye_normalize_lrs3, frontalize_lrs3,
                                     load_lrs3_landmarks)
from VisualPhoneme.train import edit_totals, upsample_ctc_logits
from VisualPhoneme.visemes import (PHONE_TO_VISUAL_GROUP, VISUAL_PHONE_GROUPS,
                                   PHONE_ID_TO_VISUAL_GROUP_ID, VISUAL_GROUPS,
                                   REST_PHONE_ID,
                                   visual_group_alternatives,
                                   visual_phone_alternatives)


class VisualPhonemeTests(unittest.TestCase):
    def test_lrs3_frontalization_preserves_missing_frames(self):
        generator = torch.Generator().manual_seed(7)
        points = torch.randn(4, 68, 2, generator=generator)
        points[:, :, 0] += torch.linspace(-1, 1, 68)
        points[2] = float("nan")
        frontal = frontalize_lrs3(points)
        self.assertTrue(torch.isnan(frontal[2]).all())
        self.assertTrue(torch.isfinite(frontal[[0, 1, 3]]).all())

    def test_autoavsr_fusion_forward(self):
        model = AutoAvsrFusionVisualPhoneme(
            classes=18, landmark_points=68, coordinate_dimensions=10,
            landmark_bottleneck=128,
        ).eval()
        with torch.inference_mode():
            logits = model(
                torch.randn(1, 3, 1, 96, 96),
                torch.randn(1, 3, 68, 10),
                torch.ones(1, 3, dtype=torch.bool),
            )
        self.assertEqual(tuple(logits.shape), (3, 1, 18))

    def test_large_transformer_fusion_forward(self):
        model = LargeTransformerFusionVisualPhoneme(
            classes=18, landmark_points=68, coordinate_dimensions=10,
            landmark_bottleneck=128,
        )
        logits = model(
            torch.randn(2, 3, 1, 96, 96),
            torch.randn(2, 3, 68, 10),
            torch.ones(2, 3, dtype=torch.bool),
        )
        self.assertEqual(tuple(logits.shape), (3, 2, 18))
        self.assertLess(model.parameter_count, 12_000_000)

    def test_lrs3_landmarks_preserve_missing_frames_and_normalize(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "landmarks.pkl"
            frames = [np.arange(136, dtype=np.float32).reshape(68, 2), None]
            with path.open("wb") as output:
                pickle.dump(frames, output, protocol=3)
            loaded = load_lrs3_landmarks(path)
            self.assertEqual(loaded.shape, (2, LRS3_LANDMARK_POINTS, 2))
            self.assertTrue(np.isnan(loaded[1]).all())
            normalized = eye_normalize_lrs3(torch.from_numpy(loaded))
            self.assertTrue(torch.isfinite(normalized[0]).all())
            self.assertTrue(torch.isnan(normalized[1]).all())

    def test_lrs3_test_manifest_loads_materialized_mouth_video(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifests").mkdir()
            (root / "test-video").mkdir()
            video = np.arange(4 * 12 * 12, dtype=np.uint8).reshape(4, 12, 12)
            with (root / "test-video/00000.npy").open("wb") as output:
                np.save(output, video, allow_pickle=False)
            row = {"clip_id": "lrs3:test/00000", "source_type": "npy-mouth",
                   "video": "test-video/00000.npy", "landmarks": None,
                   "frames": 4, "phonemes": ["B", "IY"]}
            (root / "manifests/test.jsonl").write_text(json.dumps(row) + "\n")
            dataset = Lrs3Clips(root, "test", size=16, crop="mouth")
            loaded, target, clip_id = dataset[0]
            self.assertEqual(loaded.shape, (4, 1, 16, 16))
            self.assertEqual(target.tolist(), [7, 18])
            self.assertEqual(clip_id, "lrs3:test/00000")

    def test_lrs3_teacher_words_expand_into_short_training_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifests").mkdir()
            labels = root / "teacher"
            labels.mkdir()
            video = root / "raw/trainval/talk/50001.mp4"
            video.parent.mkdir(parents=True)
            video.touch()
            row = {"clip_id": "lrs3:trainval/talk/50001", "source_type": "video",
                   "video": "raw/trainval/talk/50001.mp4", "landmarks": None,
                   "frames": 50, "fps": 25, "transcript": "the cat sat",
                   "phonemes": ["DH", "AH", "K", "AE", "T", "S", "AE", "T"]}
            (root / "manifests/train.jsonl").write_text(json.dumps(row) + "\n")
            alignment = labels / "sample.json"
            alignment.write_text(json.dumps({
                "label_status": "accepted",
                "provenance": {"teacher_transcripts": [["the", "cat", "sat"]]},
                "tiers": {
                    "words": {"entries": [[0, 0.5, "the"], [0.5, 1.1, "cat"],
                                             [1.1, 1.8, "sat"]]},
                    "phones": {"entries": [[0, .25, "DH"], [.25, .5, "AH"],
                                              [.5, .7, "K"], [.7, .9, "AE"],
                                              [.9, 1.1, "T"], [1.1, 1.3, "S"],
                                              [1.3, 1.55, "AE"], [1.55, 1.8, "T"]]},
                },
            }))
            (labels / "labels.jsonl").write_text(json.dumps({
                "source": str(video.resolve()), "alignment": str(alignment),
                "status": "accepted",
            }) + "\n")
            dataset = Lrs3Clips(root, "train", include_video=True,
                                teacher_labels_dir=labels, max_chunk_phones=5,
                                include_frame_targets=True)
            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset.rows[0]["phonemes"], ["DH", "AH", "K", "AE", "T"])
            self.assertEqual(dataset.rows[1]["phonemes"], ["S", "AE", "T"])
            self.assertEqual(dataset.rows[0]["source_clip_id"], row["clip_id"])
            self.assertEqual(len(dataset.rows[0]["frame_phone_ids"]),
                             dataset.rows[0]["frames"])

    def test_lrs3_rest_requires_transcript_boundary_and_audio_gap(self):
        row = {"clip_id": "clip", "frames": 30, "fps": 25, "transcript": "a bee"}
        document = {
            "provenance": {"teacher_transcripts": [["a", "bee"]]},
            "tiers": {
                "words": {"entries": [[0.0, 0.3, "a"], [0.5, 1.0, "bee"]]},
                "phones": {"entries": [[0.0, 0.3, "AH"], [0.5, 0.7, "B"],
                                              [0.7, 1.0, "IY"]]},
            },
        }
        from VisualPhoneme.lrs3_data import _teacher_chunks
        chunks = _teacher_chunks(row, document, 20, 1.0, True, 0.08)
        self.assertEqual(chunks[0]["target_ids"],
                         [3, REST_PHONE_ID, 7, 18])
        self.assertIn(REST_PHONE_ID, chunks[0]["frame_phone_ids"])
        no_rest = _teacher_chunks(row, document, 20, 1.0, True, 0.25)
        self.assertNotIn(REST_PHONE_ID, no_rest[0]["target_ids"])

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

    def test_large_fusion_is_separate_and_near_ten_million_parameters(self):
        model = LargeGatedFusionVisualPhoneme(
            landmark_points=68, coordinate_dimensions=10, landmark_bottleneck=128,
        )
        video = torch.zeros(1, 4, 1, 64, 64)
        landmarks = torch.zeros(1, 4, 68, 10)
        mask = torch.ones(1, 4, dtype=torch.bool)
        output = model(video, landmarks, mask)
        self.assertEqual(output.shape, (4, 1, 40))
        self.assertTrue(torch.isfinite(output).all())
        self.assertGreater(model.parameter_count, 9_000_000)
        self.assertLess(model.parameter_count, 11_000_000)

    def test_large_image_pretraining_path_matches_large_fusion(self):
        image = LargeVisualPhoneme(classes=18)
        fusion = LargeGatedFusionVisualPhoneme(
            classes=18, landmark_points=68, coordinate_dimensions=10,
        )
        output = image(torch.zeros(1, 4, 1, 64, 64))
        self.assertEqual(output.shape, (4, 1, 18))
        for prefix in ("frame_encoder.", "image_projection."):
            source = {name: value for name, value in image.state_dict().items()
                      if name.startswith(prefix)}
            target = fusion.state_dict()
            self.assertTrue(source)
            self.assertTrue(all(name in target and target[name].shape == value.shape
                                for name, value in source.items()))

    def test_visual_groups_partition_phones_and_expand_all_members(self):
        from VisualPhoneme.data import PHONEMES
        members = [phone for group in VISUAL_PHONE_GROUPS.values() for phone in group]
        self.assertEqual(set(members), set(PHONEMES))
        self.assertEqual(len(members), len(set(members)))
        alternatives = visual_phone_alternatives(["P", "V", "TH"])
        self.assertEqual(alternatives[0]["possible_phonemes"], ["B", "M", "P"])
        self.assertEqual(alternatives[1]["possible_phonemes"], ["F", "V"])
        self.assertEqual(alternatives[2]["possible_phonemes"], ["DH", "TH"])
        self.assertEqual(PHONE_TO_VISUAL_GROUP["P"], "lip-closure")
        self.assertEqual(len(VISUAL_GROUPS), 17)
        self.assertEqual(len(PHONE_ID_TO_VISUAL_GROUP_ID), 40)
        self.assertEqual(
            visual_group_alternatives(["lip-closure"])[0]["possible_phonemes"],
            ["B", "M", "P"],
        )

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

    def test_trigram_model_uses_two_token_history(self):
        transitions = fit_trigram_log_probs(
            [[1, 2, 1], [1, 2, 1], [2, 2, 2]], 3
        )
        self.assertEqual(transitions.shape, (3, 3, 3))
        self.assertTrue(torch.isfinite(transitions[:, :, 1:]).all())
        self.assertGreater(transitions[1, 2, 1], transitions[1, 2, 2])

    def test_ctc_prefix_beam_accepts_trigram_and_token_bonus(self):
        logits = torch.tensor([[0.0, 5.0, 0.0],
                               [5.0, 0.0, 0.0],
                               [0.0, 0.0, 5.0]])
        transitions = fit_trigram_log_probs([[1, 2], [1, 2]], 3)
        candidates = ctc_prefix_beam_search(
            logits.log_softmax(-1), beam_width=6, top_n=3,
            transition_log_probs=transitions, lm_weight=0.2, token_bonus=0.1,
        )
        self.assertEqual(candidates[0][0], [1, 2])

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

    def test_ctc_temporal_upsampling_repeats_emissions_and_lengths(self):
        logits = torch.arange(3 * 2 * 4).reshape(3, 2, 4)
        expanded, lengths = upsample_ctc_logits(logits, torch.tensor([3, 2]), 2)
        self.assertEqual(expanded.shape, (6, 2, 4))
        self.assertTrue(torch.equal(expanded[0], expanded[1]))
        self.assertTrue(torch.equal(expanded[4], expanded[5]))
        self.assertEqual(lengths.tolist(), [6, 4])
        with self.assertRaisesRegex(ValueError, "must be positive"):
            upsample_ctc_logits(logits, torch.tensor([3, 2]), 0)

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
