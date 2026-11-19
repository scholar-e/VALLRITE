import json
import random
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn

from VisualPhoneme.autoavsr_encoder import AutoAvsrEncoder
from VisualPhoneme.finetune_autoavsr_encoder import (Clips, PhoneFinetuner, WeightedBatches, collate,
                                                     parse_source, prepare_frames, read_rows,
                                                     word_distillation)
from VisualPhoneme.train_autoavsr_phone_head import PhoneHead


def tiny_model(trainable_blocks=1, distill=True):
    torch.manual_seed(0)
    encoder = AutoAvsrEncoder(width=32, heads=4, hidden=64, layers=3, kernel=3, dropout=0.1)
    return PhoneFinetuner(encoder, nn.Linear(32, 11), PhoneHead(32, 2, 0.1, 1, 16),
                          trainable_blocks, distill)


class SourceTest(unittest.TestCase):
    def test_parse_source_and_rows(self):
        self.assertEqual(parse_source("a/m.jsonl@root/dir"), (Path("a/m.jsonl"), Path("root/dir")))
        with self.assertRaises(ValueError):
            parse_source("a/m.jsonl")
        with tempfile.TemporaryDirectory() as folder:
            manifest = Path(folder) / "m.jsonl"
            manifest.write_text(json.dumps({"clip_id": "x", "video": "c/0.npy", "frames": 3,
                                            "transcript": "hi", "phonemes": ["HH", "AY"]}) + "\n")
            row = read_rows(manifest, Path("base"))[0]
            self.assertEqual(row["video"], "base/c/0.npy")
            self.assertEqual(len(row["phone_ids"]), 2)

    def test_weighted_batches_follow_weights_and_cover_sources(self):
        batches = iter(WeightedBatches([10, 40], [1, 3], 5, seed=0))
        drawn = [next(batches) for _ in range(400)]
        sources = Counter(0 if max(batch) < 10 else 1 for batch in drawn)
        for batch in drawn:  # one source per batch, no repeats inside a batch
            self.assertTrue(max(batch) < 10 or min(batch) >= 10)
            self.assertEqual(len(set(batch)), 5)
        self.assertAlmostEqual(sources[1] / 400, 0.75, delta=0.06)
        first_pass = [i for batch in drawn if max(batch) < 10 for i in batch][:10]
        self.assertEqual(sorted(first_pass), list(range(10)))


class AugmentTest(unittest.TestCase):
    def test_eval_is_centre_crop_and_train_varies(self):
        frames = np.random.default_rng(0).integers(0, 255, (4, 96, 96), dtype=np.uint8)
        centre = prepare_frames(frames, None)
        self.assertEqual(tuple(centre.shape), (4, 1, 88, 88))
        expected = (torch.from_numpy(frames[:, 4:92, 4:92]).float() / 255 - 0.421) / 0.165
        torch.testing.assert_close(centre[:, 0], expected)
        rng = random.Random(0)
        variants = {prepare_frames(frames, rng).sum().item() for _ in range(8)}
        self.assertGreater(len(variants), 1)

    def test_dataset_and_collate(self):
        with tempfile.TemporaryDirectory() as folder:
            for i, length in enumerate((3, 5)):
                np.save(Path(folder) / f"{i}.npy", np.zeros((length, 96, 96), np.uint8))
            rows = [{"clip_id": str(i), "video": f"{folder}/{i}.npy", "frames": n,
                     "transcript": "a", "phone_ids": [1] * (i + 1)} for i, n in enumerate((3, 5))]
            video, lengths, targets, target_lengths, indices = collate(
                [Clips(rows, augment=True)[i] for i in (0, 1)])
            self.assertEqual(tuple(video.shape), (2, 5, 1, 88, 88))
            self.assertEqual(lengths.tolist(), [3, 5])
            self.assertEqual(target_lengths.tolist(), [1, 2])
            self.assertEqual(len(targets), 3)
            self.assertEqual(indices, [0, 1])


class FinetunerTest(unittest.TestCase):
    def test_split_forward_matches_full_encoder_and_teacher(self):
        model = tiny_model().eval()
        video, lengths = torch.randn(2, 6, 1, 88, 88), torch.tensor([6, 4])
        with torch.no_grad():
            student, teacher = model(video, lengths, teacher=True)
            full = model.encoder(video, lengths)
        torch.testing.assert_close(student, full)
        torch.testing.assert_close(teacher, full)

    def test_only_top_blocks_and_head_train(self):
        model = tiny_model(trainable_blocks=1).train()
        trainable = {name for name, p in model.named_parameters() if p.requires_grad}
        self.assertTrue(all(name.startswith(("encoder.encoders.2.", "encoder.after_norm.", "head."))
                            for name in trainable))
        self.assertTrue(any(name.startswith("encoder.encoders.2.") for name in trainable))
        self.assertFalse(model.encoder.frontend.training)
        self.assertFalse(model.encoder.encoders[1].training)
        self.assertTrue(model.encoder.encoders[2].training)
        self.assertFalse(model.encoder.encoders[2].conv_module.norm.training)
        features, _ = model(torch.randn(1, 5, 1, 88, 88), torch.tensor([5]))
        model.head(features, torch.tensor([5])).sum().backward()
        self.assertIsNotNone(model.encoder.encoders[2].self_attn.linear_q.weight.grad)
        self.assertEqual(set(model.trainable_state()),
                         {f"encoders.2.{k}" for k in model.encoder.encoders[2].state_dict()}
                         | {f"after_norm.{k}" for k in model.encoder.after_norm.state_dict()})

    def test_distillation_is_zero_for_equal_logits_and_masks_padding(self):
        logits = torch.randn(2, 4, 7)
        valid = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]]).float()
        self.assertAlmostEqual(word_distillation(logits, logits, valid).item(), 0.0, places=6)
        other = logits.clone()
        other[1, 2:] = torch.randn(2, 7)  # padded frames differ only
        self.assertAlmostEqual(word_distillation(other, logits, valid).item(), 0.0, places=6)
        self.assertGreater(word_distillation(torch.randn(2, 4, 7), logits, valid).item(), 0)


if __name__ == "__main__":
    unittest.main()
