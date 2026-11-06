import unittest

import torch

from VisualPhoneme.train_autoavsr_phone_head import CLASSES, PhoneHead, greedy_phones


class AutoAvsrPhoneHeadTest(unittest.TestCase):
    def test_upsampled_shapes_and_padding_invariance(self):
        torch.manual_seed(0)
        for layers in (0, 2):
            head = PhoneHead(width=16, upsample=2, dropout=0.0, layers=layers, hidden=8).eval()
            short = torch.randn(1, 5, 16)
            padded = torch.cat((short, torch.randn(1, 3, 16)), dim=1)
            with torch.no_grad():
                alone = head(short, torch.tensor([5]))
                batched = head(torch.cat((padded, torch.randn(1, 8, 16))), torch.tensor([5, 8]))
            self.assertEqual(tuple(alone.shape), (1, 10, CLASSES))
            torch.testing.assert_close(batched[0, :10], alone[0], atol=1e-5, rtol=1e-5)

    def test_greedy_phones_collapses_repeats_and_blanks(self):
        path = torch.nn.functional.one_hot(torch.tensor([0, 3, 3, 0, 3, 5, 5]), CLASSES).float()
        self.assertEqual(greedy_phones(path), [3, 3, 5])



class HeadOptionsTest(unittest.TestCase):
    def test_time_mask_zeroes_spans_inside_each_clip_only(self):
        import numpy as np
        from VisualPhoneme.train_autoavsr_phone_head import time_mask
        features = torch.ones(2, 50, 4)
        lengths = torch.tensor([50, 20])
        masked = time_mask(features, lengths, 3, 8, np.random.default_rng(0))
        self.assertTrue(torch.equal(features, torch.ones(2, 50, 4)))  # input untouched
        zero_frames = (masked == 0).all(-1)
        self.assertGreater(int(zero_frames.sum()), 0)
        self.assertLessEqual(int(zero_frames[1].sum()), 3 * 4)  # span capped at length // 5
        self.assertTrue(torch.equal(time_mask(features, lengths, 0, 8, np.random.default_rng(0)), features))

    def test_build_head_accepts_old_and_new_configs(self):
        from VisualPhoneme.train_autoavsr_phone_head import build_head
        old = build_head({"upsample": 2, "dropout": 0.1, "context_layers": 2}, 768)
        self.assertEqual(old.project.out_features, 256)
        new = build_head({"upsample": 2, "dropout": 0.1, "context_layers": 4, "hidden": 512,
                          "input_dropout": 0.3}, 768)
        self.assertEqual((new.project.out_features, new.dropout.p, len(new.context.layers)), (512, 0.3, 4))


if __name__ == "__main__":
    unittest.main()
