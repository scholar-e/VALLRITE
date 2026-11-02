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


if __name__ == "__main__":
    unittest.main()
