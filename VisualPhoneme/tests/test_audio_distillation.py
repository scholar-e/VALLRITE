"""Verify the speech/blank/REST contract of soft audio supervision."""
import unittest

import torch

from VisualPhoneme.distill_audio import project_probabilities, soft_loss
from VisualPhoneme.visemes import REST_GROUP_ID


class AudioDistillationTest(unittest.TestCase):
    def test_family_probability_sums_exclude_blank_and_unknown(self):
        probabilities = torch.tensor([[0.5, 0.2, 0.1, 0.2]])
        target, mass = project_probabilities(probabilities,
            {"<pad>": 0, "b": 1, "p": 2, "unknown": 3}, ("lip-closure", "rest"))
        torch.testing.assert_close(mass, torch.tensor([0.3]))
        torch.testing.assert_close(target, torch.tensor([[0.0, 1.0, 0.0]]))

    def test_rest_and_low_confidence_frames_have_no_gradient(self):
        logits = torch.randn(2, 3, requires_grad=True)
        targets = torch.tensor([[0., 1., 0.], [0., 1., 0.]])
        loss = soft_loss(logits, targets, torch.tensor([0.9, 0.1]),
                         torch.tensor([REST_GROUP_ID, 1]))
        loss.backward()
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits))


if __name__ == "__main__":
    unittest.main()
