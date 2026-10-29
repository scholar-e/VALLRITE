import itertools
import unittest
import torch
from VisualPhoneme.blank_losses import expected_phone_count, phone_count_loss, masked_teacher_loss


class BlankLossTests(unittest.TestCase):
    def test_expected_count_matches_exhaustive_alignments(self):
        p = torch.tensor([[.2,.6,.2],[.5,.2,.3],[.1,.4,.5]])
        expected = 0.
        for path in itertools.product(range(3),repeat=3):
            count = sum(token!=0 and (t==0 or token!=path[t-1]) for t,token in enumerate(path))
            expected += float(torch.prod(torch.stack([p[t,c] for t,c in enumerate(path)]))) * count
        self.assertAlmostEqual(float(expected_phone_count(p.log()[:,None],torch.tensor([3]))[0]),expected,places=6)

    def test_padding_does_not_count_and_gradient_is_finite(self):
        logits = torch.randn(5,2,3,requires_grad=True)
        lengths = torch.tensor([2,5])
        count = expected_phone_count(logits,lengths)
        self.assertAlmostEqual(float(count[0].detach()),float(expected_phone_count(logits[:2,:1],torch.tensor([2]))[0].detach()),places=6)
        loss = phone_count_loss(logits,lengths,torch.tensor([1,3]))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(float(logits.grad[2:,0].abs().sum()),0.)

    def test_teacher_never_supervises_unknown_frames(self):
        logits = torch.zeros(2,1,3,requires_grad=True)
        target = torch.zeros_like(logits); target[0,0,1]=1
        mask = torch.tensor([[True],[False]])
        masked_teacher_loss(logits,target,mask).backward()
        self.assertEqual(float(logits.grad[1].abs().sum()),0.)
        self.assertLess(float(logits.grad[0,0,1]),0.)
        self.assertEqual(float(masked_teacher_loss(logits,target,torch.zeros_like(mask)).detach()),0.)

    def test_window_targets_mask_gaps_and_ambiguous_frames(self):
        from VisualPhoneme.test_blank_strategies import teacher_targets
        from VisualPhoneme.data import PHONE_TO_ID
        doc={'tiers':{'phones':{'entries':[[0,.04,'AA'],[.04,.08,'T']]}}}
        target,mask=teacher_targets({'fps':25},doc,5)
        self.assertFalse(mask[3])
        self.assertFalse(mask[4])
        self.assertEqual(float(target[:,0].sum()),0.)
        # First frame's 80-ms window overlaps AA for 40ms and T for 20ms:
        # no 70% majority, so no fabricated single-phone target.
        self.assertFalse(mask[0])
        doc={'tiers':{'phones':{'entries':[[0,.08,'AA']]}}}
        target,mask=teacher_targets({'fps':25},doc,5)
        self.assertTrue(mask[0])
        self.assertAlmostEqual(float(target[0,PHONE_TO_ID['AA']]),1.)


if __name__ == '__main__':
    unittest.main()
