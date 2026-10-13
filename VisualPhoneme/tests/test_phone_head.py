import unittest
import torch
from VisualPhoneme.train_phone_head import PhoneFamilyModel, aligned_phone_loss, ContextPhoneHead, load_phone_export
from VisualPhoneme.model import LargeVisualPhoneme
from VisualPhoneme.visemes import PHONE_ID_TO_VISUAL_GROUP_ID
from VisualPhoneme.phone_family_objectives import family_partition, family_log_probs


class PhoneHeadTests(unittest.TestCase):
    def test_family_partitions_and_probability_gradient(self):
        for name,count in [('original',17),('fine',31),('coarse',10)]:
            groups,mapping=family_partition(name)
            self.assertEqual(len(groups),count)
            logits=torch.randn(9,2,40,requires_grad=True)
            grouped=family_log_probs(logits,mapping)
            self.assertTrue(torch.allclose(grouped.exp().sum(-1),torch.ones(9,2),atol=1e-6))
            self.assertTrue(torch.allclose(grouped[...,0],logits.log_softmax(-1)[...,0]))
            (-grouped[...,1].mean()).backward()
            self.assertGreater(logits.grad.abs().sum().item(),0)

    def test_aligned_loss_ignores_unlabelled_frames(self):
        logits=torch.randn(3,1,40,requires_grad=True)
        labels=torch.tensor([[4,-100,6]])
        actual=aligned_phone_loss(logits,labels)
        expected=torch.nn.functional.cross_entropy(logits[[0,2],0],torch.tensor([4,6]))
        self.assertTrue(torch.allclose(actual,expected))
        actual.backward()
        self.assertEqual(float(logits.grad[1].abs().sum()),0)
        self.assertEqual(float(aligned_phone_loss(logits,torch.full((1,3),-100)).detach()),0)

    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        self.model=PhoneFamilyModel().eval()
        self.model.initialize_phones()

    def test_initial_family_probability_is_preserved(self):
        features=torch.randn(3,384)
        phones=self.model.phone_head(features).softmax(-1)
        families=self.model.classifier(features).softmax(-1)
        reconstructed=torch.zeros_like(families)
        reconstructed.scatter_add_(1,torch.tensor(PHONE_ID_TO_VISUAL_GROUP_ID).expand(3,-1),phones)
        self.assertTrue(torch.allclose(reconstructed,families,atol=1e-6))

    def test_export_matches_standard_phone_model(self):
        exported=LargeVisualPhoneme().eval()
        exported.load_state_dict(self.model.export_phone_state(),strict=True)
        video=torch.randn(1,8,1,48,48)
        with torch.inference_mode():
            self.assertTrue(torch.allclose(self.model(video),exported(video),atol=1e-6))
            phones,families=self.model.both(video)
        self.assertEqual(tuple(phones.shape),(8,1,40))
        self.assertEqual(tuple(families.shape),(8,1,18))

    def test_context_starts_identical_and_learns(self):
        features=torch.randn(2,13,384)
        linear=self.model.phone_head
        head=ContextPhoneHead(linear)
        self.assertTrue(torch.equal(head(features),linear(features)))
        head(features).square().mean().backward()
        self.assertGreater(head.output.weight.grad.abs().sum().item(),0)

    def test_context_export_roundtrip(self):
        self.model.phone_head=ContextPhoneHead(self.model.phone_head)
        with torch.no_grad():
            self.model.phone_head.output.weight.normal_(0,.01)
        restored=load_phone_export(dict(architecture='phone-family-context',model=self.model.state_dict())).eval()
        video=torch.randn(2,11,1,48,48)
        with torch.inference_mode():
            self.assertTrue(torch.equal(self.model(video),restored(video)))

    def test_context_uses_neighbouring_frames(self):
        head=ContextPhoneHead(self.model.phone_head).eval()
        with torch.no_grad():
            head.output.weight.normal_(0,.01)
            features=torch.randn(1,15,384)
            changed=features.clone(); changed[:,7]+=torch.randn(384)
            difference=(head(changed)-head(features)).abs().sum(-1)[0]
        self.assertGreater(difference[3:12].min().item(),0)
        self.assertEqual(difference[:3].sum().item()+difference[12:].sum().item(),0)


if __name__=='__main__':
    unittest.main()
