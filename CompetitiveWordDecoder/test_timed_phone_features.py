import unittest
import torch
from VisualPhoneme.data import PHONEMES
from CompetitiveWordDecoder.timed_phone_features import align_phone_evidence,TimedPhoneAdapter,FEATURE_DIM,bound_embedding_delta
class TimedTests(unittest.TestCase):
    def test_relative_bound_and_zero_gradient(self):
        base=torch.ones(2,8);delta=torch.ones(2,8,requires_grad=True)
        out=bound_embedding_delta(delta,base,.05)
        self.assertTrue(torch.all(out.norm(dim=-1)<=base.norm(dim=-1)*.050001))
        zero=torch.zeros(2,8,requires_grad=True);bound_embedding_delta(zero,base,.05).sum().backward()
        self.assertTrue(torch.equal(zero.grad,torch.ones_like(zero)))
    def test_repeat_alignment_and_timing(self):
        p=PHONEMES.index('P')+1
        z=torch.full((4,40),-10.);z[0,p]=10;z[1,0]=10;z[2,p]=10;z[3,0]=10
        f=align_phone_evidence(z.repeat_interleave(2,0),['P','P'],25)
        self.assertEqual(tuple(f.shape),(2,44))
        self.assertAlmostEqual(f[0,41].item(),.04,places=5)
        self.assertAlmostEqual(f[1,42].item(),.04,places=5)
        self.assertTrue(torch.allclose(f[:,:40].sum(-1),torch.ones(2)))
    def test_zero_start_and_mask_gradient(self):
        m=TimedPhoneAdapter(16);x=torch.zeros(2,3,44);x[0,1,1]=1
        self.assertEqual(m(x).abs().sum().item(),0)
        m(x).sum().backward();self.assertGreater(m.output.weight.grad.abs().sum().item(),0)
        with torch.no_grad():m.output.weight.fill_(1)
        self.assertEqual(m(x)[1].abs().sum().item(),0)
    def test_empty_phone_path(self):
        f=align_phone_evidence(torch.zeros(6,40),[],25)
        self.assertEqual(tuple(f.shape),(0,FEATURE_DIM))
if __name__=='__main__':unittest.main()
