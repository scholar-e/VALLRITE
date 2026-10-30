import unittest
import torch
from VisualPhoneme.rolling_phones import RollingPhoneAverage, smooth_repeated_logits, phone_segments


class RollingTests(unittest.TestCase):
    def test_causal_chunk_equivalence(self):
        p=torch.tensor([[.1,.8,.1],[.2,.1,.7],[.7,.2,.1],[.3,.5,.2]])
        full=RollingPhoneAverage(3).update(p)
        stream=RollingPhoneAverage(3)
        self.assertTrue(torch.equal(full,torch.cat([stream.update(p[:1]),stream.update(p[1:3]),stream.update(p[3:])])))
        self.assertTrue(torch.allclose(full[2],p[:3].mean(0)))
        self.assertTrue(torch.equal(full[:2],RollingPhoneAverage(3).update(p[:2])))
        self.assertTrue(torch.allclose(full.sum(-1),torch.ones(4)))

    def test_identity_and_exact_repeat_check(self):
        raw=torch.randn(4,3);cached=raw.repeat_interleave(2,0)
        p,logp=smooth_repeated_logits(cached,1)
        raw[:,0]-=1
        self.assertTrue(torch.equal(logp,raw.log_softmax(-1).repeat_interleave(2,0)))
        self.assertTrue(torch.allclose(p,raw.softmax(-1)))
        cached[1,1]+=1
        with self.assertRaises(ValueError):smooth_repeated_logits(cached,3)

    def test_repeats_blank_and_seconds(self):
        p=torch.tensor([[.1,.9],[.1,.9],[.9,.1],[.1,.9]])
        runs=phone_segments(p,25,['P'])
        self.assertEqual([r['phone'] for r in runs],['P',None,'P'])
        self.assertEqual(runs[0]['end_seconds'],.08)
        self.assertEqual(runs[-1]['end_seconds'],.16)

    def test_blank_preserving_keeps_instantaneous_emission_mass(self):
        torch.manual_seed(42)
        raw=torch.randn(12,4);cached=raw.repeat_interleave(2,0)
        p,_=smooth_repeated_logits(cached,3,preserve_blank=True)
        raw[:,0]-=1;original=raw.softmax(-1)
        self.assertTrue(torch.equal(p[:,0],original[:,0]))
        self.assertTrue(torch.allclose(p.sum(-1),torch.ones(12)))
        expected=original[:3,1:].sum(0);expected=expected/expected.sum()*original[2,1:].sum()
        self.assertTrue(torch.allclose(p[2,1:],expected))

    def test_invalid_input(self):
        for w in (0,-1,1.5):
            with self.assertRaises(ValueError):RollingPhoneAverage(w)
        with self.assertRaises(ValueError):RollingPhoneAverage().update(torch.tensor([[.5,.7]]))

if __name__=='__main__':unittest.main()
