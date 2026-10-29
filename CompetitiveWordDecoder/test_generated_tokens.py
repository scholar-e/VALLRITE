import unittest
from CompetitiveWordDecoder.generated_tokens import generation_record, scoring_tokens
class Tokenizer:
    eos_token_id=9
    def __len__(self):return 10
    def decode(self,ids,skip_special_tokens=True):return ''.join({1:'a',2:'b',3:'ab',8:'',9:''}[i] for i in ids)
    def encode(self,text,add_special_tokens=False):return [3] if text=='ab' else [1]
class TokenTests(unittest.TestCase):
    def test_noninvertible_tokenization(self):
        t=Tokenizer();r=generation_record([1,2,8,9,9],t)
        self.assertEqual(scoring_tokens(r,t,True)[0],[1,2,8,9])
        self.assertEqual(scoring_tokens(r,t,retokenize=True)[0],[3,9])
        self.assertEqual(r['text'],'ab')
    def test_capped_prefix_gets_no_eos(self):
        t=Tokenizer();r=generation_record([1,2],t,2)
        self.assertTrue(r['hit_generation_cap']);self.assertEqual(scoring_tokens(r,t,True)[0],[1,2])
    def test_corrupt_text_and_missing_tokens_rejected(self):
        t=Tokenizer();r=generation_record([1,9],t);r['text']='ab'
        with self.assertRaises(ValueError):scoring_tokens(r,t,True)
        with self.assertRaises(ValueError):scoring_tokens(dict(text='ab',hit_generation_cap=False),t,True)
if __name__=='__main__':unittest.main()
