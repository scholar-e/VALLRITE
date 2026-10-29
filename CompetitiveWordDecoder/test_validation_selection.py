import copy
import unittest
from CompetitiveWordDecoder.select_validation_adapter import select
class SelectionTests(unittest.TestCase):
    def test_baseline_ties_and_regression(self):
        row=dict(source_sha256='a',records=1,reference_words=1,model='m',revision='v',score_weight=1,scoring_token_modes=['exact_generated_tokens'],selection='fixed',limit=None,details=[dict(clip_id='x',reference_words=['a'])],metrics={'probability_ranked':{'word_errors':2}})
        candidate=copy.deepcopy(row)
        self.assertEqual(select([row,candidate]),0)
        candidate['metrics']['probability_ranked']['word_errors']=3
        self.assertEqual(select([row,candidate]),0)
        candidate['metrics']['probability_ranked']['word_errors']=1
        self.assertEqual(select([row,candidate]),1)
        candidate['source_sha256']='b'
        with self.assertRaises(ValueError):select([row,candidate])
if __name__=='__main__':unittest.main()
