import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from CompetitiveWordDecoder.visual_error_data import build_frozen_rows

class FrozenRowsTests(unittest.TestCase):
    def test_exact_splits_and_leakage(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp); manifests=root/'manifests';manifests.mkdir()
            for i,split in enumerate(('train','validation','evaluation')):
                directory=root/split;directory.mkdir()
                row=dict(clip_id=f'lrs3:trainval/video{i}/clip',reference_phones=['AA'],reference_words=[f'word{chr(97+i)}'],phone_hypotheses=[dict(rank=j+1,phones=['B'],ctc_log_score=-j) for j in range(5)])
                (directory/'phone-hypotheses.jsonl').write_text(json.dumps(row)+'\n')
                (manifests/f'{split}.jsonl').write_text(json.dumps(dict(clip_id=row['clip_id'],phonemes=['AA'],transcript=row['reference_words'][0]))+'\n')
            args=(root/'train/phone-hypotheses.jsonl',root/'evaluation/phone-hypotheses.jsonl',manifests)
            mixed,_=build_frozen_rows(*args);clean,_=build_frozen_rows(*args,mode='clean')
            self.assertEqual(set(mixed),{'train','validation'})
            self.assertEqual(len(mixed['train']),2)
            self.assertEqual([r['target'] for r in mixed['train']],[r['target'] for r in clean['train']])
            self.assertNotEqual(mixed['train'][0]['prompt'],mixed['train'][1]['prompt'])
            self.assertEqual(clean['train'][0]['prompt'],clean['train'][1]['prompt'])
            path=root/'evaluation/phone-hypotheses.jsonl';row=json.loads(path.read_text());row['clip_id']='lrs3:trainval/video0/other';path.write_text(json.dumps(row)+'\n')
            p=manifests/'evaluation.jsonl';ref=json.loads(p.read_text());ref['clip_id']=row['clip_id'];p.write_text(json.dumps(ref)+'\n')
            with self.assertRaisesRegex(ValueError,'leakage'):build_frozen_rows(*args)

if __name__=='__main__':unittest.main()
