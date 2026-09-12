import json
import math
from pathlib import Path
import unittest
import jsonschema
from VisualPhoneticAlphabet.core import observe, build_record, ARPABET_IPA, POINT_IDS, BASELINE_POINT_IDS


def face(aperture=0):
    points = [[0., 0.] for _ in range(468)]
    for i, p in {33:(0,0),263:(100,0),61:(20,50),291:(80,50),13:(50,50),14:(50,50+aperture),0:(50,45),17:(50,60),152:(50,100)}.items():
        points[i] = list(p)
    return points


class CoreTests(unittest.TestCase):
    def test_rigid_invariance(self):
        points = face(10)
        angle = .7
        changed = [[3*(x*math.cos(angle)-y*math.sin(angle))+12,3*(x*math.sin(angle)+y*math.cos(angle))-9] for x,y in points]
        a, b = observe(0,points), observe(0,changed)
        for key in ('inner_aperture','mouth_width','rounding','bilabial_contact'):
            self.assertAlmostEqual(a['features'][key],b['features'][key])

    def test_gap_resets_motion_and_no_false_release(self):
        a=observe(0,face())
        b=observe(40,None,a)
        c=observe(80,face(20),b)
        self.assertIsNone(c['features']['opening_velocity'])
        record=build_record('test',25,[a,b,c])
        self.assertNotIn('BCL-REL',[g['token'] for g in record['gestures']])
        self.assertIn('UNK-VIS',[g['token'] for g in record['gestures']])

    def test_schema_and_release(self):
        a=observe(0,face())
        b=observe(40,face(20),a)
        record=build_record('test',25,[a,b])
        schema=json.loads((Path(__file__).parents[1]/'schema.json').read_text())
        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.validate(record,schema)
        self.assertIn('BCL-REL',[g['token'] for g in record['gestures']])
        self.assertEqual(record['phoneme_hypotheses'],[])
        json.dumps(record,allow_nan=False)
        self.assertEqual(len(ARPABET_IPA),39)

    def test_expanded_points_preserve_baseline_speed(self):
        initial = face(10)
        moved = [p[:] for p in initial]
        moved[81][1] += 20
        a, b = observe(0, initial), observe(40, moved, observe(0, initial))
        self.assertEqual(len(POINT_IDS), 41)
        self.assertEqual(len(b['landmarks']), 41)
        self.assertNotEqual(a['landmarks']['81'], b['landmarks']['81'])
        self.assertEqual(b['features']['landmark_speed'], 0)
        # Records with the original seven points still work as previous frames.
        a['landmarks'] = {str(i): a['landmarks'][str(i)] for i in BASELINE_POINT_IDS}
        self.assertEqual(observe(40, moved, a)['features']['landmark_speed'], 0)

    def test_invalid_input(self):
        with self.assertRaises(ValueError): observe(0,[[float('nan'),0]]*468)
        with self.assertRaises(ValueError): build_record('test',0,[observe(0,None)])
        with self.assertRaises(ValueError): build_record('test',25,[observe(0,None),observe(0,None)])


if __name__=='__main__': unittest.main()
