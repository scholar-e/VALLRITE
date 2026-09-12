import importlib.util
import unittest

from VisualPhoneticAlphabet.crop import FaceNeckCropper, overlap


@unittest.skipUnless(importlib.util.find_spec('cv2'), 'optional OpenCV backend is not installed')
class CropTests(unittest.TestCase):
    def test_ambiguous_initialization(self):
        cropper = FaceNeckCropper()
        self.assertIsNone(cropper.select([(0, 0, 100, 100), (200, 0, 100, 100)]))
        self.assertEqual(cropper.segment, 0)

    def test_association_and_gap(self):
        cropper = FaceNeckCropper(max_gap=1)
        cropper.select([(0, 0, 100, 100)])
        selected = cropper.select([(10, 0, 100, 100), (200, 0, 100, 100)])
        self.assertAlmostEqual(selected[0], 6.5)
        self.assertIsNone(cropper.select([]))
        self.assertIsNone(cropper.select([]))
        self.assertEqual(cropper.select([(200, 0, 100, 100)])[0], 200)
        self.assertEqual(cropper.segment, 2)

    def test_does_not_switch_to_unrelated_face(self):
        cropper = FaceNeckCropper()
        cropper.select([(0, 0, 100, 100)])
        self.assertIsNone(cropper.select([(300, 0, 100, 100)]))

    def test_clamped_face_neck_region(self):
        import numpy as np
        class Detector:
            def detectMultiScale(self, *args, **kwargs):
                return [(0, 0, 100, 100)]
        cropper = FaceNeckCropper()
        cropper.detector = Detector()
        crop, metadata = cropper.process(np.zeros((200, 200, 3), dtype=np.uint8))
        self.assertEqual(metadata['crop_xywh'], [0, 0, 130, 185])
        self.assertEqual(crop.shape, (185, 130, 3))
        self.assertTrue(metadata['clipped'])

    def test_missing_frame_has_no_crop(self):
        import numpy as np
        crop, metadata = FaceNeckCropper().process(np.zeros((200, 200, 3), dtype=np.uint8))
        self.assertIsNone(crop)
        self.assertEqual(metadata['status'], 'missing')


class GeometryTests(unittest.TestCase):
    def test_overlap(self):
        self.assertEqual(overlap((0, 0, 10, 10), (0, 0, 10, 10)), 1)
        self.assertEqual(overlap((0, 0, 10, 10), (20, 0, 10, 10)), 0)
