import logging

from unittest import TestCase

from face_extractor.core import align_face

from .utils import download_image, download_model, MODEL_FILE_PATH, IMAGE_FILE_PATH

logger = logging.getLogger(__file__)

class ExtractTest(TestCase):
    def setUp(self):
        download_image()
        download_model()
        # self.model = get_model(MODEL_FILE_PATH)

    def tearDown(self):
        pass

    def test_extract(self):
        img = align_face(filepath=IMAGE_FILE_PATH, predictor=MODEL_FILE_PATH)[0]
        self.assertTupleEqual(img.size, (1024, 1024))
