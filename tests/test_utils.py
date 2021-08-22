import pathlib
from unittest import TestCase

import requests
import cv2

from face_utils.extract import extract

THIS_FILE_PATH=pathlib.Path(__file__)
THIS_DIR_PATH=THIS_FILE_PATH.parent
IMAGE_DIR_PATH=pathlib.Path.joinpath(THIS_DIR_PATH, "images")

pathlib.Path.mkdir(IMAGE_DIR_PATH, parents=True, exist_ok=True)

IMG_1="https://images.pexels.com/photos/3866555/pexels-photo-3866555.png?auto=compress&cs=tinysrgb&dpr=3&h=750&w=1260"

def _download_and_save(url:str=None, filepath:str=None)->None:
    r = requests.get(IMG_1, allow_redirects=True)
    if r.status_code != 200:
        raise Exception("unable to download {}".format(url))
    with open (filepath, "wb") as f:
        f.write(r.content)

class TestUtils(TestCase):

    def setUp(self) -> None:
        self.images = []
        IMG_1_PATH = pathlib.Path.joinpath(IMAGE_DIR_PATH, "img_1.jpg")
        if not pathlib.Path.exists(IMG_1_PATH):
            _download_and_save(url=IMG_1, filepath=IMG_1_PATH)
        self.images.append(IMG_1_PATH)
        return super().setUp()

    def tearDown(self) -> None:
        # do nothing
        return super().tearDown()

    def test_nothing(self):
        pass

    def test_extract_box(self):
        for ip in self.images:
            img = cv2.imread(str(ip))
            rimgs = extract(img)
            for rimg in rimgs:
                self.assertTupleEqual(rimg.shape, (256, 256, 3))
                
    def test_extract_rotate(self):
        pass

    def test_extract_rotate_vector(self):
        pass