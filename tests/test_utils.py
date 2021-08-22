import pathlib
from unittest import TestCase

import numpy as np
import requests
import cv2

from face_utils.extract import extract, extract_pitch_yaw_roll

THIS_FILE_PATH=pathlib.Path(__file__)
THIS_DIR_PATH=THIS_FILE_PATH.parent
IMAGE_DIR_PATH=pathlib.Path.joinpath(THIS_DIR_PATH, "images")

pathlib.Path.mkdir(IMAGE_DIR_PATH, parents=True, exist_ok=True)

IMG_1="https://images.pexels.com/photos/3866555/pexels-photo-3866555.png?auto=compress&cs=tinysrgb&dpr=3&h=750&w=1260"
IMG_2="https://images.unsplash.com/photo-1601412436009-d964bd02edbc?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=700&q=80"

def _download_and_save(url:str=None, filepath:str=None)->None:
    r = requests.get(url, allow_redirects=True)
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
        IMG_2_PATH = pathlib.Path.joinpath(IMAGE_DIR_PATH, "img_2.jpg")
        if not pathlib.Path.exists(IMG_2_PATH):
            _download_and_save(url=IMG_2, filepath=IMG_2_PATH)
        self.images.append(IMG_2_PATH)
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
            for i, rimg in enumerate(rimgs):
                self.assertTupleEqual(rimg.shape, (256, 256, 3))
                cv2.imwrite(str(IMAGE_DIR_PATH) + "/result-" + str(ip) + str(i.name) + ".jpg", rimg)
                
    def test_extract_rotate(self):
        for ip in self.images:
            img = cv2.imread(str(ip))
            rimgs = extract(img, correct_rotate=True)
            for i, rimg in enumerate(rimgs):
                self.assertTupleEqual(rimg.shape, (256, 256, 3))
                cv2.imwrite(str(IMAGE_DIR_PATH) + "/result-rotate-" + str(ip.name) + str(i) + ".jpg", rimg)

    def test_extract_rotate_vector(self):
        for ip in self.images:
            img = cv2.imread(str(ip))
            rimgs = extract_pitch_yaw_roll(img)
            for result in rimgs:
                assert isinstance(result, np.ndarray)
                assert len(result.shape) == 1
                assert result.shape[0] == 3
