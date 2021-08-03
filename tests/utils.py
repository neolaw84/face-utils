import os
import tempfile

import bz2
import requests

THIS_FILE_PATH = os.path.abspath(__file__)
THIS_DIR_PATH = os.path.dirname(THIS_FILE_PATH)
IMAGE_FILE_PATH = os.path.join(THIS_DIR_PATH, "bins", "test-image.jpg")
MODEL_FILE_PATH = os.path.join(THIS_DIR_PATH, "bins", "landmarks.dat")

IMAGE_URL = "https://live.staticflickr.com/2605/3721476240_bf643c709e.jpg"
MODEL_URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"

def download_image():
    if os.path.isfile(IMAGE_FILE_PATH):
        return

    r = requests.get(IMAGE_URL, allow_redirects=True)
    with open (IMAGE_FILE_PATH, "wb") as f:
        f.write(r.content)

def download_model():
    if os.path.isfile(MODEL_FILE_PATH):
        return

    temp_path = MODEL_FILE_PATH + ".bz"
    r = requests.get(MODEL_URL, allow_redirects=True)
    with open(temp_path, "wb") as f:
        f.write(r.content)

    with open(MODEL_FILE_PATH, "wb") as f:
        data = bz2.BZ2File(temp_path).read()
        f.write(data)
    
