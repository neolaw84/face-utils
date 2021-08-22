import os
import cv2
from matplotlib import pyplot as plt

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
    
def imshow_notebook(img):
    _img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(_img)
    plt.show()

def plot_it(mps):
    x = mps[:, 0]

    y = -1 * mps[:, 1]

    z = mps[:, 2]

    fig = plt.figure()

    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(x, z, y)

    plt.show()