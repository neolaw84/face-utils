from typing import Tuple, List

import cv2
from matplotlib.pyplot import flag
import numpy as np

from insightface.app import FaceAnalysis

app = FaceAnalysis(allowed_modules=["detection", "genderage", "landmark_3d_68"])
app.prepare(ctx_id=0, det_size=(640, 640))

X = 0
Y = 1

X1 = 0
Y1 = 1
X2 = 2
Y2 = 3

def _get_new_bbox_points(size=(256, 256)):
    return np.float32([
        [0, 0],
        [size[X], 0],
        [0, size[Y]], 
        [size[X], size[Y]]
    ])

def _get_width_and_height(bbox:List)->Tuple:
    return bbox[X2] - bbox[X1], bbox[Y2] - bbox[Y1]

def _get_bbox_center_width_height(bbox:List)->Tuple:
    """[summary]

    Args:
        bbox (List): [x, y, width, height]

    Returns:
        Tuple: (x, y)
    """
    width, height = _get_width_and_height(bbox)
    return (bbox[X] + width/2.0, bbox[Y] + height/2.0), width, height

def _extract(img, f, size:Tuple=(256, 256), bbox_scale:float=1.0, correct_rotate:bool=False):
    bbox = f["bbox"]

    center, width, height = _get_bbox_center_width_height(bbox)

    # scaling bbox
    if bbox_scale != 1.0:
        width = width * bbox_scale
        height = height * bbox_scale

    if width > height:
        height = width
    else:
        width = height
    bbox_points = np.float32([
        [center[X] - width/2.0, center[Y] - height/2.0], 
        [center[X] + width/2.0, center[Y] - height/2.0], 
        [center[X] - width/2.0, center[Y] + height/2.0],
        [center[X] + width/2.0, center[Y] + height/2.0]
    ])
    
    # correct rotation
    if correct_rotate:
        pass

    # M = cv2.getAffineTransform(src=bbox_points, dst=_get_new_bbox_points(size=size))
    M, _ = cv2.estimateAffinePartial2D(bbox_points, _get_new_bbox_points(size=size))
    rimg = cv2.warpAffine(img, M, dsize=size, flags=cv2.INTER_CUBIC)

    return rimg

def extract(img, size:Tuple=(256, 256), bbox_scale:float=1.0, correct_rotate:bool=False):
    """extract cv2 image(s) of the faces from the given image.

    Args:
        img (cv2_image): input image
        size (Tuple, optional): result size (x, y). Only supports square shapes. Defaults to (256, 256).
        bbox_scale (float, optional): how tight or loose the bounding box. <1 is tight while >1 is loose. Defaults to 1.0.
        correct_rotate (bool, optional): whether to correct rotation of face w.r.t Z axis of face (not camera). Defaults to False.

        It processes in this order:
            * scale bbox
            * correct rotation and
            * resize to size

    Returns:
        List[cv2_image]: a list of cv2 image(s) of the faces from the given image.
    """

    # we will never support this
    assert size[X] == size[Y]

    faces = app.get(img)
    rimgs = [_extract(img, f, size=size, bbox_scale=bbox_scale, correct_rotate=correct_rotate) for f in faces]

    return rimgs
