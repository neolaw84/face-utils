import math
from typing import Tuple, List

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from insightface.app import FaceAnalysis

from face_utils.face_model import model_points
from face_utils.utils import shift_centroid_to_origin as shift_to_o, scale_face as scale

model_points = shift_to_o(model_points)

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

def _rotate_image(image, angle, image_center):
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def combine_affine_transform(M1, M2):
    M1_ = np.vstack([M1, np.float32([[0, 0, 1]])])
    M2_ = np.vstack([M2, np.float32([[0, 0, 1]])])
    M_final = np.matmul(M2_, M1_)[0:2, :]
    return M_final

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
        landmarks = f["landmark_3d_68"]
        landmarks = shift_to_o(landmarks)
        mps_norm = scale(model_points, landmarks)

        # these two reshapes are to make sure Rotation is happy
        mps = mps_norm.reshape((-1, 3))

        # landmarks = landmarks.reshape(68*3)
        lmk = landmarks.reshape((-1, 3))
        rmat, rmsd = Rotation.align_vectors(a=lmk, b=mps)
        
        rotation = math.degrees(rmat.as_rotvec()[2])   

        # rimg_r = _rotate_image(img, rotation, center)
        M1 = cv2.getRotationMatrix2D(center, rotation, 1.0)

    # M = cv2.getAffineTransform(src=bbox_points, dst=_get_new_bbox_points(size=size))
    M, _ = cv2.estimateAffinePartial2D(bbox_points, _get_new_bbox_points(size=size))
    if correct_rotate:
        M = combine_affine_transform(M1, M)
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

def _extract_pitch_yaw_roll(img, f, size:Tuple=(256, 256), bbox_scale:float=1.0, correct_rotate:bool=False):
    bbox = f["bbox"]
    
    landmarks = f["landmark_3d_68"]
    landmarks = shift_to_o(landmarks)
    mps_norm = scale(model_points, landmarks)

    # these two reshapes are to make sure Rotation is happy
    mps = mps_norm.reshape((-1, 3))
    lmk = landmarks.reshape((-1, 3))
    rmat, rmsd = Rotation.align_vectors(a=lmk, b=mps)
    
    return rmat.as_rotvec()

def extract_pitch_yaw_roll(img):

    faces = app.get(img)
    rimgs = [_extract_pitch_yaw_roll(img, f) for f in faces]

    return rimgs