import glob
import os
import random

import cv2
import matplotlib
import numpy as np

matplotlib.rc('font', **{'size': 11})

# Set print-options
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5

# Prevent OpenCV from multi-threading (to use PyTorch DataLoader)
cv2.setNumThreads(0)


def float3(x):  # format floats to 3 decimals
    return float(format(x, '.3f'))


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)


def xyxy2xywh(x):
    # Convert bounding box format from [x1, y1, x2, y2] to [x, y, w, h]
    y = np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y


def xywh2xyxy(x):
    # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def xyxy2xyr(x1, y1, x2, y2):
    # Convert bounding box format from [x1, y1, x2, y2] to [center_x, center_y, r]
    x = int((x1 + x2) / 2)
    y = int((y1 + y2) / 2)
    w = x2 - x1
    h = y2 - y1
    r = (w + h) / 4  # average w & h to get ~2xRadius
    return x, y, r


def xyxy2xyr_array(x):
    # Convert bounding box format from [x1, y1, x2, y2] to [center_x, center_y, r]
    y = np.zeros_like(x)
    y[:, 0] = int((x[:, 0] + x[:, 2]) / 2)
    y[:, 1] = int((x[:, 1] + x[:, 3]) / 2)
    w = x[:, 2] - x[:, 0]
    h = x[:, 3] - x[:, 1]
    y[:, 2] = (w + h) / 4  # average w & h to get ~2xRadius
    return y


def scale_coords(img1_shape, coords, img0_shape):
    # Rescale coords1 (xyxy) from img1_shape to img0_shape
    gain = max(img1_shape) / max(img0_shape)  # gain  = old / new
    coords[:, [0, 2]] -= (img1_shape[1] - img0_shape[1] * gain) / 2  # x padding
    coords[:, [1, 3]] -= (img1_shape[0] - img0_shape[0] * gain) / 2  # y padding
    coords[:, :4] /= gain
    coords[:, :4] = coords[:, :4].clamp(min=0)
    return coords


def wh_iou(box1, box2):
    # Returns the IoU of wh1 to wh2. wh1 is 2, wh2 is nx2
    box2 = box2.t()

    # w, h = box1
    w1, h1 = box1[0], box1[1]
    w2, h2 = box2[0], box2[1]

    # Intersection area
    inter_area = np.min(w1, w2) * np.min(h1, h2)

    # Union Area
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area

    return inter_area / union_area  # iou


def load_image_paths(path):
    img_formats = ['.jpg', '.jpeg', '.png', '.tif']

    files = []
    if os.path.isdir(path):
        files = sorted(glob.glob('%s/*.*' % path))
    elif os.path.isfile(path):
        files = [path]

    return [x for x in files if os.path.splitext(x)[-1].lower() in img_formats]


# Annotate a class @auto_str to have it pretty-printed whenever you str() it
def auto_str(cls):
    def __str__(self):
        return '%s(%s)' % (
            type(self).__name__,
            ', '.join('%s=%s' % item for item in vars(self).items())
        )

    cls.__str__ = __str__
    return cls
