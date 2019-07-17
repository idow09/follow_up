import random
import time
from pathlib import Path

import cv2

from utils.utils import load_image_paths

NEUTRAL = (255, 0, 0)
RED = (0, 0, 255)


def put_label(img, xyr, label, color=None, line_thickness=None):
    tl = line_thickness or round(0.002 * max(img.shape[0:2])) + 1  # line thickness
    tl = tl / 6
    color = color or [random.randint(0, 255) for _ in range(3)]
    tf = max(tl - 1, 1)  # font thickness
    t_size = cv2.getTextSize(label, 0, fontScale=tl, thickness=tf)[0]
    x, y, r = xyr[0], xyr[1], xyr[2]
    c1 = (max(int(x - r), 0), max(int(y - r), 0))
    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
    cv2.rectangle(img, c1, c2, color, -1)  # filled
    white = [225, 255, 255]
    cv2.putText(img, label, (c1[0], c1[1] - 2), 0, fontScale=tl, color=white, thickness=tf, lineType=cv2.LINE_AA)


def plot_ball(img, xyr, color=None, line_thickness=None):
    tl = line_thickness or round(0.002 * max(img.shape[0:2])) + 1  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c = (xyr[0], xyr[1])
    r = xyr[2]
    if r > 0:
        cv2.circle(img, c, int(r), color, thickness=tl)


def crop_around_ball(img, xyr):
    x, y, r = xyr[0], xyr[1], xyr[2]
    half_w = half_h = int(4 * r)
    return img[y - half_h:y + half_h, x - half_w:x + half_w]


def crop_around_aoi(sample, img):
    if len(sample.labels) == 0:
        return img
    xs = []
    ys = []
    for lbl in sample.labels:
        xs.append(lbl.x)
        ys.append(lbl.y)
    for prd in sample.preds:
        xs.append(prd.x)
        ys.append(prd.y)
    # TODO: improve this +/- 70 randomness!
    x1 = max(min(xs) - 70, 0)
    y1 = max(min(ys) - 70, 0)
    x2 = min(max(xs) + 70, img.shape[1])
    y2 = min(max(ys) + 70, img.shape[0])
    return img[y1:y2, x1:x2]


def create_visualization(image_path, label_path, visualization_path, color=NEUTRAL, has_score=False, crop=False):
    image = cv2.imread(image_path)
    xyr = None
    with open(label_path, 'r') as labels_f:
        lines = labels_f.readlines()
    for line in lines:
        label = line.split()
        xyr = int(label[0]), int(label[1]), float(label[2])
        plot_ball(image, xyr, color=color)
        if has_score:
            sc = label[3]
            put_label(image, xyr, str(sc), color=color)
    if xyr is not None and len(lines) == 1 and crop:  # crop if exactly one label
        image = crop_around_ball(image, xyr)

    cv2.imwrite(visualization_path, image)


def create_visualizations(images_root, labels_root, visualizations_root, color=None, has_score=False):
    t = time.time()
    image_paths = load_image_paths(images_root)
    n_i = len(image_paths)
    color = color or (0, 0, 255) if has_score else NEUTRAL
    for i, image_path in enumerate(image_paths):
        visualization_path = str(Path(visualizations_root) / Path(image_path).name)
        label_path = str(Path(labels_root) / Path(image_path).name) + '.txt'
        create_visualization(image_path, label_path, visualization_path, color=color, has_score=has_score)
        print('%i/%i Done.' % (i + 1, n_i))
    print('Done All. (%.3fs)' % (time.time() - t))


def create_sample_visualization(sample, iou_th, color_true, color_false, crop=False, color_mode='bgr'):
    image = cv2.imread(sample.path)
    for prd in sample.preds:
        xyr = prd.x, prd.y, prd.r
        sc = prd.score
        color = color_true if prd.iou > iou_th else color_false
        plot_ball(image, xyr, color=color)
        put_label(image, xyr, str(sc), color=color)

    image = crop_around_aoi(sample, image)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if color_mode == 'rgb' else image
