import os
import random
import time
from pathlib import Path

import cv2

from utils.utils import load_image_paths

NEUTRAL = (255, 0, 0)
RED = (0, 0, 255)


def put_label(img, xyr, label, color=None, line_thickness=None):
    """
    Puts a label above a given ball (circle) in an image.
    :param img: The image to put label on
    :param xyr: A tuple containing coordinates of a circle (x, y, r)
    :param label: The text to put above the given ball (circle)
    :param color: The color for the label
    :param line_thickness: line thickness
    """
    tl = line_thickness or round(0.002 * max(img.shape[0:2])) + 1  # line thickness
    tl = tl / 8
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
    """
    Plots a ball in the given coordinates with the given color and line thickness
    :param img: The image to plot on
    :param xyr: A tuple containing coordinates of a ball (x, y, r)
    :param color: The desired color
    :param line_thickness: line thickness
    """
    tl = line_thickness or round(0.002 * max(img.shape[0:2])) + 1  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c = (xyr[0], xyr[1])
    r = xyr[2]
    if r > 0:
        cv2.circle(img, c, int(r), color, thickness=tl)


def crop_around_ball(img, xyr):
    """
    DEPRECATED
    """
    x, y, r = xyr[0], xyr[1], xyr[2]
    half_w = half_h = int(4 * r)
    return img[y - half_h:y + half_h, x - half_w:x + half_w]


def crop_around_aoi(img, sample):
    """
    Crops around area of interest
    :param img: The image to crop.
    :param sample: SampleData that contains the labels & predictions data for aoi calculation
    :return: The cropped image
    """
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
    """
    DEPRECATED:
    creates visualization based on raw label/prediction files, and not per (parsed) sample.
    """
    image = cv2.imread(image_path)
    xyr = None
    with open(label_path, 'r') as labels_f:
        lines = labels_f.readlines()
    for i, line in enumerate(lines):
        if i == 0:
            continue
        label = line.split()
        xyr = int(label[0]), int(label[1]), float(label[2])
        plot_ball(image, xyr, color=color)
        if has_score:
            sc = label[3]
            put_label(image, xyr, '%.2f' % float(sc), color=color)
    if xyr is not None and len(lines) == 1 and crop:  # crop if exactly one label
        image = crop_around_ball(image, xyr)

    cv2.imwrite(visualization_path, image)


def create_visualizations(images_root, labels_root, visualizations_root, color=None, has_score=False):
    """
    DEPRECATED:
    creates visualization based on raw label/prediction files, and not per (parsed) sample.
    """
    t = time.time()
    os.makedirs(visualizations_root, exist_ok=True)
    image_paths = load_image_paths(images_root)
    n_i = len(image_paths)
    color = color or (0, 0, 255) if has_score else NEUTRAL
    for i, image_path in enumerate(image_paths):
        visualization_path = str(Path(visualizations_root) / Path(image_path).name)
        label_path = str(Path(labels_root) / Path(image_path).name).replace('.jpg', '.txt')
        create_visualization(image_path, label_path, visualization_path, color=color, has_score=has_score)
        print('%i/%i Done.' % (i + 1, n_i))
    print('Done All. (%.3fs)' % (time.time() - t))


def create_sample_visualization(sample, iou_th, color_true, color_false, color_label, crop=False, color_mode='bgr'):
    """
    Creates an image with labels and predictions plotted on it.
    Predictions are colored differently according to their iou
    Crops the image around the area of interest if crop=True.
    :param sample: The SampleData containing the image path, the labels and the predictions
    :param iou_th: The IoU threshold which predictions are colored relative to.
    :param color_true: The color for predictions with IoU above the threshold.
    :param color_false: The color for predictions with IoU under the threshold.
    :param color_label: The color for labels.
    :param crop: Whether to crop the image around the area of interest.
    :param color_mode: The requested color mode for the returned image. 'rgb' or 'bgr'
    :return: An image in :color_mode color space, with labels and predictions plotted on it. Cropped if requested.
    """
    image = cv2.imread(sample.path)
    for lbl in sample.labels:
        xyr = lbl.x, lbl.y, lbl.r
        plot_ball(image, xyr, color=color_label, line_thickness=2)
    for prd in sample.preds:
        xyr = prd.x, prd.y, prd.r
        sc = prd.score
        color = color_true if prd.iou > iou_th else color_false
        plot_ball(image, xyr, color=color, line_thickness=2)
        put_label(image, xyr, '%.2f' % sc, color=color)

    image = crop_around_aoi(image, sample) if crop else image
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if color_mode == 'rgb' else image
