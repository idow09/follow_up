import math

from utils.utils import *


@auto_str
class SampleData:
    """
    A class to bundle all data for a single image (image_path, labels, predictions, stats, etc.)
    """

    def __init__(self, path, labels, preds, time, scale, algo_id):
        self.path = path
        self.labels = labels
        self.preds = preds
        self.stats = {}
        self.time = time
        self.scale = scale
        self.algo_id = algo_id


@auto_str
class CirclePrediction:
    """
    Contains coordinates as well as score, a (calculated) matched label, and some stats.
    """

    def __init__(self, x, y, r, score):
        self.x = x
        self.y = y
        self.r = r
        self.score = score
        self.matched_label = None
        self.iou = None
        self.center_dist = None

    def calc_rect_iou(self, label):
        """
        Prefer :calc_circle_iou
        """
        box_a = [self.x - self.r, self.y - self.r, self.x + self.r, self.y + self.r]
        box_b = [label.x - label.r, label.y - label.r, label.x + label.r, label.y + label.r]
        # determine the (x, y)-coordinates of the intersection rectangle
        x_a = max(box_a[0], box_b[0])
        y_a = max(box_a[1], box_b[1])
        x_b = min(box_a[2], box_b[2])
        y_b = min(box_a[3], box_b[3])

        # compute the area of intersection rectangle
        inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

        # compute the area of both the prediction and ground-truth
        # rectangles
        box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
        box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)

        return inter_area / float(box_a_area + box_b_area - inter_area)

    def calc_circle_iou(self, label):
        r1 = self.r
        r2 = label.r
        d = np.linalg.norm(np.array([self.x, self.y]) - np.array([label.x, label.y]))
        if d > (r1 + r2):  # No congruent
            return 0
        if d <= abs(r1 - r2):  # One inside another
            if (r1 * r2) == 0:
                return (r1 == r2) * 1.0
            iou = r1 ** 2 / r2 ** 2
            return iou if r1 < r2 else 1 / iou

        a = r1 ** 2 * np.arccos((d ** 2 + r1 ** 2 - r2 ** 2) / (2 * d * r1)) + \
            r2 ** 2 * np.arccos((d ** 2 + r2 ** 2 - r1 ** 2) / (2 * d * r2)) - \
            0.5 * math.sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))
        return a / (np.pi * r1 ** 2 + np.pi * r2 ** 2 - a)

    def calc_center_dist(self):
        if self.matched_label is None:
            return
        a = np.array([self.x, self.y])
        b = np.array([self.matched_label.x, self.matched_label.y])
        self.center_dist = np.linalg.norm(a - b)

    def match_label(self, labels, shape="circle"):
        """
        Match the prediction with the most probable (highest IoU) label from the given list.
        If None found, no matched_label will be stored.
        :param labels: The pool of labels to match with.
        """
        match_iou = 0
        match = None
        for label in labels:

            iou = self.calc_circle_iou(label)
            if iou > match_iou:
                match_iou = iou
                match = label
        self.matched_label = match
        self.iou = match_iou
        self.calc_center_dist()


@auto_str
class CircleLabel:
    """
    A class to bundle the coordinates for a label (x, y, r)
    """

    def __init__(self, x, y, r):
        self.x = x
        self.y = y
        self.r = r


@auto_str
class Prediction:
    """
    Contains coordinates as well as score, a (calculated) matched label, and some stats.
    """
    def __init__(self, x1, y1, x2, y2, score):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.score = score
        self.matched_label = None
        self.iou = None
        self.center_dist = None

    def calc_rect_iou(self, label):
        """
        Prefer :calc_iou
        """
        box_a = [self.x1, self.y1, self.x2, self.y2]
        box_b = [label.x1, label.y1, label.x2, label.y2]
        # determine the (x, y)-coordinates of the intersection rectangle
        x_a = max(box_a[0], box_b[0])
        y_a = max(box_a[1], box_b[1])
        x_b = min(box_a[2], box_b[2])
        y_b = min(box_a[3], box_b[3])

        # compute the area of intersection rectangle
        inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

        # compute the area of both the prediction and ground-truth
        # rectangles
        box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
        box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)

        return inter_area / float(box_a_area + box_b_area - inter_area)

    def calc_center_dist(self):
        if self.matched_label is None:
            return
        w = self.x2 - self.x1
        h = self.y2 - self.y1
        cx = int(self.x1 + w/2)
        cy = int(self.y1 + h/2)

        lw = self.matched_label.x2 - self.matched_label.x1
        lh = self.matched_label.y2 - self.matched_label.y1
        lcx = int(self.matched_label.x1 + lw / 2)
        lcy = int(self.matched_label.y1 + lh / 2)

        a = np.array([cx, cy])
        b = np.array([lcx, lcy])
        self.center_dist = np.linalg.norm(a - b)

    def match_label(self, labels):
        """
        Match the prediction with the most probable (highest IoU) label from the given list.
        If None found, no matched_label will be stored.
        :param labels: The pool of labels to match with.
        """
        match_iou = 0
        match = None
        for label in labels:

            iou = self.calc_rect_iou(label)
            if iou > match_iou:
                match_iou = iou
                match = label
        self.matched_label = match
        self.iou = match_iou
        self.calc_center_dist()


@auto_str
class Label:
    """
    A class to bundle the coordinates for a label (x1, y1, x2, y2)
    """

    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
