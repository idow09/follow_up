import shutil
from pathlib import Path

from benchmark_framework.sample_data import SampleData, Label, Prediction
from benchmark_framework.benchmark_utils import generate_fake_preds
from utils.utils import *

IOU_THRESHOLD = 0.5


def parse_labels(labels_path):
    labels = []
    with open(labels_path, 'r') as gt_f:
        for line in gt_f.readlines():
            label = line.split()
            labels.append(Label(int(label[0]), int(label[1]), float(label[2])))
    return labels


def parse_preds(preds_path, labels):
    preds = []
    time = None
    with open(preds_path, 'r') as pred_f:
        for line in pred_f.readlines():
            log = [float(x) for x in line.split()]
            pred = Prediction(int(log[0]), int(log[1]), log[2], log[3])
            pred.match_label(labels)
            preds.append(pred)
            time = log[4]
    return preds, time


def calc_sample_precision_recall(sample, thresholds):
    for th in thresholds:
        matched_labels = []
        tot_pos = 0
        true_pos = 0
        for pred in sample.preds:
            if pred.score > th:
                tot_pos += 1
                if pred.iou > IOU_THRESHOLD:
                    true_pos += 1
                    matched_labels.append(pred.matched_label)
        p = true_pos / tot_pos if tot_pos > 0 else 1.0
        r = len(set(matched_labels)) / len(sample.labels) if len(sample.labels) > 0 else 1.0
        sample.stats[th] = {'precision': p, 'recall': r}


class Benchmark:
    def __init__(self, algo_id, scale=None, fake=False, persist=False):
        self.sample_list = []
        self.algo_id = algo_id
        self.scale = scale
        self.roc = None
        self.center_dist_list = None

        self.fake = fake
        self.persist = persist

    def parse_sample_results(self, image_path, gts_root, preds_root):
        labels_path = str(Path(gts_root) / Path(image_path).name) + '.txt'
        preds_path = str(Path(preds_root) / Path(image_path).name) + '.txt'
        labels = parse_labels(labels_path)
        if self.fake:
            preds, time = generate_fake_preds(preds_path, labels, persist=self.persist)
        else:
            preds, time = parse_preds(preds_path, labels)
        return SampleData(image_path, labels, preds, time, self.scale, self.algo_id)

    def parse_experiment_results(self, images_root, labels_root, preds_root):
        if self.fake and self.persist:
            if os.path.exists(preds_root):
                shutil.rmtree(preds_root)
            os.makedirs(preds_root)

        image_paths = load_image_paths(images_root)
        for image_path in image_paths:
            sample_data = self.parse_sample_results(image_path, labels_root, preds_root)
            self.sample_list.append(sample_data)

    def calc_stats(self, thresholds=None):
        self.calc_roc(thresholds=thresholds)
        self.calc_hist()

    def calc_hist(self):
        self.center_dist_list = []
        for sample in self.sample_list:
            for pred in sample.preds:
                if pred.center_dist is not None:
                    self.center_dist_list.append(pred.center_dist)

    def calc_roc(self, thresholds=None):
        if thresholds is None:
            thresholds = np.linspace(0, 1, 11)
        self.roc = {}

        th2samples_precision_list = {th: [] for th in thresholds}
        th2samples_recall_list = {th: [] for th in thresholds}

        for sample in self.sample_list:
            calc_sample_precision_recall(sample, thresholds)

            for th, p_r_dict in sample.stats.items():
                th2samples_precision_list[th].append(p_r_dict['precision'])
                th2samples_recall_list[th].append(p_r_dict['recall'])

        for th in thresholds:
            p = sum(th2samples_precision_list[th]) / len(th2samples_precision_list[th])
            r = sum(th2samples_recall_list[th]) / len(th2samples_recall_list[th])
            self.roc[th] = {'precision': p, 'recall': r}