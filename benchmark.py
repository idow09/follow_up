import shutil
from pathlib import Path
from data_structures import SampleData, Label, Prediction, CirclePrediction, CircleLabel, MultiLabel, MultiPrediction
from utils.fake_generator import generate_fake_preds
from utils.utils import *

IOU_THRESHOLD = 0.5


def parse_labels(labels_path, shape="rect"):
    """
    Returns a list of Label objects parsed from a given txt file
    :param labels_path: The path to the txt file that contains labels in the (x, y, r) format.
    The labels are line separated and the coordinates are space separated.
    :return: A list of Label objects parsed from the given file.
    """
    labels = []
    with open(labels_path, 'r') as gt_f:
        for line in gt_f.readlines():
            label = line.split()
            if shape == "circle":
                labels.append(CircleLabel(int(label[0]), int(label[1]), float(label[2])))
            elif shape == 'multi':
                labels.append(MultiLabel(int(label[0]), int(label[1]), int(label[2]), int(label[3]), label[4]))
            else:
                labels.append(Label(int(label[0]), int(label[1]), int(label[2]), int(label[3])))
    return labels


def parse_preds(preds_path, labels, shape="rect"):
    """
    Returns a list of Prediction objects parsed from a given txt file, and matched with a corresponding label.
    :param preds_path: The path to the txt file that contains predictions in the (x, y, r) format.
    The predictions are line separated and the coordinates are space separated.
    :param labels: A list of labels to match with.
    :return: A list of Prediction objects parsed from the given file, each matched with a single Label object from the
    labels list (or with None if not found).
    """
    preds = []
    with open(preds_path, 'r') as pred_f:
        time = float(pred_f.readline())
        for line in pred_f.readlines():
            log = [float(x) for x in line.split()]
            if shape == "circle":
                pred = CirclePrediction(int(log[0]), int(log[1]), log[2], log[3])
                pred.match_label(labels, shape=shape)
            elif shape == "multi":
                pred = MultiPrediction(int(log[0]), int(log[1]), int(log[2]), int(log[3]), int(log[4]), log[5])
                pred.match_label(labels)
            else:
                pred = Prediction(int(log[0]), int(log[1]), int(log[2]), int(log[3]), log[4])
                pred.match_label(labels)
            if pred.matched_label:
                if shape != 'multi' or not pred.is_dont_care:
                    preds.append(pred)
    return preds, time


def calc_sample_precision_recall(sample, thresholds, shape):
    """
    Calculates precision & recall for a single SampleData (single image) for all given thresholds.
    Stores the results in a dict: sample.stats
    :param sample: The sample to calculate stats for.
    :param thresholds: A list of thresholds (in [0, 1]) to calculate stats for.
    """
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

        if shape == 'multi':
            n_labels = len([x for x in sample.labels if x.class_id != -1])
        else:
            n_labels = len(sample.labels)
        r = len(set(matched_labels)) / n_labels if n_labels > 0 else 1.0
        sample.stats[th] = {'precision': p, 'recall': r}


class Benchmark:
    def __init__(self, algo_id, scale=None, fake=False, persist=False, shape='rect'):
        """
        :param algo_id: Only for documentation purposes
        :param scale: Only for documentation purposes
        :param fake: ~IRRELEVANT~ If true, the Benchmark produces fake results to test.
        :param persist: ~IRRELEVANT~ If true, the Benchmark saves the fake results to a directory
        """
        self.sample_list = []
        self.algo_id = algo_id
        self.scale = scale
        self.roc = None
        self.center_dist_list = None
        self.run_time = None

        self.fake = fake
        self.persist = persist
        self.shape = shape

    def parse_sample_results(self, image_path, labels_root, preds_root):
        """
        Parse a single sample data into a SampleData, given its image path and the roots of labels & predictions.
        :param image_path: The path to the image.
        :param labels_root: The path to the labels directory.
        :param preds_root: The path to the predictions directory.
        :return: A SampleData for the image, containing parsed labels and predictions.
        """
        labels_path = str(Path(labels_root) / Path(image_path).name).replace('.jpg', '.txt')
        preds_path = str(Path(preds_root) / Path(image_path).name).replace('.jpg', '.txt')
        # print("preds_path", preds_path)
        labels = parse_labels(labels_path, shape=self.shape)
        if self.fake:
            preds, time = generate_fake_preds(preds_path, labels, persist=self.persist)
        else:
            preds, time = parse_preds(preds_path, labels, shape=self.shape)
        return SampleData(image_path, labels, preds, time, self.scale, self.algo_id)

    def parse_experiment_results(self, images_source, labels_root, preds_root):
        """
        :param images_source: Images root directory OR txt file containing paths
        :param labels_root: Labels root directory
        :param preds_root: Results (preds) root directory
        """
        if self.fake and self.persist:
            if os.path.exists(preds_root):
                shutil.rmtree(preds_root)
            os.makedirs(preds_root)

        image_paths = load_image_paths(images_source)
        for image_path in image_paths:
            sample_data = self.parse_sample_results(image_path, labels_root, preds_root)
            self.sample_list.append(sample_data)
        if len(self.sample_list) < 1:
            print('WARNING! No samples were found during parsing.')

    def calc_stats(self, thresholds=None):
        """
        Calculates the ROC for the given thresholds, and a histogram of center-distances.
        :param thresholds: A list of thresholds (in [0, 1]) to calculate stats for.
        """
        self.calc_roc(thresholds=thresholds)
        self.calc_hist()
        self.calc_time()

    def calc_time(self):
        time_sum = 0.0
        for sample in self.sample_list:
            time_sum += sample.time
        self.run_time = time_sum

    def calc_hist(self):
        """
        Calculates a histogram of center-distances.
        """
        dists = [pred.center_dist for sample in self.sample_list for pred in sample.preds]
        self.center_dist_list = list(filter(lambda dist: dist is not None, dists))

    def calc_roc(self, thresholds=None):
        """
        Calculates the ROC for the given thresholds
        :param thresholds: A list of thresholds (in [0, 1]) to calculate stats for.
        """
        if thresholds is None:
            thresholds = np.linspace(0, 1, 11)
        self.roc = {}

        th2samples_precision_list = {th: [] for th in thresholds}
        th2samples_recall_list = {th: [] for th in thresholds}

        for sample in self.sample_list:
            calc_sample_precision_recall(sample, thresholds, self.shape)

            for th, p_r_dict in sample.stats.items():
                th2samples_precision_list[th].append(p_r_dict['precision'])
                th2samples_recall_list[th].append(p_r_dict['recall'])

        for th in thresholds:
            p = sum(th2samples_precision_list[th]) / len(th2samples_precision_list[th])
            r = sum(th2samples_recall_list[th]) / len(th2samples_recall_list[th])
            self.roc[th] = {'precision': p, 'recall': r}

    @staticmethod
    def has_labels(sample):
        return len(sample.labels) >= 1

    @staticmethod
    def no_preds(sample):
        return len(sample.preds) < 1 and Benchmark.has_labels(sample)

    @staticmethod
    def low_recall(sample):
        return sample.stats[0]['recall'] < 1

    @staticmethod
    def high_recall(sample):
        return sample.stats[0]['recall'] > 0.9

    @staticmethod
    def high_precision(sample):
        return sample.stats[0]['precision'] > 0.9

    @staticmethod
    def high_stats(sample):
        return Benchmark.high_recall(sample) and \
               Benchmark.high_precision(sample) and \
               Benchmark.has_labels(sample)

    def choose_samples(self, num_samples=9, cond=lambda s: True):
        """
        Returns :num_samples samples from the benchmark's sample_list, that conforms to the specified condition.
        :param num_samples: How many samples to return (max)
        :param cond: A predicate that returns True/False given a SampleData.
        :return: A list of samples that conforms to the condition.
        """
        candidates = list(filter(cond, self.sample_list))
        if len(candidates) < num_samples:
            return candidates
        return np.random.choice(candidates, num_samples, replace=False)

    def count_samples(self, cond=lambda s: True):
        """
        Returns the number of samples from the benchmark's sample_list, that conforms to the specified condition.
        :param cond: A predicate that returns True/False given a SampleData.
        :return: The number of samples that conforms to the condition.
        """
        return len(list(filter(cond, self.sample_list)))


if __name__ == "__main__":
    project_root = 'C:\\Users\\dana\\Documents\\Ido\\follow_up_project'
    data_root = os.path.join(project_root, 'datasets', 'walking_benchmark')
    images_root = os.path.join(data_root, 'images')
    labels_root = os.path.join(data_root, 'labels')
    results_root = os.path.join(project_root, 'benchmark', 'walking_benchmark', '2019_09_05_multitracker', 'results')
    images_root = r'C:\\Users\dana\Documents\Ido\follow_up_project\datasets\walking_benchmark\filenames_from13.txt'

    bm = Benchmark("experiment", fake=False, persist=False, shape='rect')
    # bm = Benchmark("experiment")
    bm.parse_experiment_results(images_root, labels_root, results_root)
    bm.calc_stats()
    stop = 0
