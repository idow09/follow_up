
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import mpld3
from pathlib import Path

from benchmark import Benchmark
from utils.utils import *
from utils.visual_utils import *


project_root = 'C:\\Users\\dana\\Documents\\Ido\\follow_up_project'
data_root = os.path.join(project_root, 'datasets', 'walking_benchmark')
images_root = os.path.join(data_root, 'images')
labels_root = os.path.join(data_root, 'labels')
results_root =os.path.join(project_root, 'benchmark', 'walking_benchmark','2019_09_05_multitracker')
images_root = r'C:\\Users\dana\Documents\Ido\follow_up_project\datasets\walking_benchmark\images\filenames_from10.txt'


bm = Benchmark("experiment", fake=False, persist=False)
# bm = Benchmark("experiment")
bm.parse_experiment_results(images_root, labels_root, results_root)
bm.calc_stats(thresholds=np.linspace(2, 3, 2))
# bm.calc_stats()

GREEN = (0, 128, 0)
RED = (0, 0, 255)
NEUTRAL = (255, 0, 0)

samples = bm.choose_samples(num_samples=9, cond=Benchmark.has_labels)

_, axs = plt.subplots(3, 3, figsize=(12, 12))
axs = axs.flatten()
for sample, ax in zip(samples, axs):
    im = create_sample_visualization(sample, 0.5, GREEN, RED, NEUTRAL, crop=True, color_mode='rgb')
    ax.imshow(im)
    ax.set_title(Path(sample.path).name)