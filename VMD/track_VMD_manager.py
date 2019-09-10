import argparse
import os
from time import time, sleep

import cv2
import cv2 as cv
import numpy as np
import torchvision.transforms as transforms
from torchvision import models
from tqdm import tqdm

from VMD.VMD_utils import VMdetector
from VMD.classification import FastClassifier
from VMD.trackers import MultiTracker
from models import resnet
from utils.utils import resize_image

YELLOW_LOWER_HUE = 20
YELLOW_UPPER_HUE = 30
BRIGHTNESS_THRESHOLD = 173
SATURATION_THRESHOLD = 173


class time_analyzer:
    def __init__(self):
        self.time_sum = 0
        self.time_counters = {}

    def add_time(self, type_name, start_time):
        time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
        if type_name in self.time_counters:
            self.time_counters[type_name] += time
        else:
            self.time_counters[type_name] = time


def get_image(path):
    for root, dirs, files in os.walk(path):
        for name in files:
            frame = cv.imread(os.path.join(root, name))
            yield (frame, name)
    return


def predict_ball_location(image, color_mode='bgr'):
    if color_mode == 'bgr':
        img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    elif color_mode == 'rgb':
        img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    else:
        raise AttributeError('Supported color modes: BGR, RGB')

    yellows = np.logical_and(img_hsv[:, :, 0] > YELLOW_LOWER_HUE, img_hsv[:, :, 0] < YELLOW_UPPER_HUE)
    brights = np.logical_and(img_hsv[:, :, 1] > SATURATION_THRESHOLD, img_hsv[:, :, 2] > BRIGHTNESS_THRESHOLD)
    ball_pxls = np.logical_and(yellows, brights)
    ball_pxls_sum = np.sum(ball_pxls)
    not_ball_pxls = np.sum(np.ones(image.shape[:2])) - ball_pxls_sum
    # ys, xs = np.where(ball_pxls)
    # try:
    #     x1 = min(xs)
    #     x2 = max(xs)
    #     y1 = min(ys)
    #     y2 = max(ys)
    #     return (x1, y1, x2, y2), 1.0
    # except ValueError:
    #     return (0, 0, 0, 0), 0.0
    if ball_pxls_sum / float(not_ball_pxls) > 0.3:
        return True
    else:
        return False


def analyze_frame(vm_detector, catched, multy_tracker, detector_counter, timer_analyzer, orig_frame):
    if detector_counter > 10:
        catched = False
    if multy_tracker and catched == True and multy_tracker.is_working:  # VMD in i-1
        start_track = cv2.getTickCount()
        detection_bbs = operate_tracker(multy_tracker)
        timer_analyzer.add_time("tracking", start_track)
    if not catched:
        detection_bbs = vm_detector.detect(orig_frame)
        if len(detection_bbs) != 0 and multy_tracker:
            for detected_bb in detection_bbs:
                track_bbox = (
                    detected_bb[0], detected_bb[1], detected_bb[2] - detected_bb[0], detected_bb[3] - detected_bb[1])

            tracker_id = multy_tracker.init_new_tracker(orig_frame, track_bbox)
            if tracker_id == -1:
                catched = False

    return detection_bbs

def operate_tracker(tracker, detected_bb, catched, detector_counter):
    tracker_oks, bboxs = tracker.update(frame)
    detections = []
    # Draw bounding box
    for k, ok in enumerate(tracker_oks):
        if ok:
            # Tracking success
            bbox = bboxs[k]
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
            detected_bb = (p1[0], p1[1], p2[0], p2[1])
            detections.append(detected_bb)
        else:
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255),
                        2)
            catched = False
    # Display tracker type on frame
    cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);
    if len(detections) > 0:
        detector_counter += 1
    return detections


def write_detections(preds_path, frame_time, bbs):
    with open(preds_path, 'w') as file:
        file.write('%g\n' % frame_time)
    sc = 1.0
    for bb in bbs:
        # w=(bb[2]-bb[0])
        # h=(bb[3]-bb[1])
        # c_x = bb[0] + w/2
        # c_y = bb[1] + h/2
        # r = (w+h)/4.0
        with open(preds_path, 'a') as file:
            file.write(('%g ' * 5 + '\n') % (bb[0], bb[1], bb[2], bb[3], sc))


def write_time(destinate_frames_path, images_path, cur_idx):
    path = os.path.join(destinate_frames_path, os.path.basename(images_path) + ".txt")
    print("times analyzer:")
    with open(path, 'w') as time_f:
        for key, val in timer_analyzer.time_counters.items():
            line = str(key) + " " + str(val / cur_idx) + "\n"
            time_f.write(line)
            print(line)


if __name__ == '__main__':
    # images_path = r"C:\Users\dana\Documents\Ido\follow_up_project\datasets\efi\images\try_set1"
    # destinate_frames_path = r"C:\Users\dana\Documents\Ido\follow_up_project\benchmark\2019_08_21_MOG2\try_set1\pipe"
    p = argparse.ArgumentParser()
    p.add_argument('--images_path',  type=str, default= r'C:\Users\dana\Documents\Ido\follow_up_project\datasets\walking_benchmark\images')
    p.add_argument('--destinate_frames_path',  type=str, default=r'C:\Users\dana\Documents\Ido\follow_up_project\benchmark\walking_benchmark\2019_09_08_test')
    p.add_argument('--destinate_frames_path2',  type=str, default=r"C:\Users\dana\Documents\Ido\follow_up_project\benchmark\walking_benchmark\2019_09_08_test2")
    p.add_argument('--model_type', type=str, default='torchvision')
    args = p.parse_args()

    os.makedirs(args.destinate_frames_path, exist_ok=True)
    os.makedirs(args.destinate_frames_path2, exist_ok=True)
    # for j in range(1,4):
    #     images_path = r'C:\Users\dana\Documents\Ido\follow_up_project\datasets\efi\images\slomo{}'.format(j)

    # load model
    if args.model_type == 'torchvision':
        # model = models.resnet50(pretrained=True)
        model = models.resnet50(pretrained=True)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        classes_file_path = r'C:\Users\dana\Documents\Ido\follow_up_project\datasets\imagenet_classes.txt'
    elif args.model_type in ['cifar100', 'cifar10']:
        model = resnet.cifar_resnet32(pretrained=args.model_type)
        normalize = transforms.Normalize(
            mean=[0.5071, 0.4865, 0.4409]  # for cifar100
            , std=[0.2009, 0.1984, 0.2023])  # for cifar100)
        classes_file_path = r'C:\Users\dana\Documents\Ido\follow_up_project\datasets\cifar_100_classes.txt'

    # classes initialization
    timer_analyzer = time_analyzer()
    denoised_mask = None
    begin = time()
    fgMask = None
    catched = False
    detector_counter = 0
    classifier = FastClassifier(model, normalize, classes_file_path)
    vm_detector = VMdetector()
    detector_frames_init = 10
    detected_bb = None
    tracker_type = 'CSRT'
    multy_tracker = None

    # running over frames
    capture = get_image(args.images_path)
    frame, name = next(capture)
    for i, cap in tqdm(enumerate(capture)):
        orig_frame, name = cap
        timer = cv2.getTickCount()
        scale=1
        # run over current frame
        if i < detector_frames_init:
            vm_detector.apply_image(frame)
        detection_bbs = analyze_frame(vm_detector, catched, multy_tracker, detector_counter, timer_analyzer, orig_frame)


        # analyze detection
        if len(detection_bbs) > 0:
            scaled_detection_bbs = []
            for detected_bb in detection_bbs:
                detected_bbs_scaled = (
                    int(detected_bb[0] / scale), int(detected_bb[1] / scale), int(detected_bb[2] / scale),
                    int(detected_bb[3] / scale))
                scaled_detection_bbs.append(detected_bbs_scaled)
            detection_bbs = scaled_detection_bbs

        frame_tics = cv2.getTickCount() - timer
        fps = cv2.getTickFrequency() / (frame_tics)
        frame_time = 1 / float(fps)
        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

        if i == 10:
            if tracker_type != "":
                multy_tracker = MultiTracker(tracker_type)
        if i < 11:
            continue

        dest_img_path = os.path.join(args.destinate_frames_path, name)
        dest_img_path2 = os.path.join(args.destinate_frames_path2, name)
        cv.imwrite(dest_img_path, frame)
        cv.imwrite(dest_img_path2, frame)
        # cv.imwrite(os.path.join(destinate_frames_path, name.replace(".jpg", "_mask.jpg")), fgMask)
        # cv.imwrite(os.path.join(destinate_frames_path, name.replace(".jpg", "_mask_de.jpg")), denoised_mask)
        pred_path = os.path.join(args.destinate_frames_path, name.replace(".jpg", ".txt"))
        print("detected_bb: ", detected_bb)
        write_detections(pred_path, frame_time, detection_bbs)

    end = time()
    print("time: ", (end - begin) / i)
    write_time(args.destinate_frames_path, args.images_path, i)