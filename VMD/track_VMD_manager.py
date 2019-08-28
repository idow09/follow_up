import cv2 as cv
import cv2
import numpy as np
import os
from time import time
from VMD.VMD_utils import VMdetector
from VMD.classification import FastClassifier
from VMD.trackers import FastTracker
from copy import deepcopy

YELLOW_LOWER_HUE = 20
YELLOW_UPPER_HUE = 30
BRIGHTNESS_THRESHOLD = 173
SATURATION_THRESHOLD = 173


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


def analyze_frame():
    global detected_bb, catched, fgMask, denoised_mask, tracker
    if tracker and catched == True and tracker.is_inited:  # VMD in i-1
        operate_tracker(tracker)
    if not catched:
        operate_VMD_detection()
    return detected_bb


def operate_VMD_detection():
    global fgMask, denoised_mask, detected_bb, tracker, catched
    fgMask = vm_detector.apply_image(orig_frame)
    denoised_mask = vm_detector.denoise()
    detections = vm_detector.detect()
    detect_bool = False
    for det in detections:
        bb_img = orig_frame[det[1]:det[3], det[0]:det[2], :]
        # idxs=(284, 621, 324, 665),image = image[621:665, 284:324, :]
        class_idx, class_name, cur_percentage = classifier.apply(bb_img)
        # # if class_idx == 723 or class_idx == 417 or class_idx == 722 or class_idx == 574 or class_idx == 574 or class_idx == 920:
        #     detected_bb = det
        #     cv2.rectangle(frame, (det[0], det[1]), (det[2], det[3]), (255, 0, 0), 2, 1)  #frame is with BBs
        #     detect_bool = True
        #     break
        is_ball = predict_ball_location(bb_img)
        if is_ball:
            detected_bb = det
            cv2.rectangle(frame, (det[0], det[1]), (det[2], det[3]), (255, 0, 0), 2, 1)  # frame is with BBs
            detect_bool = True
            break
    if detect_bool == True and tracker:
        # track_bbox=(286, 622, 40, 42)
        track_bbox = (
            detected_bb[0], detected_bb[1], detected_bb[2] - detected_bb[0], detected_bb[3] - detected_bb[1])
        tracker = FastTracker(tracker_type)
        tracker_ok = tracker.init_tracker(orig_frame, track_bbox)
        if tracker_ok:
            catched = True
        else:
            catched = False


def operate_tracker(tracker):
    global detected_bb, catched
    tracker_ok, bbox = tracker.update(frame)
    # Draw bounding box
    if tracker_ok:
        # Tracking success
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        detected_bb = (p1[0], p1[1], p2[0], p2[1])

    else:
        # Tracking failure
        cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255),
                    2)
        catched = False
    # Display tracker type on frame
    cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

def write_detection(preds_path, frame_time, bb):
    if bb == ():
        with open(preds_path, 'w') as file:
            file.write('%g\n' % frame_time)
            return
    w=(bb[2]-bb[0])
    h=(bb[3]-bb[1])
    c_x = bb[0] + w/2
    c_y = bb[1] + h/2
    r = (w+h)/4.0
    sc = 1.0
    with open(preds_path, 'w') as file:
        file.write('%g\n' % frame_time)
        file.write(('%g ' * 4 + '\n') % (c_x, c_y, r, sc))


if __name__ == '__main__':
    # images_path = r"C:\Users\dana\Documents\Ido\follow_up_project\datasets\efi\images\try_set1"
    # destinate_frames_path = r"C:\Users\dana\Documents\Ido\follow_up_project\benchmark\2019_08_21_MOG2\try_set1\pipe"
    images_path = r'C:\Users\dana\Documents\Ido\follow_up_project\datasets\efi\images\slomo2'
    destinate_frames_path = r'C:\Users\dana\Documents\Ido\follow_up_project\benchmark\2019_08_28_VMD_and_tracker'
    # destinate_frames_path = r"C:\Users\dana\Documents\Ido\follow_up_project\benchmark\2019_08_21_MOG2\try_set2\pipe_noslomo"

    tracker_type = 'CSRT'

    vm_detector = VMdetector()
    capture = get_image(images_path)
    frame, name = next(capture)
    denoised_mask = None
    begin = time()
    fgMask = None
    catched = False
    classifier = FastClassifier()
    detected_bb = None
    if tracker_type != "":
        tracker = FastTracker(tracker_type)
    else:
        tracker = None

    for i, cap in enumerate(capture):
        # if i > 100:
        #     break
        if i == 0:
            fgMask = vm_detector.apply_image(frame)
            continue
        print(i)
        frame, name = cap
        orig_frame = deepcopy(frame)
        timer = cv2.getTickCount()

        detected_bb = ()


        detected_bb = analyze_frame()

        frame_tics = cv2.getTickCount() - timer
        fps = cv2.getTickFrequency() / (frame_tics)
        frame_time = 1/float(fps)
        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

        dest_img_path = os.path.join(destinate_frames_path, name)
        cv.imwrite(dest_img_path, frame)
        # cv.imwrite(os.path.join(destinate_frames_path, name.replace(".jpg", "_mask.jpg")), fgMask)
        # cv.imwrite(os.path.join(destinate_frames_path, name.replace(".jpg", "_mask_de.jpg")), denoised_mask)
        pred_path = os.path.join(destinate_frames_path, name.replace(".jpg", ".txt"))
        print("detected_bb: ",detected_bb)
        write_detection(pred_path, frame_time, detected_bb)

    end = time()
    print("time: ", (end - begin) / 100)
