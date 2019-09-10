import os
from time import time

import cv2
import cv2 as cv
import numpy as np

from utils.utils import resize_image
from VMD.VMD2Detection import export_contour_detection


def get_image(path):
    for root, dirs, files in os.walk(path):
        for name in files:
            frame = cv2.imread(os.path.join(root, name))
            yield (frame, name)
    return


def denoise_foreground(img, fgmask):

    img_bw = 255 * (fgmask > 5).astype('uint8')
    # mask = img_bw
    se0 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 5))
    se1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 7))
    se2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    mask1 = cv2.morphologyEx(img_bw, cv2.MORPH_CLOSE, se0)
    mask2 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, se1)
    mask3 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, se2)
    mask = np.dstack([mask3, mask3, mask3]) / 255
    # img_dn = img * mask
    # if True:
        # path = r'C:\Users\dana\Documents\Ido\follow_up_project\benchmark\walking_benchmark\try'
        # cv2.imwrite(os.path.join(path, "orig.jpg"), img)
        # mask1 = np.dstack([mask1, mask1, mask1])
        # mask2 = np.dstack([mask2, mask2, mask2])
        # mask3 = mask * 255
        # cv2.imwrite(os.path.join(path, "vmd.jpg"), fgmask)
        # cv2.imwrite(os.path.join(path, "mask0.jpg"), img_bw)
        # cv2.imwrite(os.path.join(path, "mask1.jpg"), mask1)
        # cv2.imwrite(os.path.join(path, "mask2.jpg"), mask2)
        # cv2.imwrite(os.path.join(path, "mask3.jpg"), mask3)
    return mask * 255


class VMdetector:
    def __init__(self, valid_class_list=None, backsub=None, classifier=None, frame_size=416):
        if valid_class_list is None:
            valid_class_list = [723]
        if backsub != None:
            self.backsub = backsub
        else:
            self.backsub = cv2.createBackgroundSubtractorMOG2(history=5, varThreshold=10, detectShadows=True)
        self.denoised_mask = None
        self.valid_class_list = valid_class_list
        self.classifier = classifier
        self.frame_size = frame_size

    def apply_image(self, frame, resize=True):
        if resize:
            frame, _ = resize_image(frame, self.frame_size)
        self.fgMask = self.backsub.apply(frame, learningRate=0.05)
        return self.fgMask

    def denoise(self, frame, resize=True):
        if resize:
            frame, _ = resize_image(frame, self.frame_size)
        self.denoised_mask = denoise_foreground(frame, self.fgMask)
        return self.denoised_mask

    def filter_detections(self, orig_frame, detections, scale, bb_size=None):
        detect_bool = False
        correct_detections = []
        for det in detections:
            det = np.array(det)
            if type(scale) == tuple:
                det[[0, 2]] = det[[0, 2]] / scale[0]
                det[[1, 3]] = det[[1, 3]] / scale[1]
            else:
                det = (det / scale).astype(np.int)
            bb_img = orig_frame[det[1]:det[3], det[0]:det[2], :]
            if bb_size:
                bb_img = cv2.resize(bb_img, scale)
            class_idx, class_name, cur_percentage = self.classifier.apply(bb_img)
            print(class_name)
            if class_idx in self.valid_class_list:
                cv2.rectangle(frame, (det[0], det[1]), (det[2], det[3]), (255, 0, 0), 2, 1)  # frame is with BBs
                detect_bool = True
            if detect_bool:
                cv2.rectangle(frame, (det[0], det[1]), (det[2], det[3]), (255, 0, 0), 2, 1)  # frame is with BBs
                detect_bool = True
                correct_detections.append(det)
        return detect_bool, correct_detections

    def detect(self, orig_frame, denoised=True):
        frame, scale = resize_image(orig_frame, self.frame_size)
        self.apply_image(frame,  resize=False)
        if not denoised:
            mask = self.fgMask
        else:
            self.denoise(frame,  resize=False)
            mask = self.denoised_mask

        mask = cv2.cvtColor(mask.astype('uint8'), cv2.COLOR_BGR2GRAY)
        detections = export_contour_detection(mask)

        if self.classifier:
            detect_bool, detections = self.filter_detections(orig_frame, detections, scale)
            if not detect_bool:
                detections = []

        return detections



if __name__ == '__main__':

    images_path = r"C:\Users\dana\Documents\Ido\follow_up_project\datasets\efi\images\try_set1"
    destinate_frames_path = r"C:\Users\dana\Documents\Ido\follow_up_project\benchmark\2019_08_21_MOG2\try_set1\frames"
    destinate_orig_frames_path = r"C:\Users\dana\Documents\Ido\follow_up_project\benchmark\2019_08_21_MOG2\try_set1\frames_orig"
    dest_path = r"C:\Users\dana\Documents\Ido\follow_up_project\benchmark\2019_08_21_MOG2\try_set1\mask"

    method = "MOG2"
    if method == 'MOG2':
        vm_detector = VMdetector()
        # backSub = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=300, detectShadows=True)
        # cv.BackgroundSubtractorMOG2.setBackgroundRatio(backSub,0.3)
    else:
        backSub = cv2.createBackgroundSubtractorKNN()
        vm_detector = VMdetector(backSub)

    capture = get_image(images_path)

    frame, name = next(capture)
    begin = time()

    for i, cap in enumerate(capture):
        if i > 100:
            break
        frame, name = cap
        # frame = cv2.medianBlur(frame, 9)
        fgMask = vm_detector.apply_image(frame)
        # fgdn=frame
        # fgdn = denoise_foreground(frame, fgMask)

        # cv.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
        # cv.putText(frame, str(i+1), (15, 15),
        #            cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
        #
        denoised_frame = vm_detector.denoise()
        # frame = cv2.medianBlur(frame, 5)
        # fgMask = cv2.medianBlur(fgMask, 5)
        cv.imwrite(os.path.join(destinate_orig_frames_path, name), frame)
        cv.imwrite(os.path.join(destinate_frames_path, name), denoised_frame)
        cv.imwrite(os.path.join(dest_path, name.replace(".jpg", "_mask.jpg")), fgMask)
        # keyboard = cv.waitKey(30)
        # if keyboard == 'q' or keyboard == 27:
        #     break
    end = time()
    print("time: ", (end - begin) / 100)
