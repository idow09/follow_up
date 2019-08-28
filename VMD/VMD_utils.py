
import cv2
import cv2 as cv
from time import time
import os
import numpy as np
from VMD.VMD2Detection import export_detections

def get_image(path):
    for root, dirs, files in os.walk(path):
        for name in files:
            frame = cv2.imread(os.path.join(root, name))
            yield(frame, name)
    return

def denoise_foreground(img, fgmask):
    img_bw = 255*(fgmask > 5).astype('uint8')
    # se1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1,1))
    se2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6,6))
    # mask = cv2.morphologyEx(img_bw, cv2.MORPH_CLOSE, se1)
    mask = cv2.morphologyEx(img_bw, cv2.MORPH_OPEN, se2)
    mask = np.dstack([mask, mask, mask]) / 255
    img_dn = img * mask
    return mask*255


class VMdetector:
    def __init__(self, class_idx=723, backsub=None):
        if backsub != None:
            self.backsub = backsub
        else:
            self.backsub = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=30, detectShadows=True)
        self.denoised_mask = None
        self.class_idx = class_idx

    def apply_image(self, frame):
        self.cur_frame = frame
        self.fgMask = self.backsub.apply(frame)
        return self.fgMask

    def denoise(self):
        self.denoised_mask = denoise_foreground(self.cur_frame, self.fgMask)
        return self.denoised_mask

    def detect(self, denoised=True):
        if not denoised:
            mask = self.fgMask
        else:
            mask = self.denoised_mask
        # mask = cv2.cvtColor(mask.astype('float32'), cv2.COLOR_BGR2GRAY)
        mask = cv2.cvtColor(mask.astype('uint8'), cv2.COLOR_BGR2GRAY)
        detections = export_detections(mask)
        return detections

    # def filter_detections(self, detections):
    #     for det in detections:
    #         bb_img = orig_frame[det[1]:det[3], det[0]:det[2], :]
    #         # # idxs=(284, 621, 324, 665),(516, 1006, 721, 1083)
    #         # image = image[621:665, 284:324, :]
    #         class_idx, class_name, cur_percentage = classifier.apply(bb_img)
    #         if class_idx == self.class_idx:
    #             detected_bb = det
    #             detect_bool = True
    #             break

if __name__ == '__main__':


    images_path = r"C:\Users\dana\Documents\Ido\follow_up_project\datasets\efi\images\try_set1"
    destinate_frames_path = r"C:\Users\dana\Documents\Ido\follow_up_project\benchmark\2019_08_21_MOG2\try_set1\frames"
    destinate_orig_frames_path = r"C:\Users\dana\Documents\Ido\follow_up_project\benchmark\2019_08_21_MOG2\try_set1\frames_orig"
    dest_path = r"C:\Users\dana\Documents\Ido\follow_up_project\benchmark\2019_08_21_MOG2\try_set1\mask"


    method="MOG2"
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
        cv.imwrite(os.path.join(destinate_frames_path,name), denoised_frame)
        cv.imwrite(os.path.join(dest_path,name.replace(".jpg", "_mask.jpg")), fgMask)
        # keyboard = cv.waitKey(30)
        # if keyboard == 'q' or keyboard == 27:
        #     break
    end = time()
    print("time: ", (end-begin)/100)