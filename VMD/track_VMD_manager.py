import cv2 as cv
import cv2
import numpy as np
import os
from time import time
from VMD.VMD_utils import VMdetector
from VMD.classification import FastClassifier
from VMD.trackers import FastTracker
from copy import deepcopy


def get_image(path):
    for root, dirs, files in os.walk(path):
        for name in files:
            frame = cv.imread(os.path.join(root, name))
            yield (frame, name)
    return


images_path = r"C:\Users\dana\Documents\Ido\follow_up_project\datasets\efi\images\try_set1"
destinate_frames_path = r"C:\Users\dana\Documents\Ido\follow_up_project\benchmark\2019_08_21_MOG2\try_set1\pipe"
# dest_path = r"C:\Users\dana\Documents\Ido\follow_up_project\benchmark\2019_08_21_MOG2\try_set1\mask"
tracker_type = 'CSRT'

if __name__ == '__main__':
    vm_detector = VMdetector()
    capture = get_image(images_path)
    frame, name = next(capture)

    begin = time()
    catched = False
    classifier = FastClassifier()
    detected_bb=None
    tracker = FastTracker(tracker_type)

    for i, cap in enumerate(capture):
        if i > 100:
            break
        if i==0:
            fgMask = vm_detector.apply_image(frame)
            continue
        print(i)
        if i==25:
            c=2
        frame, name = cap
        orig_frame = deepcopy(frame)
        timer = cv2.getTickCount()
        if catched == True and tracker.is_inited: #VMD in i-1
            ok, bbox = tracker.update(frame)
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
            # Draw bounding box
            if ok:
                # Tracking success
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
            else:
                # Tracking failure
                cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255),
                            2)
                catched = False

            # Display tracker type on frame
            cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

            # Display FPS on frame
            cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);





        if catched == False:
            fgMask = vm_detector.apply_image(orig_frame)
            denoised_mask = vm_detector.denoise()
            detections = vm_detector.detect()

            detect_bool = False
            for det in detections:
                bb_img = orig_frame[det[1]:det[3],det[0]:det[2],:]
                # # idxs=(284, 621, 324, 665),(516, 1006, 721, 1083)
                # image = image[621:665, 284:324, :]
                class_idx, class_name, cur_percentage = classifier.apply(bb_img)
                if class_idx == 723 or class_idx == 417 or class_idx == 722 or class_idx == 574:
                    detected_bb = det
                    cv2.rectangle(frame, (det[0], det[1]), (det[2], det[3]), (255, 0, 0), 2, 1)  #frame is with BBs
                    detect_bool = True
                    break
            if detect_bool==True:
                # track_bbox=(286, 622, 40, 42)
                track_bbox = (detected_bb[0], detected_bb[1], detected_bb[2]-detected_bb[0], detected_bb[3]-detected_bb[1])
                ok = tracker.init_tracker(orig_frame, track_bbox)
                catched = True


        cv.imwrite(os.path.join(destinate_frames_path, name), frame)
        # cv.imwrite(os.path.join(dest_path, name.replace(".jpg", "_mask.jpg")), fgMask)

    end = time()
    print("time: ", (end - begin) / 100)


