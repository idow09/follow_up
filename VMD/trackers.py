
import cv2
import os
import sys
from time import time

frame_path = r'C:\Users\dana\Documents\Ido\follow_up_project\benchmark\2019_08_21_MOG2\try_set1\mask\efi_slomo_vid_1_0036_mask.jpg'

frame = cv2.imread(frame_path)

# (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

params_root=r'C:\Users\dana\Documents\Ido\follow_up_project\benchmark\2019_08_21_MOG2\try_set2\params'

def create_tracker(tracker_type):
    if tracker_type == 'BOOSTING':
        tracker = cv2.TrackerBoosting_create()
    if tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    if tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
        tracker.save(os.path.join(params_root,'paramsKCF.json'))

        # read
        tracker = cv2.TrackerKCF_create()
        fs = cv2.FileStorage(os.path.join(params_root,'paramsKCF2.json'), cv2.FileStorage_READ)
        tracker.read(fs.getFirstTopLevelNode())
    if tracker_type == 'TLD':
        tracker = cv2.TrackerTLD_create()
    if tracker_type == 'MEDIANFLOW':
        tracker = cv2.TrackerMedianFlow_create()
    if tracker_type == 'GOTURN':
        tracker = cv2.TrackerGOTURN_create()
    if tracker_type == 'MOSSE':
        tracker = cv2.TrackerMOSSE_create()
    if tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT_create()
    return tracker


class FastTracker():
    def __init__(self, tracker_type):
        self.tracker = create_tracker(tracker_type)
        self.is_inited = False

    def init_tracker(self, frame, bbox):
        ok = self.tracker.init(frame, bbox)
        self.is_inited = ok
        return ok

    def update(self, frame):
        ok, bbox = self.tracker.update(frame)
        return ok, bbox


if __name__ == '__main__':

    # Set up tracker.
    # Instead of MIL, you can also use
    dest_root = r"C:\Users\dana\Documents\Ido\follow_up_project\benchmark\2019_08_21_MOG2\try_set2\track_KCF2_2"
    if not os.path.exists(dest_root):
        os.makedirs(dest_root)
    tracker_types1= ['BOOSTING', 'MIL', 'TLD', 'MEDIANFLOW', 'GOTURN']
    tracker_types = ['KCF', 'MOSSE', 'CSRT']
    tracker_type = tracker_types[0]
    tracker = create_tracker(tracker_type)

    # Read video
    video = cv2.VideoCapture(r"C:\Users\dana\Documents\Ido\follow_up_project\datasets\efi\videos\efi_slomo_vid_1.mp4")

    # Exit if video not opened.
    if not video.isOpened():
        print
        "Could not open video"
        sys.exit()

    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print
        'Cannot read video file'
        sys.exit()

    # Define an initial bounding box
    # bbox = (287, 23, 86, 320)
    cv2.startWindowThread()
    cv2.namedWindow("preview")

    # Uncomment the line below to select a different bounding box
    cv2.resizeWindow('image', 600, 300)
    clone1 = cv2.resize(frame, (int(frame.shape[1]/ 2), int(frame.shape[0] / 2)), interpolation=cv2.INTER_AREA)
    refPt = cv2.selectROI(clone1, False)
    print(refPt)
    a, b, c, d = (refPt[0]) * 2, (refPt[1]) * 2, (refPt[2]) * 2, (refPt[3]) * 2
    bbox=(a, b,c,d)

    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)

    i=0
    strart = time()
    while True and i<200:
        i+=1
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break

        if i == 88:
            tracker = create_tracker(tracker_type)
            # Uncomment the line below to select a different bounding box
            clone1 = cv2.resize(frame, (int(frame.shape[1] / 2), int(frame.shape[0] / 2)), interpolation=cv2.INTER_AREA)
            refPt = cv2.selectROI(clone1, False)
            print(refPt)
            a, b, c, d = (refPt[0]) * 2, (refPt[1]) * 2, (refPt[2]) * 2, (refPt[3]) * 2
            bbox = (a, b, c, d)
            print("bbox: ", bbox)
            ok = tracker.init(frame, bbox)
        # Start timer
        timer = cv2.getTickCount()

        # Update tracker
        ok, bbox = tracker.update(frame)

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        else:
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Display tracker type on frame
        cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

        # Display result
        cv2.imshow("Tracking", frame)

        cv2.imwrite(os.path.join(dest_root, 'tracking{}.jpg'.format(i)), frame)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27: break

    end = time()
    print("time:",end-strart)
