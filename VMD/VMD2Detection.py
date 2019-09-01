
import cv2
import numpy as np


def imshow_components(labels, show=False):
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0

    # cv2.imshow('labeled.png', labeled_img)
    if show:
        cv2.imwrite(r"C:\Users\dana\Documents\Ido\follow_up_project\benchmark\2019_08_21_MOG2\try_set2\try\conCom.jpg", labeled_img)
    # cv2.waitKey()

def export_contour_detection(img):
    # mask = np.zeros(img.shape, np.uint8)
    img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]  # ensure binary
    contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # draw the white paper and eliminate the small pieces (less than 1000000 px). This px count is the same as the QR code dectection
    detections = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 30:
            # cv2.drawContours(mask, [cnt], 0, 255,
            #                  -1)  # the [] around cnt and 3rd argument 0 mean only the particular contour is drawn

            # Build a ROI to crop the QR
            x, y, w, h = cv2.boundingRect(cnt)
            # print(x, y, w, h)
            # roi = mask[y:y + h, x:x + w]
            # crop the original QR based on the ROI
            # QR_crop = orig[y:y + h, x:x + w]
            # use cropped mask image (roi) to get rid of all small pieces
            # QR_final = QR_crop * (roi / 255)
            detections.append((x, y, x+w, y+h))
    return detections


def export_detections(img, show=False):
    # img = np.array(img, dtype=np.uint8)
    img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]  # ensure binary
    ret, labels = cv2.connectedComponents(img, connectivity=8)
    # if show:
    #     imshow_components(labels)
    detections=[]
    for idx in range(ret):
        if np.sum(([labels == idx])) > 150 and idx != 0:
            label_ar = np.array(labels)
            points = np.where(label_ar == idx)
            # ymin = max(min(points[0]) - 5,0)
            # ymax = min(max(points[0]) + 5, img.shape[0]-1)
            # xmin = max(min(points[1]) - 5,0)
            # xmax = min(max(points[1]) + 5, img.shape[1]-1)
            ymin = min(points[0])
            ymax = max(points[0])
            xmin = min(points[1])
            xmax = max(points[1])
            detection = (xmin, ymin, xmax, ymax)
            detections.append(detection)
            if show:
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2, 1)
    if show:
        return detections, img
    return detections

# def detect_contours():
#     ret, thresh = cv2.threshold(imgray, 127, 255, 0)
#     im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


if __name__ == '__main__':
    # mask_path = r"C:\Users\dana\Documents\Ido\follow_up_project\benchmark\2019_08_21_MOG2\try_set1\mask\efi_slomo_vid_1_0088_mask.jpg"
    mask_path = r"C:\Users\dana\Documents\Ido\follow_up_project\benchmark\2019_08_21_MOG2\try_set1\frames\efi_slomo_vid_1_0088.jpg"
    orig_path = r'C:\Users\dana\Documents\Ido\follow_up_project\datasets\efi\images\slomo1\efi_slomo_vid_1_0088.jpg'
    orig = cv2.imread(orig_path)
    img0=cv2.imread(mask_path)
    # img = cv2.imread(mask_path, 0)
    img = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    # ret, thresh = cv2.threshold(img, 127, 255, 0)
    # rows, cols=np.nonzero(img)




    # detections, img = export_detections(img, True)
    # print(detections)


    # using findContours func to find the none-zero pieces
    mask = np.zeros(img.shape, np.uint8)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # draw the white paper and eliminate the small pieces (less than 1000000 px). This px count is the same as the QR code dectection
    for cnt in contours:
        if cv2.contourArea(cnt) > 100:
            # cv2.drawContours(mask, [cnt], 0, 255,
            #                  -1)  # the [] around cnt and 3rd argument 0 mean only the particular contour is drawn

            # Build a ROI to crop the QR
            x, y, w, h = cv2.boundingRect(cnt)
            print(x, y, w, h)
            roi = mask[y:y + h, x:x + w]
            # crop the original QR based on the ROI
            QR_crop = orig[y:y + h, x:x + w]
            # use cropped mask image (roi) to get rid of all small pieces
            # QR_final = QR_crop * (roi / 255)
            cv2.imwrite(
                r'C:\Users\dana\Documents\Ido\follow_up_project\benchmark\2019_08_21_MOG2\try_set1\try\bbs2.jpg',
                QR_crop)

    cv2.imwrite(
            r'C:\Users\dana\Documents\Ido\follow_up_project\benchmark\2019_08_21_MOG2\try_set1\try\bbs2.jpg',
            img)
