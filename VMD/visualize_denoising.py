
path =

img_bw = 255 * (fgmask > 5).astype('uint8')
mask = img_bw
se0 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 5))
se1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 7))
se2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, se0)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se1)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, se2)
mask = np.dstack([mask, mask, mask]) / 255
# img_dn = img * mask
mask * 255