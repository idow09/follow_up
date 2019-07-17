import glob
import os
import argparse
import cv2


def break_videos(from_path, to_dir):
    vid_formats = ['.mov', '.avi', '.mp4']
    files = []
    if os.path.isdir(from_path):
        files = sorted(glob.glob('%s/*.*' % from_path))
    elif os.path.isfile(from_path):
        files = [from_path]
    videos = [x for x in files if os.path.splitext(x)[-1].lower() in vid_formats]
    n_v = len(videos)

    if not os.path.exists(to_dir):
        os.makedirs(to_dir)

    for v, vid in enumerate(videos):
        cap = cv2.VideoCapture(vid)
        i = 0
        ret = True
        im_path = to_dir + os.sep + os.path.split(vid)[1].replace('.mp4', '.jpg')
        while cap.isOpened() and ret:
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(im_path.replace('.jpg', '_%04d.jpg' % i), frame)
                i += 1

        # When everything done, release the capture
        cap.release()
        print('video %d/%d: %d frames' % (v + 1, n_v, i))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--from-path', type=str, default='data/videos', help='path of source videos')
    parser.add_argument('--to-dir', type=str, default='data/frames', help='path of destination frames')
    opt = parser.parse_args()
    print(opt)

    break_videos(opt.from_path, opt.to_dir)
