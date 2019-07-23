import argparse
import shutil
import time
from pathlib import Path

from utils.visual_utils import plot_ball
from utils.utils import *

YELLOW_LOWER_HUE = 20
YELLOW_UPPER_HUE = 30
BRIGHTNESS_THRESHOLD = 173
SATURATION_THRESHOLD = 173


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
    ys, xs = np.where(ball_pxls)
    try:
        x1 = min(xs)
        x2 = max(xs)
        y1 = min(ys)
        y2 = max(ys)
        return (x1, y1, x2, y2), 1.0
    except ValueError:
        return (0, 0, 0, 0), 0.0


def extract_vid_name(path):
    name_parts = Path(path).name.split('_')
    name_parts = name_parts[:-1]
    return '_'.join(name_parts)


def detect_ball(source, output, visualize, leave_trace=True):
    if os.path.exists(output):
        shutil.rmtree(output)  # delete output folder
    os.makedirs(output)  # make new output folder

    image_paths = load_image_paths(source)
    n_i = len(image_paths)
    current_vid = None

    tot = time.time()
    for i, path in enumerate(image_paths):
        save_path = str(Path(output) / Path(path).name)

        if leave_trace:
            vid_name = extract_vid_name(path)
            if current_vid != vid_name:
                current_vid = vid_name
                pts = np.empty((0, 2), dtype=np.int32)

        tic = time.time()
        img = cv2.imread(path)
        xyxy, sc = predict_ball_location(img)
        t = time.time() - tic

        xyr = xyxy2xyr(*xyxy)
        if visualize:
            plot_ball(img, xyr, color=(255, 0, 0))
            if leave_trace:
                if sc == 0:
                    pts = np.empty((0, 2), dtype=np.int32)
                else:
                    # noinspection PyUnboundLocalVariable
                    pts = np.vstack((pts, np.array([xyr[0], xyr[1]], dtype=np.int32)))
                if len(pts) > 0:
                    cv2.polylines(img, [pts], isClosed=False, color=(255, 0, 0), thickness=2)
            cv2.imwrite(save_path, img)
        with open(save_path + '.txt', 'a') as file:
            # file.write('%g\n' % t)
            if sc != 0:
                file.write(('%g ' * 3 + '\n') % xyr)
        print('%i/%i Done. (%.3fs)' % (i + 1, n_i, t))
    print('Done All. (%.3fs)' % (time.time() - tot))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='data/samples', help='path to images')
    parser.add_argument('--output', type=str, default='output', help='specifies the output path for images and videos')
    parser.add_argument('--visualize', type=str, default=True, help='specifies whether to visualize')
    opt = parser.parse_args()
    print(opt)

    detect_ball(
        source=opt.source,
        output=opt.output,
        visualize=opt.visualize
    )
