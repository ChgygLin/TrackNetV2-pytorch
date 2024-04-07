import cv2
import os
import sys
from natsort import natsorted

import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('imgs_base', type=str, default=None, help='Path to the imgs dir.')
    parser.add_argument('--ori', type=str, default=None, help='ori video file.')
    parser.add_argument('--fps', type=str, default=None, help='fps')
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = parse_opt()

    assert(opt.imgs_base is not None)

    fps = 30
    if opt.fps is not None:
        fps = int(opt.fps)
    else:
        if opt.ori is None:
            opt.ori = opt.imgs_base.replace("images", "videos")
            opt.ori = f"{opt.ori}.mp4"

            if os.path.exists(opt.ori):
                print(f"found ori video {opt.ori}")
            else:
                opt.ori = None

        if opt.ori is not None:
            cap = cv2.VideoCapture(opt.ori)
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"get ori video {opt.ori} fps: {fps}")

    _, file_name = os.path.split(opt.imgs_base)

    imgs_files = [file for file in os.listdir(opt.imgs_base) if file.endswith('.jpg')]
    sorted_imgs_files = natsorted(imgs_files)

    img = cv2.imread(opt.imgs_base + '/' + sorted_imgs_files[0])
    h, w = img.shape[0], img.shape[1]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f'{file_name}.mp4', fourcc, fps, (w, h))

    for file in sorted_imgs_files:
        # print(f"handle {file}")
        img = cv2.imread(opt.imgs_base + '/' + file)
        out.write(img)

    print(f"save to {file_name}.mp4, fps:{fps}")
    out.release()