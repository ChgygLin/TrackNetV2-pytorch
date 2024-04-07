from pathlib import Path
import cv2
import pandas as pd
import numpy as np
import os
import argparse
import sys
import time
import json

import scipy
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from utils.general import Pcalib, visualize_court, court

# python show_label_all.py ./dataset/match1/videos/1_10_12.mp4 --label dataset/match/labels
# 若不加label,则会默认从mp4文件同级目录读取

COLOR_calc = (0, 255, 0)
COLOR_label = (0, 255, 255)


# state 0:hidden  1:visible
state_name = ['HIDDEN', 'VISIBLE']

keybindings = {
    'next':          [ ord('n') ],
    'prev':          [ ord('p')],

    'piece_start':   [ ord('s'), ],     # 裁剪开始帧
    'piece_end':     [ ord('e'), ],     # 裁剪结束帧

    'first_frame':   [ ord('z'), ],
    'last_frame':    [ ord('x'), ],

    'forward_frames':   [ ord('f'), ],      #   前进36帧
    'backward_frames':  [ ord('b'), ],      #   后退36帧

    'circle_grow':   [ ord('='), ord('+') ],
    'circle_shrink': [ ord('-'), ],

    'quit':          [ ord('q'), ],
}

def reprojection_error(P, points3D, points2D):
    P = P.reshape(3, 4)
    projected_points = np.dot(P, np.hstack((points3D, np.ones((points3D.shape[0], 1)))).T).T
    projected_points = projected_points[:, :2] / projected_points[:, 2][:, np.newaxis]
    error = np.linalg.norm(projected_points - points2D)
    return error

def ransac_reprojection(initial_guess_P, points_3d, points_2d, threshold=5, max_iterations=10):
    best_P = None
    best_inliers = []

    for i in range(max_iterations):
        # 随机采样4个点
        sample_indices = np.random.choice(len(points_3d), size=6, replace=False)
        sample_points_3d = points_3d[sample_indices]
        sample_points_2d = points_2d[sample_indices]

        # 计算投影矩阵
        ransac_result = minimize(reprojection_error, initial_guess_P, args=(sample_points_3d, sample_points_2d), method='BFGS')
        P = ransac_result.x.reshape(3, 4)

        # 计算所有点的重投影误差
        uv2 = np.dot( P, np.concatenate( (points_3d.T, np.ones((1, len(points_3d)))) ) )
        uv2 = ((uv2 / uv2[2, :]).T).astype(np.int32)
        all_errors = np.sqrt(np.sum((uv2[:, :2] - points_2d)**2, axis=1))

        # 计算内点
        inliers = np.where(all_errors < threshold)[0]

        # 如果内点数量超过当前最优解，则更新最优解
        if len(inliers) > len(best_inliers):
            best_P = P
            best_inliers = inliers

    return best_P, best_inliers

def fit_3d_2d(kps_int, num, Ps):
    kps_court = np.zeros((32, 7), dtype=np.float32)
    kps_court[:, :3] = kps_int
    kps_court[:, 3:6] = court
    kps_court[:, 6] = [i for i in range(32)]

    selection = kps_court[kps_court[:, 2] == 1]
    uv = selection[:, :2]
    xyz = selection[:, 3:6]
    mask_label = selection[:, 6]

    if kps_int[30][2]==0 or kps_int[31][2]==0:
        if num == 0:
            initial_guess_P = None
        else:
            if Ps[-1] is not None:
                initial_guess_P = Ps[-1].ravel()
            else:
                initial_guess_P = None

        result = scipy.optimize.minimize(reprojection_error, initial_guess_P, args=(xyz, uv))
        P = result.x.reshape(3, 4)
        # best_P, inliers = ransac_reprojection(initial_guess_P, xyz, uv)
    else:

        P, err = Pcalib(xyz, uv)

    uv2 = np.dot( P, np.concatenate( (xyz.T, np.ones((1, len(xyz)))) ) )
    uv2 = ((uv2 / uv2[2, :]).T).astype(np.int32)
    errors = np.sqrt(np.sum((uv2[:, :2] - uv)**2, axis=1))
    errors = np.vstack((errors, mask_label)).T
    errors = errors[errors[:, 0] > 15]

    return P, errors




class VideoPlayer():
    def __init__(self, opt) -> None:
        self.jump = 36

        self.cap = cv2.VideoCapture(opt.video_path)
        self.width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.bitrate = self.cap.get(cv2.CAP_PROP_BITRATE)

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        self.circle_size = 5

        self.window = cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Frame', 1280, 720)

        _, self.frame = self.cap.read()
        self.frame_num = 0

        self.piece_start = 0
        self.piece_end = 0

        self.Ps = []
        self.kps_all_ori = []
        self.ball = []
        self.all_errors = []

        if opt.label is None:
            self.ball = pd.read_csv(opt.csv)[['Visibility', 'X', 'Y']].fillna(0).values.astype(np.int32)

            for num in range(self.frames):
                num_json = os.path.join(opt.court, "{}.json".format(num))
                if os.path.exists(num_json):
                    with open(num_json) as file:
                        data = json.load(file)
                        kps = []                    # 一张图对应的标签关键点
                        for _ in range(32):
                            kps.append([0, 0, 0])

                        for shape in data["shapes"]:
                            index = shape["label"]
                            point = shape["points"][0]  # 嵌套了list

                            point.append(1) # visible

                            index = int(index) - 1      # 标注时使用1-32， 替换为0-31
                            assert index <= 31 and index >= 0

                            kps[index] = point

                        kps_int = np.array(kps).astype(int)
                        self.kps_all_ori.append(kps_int)

                        P, errors = fit_3d_2d(kps_int, num, self.Ps)

                        self.Ps.append(P)
                        self.all_errors.append(errors)
                else:
                    print("warning: {} is not found".format(num_json))
                    exit()

        else:
            assert(os.path.exists(opt.label))
            with open(opt.label) as file:
                self.label_data = json.load(file)

                for index in range(self.frames):
                    x = int(self.label_data[index]['shuttle'][0]*self.width)
                    y = int(self.label_data[index]['shuttle'][1]*self.height)
                    vis = int(self.label_data[index]['shuttle'][2])

                    self.ball.append([vis, x, y])

                    kps_frac = np.array(self.label_data[index]['court'])
                    assert len(kps_frac) == 32
                    kps_int = (kps_frac * np.array([self.width, self.height, 1]).T).astype(int)

                    self.kps_all_ori.append(kps_int)

                    P, errors = fit_3d_2d(kps_int, index, self.Ps)

                    self.Ps.append(P)
                    self.all_errors.append(errors)

        self.display()


    def display(self):
        res_frame = self.frame.copy()
        res_frame = cv2.putText(res_frame, "Frame: {}/{}".format(int(self.frame_num), int(self.frames-1)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.circle(res_frame, (50, 70), 3, COLOR_calc, -1)
        res_frame = cv2.putText(res_frame, "Calc Pos", (70, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_calc, 2, cv2.LINE_AA)

        cv2.circle(res_frame, (50, 100), 3, COLOR_label, -1)
        res_frame = cv2.putText(res_frame, "Label Pos", (70, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_label, 2, cv2.LINE_AA)

        vis = self.ball[self.frame_num][0]
        x = self.ball[self.frame_num][1]
        y = self.ball[self.frame_num][2]
        if vis:
            cv2.circle(res_frame, (x, y), self.circle_size, (0, 0, 255), -1)

        P = self.Ps[self.frame_num]
        kps_ori = np.array(self.kps_all_ori[self.frame_num]).astype(np.int32)

        kps_int = np.array(self.kps_all_ori[self.frame_num]).astype(np.int32)
        uv2 = np.dot( P, np.concatenate( (court.T, np.ones((1, len(court)))) ) )
        uv2 = np.array((uv2 / uv2[2, :]).T).astype(np.int32)
        uv2 = np.clip(uv2, -9999, 9999)
        kps_int[:, :2] = uv2[:, 0:2]

        res_frame = visualize_court(res_frame, kps_int)

        errors = self.all_errors[self.frame_num]
        for err, label_index in errors:
            x, y = kps_int[int(label_index)][0], kps_int[int(label_index)][1]
            xo, yo = kps_ori[int(label_index)][0], kps_ori[int(label_index)][1]

            cv2.line(res_frame, (x, y), (xo, yo), (0, 0, 255), 2)

            cv2.circle(res_frame, (x, y), 3, COLOR_calc, -1)
            cv2.circle(res_frame, (xo, yo), 3, COLOR_label, -1)
            res_frame = cv2.putText(res_frame, "{}-{:.1f}".format(int(label_index), err), (x+5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255))
            print("frame: {} label index:{}, label pos:({},{}), calc pos:({},{}), err pix:{:.1f}".format(self.frame_num, int(label_index)+1, xo, yo, x, y, err))

        cv2.imshow('Frame', res_frame)

    #    frame_num   0---->frames-1
    def main_loop(self):
        key = cv2.waitKeyEx(0)
        if key in keybindings['first_frame']:
            self.frame_num = 0
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_num)
            ret, self.frame = self.cap.read()

            assert(ret==True)

        elif key in keybindings['last_frame']:
            self.frame_num = self.frames-1                  
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_num)       # cap.set时, frame_num不用加1
            ret, self.frame = self.cap.read()

            print(type(self.frame))
            assert(ret==True)


        elif key in keybindings['next']:
            if self.frame_num < self.frames-1:
                ret, self.frame = self.cap.read()
                self.frame_num += 1

                assert(ret==True)

        elif key in keybindings['prev']:
            time.sleep(0.01)
            if self.frame_num > 0:
                self.frame_num -= 1
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_num)
                _, self.frame = self.cap.read()


        elif key in keybindings['forward_frames']:
            if self.frame_num < self.frames-1:
                for _ in range(self.jump):
                    if self.frame_num == self.frames-2:      # 倒数第二帧，最后一帧使用read()
                        break

                    self.cap.grab()                         # cap.grab跳过帧, frame_num加1
                    self.frame_num += 1
   
                ret, self.frame = self.cap.read()
                self.frame_num += 1

        
        elif key in keybindings['backward_frames']:
            if self.frame_num < self.jump:
                self.frame_num = 0
            else:
                self.frame_num -= self.jump

            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_num)
            ret, self.frame = self.cap.read()

            assert(ret==True)


        elif key in keybindings['circle_grow']:
            self.circle_size += 1
        elif key in keybindings['circle_shrink']:
            self.circle_size -= 1


        elif key in keybindings['quit']:
            self.finish()
            return

        self.display()


    def finish(self):
        self.cap.release()
        cv2.destroyAllWindows()


    def __del__(self):
        self.finish()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path', type=str, nargs='?', default=None, help='Path to the video file.')
    parser.add_argument('--label', type=str, default=None, help='Path to the directory where all file should be saved. If not specified, csv file will be saved in the same directory as the video file.')
    parser.add_argument('--court', type=str, default=None, help='Path to the directory where court file should be saved. If not specified, csv file will be saved in the same directory as the video file.')
    parser.add_argument('--csv', type=str, default=None, help='Path to the directory where shuttle file should be saved. If not specified, csv file will be saved in the same directory as the video file.')
    opt = parser.parse_args()
    return opt



if __name__ == '__main__':
    opt = parse_opt()

    if opt.label is None:
        directory, file_name = os.path.split(opt.video_path)
        new_file_name = file_name.replace(".mp4", "_all.json")

        opt.label = os.path.join(directory, new_file_name)
        if os.path.exists(opt.label) == False:
            opt.label = None

    if opt.label is None:
        assert(opt.court is not None)
        directory, file_name = os.path.split(opt.video_path)
        opt.court = os.path.join(opt.court, file_name.split(".")[0])

        if opt.csv is None:
            opt.csv = Path(opt.video_path).with_suffix('.csv')
            assert(os.path.exists(opt.csv))

    player = VideoPlayer(opt)
    while(player.cap.isOpened()):
        player.main_loop()
