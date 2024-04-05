from pathlib import Path
import cv2
import pandas as pd
import numpy as np
import os
import argparse
import sys
import time
import json

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from utils.general import Pcalib, visualize_court, court

# python show_label_all.py ./dataset/match1/videos/1_10_12.mp4 --label dataset/match/labels
# 若不加label,则会默认从mp4文件同级目录读取


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

        if opt.label is None:
            self.ball = pd.read_csv(opt.csv)[['Visibility', 'X', 'Y']].values.astype(int)

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
                        
                        self.kps_all_ori.append(kps)

                        if kps[30][2]==0 or kps[31][2]==0:
                            self.Ps.append(None)
                        else:
                            kps_int = np.array(kps).astype(int)
                            kps_court = np.zeros((32, 6), dtype=np.float32)
                            kps_court[:, :3] = kps_int
                            kps_court[:, 3:6] = court

                            selection = kps_court[kps_court[:, 2] == 1]
                            uv = selection[:, :2]
                            xyz = selection[:, 3:6]
                            P, err = Pcalib(xyz, uv)

                            self.Ps.append(P)

                            # uv2 = np.dot( P, np.concatenate( (court.T, np.ones((1, len(court)))) ) )
                            # uv2 = np.array((uv2 / uv2[2, :]).T).astype(np.int32)
                            # uv2 = np.clip(uv2, -9999, 9999)

                            # kps_int[:, :2] = uv2[:, 0:2]
                            # self.kps_all.append(kps_int)
                else:
                    print("warning: {} is not found".format(num_json))
                    self.Ps.append(None)
                    self.kps_all_ori.append(None)

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

                    kps_court = np.zeros((32, 6))
                    kps_court[:, :3] = kps_int
                    kps_court[:, 3:6] = court

                    selection = kps_court[kps_court[:, 2] == 1]
                    uv = selection[:, :2]
                    xyz = selection[:, 3:6]
                    P, err = Pcalib(xyz, uv)

                    uv2 = np.dot( P, np.concatenate( (court.T, np.ones((1, len(court)))) ) )
                    uv2 = np.array((uv2 / uv2[2, :]).T).astype(np.int32)
                    uv2 = np.clip(uv2, -9999, 9999)

                    kps_int[:, :2] = uv2[:, 0:2]
                    self.kps_all.append(kps_int)

        self.display()


    def save_piece(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.piece_start)

        out = cv2.VideoWriter('{}_{}.mp4'.format(self.piece_start+1, self.piece_end+1), self.fourcc, self.fps, (self.width, self.height))

        frame_cnt = self.piece_start
        while frame_cnt <= self.piece_end:
            ret, frame = self.cap.read()
            out.write(frame)

            frame_cnt += 1
        
        out.release()

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_num)
        print("save piece succefully!")


    def display(self):
        res_frame = self.frame.copy()
        # res_frame = cv2.putText(res_frame, state_name[self.info['visible'][self.frame_num]], (100, 110), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)
        res_frame = cv2.putText(res_frame, "Frame: {}/{}".format(int(self.frame_num), int(self.frames-1)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.circle(res_frame, (50, 70), 3, (0, 255, 0), -1)
        res_frame = cv2.putText(res_frame, "Current frame", (70, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.circle(res_frame, (50, 100), 3, (0, 255, 255), -1)
        res_frame = cv2.putText(res_frame, "Previous frame", (70, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        # res_frame = cv2.putText(res_frame, "Piece: {}-{}".format(int(self.piece_start+1), int(self.piece_end+1)), (100, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        vis = self.ball[self.frame_num][0]
        x = self.ball[self.frame_num][1]
        y = self.ball[self.frame_num][2]
        if vis:
            cv2.circle(res_frame, (x, y), self.circle_size, (0, 0, 255), -1)

        P = self.Ps[self.frame_num]
        kps_int = self.kps_all_ori[self.frame_num]
        if kps_int is not None:
            if P is not None:
                kps_int = np.array(self.kps_all_ori[self.frame_num]).astype(np.int32)

                uv2 = np.dot( P, np.concatenate( (court.T, np.ones((1, len(court)))) ) )
                uv2 = np.array((uv2 / uv2[2, :]).T).astype(np.int32)
                uv2 = np.clip(uv2, -9999, 9999)

                kps_int[:, :2] = uv2[:, 0:2]
                res_frame = visualize_court(res_frame, kps_int)
            else:
                if self.frame_num >= 1:
                    kps_int = np.array(self.kps_all_ori[self.frame_num]).astype(np.int32)
                    kps_int_pre = np.array(self.kps_all_ori[self.frame_num-1]).astype(np.int32)

                    for i in range(32):
                        x, y, v = kps_int[i][0], kps_int[i][1], kps_int[i][2]
                        xo, yo, vo = kps_int_pre[i][0], kps_int_pre[i][1], kps_int_pre[i][2]

                        if v:
                            cv2.circle(res_frame, (x, y), 3, (0, 255, 0), -1)
                        if vo:
                            cv2.circle(res_frame, (xo, yo), 3, (0, 255, 255), -1)
                        if v and vo:
                            cv2.line(res_frame, (x, y), (xo, yo), (0, 0, 255), 2)

        cv2.imshow('Frame', res_frame)

    #    frame_num   0---->frames-1
    def main_loop(self):
        key = cv2.waitKeyEx(1)
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


        elif key in keybindings['piece_start']:
            self.piece_start = self.frame_num

        elif key in keybindings['piece_end']:
            self.piece_end = self.frame_num
            self.save_piece()


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
