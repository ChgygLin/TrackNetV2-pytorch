
#   match1/csv/xxx_ball.csv         --->    match1/labels/xxx.csv
#   match1/video                    --->    match1/videos

#   Frame,Visibility,X,Y            --->    frame_num,visible,x,y
#   11,1,621,305                    --->    11,1,0.48515625,0.423611111


import os
import sys
import glob
import cv2
import pandas as pd

try:
	rallyPath = sys.argv[1]
	if not rallyPath :
		raise ''
except:
	print('usage: python3 handle_dataset.py <rallyPath>')
	exit(1)


for dir in os.listdir(rallyPath):

    # video_path = os.path.join(rallyPath, dir, "video")      # Test/match1/video
    # csv_path = os.path.join(rallyPath, dir, "csv")          # Test/match1/csv
	
    # assert os.path.isdir(video_path), '这不是一个目录'
    # assert os.path.isdir(csv_path), '这不是一个目录'
	
    # # 目录更名, video->videos, csv->labels
    # new_video_path = os.path.join(rallyPath, dir, "videos")     # Test/match1/videos
    # new_csv_path = os.path.join(rallyPath, dir, "labels")       # Test/match1/labels
	
    # os.rename(video_path, new_video_path)
    # os.rename(csv_path, new_csv_path)
	
    new_video_path = os.path.join(rallyPath, dir, "videos")      # Test/match1/video
    new_csv_path = os.path.join(rallyPath, dir, "labels")          # Test/match1/csv

    # Test/match1/videos/1_05_02.mp4
    # Test/match1/labels/1_05_02.csv
    for file in os.listdir(new_csv_path):
        filename, _ = os.path.splitext(file)

        file_path = os.path.join(new_csv_path, file)                            # Test/match1/labels/1_07_03.csv
        video_path = os.path.join(new_video_path, '{}.mp4'.format(filename))    # Test/match1/videos/1_07_03.mp4

        print("handle video: {}".format(video_path))

        if os.path.exists(video_path):
            # 确定视频宽高
            cap = cv2.VideoCapture(video_path)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
        else:
            frame_width, frame_height = 1280, 720

        # 将坐标转化为比例
        df = pd.read_csv(file_path)
        df.columns = ['frame_num', 'visible', 'x', 'y']
        # df = df.rename(columns={'Frame': 'frame_num', 'Visibility': 'visible', 'X': 'x', 'Y': 'y'})

        df['x'] = df['x'].astype(float)
        df['y'] = df['y'].astype(float)

        df.loc[:, 'x'] /= frame_width
        df.loc[:, 'y'] /= frame_height

        df.to_csv(file_path, index=False)
