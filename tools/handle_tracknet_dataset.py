
# 预处理tracknet数据集  https://hackmd.io/Nf8Rh1NrSrqNUzmO0sQKZw

#   match1/csv/xxx_ball.csv         --->    match1/labels/xxx.csv
#   match1/video                    --->    match1/videos


#   输入参数
#   TrackNetV2/Professional
#   TrackNetV2/Amateur
#   TrackNetV2/Test

import os
import sys
import glob

try:
	rallyPath = sys.argv[1]
	if not rallyPath :
		raise ''
except:
	print('usage: python3 handle_dataset.py <rallyPath>')
	exit(1)


for dir in os.listdir(rallyPath):

    video_path = os.path.join(rallyPath, dir, "video")
    csv_path = os.path.join(rallyPath, dir, "csv")
	
    assert os.path.isdir(video_path), '这不是一个目录'
    assert os.path.isdir(csv_path), '这不是一个目录'
	
    # csv更名,去掉'_ball'
    for file in glob.glob(os.path.join(csv_path, "*.csv")):
        new_name = file.replace('_ball', '')
        os.rename(file, new_name)
	
    # 目录更名, video->videos, csv->labels
    new_video_path = os.path.join(rallyPath, dir, "videos")
    new_csv_path = os.path.join(rallyPath, dir, "labels")
	
    os.rename(video_path, new_video_path)
    os.rename(csv_path, new_csv_path)
	

