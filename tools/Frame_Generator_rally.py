# 将一个目录下所有的match比赛，分解成图像帧
# python Frame_Generator_rally.py Test      --->    Test/match1、Test/match2、Test/match3等

import os
import sys

from Frame_Generator_batch import extract_videos


def extract_rally(rallyPath):
    for dir in os.listdir(rallyPath):
        video_path = os.path.join(rallyPath, dir, "videos")     # Test/match1/videos

        print(video_path)
        extract_videos(video_path)



if __name__ == "__main__":
    try:
        rallyPath = sys.argv[1]
        if not rallyPath :
            raise ''
    except:
        print('usage: python3 handle_dataset.py <rallyPath>')
        exit(1)


    extract_rally(rallyPath)
