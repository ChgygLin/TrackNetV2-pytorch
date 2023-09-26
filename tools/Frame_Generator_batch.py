# 将一个目录下所有的视频依次分解成图像帧, 自动创建同级images目录
# python Frame_Generator_batch.py Test/match1/videos

import os
import sys
from glob import glob

from Frame_Generator import extract_video

# Test/match1/videos/1_05_02.mp4		---->	Test/match1/images/1_05_02/*.jpg
# Test/match1/videos/1_05_02.mp4		---->	Test/match1/images/1_05_02/*.jpg
#											...


def extract_videos(videosPath):
	for filePath in glob(os.path.join(videosPath, '*mp4')):
		tmp, _ = os.path.splitext(filePath)					# Test/match1/videos/1_05_02.mp4 ---> Test/match1/videos/1_05_02
		imagePath = tmp.replace('videos', 'images')				# Test/match1/videos/1_05_02	 ---> Test/match1/images/1_05_02

		extract_video(filePath, imagePath)


if __name__ == "__main__":
	try:
		videosPath = sys.argv[1]
		if not videosPath:
			raise ''
	except:
		print('usage: python3 Frame_Generator.py <videosPath>')
		exit(1)


	extract_videos(videosPath)
