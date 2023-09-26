# 将单个视频分解成图像帧
# python Frame_Generator.py Test/match1/videos/1_05_02.mp4 Test/match1/images/1_05_02

import cv2
import os
import sys
import shutil


def extract_video(filePath, outputPath):
	if os.path.exists(outputPath):
		shutil.rmtree(outputPath)

	os.makedirs(outputPath)

	#Segment the video into frames
	cap = cv2.VideoCapture(filePath)
	success, count = True, 0
	success, image = cap.read()

	while success:
		imageFile = os.path.join(outputPath, '{}.jpg'.format(count))
		print(imageFile)
		cv2.imwrite(imageFile, image)
		count += 1
		success, image = cap.read()


if __name__ == "__main__":
	try:
		filePath = sys.argv[1]
		outputPath = sys.argv[2]
		if (not filePath) or (not outputPath):
			raise ''
	except:
		print('usage: python3 Frame_Generator.py <filePath> <outputFolder>')
		exit(1)


	extract_video(filePath, outputPath)


