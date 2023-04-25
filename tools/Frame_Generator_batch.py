import cv2
import csv
import os
import sys
import shutil
from glob import glob

# dataset/videos/*.mp4		---->	dataset/images/1/*.jpg
#									dataset/images/2/*.jpg
#											...

match = 'dataset'
p = os.path.join(match, 'videos', '*mp4')
video_list = glob(p)

if not os.path.exists(match + '/images/'):
	os.makedirs(match + '/images/')

if not os.path.exists(match + '/labels/'):
	os.makedirs(match + '/labels/')

for video_path in video_list:
	videos_dir = os.path.join(match, 'videos')
	video_name = video_path[len(videos_dir)+1:-4]

	outputPath = os.path.join(match, 'images', video_name)
	outputPath += '/'

	if not os.path.exists(outputPath):
		os.makedirs(outputPath)

	cap = cv2.VideoCapture(video_path)
	success, count = True, 0
	success, image = cap.read()

	while success:
		cv2.imwrite(outputPath + '%d.jpg' %(count), image)
		count += 1
		print("{}: {}".format(video_path, count))
		success, image = cap.read()

