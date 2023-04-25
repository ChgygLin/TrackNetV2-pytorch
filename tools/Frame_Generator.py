import cv2
import csv
import os
import sys
import shutil

# videoPath.mp4		--->	outputFolder/*.jpg

try:
	videoName = sys.argv[1]
	outputPath = sys.argv[2]
	if (not videoName) or (not outputPath):
		raise ''
except:
	print('usage: python3 Frame_Generator.py <videoPath> <outputFolder>')
	exit(1)

if outputPath[-1] != '/':
	outputPath += '/'
	
if os.path.exists(outputPath):
	shutil.rmtree(outputPath)

os.makedirs(outputPath)

#Segment the video into frames
cap = cv2.VideoCapture(videoName)
success, count = True, 0
success, image = cap.read()
while success:
	cv2.imwrite(outputPath + '%d.jpg' %(count), image)
	count += 1
	print(count)
	success, image = cap.read()
