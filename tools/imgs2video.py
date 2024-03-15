import cv2
import os
from natsort import natsorted

imgs_base = "./test"

imgs_files = [file for file in os.listdir(imgs_base) if file.endswith('.jpg')]
sorted_imgs_files = natsorted(imgs_files)

img = cv2.imread(imgs_base + '/' + sorted_imgs_files[0])
h, w = img.shape[0], img.shape[1]

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('./test.mp4', fourcc, 30, (w, h))

for file in sorted_imgs_files:
    print("handle {}".format(file))
    img = cv2.imread(imgs_base + '/' + file)
    out.write(img)

out.release()