import io
import json

import sys
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # 1 -> root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

import cv2
import tempfile
import numpy as np
from argparse import ArgumentParser

import torch
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request

from models.tracknet import TrackNet
from utils.general import get_shuttle_position

# reference:  https://pytorch.org/tutorials/intermediate/flask_rest_api_tutorial.html


def prediction(video, model, device, imgsz):
    vid_cap = cv2.VideoCapture(video)
    fps = vid_cap.get(cv2.CAP_PROP_FPS)
    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("./predict.mp4", fourcc, fps, (w, h))

    count = 0
    video_end = False
    while vid_cap.isOpened():
        imgs = []
        for _ in range(3):
            ret, img = vid_cap.read()

            if not ret:
                video_end = True
                break
            imgs.append(img)

        if video_end:
            break

        imgs_torch = []
        for img in imgs:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img_torch = transforms.ToTensor()(img).to(device)   # already [0, 1]
            img_torch = transforms.functional.resize(img_torch, imgsz, antialias=True)

            imgs_torch.append(img_torch)

        imgs_torch = torch.cat(imgs_torch, dim=0).unsqueeze(0)

        preds = model(imgs_torch)
        preds = preds[0].detach().cpu().numpy()

        y_preds = preds > 0.5
        y_preds = y_preds.astype('float32')
        y_preds = y_preds*255
        y_preds = y_preds.astype('uint8')

        for i in range(3):
            (visible, cx_pred, cy_pred) = get_shuttle_position(y_preds[i])
            (cx, cy) = (int(cx_pred*w/imgsz[1]), int(cy_pred*h/imgsz[0]))
            if visible:
                cv2.circle(imgs[i], (cx, cy), 8, (0,0,255), -1)


            out.write(imgs[i])
            print("{} ---- visible: {}  cx: {}  cy: {}".format(count, visible, cx, cy))

            count += 1

    out.release()
    vid_cap.release()





def parse_opt():
    parser = ArgumentParser()

    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[288, 512], help='image size h,w')
    parser.add_argument('--weights', type=str, default=ROOT / 'best.pt', help='Path to trained model weights.')

    opt = parser.parse_args()

    return opt


def main(opt):
    f_weights = str(opt.weights)
    imgsz = opt.imgsz

    device = "cuda"

    model = TrackNet().to(device)
    model.load_state_dict(torch.load(f_weights))
    model.eval()
    print("initialize TrackNet, load weights: {}".format(f_weights))

    app = Flask(__name__)

    @app.route('/predict', methods=['POST'])
    def predict():
        if request.method == 'POST':
            file = request.files['file']

            # file.save('video.mp4')
            file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)

            with tempfile.NamedTemporaryFile() as temp:
                temp.write(file_bytes)

                prediction(temp.name, model, device, imgsz)

            return 'Video processed successfully'
    
    app.run()


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)