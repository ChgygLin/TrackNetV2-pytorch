import torch
import torchvision

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from argparse import ArgumentParser

from models.tracknet import TrackNet


# from yolov5 detect.py
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def get_ball_position(img, original_img_=None):
    ret, thresh = cv2.threshold(img, 128, 1, 0)
    thresh = cv2.convertScaleAbs(thresh)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print(len(contours))
    if len(contours) != 0:

        #find the biggest area of the contour
        c = max(contours, key = cv2.contourArea)

        if original_img_ is not None:
            # the contours are drawn here
            cv2.drawContours(original_img_, [c], -1, 255, 3)

        x,y,w,h = cv2.boundingRect(c)
        print("Center: ({}, {}) | Width: {} | Height: {}".format(x, y, w, h))
        
        return x, y, w, h


def parse_opt():
    parser = ArgumentParser()

    parser.add_argument('--source', type=str, default=ROOT / 'dataset/match2/videos/1_10_12.mp4', help='Path to video.')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.csv')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--weights', type=str, default=ROOT / 'best.pth', help='Path to trained model weights.')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')

    opt = parser.parse_args()

    return opt


def main(opt):
    imgsz = [288, 512]

    source_name = os.path.splitext(os.path.basename(opt.source))[0]
    b_save_txt = opt.save_txt
    b_view_img = opt.view_img
    d_save_dir = opt.project
    f_weights = opt.weights

    if not os.path.exists(d_save_dir):
        os.makedirs(d_save_dir)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = TrackNet().to(device)
    model.load_state_dict(torch.load(f_weights))
    model.eval()

    vid_cap = cv2.VideoCapture(opt.source)
    video_end = False

    video_len = vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = vid_cap.get(cv2.CAP_PROP_FPS)
    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('{}/{}.mp4'.format(d_save_dir, source_name), fourcc, fps, (w, h))

    if b_save_txt:
        f_save_txt = open('{}/{}.csv'.format(d_save_dir, source_name), 'w')
        f_save_txt.write('frame_num,visible,x,y\n')

    if b_view_img:
        cv2.namedWindow(source_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(source_name, (w, h))

    count = 0
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
            img_torch = torch.tensor(img).permute(2, 0, 1).float().to(device) / 255.0
            img_torch = torchvision.transforms.functional.resize(img_torch, imgsz, antialias=True)
            imgs_torch.append(img_torch)

        imgs_torch = torch.cat(imgs_torch, dim=0).unsqueeze(0)

        preds = model(imgs_torch)
        preds = preds[0].detach().cpu().numpy()

        y_preds = preds > 0.5
        y_preds = y_preds.astype('float32')
        y_preds = y_preds*255
        y_preds = y_preds.astype('uint8')

        for i in range(3):
            if np.amax(y_preds[i]) <= 0:
                if b_save_txt:
                    f_save_txt.write('{},0,0,0\n'.format(count))

                if b_view_img:
                    cv2.imshow(source_name, imgs[i])
                    cv2.waitKey(1)

                out.write(imgs[i])
                print('{} cx: 0  cy: 0'.format(count))
                
            else:
                pred_img = cv2.resize(y_preds[i], (w, h), interpolation=cv2.INTER_AREA)
                cv2.imwrite('{}/{}.png'.format(d_save_dir, count), pred_img)

                # x, y, w, h = get_ball_position(pred_frame, original_img_=frames[i])
                (cnts, _) = cv2.findContours(pred_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                rects = [cv2.boundingRect(ctr) for ctr in cnts]
                max_area_idx = 0
                max_area = rects[max_area_idx][2] * rects[max_area_idx][3]

                for ii in range(len(rects)):
                    area = rects[ii][2] * rects[ii][3]
                    if area > max_area:
                        max_area_idx = ii
                        max_area = area

                target = rects[max_area_idx]


                if b_save_txt:
                    f_save_txt.write('{},1,{},{}\n'.format(count, target[0]/w, target[1]/h))
                
                if b_view_img:
                    cv2.circle(imgs[i], (target[0], target[1]), 8, (0,0,255), -1)
                    cv2.imshow(source_name, imgs[i])
                    cv2.waitKey(1)

                out.write(imgs[i])
                print("{} cx: {}  cy: {}".format(count, target[0], target[1]))


            count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    if b_save_txt:
        # 每次识别3张，最后可能有1-2张没有识别，补0
        while count < video_len:
            f_save_txt.write('{},0,0,0\n'.format(count))
            count += 1

        f_save_txt.close()

    out.release()
    vid_cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
