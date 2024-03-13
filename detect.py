import torch
import torchvision

import os
import sys
import cv2
import json
import numpy as np
from pathlib import Path
from argparse import ArgumentParser

from models.tracknet import TrackNet
from utils.general import get_shuttle_position, postprocess_court, visualize_kps, visualize_court

# from yolov5 detect.py
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def parse_opt():
    parser = ArgumentParser()

    parser.add_argument('--source', type=str, default=ROOT / 'example_dataset/match/videos/1_10_12.mp4', help='Path to video.')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.csv')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[288, 512], help='image size h,w')
    parser.add_argument('--weights', type=str, default=ROOT / 'best.pt', help='Path to trained model weights.')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')

    opt = parser.parse_args()

    return opt


def main(opt):
    # imgsz = [288, 512]
    # imgsz = [360, 640]

    source_name = os.path.splitext(os.path.basename(opt.source))[0]
    b_save_txt = opt.save_txt
    b_view_img = opt.view_img
    d_save_dir = str(opt.project)
    f_weights = str(opt.weights)
    f_source = str(opt.source)
    imgsz = opt.imgsz
    sq = 3

    # video_name ---> video_name_pred
    source_name = '{}_predict'.format(source_name)

    # runs/detect
    if not os.path.exists(d_save_dir):
        os.makedirs(d_save_dir)

    # runs/detect/video_name
    img_save_path = '{}/{}'.format(d_save_dir, source_name)
    if not os.path.exists(img_save_path):
        os.makedirs(img_save_path)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = TrackNet().to(device)
    model.load_state_dict(torch.load(f_weights))
    model.eval()

    vid_cap = cv2.VideoCapture(f_source)
    video_end = False

    video_len = vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = vid_cap.get(cv2.CAP_PROP_FPS)
    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('{}/{}.mp4'.format(d_save_dir, source_name), fourcc, fps, (w, h))

    if b_save_txt:
        f_save_txt = open('{}/{}.json'.format(d_save_dir, source_name), 'w')
        js_court_data = []
        js_shuttle_data = []

    if b_view_img:
        cv2.namedWindow(source_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(source_name, (w, h))

    # vid_cap.set(cv2.CAP_PROP_POS_FRAMES, 59)
    count = 0
    P = None
    while vid_cap.isOpened():
        imgs = []
        for _ in range(sq):
            ret, img = vid_cap.read()

            if not ret:
                video_end = True
                break
            imgs.append(img)

        if video_end:
            break

        imgs_torch = []
        for img in imgs:
            # https://www.geeksforgeeks.org/converting-an-image-to-a-torch-tensor-in-python/
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img_torch = torchvision.transforms.ToTensor()(img).to(device)   # already [0, 1]
            img_torch = torchvision.transforms.functional.resize(img_torch, imgsz, antialias=True)

            imgs_torch.append(img_torch)

        imgs_torch = torch.cat(imgs_torch, dim=0).unsqueeze(0)

        preds = model(imgs_torch)
        preds = preds[0].detach().cpu().numpy()

        y_preds = preds > 0.5
        y_preds = y_preds.astype('float32')
        y_preds = y_preds*255
        y_preds = y_preds.astype('uint8')

        for si in range(sq):
            kps = np.zeros((33, 3), dtype=np.int32)
            for i in range(33):
                (visible, cx_pred, cy_pred) = get_shuttle_position(y_preds[33*si+i])
                (cx, cy) = (int(cx_pred*w/imgsz[1]), int(cy_pred*h/imgsz[0]))
                if visible:
                    kps[i] = [cx, cy, 1]

            print("{} ".format(count),  end="")
            # postprocess kps
            # import time
            # t1 = time.time()
            kps_court = kps[:32, :]
            kps_court, P = postprocess_court(kps_court, last_P=P, img=imgs[si])
            # t2 = time.time()
            # print("postprocess_court: {}ms".format(t2-t1))

            if b_save_txt:
                court_data = {}

                court_data['frame_num'] = count
                if P is not None:
                    court_data['visible'] = 1
                    court_data['P'] = P.tolist()
                else:
                    court_data['visible'] = 0
                    court_data['P'] = np.zeros((3, 4)).tolist()

                js_court_data.append(court_data)

                shuttle_data = {}
                shuttle_data["frame_num"] = count
                shuttle_data["visible"] = int(kps[32][2])
                shuttle_data["x"] = int(kps[32][0])
                shuttle_data["y"] = int(kps[32][1])

                js_shuttle_data.append(shuttle_data)

            print("{} ---- shuttle visible: {}-({}, {}),  court visible: {}".format(count, kps[32][2], kps[32][0], kps[32][1], 1 if P is not None else 0))
            if b_view_img:
                if kps[32][2]:  # visible
                    cv2.circle(imgs[si], (kps[32][0], kps[32][1]), 8, (0,0,255), -1)

                if P is not None:
                    imgs[si] = visualize_court(imgs[si], kps)
                    cv2.imwrite('{}/{}.png'.format(img_save_path, count), imgs[si])
                    cv2.imshow(source_name, imgs[si])
                    cv2.waitKey(1)
                # else:
                    # print("detect frame-{} Error!!!!!!!!!!!!!!!!!!!!!!!".format(count))
                    # cv2.imwrite("./runs/detect/detect-error-{}.jpg".format(count), imgs[0])
                    # raise("P error!")

            out.write(imgs[si])

            count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if b_save_txt:
        # 每次识别3张，最后可能有1-2张没有识别，补0
        # while count < video_len:
        #     f_save_txt.write('{},0,0,0\n'.format(count))
        #     count += 1

        js_all_data = {}
        js_all_data["court"] = js_court_data
        js_all_data["shuttle"] = js_shuttle_data

        json.dump(js_all_data, f_save_txt)
        f_save_txt.close()


    out.release()
    vid_cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
