import torch
import torchvision

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser

from models.tracknet import TrackNet
from utils.dataloaders import create_dataloader
from utils.general import check_dataset, outcome, evaluation


# from yolov5 detect.py
FILE = Path(__file__).resolve()
ABS_ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ABS_ROOT) not in sys.path:
    sys.path.append(str(ABS_ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ABS_ROOT, Path.cwd()))  # relative




def wbce_loss(y_true, y_pred):
    return -1*(
        ((1-y_pred)**2) * y_true * torch.log(torch.clamp(y_pred, min=1e-07, max=1))  +
        (y_pred**2) * (1-y_true) * torch.log(torch.clamp(1-y_pred, min=1e-07, max=1))
    ).sum()


def validation_loop(device, model, val_loader, save_dir):
    model.eval()

    loss_sum = 0
    TP = TN = FP1 = FP2 = FN = 0

    with torch.inference_mode():
        pbar = tqdm(val_loader, ncols=180)
        for batch_index, (X, y) in enumerate(pbar):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)

            loss_sum += wbce_loss(y, y_pred).item()

            y_ = y.detach().cpu().numpy()
            y_pred_ = y_pred.detach().cpu().numpy()

            y_pred_ = (y_pred_ > 0.5).astype('float32')
            (tp, tn, fp1, fp2, fn) = outcome(y_pred_, y_)
            TP += tp
            TN += tn
            FP1 += fp1
            FP2 += fp2
            FN += fn

            (accuracy, precision, recall) = evaluation(TP, TN, FP1, FP2, FN)
            
            pbar.set_description('Val loss: {:.6f}  |  TP: {}, TN: {}, FP1: {}, FP2: {}, FN: {}  |  Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}'.format( \
                loss_sum / ((batch_index+1)*X.shape[0]), TP, TN, FP1, FP2, FN, accuracy, precision, recall))

    return loss_sum/len(val_loader)



def parse_opt():
    parser = ArgumentParser()

    parser.add_argument('--data', type=str, default=ROOT / 'data/match/test.yaml', help='Path to dataset.')
    parser.add_argument('--weights', type=str, default=ROOT / 'best.pt', help='Path to trained model weights.')
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--project', default=ROOT / 'runs/val', help='save results to project/name')

    opt = parser.parse_args()
    return opt


def main(opt):
    d_save_dir = str(opt.project)
    f_weights = str(opt.weights)
    batch_size = opt.batch_size
    f_data = str(opt.data)


    data_dict = check_dataset(f_data)
    train_path, val_path = data_dict['train'], data_dict['val']

    if not os.path.exists(d_save_dir):
        os.makedirs(d_save_dir)


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TrackNet().to(device)

    assert os.path.exists(f_weights), f_weights+" is invalid"
    print("load pretrain weights {}".format(f_weights))
    model.load_state_dict(torch.load(f_weights))


    val_loader = create_dataloader(val_path, batch_size=batch_size)

    validation_loop(device, model, val_loader, d_save_dir)



if __name__ == '__main__':
    opt = parse_opt()
    main(opt)