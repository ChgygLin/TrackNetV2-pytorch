import torch
import torchvision

import os
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser
from tensorboardX import SummaryWriter

from models.tracknet import TrackNet
from utils.dataloaders import create_dataloader
from utils.general import check_dataset, outcome, evaluation, tensorboard_log


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


def validation_loop(device, model, val_loader, log_writer, epoch):
    model.eval()

    loss_sum = 0
    TP = TN = FP1 = FP2 = FN = 0

    with torch.inference_mode():
        pbar = tqdm(val_loader, ncols=180)
        for batch_index, (X, y, _, _) in enumerate(pbar):
            y = y[:,0,:,:,:]    # [batch][sq][32][h][w]
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

            pbar.set_description('Val   loss: {:.6f}  |  TP: {}, TN: {}, FP1: {}, FP2: {}, FN: {}  |  Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}'.format( \
                loss_sum / ((batch_index+1)*X.shape[0]), TP, TN, FP1, FP2, FN, accuracy, precision, recall))

        tensorboard_log(log_writer, "Val", loss_sum / ((batch_index+1)*X.shape[0]), TP, TN, FP1, FP2, FN, epoch)

    return loss_sum/len(val_loader)


def training_loop(device, model, optimizer, lr_scheduler, train_loader, val_loader, start_epoch, epochs, save_dir):
    best_val_loss = float('inf')

    checkpoint_period = 3
    log_period = 100
    
    log_dir = '{}/logs'.format(save_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_writer = SummaryWriter(log_dir)

    for epoch in range(start_epoch, epochs):
        print("\n==================================================================================================")
        tqdm.write("Epoch: {} / {}\n".format(epoch, epochs))
        running_loss = 0.0
        TP = TN = FP1 = FP2 = FN = 0

        model.train()
        pbar = tqdm(train_loader, ncols=180)
        for batch_index, (X, y, _, _) in enumerate(pbar):
            y = y[:,0,:,:,:]    # [batch][sq][32][h][w]
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()

            y_pred = model(X)

            loss = wbce_loss(y, y_pred)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

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

            pbar.set_description('Train loss: {:.6f}  |  TP: {}, TN: {}, FP1: {}, FP2: {}, FN: {}  |  Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}'.format( \
                running_loss / ((batch_index+1)*X.shape[0]), TP, TN, FP1, FP2, FN, accuracy, precision, recall))

            if batch_index % log_period == 0:
                with torch.inference_mode():
                    images = [
                        torch.unsqueeze(y[0,0,:,:], 0).repeat(3,1,1).cpu(),
                        torch.unsqueeze(y_pred[0,0,:,:], 0).repeat(3,1,1).cpu(),
                    ]

                    images.append(X[0,(0,1,2),:,:].cpu())
                    res = X[0, (0,1,2),:,:] * y[0,0,:,:]

                    images.append(res.cpu())
                    grid = torchvision.utils.make_grid(images, nrow=1)

                    torchvision.utils.save_image(grid, '{}/epoch_{}_batch{}.png'.format(log_dir, epoch, batch_index))

        if val_loader is not None:
            best = False
            val_loss = validation_loop(device, model, val_loader, log_writer, epoch)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best = True

            if epoch % checkpoint_period == checkpoint_period - 1:
                tqdm.write('\n--- Saving weights to: {}/last.pt ---'.format(save_dir))
                torch.save(model.state_dict(), '{}/last.pt'.format(save_dir))

                if best:
                    tqdm.write('--- Saving weights to: {}/best.pt ---'.format(save_dir))
                    torch.save(model.state_dict(), '{}/best.pt'.format(save_dir))

        tensorboard_log(log_writer, "Train", running_loss / ((batch_index+1)*X.shape[0]), TP, TN, FP1, FP2, FN, epoch)

        print('lr: {}'.format(lr_scheduler.get_last_lr()))
        lr_scheduler.step()

        if epoch%10 == 0:
            ckpt_dir = '{}/checkpoint'.format(save_dir)
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)

            cur_ckpt = '{}/{}/ckpt_{}.pt'.format(ABS_ROOT, ckpt_dir, epoch)
            latest_ckpt = '{}/{}/ckpt_latest.pt'.format(ABS_ROOT, ckpt_dir)

            checkpoint = {
                "net": model.state_dict(),
                'optimizer': optimizer.state_dict(),
                "epoch": epoch+1,
                'lr_scheduler': lr_scheduler.state_dict()
            }

            torch.save(checkpoint, cur_ckpt)
            os.system('ln -sf {} {}'.format(cur_ckpt, latest_ckpt))
            print("save checkpoint {}".format(cur_ckpt))


def parse_opt():
    parser = ArgumentParser()

    parser.add_argument('--data', type=str, default=ROOT / 'data/match/test.yaml', help='Path to dataset.')
    parser.add_argument('--weights', type=str, default=ROOT / 'best.pt', help='Path to trained model weights.')
    parser.add_argument('--epochs', type=int, default=100, help='total training epochs')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[288, 512], help='image size h,w')
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--project', default=ROOT / 'runs/train', help='save results to project/name')
    parser.add_argument('--resume', action='store_true', help='whether load checkpoint for resume')

    opt = parser.parse_args()
    return opt


def main(opt):
    d_save_dir = str(opt.project)
    f_weights = str(opt.weights)
    epochs = opt.epochs
    batch_size = opt.batch_size
    f_data = str(opt.data)
    imgsz = opt.imgsz

    start_epoch = 0

    data_dict = check_dataset(f_data)
    train_path, val_path = data_dict['train'], data_dict['val']

    if not os.path.exists(d_save_dir):
        os.makedirs(d_save_dir)


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TrackNet().to(device)

    optimizer = torch.optim.Adadelta(model.parameters(), lr=0.99)

    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9)

    if opt.resume:
        default_ckpt = "{}/checkpoint/ckpt_latest.pt".format(d_save_dir)
        checkpoint = torch.load(default_ckpt)

        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        print("load checkpoint {}".format(default_ckpt))
    else:
        if os.path.exists(f_weights):
            print("load pretrain weights {}".format(f_weights))
            model.load_state_dict(torch.load(f_weights))
        else:
            print("train from scratch")

    train_loader = create_dataloader(train_path, imgsz, batch_size=batch_size, sq=1, shuffle=True)
    val_loader = create_dataloader(val_path, imgsz, batch_size=batch_size, sq=1)


    training_loop(device, model, optimizer, lr_scheduler, train_loader, val_loader, start_epoch, epochs, d_save_dir)



if __name__ == '__main__':
    opt = parse_opt()
    main(opt)