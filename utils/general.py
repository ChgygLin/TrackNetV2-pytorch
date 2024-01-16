import yaml
from pathlib import Path
import cv2
import numpy as np



FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory


def yaml_load(file):
    # Single-line safe yaml loading
    with open(file, errors='ignore') as f:
        return yaml.safe_load(f)

def check_dataset(data):
    if isinstance(data, (str, Path)):
        data = yaml_load(data)  # dictionary
    
    path = Path(data.get('path'))  # optional 'path' default to '.'
    if not path.is_absolute():
        path = (ROOT / path).resolve()
        data['path'] = path  # download scripts
    
    for k in 'train', 'val':
        if data.get(k):  # prepend path
            if isinstance(data[k], str):
                x = (path / data[k]).resolve()
                if not x.exists() and data[k].startswith('../'):
                    x = (path / data[k][3:]).resolve()
                data[k] = str(x)
            else:
                data[k] = [str((path / x).resolve()) for x in data[k]]
    
    return data


# img: 0/1 binary image, numpy array.
def get_shuttle_position(img):
    if np.amax(img) <= 0:
        # (visible, cx, cy)
        return (0, 0, 0)

    else:
        (cnts, _) = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        assert (len(cnts) != 0)

        rects = [cv2.boundingRect(ctr) for ctr in cnts]
        max_area_idx = 0
        max_area = rects[max_area_idx][2] * rects[max_area_idx][3]

        for ii in range(len(rects)):
            area = rects[ii][2] * rects[ii][3]
            if area > max_area:
                max_area_idx = ii
                max_area = area

        target = rects[max_area_idx]
        (cx, cy) = (int(target[0] + target[2] / 2), int(target[1] + target[3] / 2))

        # (visible, cx, cy)
        return (1, cx, cy)


def outcome(y_pred, y_true, tol=3): # [batch, 3, h, w]
    n = y_pred.shape[0]
    i = 0
    TP = TN = FP1 = FP2 = FN = 0
    while i < n:
        for j in range(3):
            if np.amax(y_pred[i][j]) == 0 and np.amax(y_true[i][j]) == 0:
                TN += 1
            elif np.amax(y_pred[i][j]) > 0 and np.amax(y_true[i][j]) == 0:
                FP2 += 1
            elif np.amax(y_pred[i][j]) == 0 and np.amax(y_true[i][j]) > 0:
                FN += 1
            elif np.amax(y_pred[i][j]) > 0 and np.amax(y_true[i][j]) > 0:
                h_pred = y_pred[i][j] * 255
                h_true = y_true[i][j] * 255
                h_pred = h_pred.astype('uint8')
                h_true = h_true.astype('uint8')

                (_, cx_pred, cy_pred) = get_shuttle_position(h_pred)
                (_, cx_true, cy_true) = get_shuttle_position(h_true)

                dist = np.sqrt(pow(cx_pred-cx_true, 2)+pow(cy_pred-cy_true, 2))

                if dist > tol:
                    FP1 += 1
                else:
                    TP += 1
        i += 1
    return (TP, TN, FP1, FP2, FN)


def evaluation(TP, TN, FP1, FP2, FN):
    try:
        accuracy = (TP + TN) / (TP + TN + FP1 + FP2 + FN)
    except:
        accuracy = 0
    try:
        precision = TP / (TP + FP1 + FP2)
    except:
        precision = 0
    try:
        recall = TP / (TP + FN)
    except:
        recall = 0

    return (accuracy, precision, recall)

def tensorboard_log(log_writer, type, avg_loss, TP, TN, FP1, FP2, FN,  epoch):
    log_writer.add_scalar('{}/loss'.format(type), avg_loss, epoch)
    log_writer.add_scalar('{}/TP'.format(type), TP, epoch)
    log_writer.add_scalar('{}/TN'.format(type), TN, epoch)
    log_writer.add_scalar('{}/FP1'.format(type), FP1, epoch)
    log_writer.add_scalar('{}/FP2'.format(type), FP2, epoch)
    log_writer.add_scalar('{}/FN'.format(type), FN, epoch)
    log_writer.add_scalar('{}/TP'.format(type), TP, epoch)

    (accuracy, precision, recall) = evaluation(TP, TN, FP1, FP2, FN)

    log_writer.add_scalar('{}/Accuracy'.format(type), accuracy, epoch)
    log_writer.add_scalar('{}/precision'.format(type), precision, epoch)
    log_writer.add_scalar('{}/precision'.format(type), precision, epoch)