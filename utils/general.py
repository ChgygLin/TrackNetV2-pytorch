import yaml
from pathlib import Path
import cv2
import numpy as np



FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory


class Metrics:
    def __init__(self, tp=0, tn=0, fp1=0, fp2=0, fn=0):
        self.TP = tp
        self.TN = tn
        self.FP1 = fp1
        self.FP2 = fp2
        self.FN = fn

    def __add__(self, other):
        return Metrics(self.TP + other.TP,
                             self.TN + other.TN,
                             self.FP1 + other.FP1,
                             self.FP2 + other.FP2,
                             self.FN + other.FN)

    def evaluation(self):
        try:
            accuracy = (self.TP + self.TN) / (self.TP + self.TN + self.FP1 + self.FP2 + self.FN)
        except:
            accuracy = 0
        try:
            precision = self.TP / (self.TP + self.FP1 + self.FP2)
        except:
            precision = 0
        try:
            recall = self.TP / (self.TP + self.FN)
        except:
            recall = 0

        return (accuracy, precision, recall) 


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


court = np.array( [   [0.02, 0.02, 0],   [0.48, 0.02, 0],   [3.05, 0.02, 0],   [5.62, 0.02, 0],   [6.08, 0.02, 0],
            [0.02, 0.78, 0],   [0.48, 0.78, 0],   [3.05, 0.78, 0],   [5.62, 0.78, 0],   [6.08, 0.78, 0],
            [0.02, 4.7, 0],    [0.48, 4.7, 0],    [3.05, 4.7, 0],    [5.62, 4.7, 0],    [6.08, 4.7, 0],
            [0.02, 8.7, 0],    [0.48, 8.7, 0],    [3.05, 8.7, 0],    [5.62, 8.7, 0],    [6.08, 8.7, 0],
            [0.02, 12.62, 0],  [0.48, 12.62, 0],  [3.05, 12.62, 0],  [5.62, 12.62, 0],  [6.08, 12.62, 0],
            [0.02, 13.38, 0],  [0.48, 13.38, 0],  [3.05, 13.38, 0],  [5.62, 13.38, 0],  [6.08, 13.38, 0], [0, 6.7, 1.55], [6.1, 6.7, 1.55] ])

def normalization(nd, x):
    m, s = np.mean(x, 0), np.std(x)

    if nd == 2:
        Tr = np.array([[s, 0, m[0]], [0, s, m[1]], [0, 0, 1]])
    else:
        Tr = np.array([[s, 0, 0, m[0]], [0, s, 0, m[1]], [0, 0, s, m[2]], [0, 0, 0, 1]])

    Tr = np.linalg.inv(Tr)
    x = np.dot( Tr, np.concatenate( (x.T, np.ones((1,x.shape[0]))) ) )
    x = x[0:nd, :].T

    return Tr, x

def Pcalib(xyz, uv):
    n = xyz.shape[0]

    Txyz, xyzn = normalization(3, xyz)
    Tuv, uvn = normalization(2, uv)

    A = []

    for i in range(n):
        x, y, z = xyzn[i, 0], xyzn[i, 1], xyzn[i, 2]
        u, v = uvn[i, 0], uvn[i, 1]
        A.append( [x, y, z, 1, 0, 0, 0, 0, -u * x, -u * y, -u * z, -u] )
        A.append( [0, 0, 0, 0, x, y, z, 1, -v * x, -v * y, -v * z, -v] )

    A = np.asarray(A)
    U, S, V = np.linalg.svd(A)

    L = V[-1, :] / V[-1, -1]
    P = L.reshape(3, 4)

    P = np.dot( np.dot( np.linalg.pinv(Tuv), P ), Txyz )
    P = P / P[-1, -1]


    uv2 = np.dot( P, np.concatenate( (xyz.T, np.ones((1, n))) ) )
    uv2 = ((uv2 / uv2[2, :]).T).astype(np.int32)

    err =  np.mean( np.sqrt(np.sum( (uv2[:, :2] - uv)**2, 1)) )
    return P, err


def postprocess_court(kps, last_P=None, img=None):
    kps_court = np.zeros((32, 6))
    kps_court[:, :3] = kps
    kps_court[:, 3:6] = court

    net = kps_court[30:32, :]
    place = kps_court[:30, :]

    best_P = last_P
    # net exist
    if net[0][2] and net[1][2]:
        place = place[place[:, 2] == 1]     # xyv xyz

        if best_P is not None:
            uv_all = place[:, :2]
            xyz_all = place[:, 3:6]

            uv2 = np.dot( best_P, np.concatenate( (xyz_all.T, np.ones((1, len(place)))) ) )
            uv2 = ((uv2 / uv2[2, :]).T).astype(np.int32)

            err_all = np.sqrt(np.sum((uv2[:, :2] - uv_all)**2, axis=1))

            place = place[err_all[:] < 20]  # 使用上一帧的P剔除明显错误的点
            min_err = np.mean(err_all[err_all[:] < 20]) # 不计算明显错误点的误差
        else:
            min_err = np.inf

        if len(place) >= 6:
            for _ in range(3):
                selection = np.vstack((place, net))

                uv = selection[:, :2]
                xyz = selection[:, 3:6]
                P, _ = Pcalib(xyz, uv)

                ##########################################################################
                uv_all = place[:, :2]
                xyz_all = place[:, 3:6]

                uv2 = np.dot( P, np.concatenate( (xyz_all.T, np.ones((1, len(place)))) ) )
                uv2 = ((uv2 / uv2[2, :]).T).astype(np.int32)
                err_all = np.sqrt(np.sum((uv2[:, :2] - uv_all)**2, axis=1))

                place = place[err_all[:] < 20]
                if len(place) < 6:
                    break

                err = np.mean(err_all)
                big_err = np.array([err_all[:] >= 20]).astype(np.uint32)
                if big_err.sum() == 0:
                    if err < min_err:
                        min_err = err
                        best_P = P

                    # print("Found court: min_err: {} num_kps: {}".format(min_err, len(place)), end="")
                    break   # 只要big_err为0就退出
                # else:
                #     print("remaining big error: {}".format(big_err.sum()))

            if best_P is None:
                print("Notice:!!!!! found no P")

    if best_P is not None:
        uv2 = np.dot( best_P, np.concatenate( (court.T, np.ones((1, len(court)))) ) )
        uv2 = np.array((uv2 / uv2[2, :]).T).astype(np.int32)

        kps[:, :2] = uv2[:, 0:2]
        print(" ")

        return kps, best_P
    else:
        print(" ")
        return None, None

def visualize_kps(img, kps):    # x_int, y_int, visible
    im = img.copy()

    for index in range(len(kps)):
        cx, cy, visible = kps[index][0], kps[index][1], kps[index][2]

        if visible:
            cv2.circle(im, (cx, cy), 3, (0,0,255), -1)

    return im


def _visualize_court(img, kps):    # x_int, y_int, visible
    color = (255, 0, 0)
    cv2.line(img, kps[0], kps[25], color, 2)
    cv2.line(img, kps[0], kps[4], color, 2)
    cv2.line(img, kps[4], kps[29], color, 2)
    cv2.line(img, kps[25], kps[29], color, 2)

    cv2.line(img, kps[5], kps[9], color, 2)
    cv2.line(img, kps[10], kps[14], color, 2)
    cv2.line(img, kps[15], kps[19], color, 2)
    cv2.line(img, kps[20], kps[24], color, 2)

    cv2.line(img, kps[1], kps[26], color, 2)
    cv2.line(img, kps[3], kps[28], color, 2)

    cv2.line(img, kps[2], kps[12], color, 2)
    cv2.line(img, kps[17], kps[27], color, 2)

    cv2.line(img, kps[30], kps[31], color, 2)



    return img


def visualize_court(img, kps):    # x_int, y_int, visible
    im = img.copy()
    w , h = im.shape[1], im.shape[0]

    delta_xy = np.min(kps, axis=0)
    delta_x = np.abs(delta_xy[0]) if delta_xy[0] < 0 else 0
    delta_y = np.abs(delta_xy[1]) if delta_xy[1] < 0 else 0

    kps_tmp = kps[:, :2] + np.array([delta_x, delta_y])

    new_size = np.max(kps_tmp, axis=0)
    new_size = np.maximum(new_size, np.array([delta_x+w, delta_y+h]))

    new_image = np.zeros((new_size[1], new_size[0], 3), dtype=np.uint8)
    new_image[delta_y:(delta_y+h), delta_x:(delta_x+w)] = im

    im = _visualize_court(new_image, kps_tmp)
    im = im[delta_y:delta_y+h, delta_x:delta_x+w]

    return im


def outcome(y_pred, y_true, tol=3): # [batch, 33*sq, h, w]
    n = y_pred.shape[0]
    kps_len = y_pred.shape[1]
    i = 0
    TP = TN = FP1 = FP2 = FN = 0
    while i < n:
        for j in range(kps_len):
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
    return Metrics(TP, TN, FP1, FP2, FN)


def outcome_all(y_pred, y_true): # [batch, 33*sq, h, w]
    sq = int(y_pred.shape[1] / 33)

    # get shuttle   33-1  66-1    99-1
    y_pred_shuttle = y_pred[:, [32, 65, 98], :, :]
    y_true_shuttle = y_true[:, [32, 65, 98], :, :]

    # get net   33-3 -> 33-1
    y_pred_net = y_pred[:, [30,31, 63,64, 96,97], :, :]
    y_true_net = y_true[:, [30,31, 63,64, 96,97], :, :]

    # get place 0 ->  33-3
    indices = np.concatenate([np.arange(0, 30), np.arange(33, 63), np.arange(66, 96)])
    y_pred_ground = y_pred[:, indices, :, :]
    y_true_ground = y_true[:, indices, :, :]


    m_shuttle = outcome(y_pred_shuttle, y_true_shuttle)
    m_net = outcome(y_pred_net, y_true_net)
    m_ground = outcome(y_pred_ground, y_true_ground)

    return m_shuttle, m_net, m_ground



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

def tensorboard_log(log_writer, type, avg_loss, m_Metrics,  epoch):
    log_writer.add_scalar('{}/loss'.format(type), avg_loss, epoch)
    log_writer.add_scalar('{}/TP'.format(type), m_Metrics.TP, epoch)
    log_writer.add_scalar('{}/TN'.format(type), m_Metrics.TN, epoch)
    log_writer.add_scalar('{}/FP1'.format(type), m_Metrics.FP1, epoch)
    log_writer.add_scalar('{}/FP2'.format(type), m_Metrics.FP2, epoch)
    log_writer.add_scalar('{}/FN'.format(type), m_Metrics.FN, epoch)
    log_writer.add_scalar('{}/TP'.format(type), m_Metrics.TP, epoch)

    (accuracy, precision, recall) = m_Metrics.evaluation()

    log_writer.add_scalar('{}/Accuracy'.format(type), accuracy, epoch)
    log_writer.add_scalar('{}/precision'.format(type), precision, epoch)
    log_writer.add_scalar('{}/precision'.format(type), precision, epoch)