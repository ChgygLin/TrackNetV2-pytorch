import torch
import torchvision

from torch.utils.data import Dataset, DataLoader

import os
import cv2
import json
import pandas as pd
import numpy as np

from utils.augmentations import random_perspective, Albumentations, augment_hsv, random_flip


class ToTensor:
    # YOLOv5 ToTensor class for image preprocessing
    def __init__(self, half=False):
        super().__init__()
        self.half = half

    def __call__(self, im):  # im = np.array HWC in BGR order
        im = np.ascontiguousarray(im.transpose((2, 0, 1))[::-1])  # HWC to CHW -> BGR to RGB -> contiguous
        im = torch.from_numpy(im)  # to torch
        im = im.half() if self.half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0-255 to 0.0-1.0
        return im


# num_workers   https://zhuanlan.zhihu.com/p/568076554
# batch size 20    nw 1 ---> 1.35   nw 8 ---> 1.75
def create_dataloader(path,
                      imgsz=[288, 512],
                      batch_size=1,
                      sq=3,
                      augment=False,
                      workers=8,
                      shuffle=False):
    print("create dataloader image size: {}".format(imgsz))

    dataset = LoadImagesAndLabels(path, imgsz, batch_size, sq, augment)

    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers
    # print("num_workers: {}".format(nw))

    return DataLoader(dataset, batch_size=batch_size, num_workers=nw, shuffle=shuffle) 

# assert match/images exist, return path/images/xxx list
def get_match_image_list(match_path):
    base_images = os.path.join(match_path, "images")
    assert os.path.exists(base_images), base_images+" is invalid"

    image_dir_list = [f.path for f in os.scandir(base_images) if f.is_dir()]
    return image_dir_list

# return all match image list
def get_rally_image_list(rally_path):
    image_dir_list = []

    for match_name in os.listdir(rally_path):
        image_dir_list.extend(get_match_image_list(os.path.join(rally_path, match_name)))

    return image_dir_list

class LoadImagesAndLabels(Dataset):
    def __init__(self, 
                 path, 
                 imgsz=[288, 512], 
                 batch_size=1,
                 sq=3,# 网络输入几张图
                 augment=False,
                 ):
        self.imgsz = imgsz
        self.path = path
        self.batch_size = batch_size
        self.sq = sq
        self.augment = augment
        self.albumentations = Albumentations(imgsz) if augment else None

        self.image_dir_list = []               # 所有的图片目录
        self.label_path_list = []               # 所有的样本路径，文件， 与image_dir_list的元素一一对应
        self.label_data_list = []       # labels
        self.lens = []   # 总共的样本数量


        # image : "./dataset1/match2/images/1_0_1"
        # image list : ["./dataset1/match2/images/1_0_1", "./dataset1/match2/images/1_1_1"]

        # match : "./dataset1/match2"
        # match list : ["./dataset1/match1", "./dataset1/match2"]

        # rally : "Professional"
        # rally list : ["Amateur", "Professional"]

        for p in path if isinstance(path, list) else [path]:
            if p[-1] == '/':
                p = p[:-1]

            if "images" in p:   # image 
                self.image_dir_list.append(p)
            elif "match" in p:  # match
                self.image_dir_list.extend(get_match_image_list(p))
            else:               # rally
                self.image_dir_list.extend(get_rally_image_list(p))


            # 校验csv标签长度和img目录文件数量是否一致
        
        print("\n")
        print(self.image_dir_list)

        # TODO::::::::
        # Check cache

        # 读取csv
        print("\n")
        for image_dir in self.image_dir_list:
            label_path = "{}.{}".format(image_dir.replace('images', 'labels'), "json")
            self.label_path_list.append(label_path)

            with open(label_path, 'r') as file:
                label_data = json.load(file)           # 一个json文件保存一个图片目录的多张图片
                label_data_len = len(label_data)

                self.lens.append(label_data_len - self.sq + 1)     # 123 -> 234 -> 345
                self.label_data_list.append(label_data)

                print("{} len: {}".format(label_path, label_data_len))
                # img_name = js_data[img_num]['image']
                # kps = js_data[img_num]['kps']

        print("\n")


    def __len__(self):
        return sum(self.lens)
    
    def __getitem__(self, index):
        rel_index = index

        # 判断当前样本在哪个df中, index从0开始
        for ix in range(len(self.lens)):
            if rel_index < self.lens[ix]:
                break
            else:
                rel_index -= self.lens[ix]

        #print("sample {}  use label:{}  relative index: {}".format(index, self.label_path_list[ix], rel_index))

        # 获取sq张图片
        # images :      [sq][3][h][w]
        # hms_kps:      [sq][32][h][w]
        # images_kps:   [sq][32][x][y]
        # images_name:  [sq]
        images, hms_kps, images_kps, images_name = self._get_sample(self.image_dir_list[ix], self.label_data_list[ix], rel_index)

        return images, hms_kps, images_kps, images_name


    def _get_sample(self, image_dir, label_data, image_rel_index):
        images = []
        hms_kps = []
        images_kps = []
        images_name = []

        w = self.imgsz[1]
        h = self.imgsz[0]

        for i in range(self.sq):
            image_path = image_dir + "/" + label_data[image_rel_index+i]['image']
            img = cv2.imread(image_path)  # BGR

            interp = cv2.INTER_LINEAR if (self.augment) else cv2.INTER_AREA
            img = cv2.resize(img, (w, h), interpolation=interp)


            # 32个点
            kps_frac = np.array(label_data[image_rel_index+i]['kps'])
            assert len(kps_frac) == 32

            # kps_frac -> kps_int
            kps_int = kps_frac * np.array([w, h, 1]).T      # width, height, visible
            kps_xy = kps_int[:, :2]    # xy

            if self.augment:
                img, kps_xy = random_perspective(img, kps_xy)

                img, kps_xy = self.albumentations(img, kps_xy)

                augment_hsv(img, hgain=0.015, sgain=0.7, vgain=0.4)

                img, kps_xy = random_flip(img, kps_xy)

                # kps_int will return
                kps_int[:, :2] = kps_xy
                kps_int = kps_int.astype(int)

                # from utils.general import visualize_kps
                # visualize_kps(img, kps_int)


            hm_kps = np.zeros((len(kps_int), h, w), dtype=np.float32)
            for i in range(len(kps_int)):
                # x, y, visible
                if kps_int[i][2]:
                    x = kps_int[i][0]
                    y = kps_int[i][1]

                    heatmap = self._gen_heatmap(w, h, x, y)
                    hm_kps[i] = heatmap
                else:
                    heatmap = self._gen_heatmap(w, h, -1, -1)
                    hm_kps[i] = heatmap

            img = ToTensor()(img)

            images.append(img)
            hms_kps.append(hm_kps)
            images_kps.append(kps_int)
            images_name.append(image_path)

        images = torch.concatenate(images)  # 平铺RGB维度
        hms_kps = torch.tensor(np.array(hms_kps), requires_grad=False, dtype=torch.float32)

        return images, hms_kps, images_kps, images_name


    def _gen_heatmap(self, w, h, cx, cy, r=2.5, mag=1):
        if cx < 0 or cy < 0:
            return np.zeros((h, w))
        x, y = np.meshgrid(np.linspace(1, w, w), np.linspace(1, h, h))
        heatmap = ((y - (cy + 1))**2) + ((x - (cx + 1))**2)
        heatmap[heatmap <= r**2] = 1
        heatmap[heatmap > r**2] = 0
        
        return heatmap*mag


    def _make_gaussian(size=(1920, 1080), center=(0.5, 0.5), fwhm=(5, 5)):
        """ Make a square gaussian kernel.

        size:   side of the square
        center: central point
        fwhm:   Diameter

        source: https://stackoverflow.com/questions/7687679/how-to-generate-2d-gaussian-with-python
        """

        x = np.arange(0, size[0], 1, float)
        y = np.arange(0, size[1], 1, float)[:,np.newaxis]

        x0 = size[0]*center[0]
        y0 = size[1]*center[1]

        return np.exp(-4*np.log(2) * ((x-x0)**2/fwhm[0]**2 + (y-y0)**2/fwhm[1]**2))


if __name__ == "__main__":
    if not os.path.exists('./runs/loader_test'):
        os.makedirs('./runs/loader_test')

    batch_size = 1
    sq = 3
    test_loader = create_dataloader("/home/chg/Documents/Badminton/CourtV1/match3/images/03", batch_size=batch_size, sq=sq)

    for index, (_images, _hms_kps, _, _) in enumerate(test_loader):
        hms = []
        for ii in range(sq):
            # 叠加32张热力图
            hm_kps = _hms_kps[0][ii]

            heatmaps = np.zeros((3, hm_kps[0].shape[0], hm_kps[0].shape[1]), dtype=np.float32)
            for heatmap in hm_kps:
                heatmap = heatmap.repeat(3,1,1)
                heatmaps = cv2.addWeighted(heatmaps, 1, np.array(heatmap), 1, gamma=0)

            heatmaps = torch.tensor(np.array(heatmaps), requires_grad=False, dtype=torch.float32)
            hms.append(heatmaps)
        # torchvision.utils.save_image(hms, './runs/loader_test/batch{}_{}_heatmap.png'.format(index, jj))

        ims = []
        for ii in range(sq):
            im = _images[0,(0+ii*3,1+ii*3,2+ii*3),:,:]
            ims.append(im)
            hms.append(im)
        # torchvision.utils.save_image(ims, './runs/loader_test/batch{}_{}_image.png'.format(index, jj))

        print(len(hms))
        hms = torchvision.utils.make_grid(hms, nrow=sq)
        torchvision.utils.save_image(hms, './runs/loader_test/batch{}_{}.png'.format(index, 0))

        if index >= 10:
            break


