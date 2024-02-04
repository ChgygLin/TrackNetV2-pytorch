import torch
import torchvision

from torch.utils.data import Dataset, DataLoader

import os
import cv2
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
        self.df_list = []       # labels
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
        for csv_path in self.image_dir_list:
            label_path = "{}.{}".format(csv_path.replace('images', 'labels'), "csv")
            self.label_path_list.append(label_path)

            df = pd.read_csv(label_path)
            df_len = len(df.index)

            self.lens.append(df_len - self.sq + 1)
            self.df_list.append(df)
            print("{} len: {}".format(csv_path, df_len))
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

        w = self.imgsz[1]
        h = self.imgsz[0]
        df = self.df_list[ix]

        heatmaps = []

        for iy in range(self.sq):
            visible = df['visible'][rel_index+iy]
            x = df['x'][rel_index+iy]
            y = df['y'][rel_index+iy]

            if visible == 0:
                heatmap = self._gen_heatmap(w, h, -1, -1)
            else:
                heatmap = self._gen_heatmap(w, h, int(w*x), int(h*y))
            
            heatmaps.append(heatmap)

        heatmaps = torch.tensor(np.array(heatmaps), requires_grad=False, dtype=torch.float32)
        images = self._get_sample(self.image_dir_list[ix], rel_index)

        return images, heatmaps


    def _get_sample(self, image_dir, image_rel_index):
        images = []

        for i in range(self.sq):
            image_path = image_dir + "/" + str(image_rel_index+i) + ".jpg"
            img = torchvision.io.read_image(image_path)

            img = torchvision.transforms.functional.resize(img, self.imgsz, antialias=True)
            img = img.type(torch.float32)
            img *= 1 / 255
            images.append(img)

        return torch.concatenate(images)


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
    test_loader = create_dataloader("./example_dataset/match/images/1_10_12", batch_size=batch_size)

    for index, (_images,_heatmaps) in enumerate(test_loader):
        jj = 0
        # for jj in range(batch_size):
        hms = []
        for ii in range(3):
            hm = _heatmaps[jj,ii,:,:].repeat(3,1,1)              # 奇怪，为什么不用*255就能直接得到灰度图像
            hms.append(hm)
        # torchvision.utils.save_image(hms, './runs/loader_test/batch{}_{}_heatmap.png'.format(index, jj))

        ims = []
        for ii in range(3):
            im = _images[jj,(0+ii*3,1+ii*3,2+ii*3),:,:]          # 奇怪，为什么不用*255就能直接得到彩色图像
            ims.append(im)
            hms.append(im)
        # torchvision.utils.save_image(ims, './runs/loader_test/batch{}_{}_image.png'.format(index, jj))

        hms = torchvision.utils.make_grid(hms, nrow=3)
        torchvision.utils.save_image(hms, './runs/loader_test/batch{}_{}.png'.format(index, jj))

        if index >= 10:
            break


