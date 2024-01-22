# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""Image augmentation functions."""

import math
import random

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import albumentations as A


class Albumentations:
    # YOLOv5 Albumentations class (optional, only used if package is installed)
    def __init__(self, imgsz=[288, 512]):
        self.transform = None
        prefix = "albumentations: "

        T = [
            A.RandomResizedCrop(height=imgsz[0], width=imgsz[1], scale=(0.8, 1.0), ratio=(0.9, 1.11), p=0.0),
            A.Blur(p=0.1),
            A.MedianBlur(p=0.1),
            A.ToGray(p=0.1),
            A.CLAHE(p=0.1),
            A.RandomBrightnessContrast(p=0.0),
            A.RandomGamma(p=0.0),
            A.ImageCompression(quality_lower=75, p=0.0),
        ]  # transforms
        self.transform = A.Compose(T, keypoint_params=A.KeypointParams(format="xy", remove_invisible=False))

        print(prefix + ", ".join(f"{x}".replace("always_apply=False, ", "") for x in T if x.p))


    def __call__(self, im, kps, p=1.0):
        if self.transform and random.random() < p:
            new = self.transform(image=im, keypoints=kps)  # transformed
            im, kps = new["image"], np.array(new["keypoints"])

        return im, kps


def augment_hsv(im, hgain=0.5, sgain=0.5, vgain=0.5):
    # HSV color-space augmentation
    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
        dtype = im.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im)  # no return needed


def random_perspective(
    im, kps, degrees=10, translate=0.1, scale=0.1, shear=10, perspective=0.0, border=(0, 0)
):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-10, 10))

    height = im.shape[0] + border[0] * 2  # shape(h,w,c)
    width = im.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -im.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -im.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            im = cv2.warpPerspective(im, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            im = cv2.warpAffine(im, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Visualize
    # import matplotlib.pyplot as plt
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(im[:, :, ::-1])  # base
    # ax[1].imshow(im2[:, :, ::-1])  # warped
    # plt.show()

    # Transform label coordinates
    n = len(kps)

    xy = np.ones((n, 3))
    xy[:, :2] = kps         # 齐次化坐标

    xy = xy @ M.T  # transform
    xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2])  # perspective rescale or affine

    # clip
    xy[:, 0] = xy[:, 0].clip(0, width)
    xy[:, 1] = xy[:, 1].clip(0, height)

    return im, xy


def random_flip(im, kps, p=0.5):
    # Flip left-right
    if random.random() < p:
        im = np.fliplr(im)
        kps[:, 0] = im.shape[1] - kps[:, 0]     # width

    return im, kps

