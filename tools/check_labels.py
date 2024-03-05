import os
import glob
import json
import pandas as pd
import numpy as np

# 基本的数据校验

base_path = "/home/chg/Documents/Badminton/merge_dataset/TrackNetV2/"


# Amateur/match1
def handle_rally(match_path):
    images_path = "{}/images".format(match_path)
    labels_path = "{}/labels".format(match_path)

    # 校验json标签
    for images_dir in os.listdir(images_path):
        for js_dir in glob.glob(os.path.join(images_path+"/"+images_dir, '*.json')):
            print(js_dir)

            with open(js_dir, 'r') as file:
                data = json.load(file)

                for shape in data["shapes"]:
                    index = shape["label"]
                    
                    try:
                        index = int(index)
                    except ValueError:
                        print("Error index: {}".format(index))
                        exit()

                    if index<=0 or index >=33:
                        print("Error index: {}".format(index))
                        exit()

    # 校验csv标签
    # for labels_dir in os.listdir(labels_path):
    for csv_dir in glob.glob(os.path.join(labels_path+"/", '*.csv')):
        images_base_path = csv_dir.replace("labels", "images").split(".")[0]

        print(csv_dir)
        df = pd.read_csv(csv_dir)
        frame_nums = df["frame_num"].values

        assert(np.all(np.ediff1d(frame_nums) == 1))

        for frame_num in frame_nums:
            if not os.path.exists("{}/{}.jpg".format(images_base_path, frame_num)):
                print("{}/{}.jpg".format(images_base_path, frame_num))
                exit()

# Amateur
def handle_rally_batch(batch_path):
    for rally_dir in os.listdir(batch_path):

        match_path = os.path.join(batch_path, rally_dir)
        handle_rally(match_path)

# from_path: 
def handle_base_path(base_path):
    for to_batch_dir in os.listdir(base_path):
        to_batch_path = os.path.join(base_path, to_batch_dir)

        handle_rally_batch(to_batch_path)


handle_base_path(base_path)