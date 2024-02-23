import os
import sys
import json
import natsort
from glob import glob
from PIL import Image
import pandas as pd


################### 将一个目录下所有的jpg图片的json标签, 以及对应羽毛球的csv，合并为一个json文件 #####################




##################### 标注工具使用labelme, 使用以下补丁增加关键点序号显示 #####################
# diff --git a/labelme/shape.py b/labelme/shape.py
# index 527ab95..8f08191 100644
# --- a/labelme/shape.py
# +++ b/labelme/shape.py
# @@ -257,6 +257,11 @@ class Shape(object):
#              if vrtx_path.length() > 0:
#                  painter.drawPath(vrtx_path)
#                  painter.fillPath(vrtx_path, self._vertex_fill_color)
# +
# +                point = self.points[i]
# +                painter.drawText(int(point.x()+4), int(point.y()+4), str(self.label))
# +                print("drawText: {}".format(self.label))
# +
#              if self.fill and self.mask is None:
#                  color = self.select_fill_color if self.selected else self.fill_color
#                  painter.fillPath(line_path, color)



# 将一个目录下所有的jpg图片的json标签, 以及对应羽毛球的csv，合并为一个json文件
# image_dir: match/images/1     --->        match/labels/1.json
def merge_dir_labels(image_dir):
    # 获取该目录对应的csv
    csv_path = "{}.csv".format( image_dir.replace("images", "labels") )
    df = pd.read_csv(csv_path)


    # 获取目录下所有的jpg,全路径
    file_paths = glob(os.path.join(image_dir, '*jpg'))

    # 对文件名按序号进行排序
    sorted_file_paths = natsort.natsorted(file_paths, alg=natsort.ns.IGNORECASE)

    images_labels_json = []     # 一个目录所有图生成一个json, 包括羽毛球

    for file_path in sorted_file_paths:
        image_labels_json = {}      # 一张图对应的标签dict

        # 获取羽毛球标签
        frame_num = os.path.basename(file_path).split(".")[0]
        shuttle = df[df['frame_num'] == int(frame_num)].drop('frame_num', axis=1).values

        img = Image.open(file_path)

        label_path = file_path.replace('.jpg', '.json')
        with open(label_path, 'r') as file:
            data = json.load(file)
            print(label_path)

            kps = []                    # 一张图对应的标签关键点
            for _ in range(32):
                kps.append([0, 0, 0])

            for shape in data["shapes"]:
                index = shape["label"]
                point = shape["points"][0]  # 嵌套了list

                point[0] /= img.size[0]
                point[1] /= img.size[1]
                point.append(1) # visible

                index = int(index) - 1      # 标注时使用1-32， 替换为0-31
                assert index <= 31 and index >= 0

                kps[index] = point

            image_labels_json['image'] = os.path.basename(file_path)    # 只保存文件名,不要路径
            image_labels_json['shuttle'] = [shuttle[0][1], shuttle[0][2], int(shuttle[0][0])]
            image_labels_json['court'] = kps

            images_labels_json.append(image_labels_json)

    label_dir = image_dir.replace("images", "labels")
    with open(label_dir + '.json', 'w') as file:
        json.dump(images_labels_json, file)


# 处理一个match目录下的所有的图片目录
# 目录结构: match/images/1  match/images/2  match/images/3  ....
# math_path: match      --->        match/labels/1.json     match/labels/2.json     match/labels/3.json     ....
def merge_dir_labels_batch(math_path):
    images_dir = os.path.join(math_path, "images")     # xxx/match ---> xxx/match/images
    labels_dir = os.path.join(math_path, "labels")     # xxx/match/labels

    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)

    for dir in os.listdir(images_dir):
        image_dir_tmp = os.path.join(images_dir, dir)

        # 只处理目录
        if os.path.isdir(image_dir_tmp):
            merge_dir_labels(image_dir_tmp)



if __name__ == "__main__":
    try:
        math_path = sys.argv[1]
        if not math_path:
            raise ''
    except:
        print('usage: python merge_labelme.py match_path')
        exit(1)
          
    # folder_path = "xxx/match/court1"
    # merge_dir_labels(folder_path)
    
    merge_dir_labels_batch(math_path)
