# TrackNetV2-pytorch

Paper:	**TrackNetV2: Efﬁcient Shuttlecock Tracking Network**

Original Project（tensorflow）:	https://nol.cs.nctu.edu.tw:234/open-source/TrackNetv2

> <del>官方上传的标注工具、数据集均已失效。</del>del>
>
> The author has now reuploaded the dataset。

Paper reading：[TrackNetV2论文记录与pytorch复现](https://zhuanlan.zhihu.com/p/624900770)



## Inference with pytorch weights converted from tensorflow weights:

```shell
git apply tf2torch/diff.txt
python detect.py --source xxx.mp4 --weights  ./tf2torch/track.pt --view-img		# TrackNetv2/3_in_3_out/model906_30
```



## Inference:

```
python detect.py --source xxx.mp4 --weights  xxx.pt --view-img
```



## Training:

```
# training from scratch
python train.py --data data/match.yaml

# training from pretrain weight
python train.py --weights xxx.pt --data data/match.yaml

# resume training
python train.py --data data/match.yaml --resume
```



## Evaluation:

```shell
python val.py --weights xxx.pt --data data/match.yaml
```





## DataSet:

```
# TrackNetV2 dataset
#	/home/chg/Badminton/TrackNetV2
#	- Amateur  
#	- Professional  
#	- Test

python tools/handle_tracknet_dataset.py /home/chg/Badminton/TrackNetV2/Amateur
python tools/handle_tracknet_dataset.py /home/chg/Badminton/TrackNetV2/Professional
python tools/handle_tracknet_dataset.py /home/chg/Badminton/TrackNetV2/Test

python tools/Frame_Generator_rally.py /home/chg/Badminton/TrackNetV2/Amateur
python tools/Frame_Generator_rally.py /home/chg/Badminton/TrackNetV2/Professional
python tools/Frame_Generator_rally.py /home/chg/Badminton/TrackNetV2/Test


# TrackNetV2 dataset config : data/match.yaml
path: /home/chg/Documents/Badminton/TrackNetV2
train:
    - Amateur
    - Professional 
val:
    - Test
    
# also you can use follow config for testing
train:
    - Test/match1/images/1_05_02
val:
    - Test/match2/images/1_03_03

# or
train:
    - Test/match1
val:
    - Test/match2

```



## Reference：

https://github.com/mareksubocz/TrackNet

https://nol.cs.nctu.edu.tw:234/open-source/TrackNetv2

https://github.com/ultralytics/yolov5