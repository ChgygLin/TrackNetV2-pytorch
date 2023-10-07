# TrackNetV2-pytorch

论文:	**TrackNetV2: Efﬁcient Shuttlecock Tracking Network**

官方代码（tensorflow版）:	https://nol.cs.nctu.edu.tw:234/open-source/TrackNetv2

> <del>官方上传的标注工具、数据集均已失效。</del>del>
>
> 目前原作者已重新上传数据集。

论文解析：[TrackNetV2论文记录与pytorch复现](https://zhuanlan.zhihu.com/p/624900770)



使用tensorflow转换的pytorch权重(转换方法详见上文)

```shell
git apply tf2torch/diff.txt
python detect.py --source xxx.mp4 --weights  ./tf2torch/track.pt --view-img
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



## DataSet:

```
# TrackNetV2 dataset
#	/home/chg/Badminton/TrackNetV2
#	- Amateur  
#	- Professional  
#	- Test

python tools/handle_tracknet_dataset.py /home/chg/Badminton/TrackNetV2

python Frame_Generator_rally.py /home/chg/Badminton/TrackNetV2/Amateur
python Frame_Generator_rally.py /home/chg/Badminton/TrackNetV2/Professional
python Frame_Generator_rally.py /home/chg/Badminton/TrackNetV2/Test


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



参考代码：

https://github.com/mareksubocz/TrackNet

https://nol.cs.nctu.edu.tw:234/open-source/TrackNetv2

https://github.com/ultralytics/yolov5