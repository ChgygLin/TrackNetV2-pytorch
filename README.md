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



参考代码：

https://github.com/mareksubocz/TrackNet

https://nol.cs.nctu.edu.tw:234/open-source/TrackNetv2

https://github.com/ultralytics/yolov5