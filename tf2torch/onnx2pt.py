import torch
import onnx
from onnx2torch import convert
from models.tracknet import TrackNet
from torchsummary import summary

# tensorflow weight
#https://nol.cs.nctu.edu.tw:234/open-source/TrackNetv2/tree/master/3_in_3_out/model906_30

# model906_30 ---> mode_save ---> track.onnx
#

# Path to ONNX model
onnx_model_path = './track.onnx'
# You can pass the path to the onnx model to convert it or...


# onnx_model = onnx.load(onnx_model_path)
# # https://stackoverflow.com/questions/53176229/strip-onnx-graph-from-its-constants-initializers
# onnx_model.graph.ClearField('initializer')
# torch_model = convert(onnx_model)
torch_model = convert(onnx_model_path)


onnx_dict = torch_model.state_dict()

del onnx_dict['initializers.onnx_initializer_0']
del onnx_dict['initializers.onnx_initializer_1']
del onnx_dict['initializers.onnx_initializer_2']
del onnx_dict['initializers.onnx_initializer_3']
del onnx_dict['initializers.onnx_initializer_4']
del onnx_dict['initializers.onnx_initializer_5']




track_model = TrackNet()
# print(summary(track_model,(9, 288, 512), device="cpu"))
track_dict = track_model.state_dict()



assert(len(onnx_dict)==len(track_dict))

convert_dict = {}
# print(track_dict.keys())
# print(onnx_dict.keys())

for k1, k2 in zip(onnx_dict.keys(), track_dict.keys()):
    # print(f'onnx: {k1} shape:{onnx_dict[k1].shape}     track: {k2} shape:{track_dict[k2].shape}')
    convert_dict[k2] = onnx_dict[k1]


torch.save(convert_dict, './track.pt')




# ValueError: Unknown layer: BatchNormalization.
# ValueError: Unknown loss function: custom_loss.

# /home/chg/anaconda3/envs/track/lib/python3.10/site-packages/tf2onnx/tf_loader.py

# line 645
# def custom_loss(y_true, y_pred):
#     from tensorflow.python.keras import backend as K
#     loss = (-1)*(K.square(1 - y_pred) * y_true * K.log(K.clip(y_pred, K.epsilon(), 1)) + K.square(y_pred) * (1 - y_true) * K.log(K.clip(1 - y_pred, K.epsilon(), 1)))
#     return K.mean(loss)

# def from_keras(model_path, input_names, output_names):
#     ...
#     custom_objects = {'custom_loss':custom_loss}


# size mismatch for conv2d_1.bn.weight: copying a param with shape torch.Size([512]) from checkpoint, the shape in current model is torch.Size([64]).
# 	size mismatch for conv2d_1.bn.bias: copying a param with shape torch.Size([512]) from checkpoint, the shape in current model is torch.Size([64]).
# onnx: BatchNormalization_0.weight shape:torch.Size([512])     track: conv2d_1.bn.weight shape:torch.Size([64])

# tensorflow中conv2d默认channels_last,默认情况下BatchNormalization会对最后一个通道C通道进行BN,而使用channels_first后，
# BatchNormalization 输入(N, C, H, W), 学习参数, C


# https://keras.io/api/layers/normalization_layers/batch_normalization/
# 在keras和tensorflow中，BatchNormalization默认是对输入的最后一个维度进行归一化，即axis=-1，这通常是特征维度。在pytorch中，BatchNormalization有不同的变体，如BatchNorm1d，BatchNorm2d等，它们默认是对输入的第二个维度进行归一化，即num_features。