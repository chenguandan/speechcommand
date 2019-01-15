# Speech Commands Example

This is a basic speech recognition example. For more information, see the
tutorial at https://www.tensorflow.org/versions/master/tutorials/audio_recognition.


_fb代表使用filter_bank特征
_p代表使用了说话人向量信息
_pair表示使用采样pair学习是否属于同一个说话人，用于提取说话人向量。

网络结构在：
resnet.py
SE_ResNeXt.py
dual_path_network.py
