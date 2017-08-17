# Brouhaha:基于iOS Metal的深度学习运算库

Brouhah是一个基于iOS Metal的深度学习运算库。这个库可以方便的调用iOS Metal Shader执行深度学习算法。

## 地址
github:https://github.com/amazingyyc/Brouhaha
<br>
码云:https://gitee.com/JingQiManHua/Brouhaha

## 介绍
Brouhaha只包含深度学习的前向运算，并不能用于训练一个深度学习模型。在使用Brouhaha之前必须有一个使用其他的训练库（比如：Caffe，Tensorflow，Torch）训练好的深度学习model。Brouhaha包含常用的卷积（包括转置卷积，Dilated卷积），池化，激活，全联接，BatchNormalize和方便图片转换的转换层。主要包括以下三个部分：
1. **BrouhahaMetal:** 使用Metal Shader编写的核心运算函数，用于加速计算。
2. **Brouhaha:** 包含常用的神经网络层的抽象，使用Objective-c开发，为了加速引入了一些汇编。
3. **Brouhaha-Demo:** 包含两个Demo，演示怎么使用这个库。LeNet是一个使用卷积神经网络识别图片中的数字的模型。ArtTransform类似于Prisma，用于图片风格的转换。

为了速度模型中的float32类型的数值会转换成float16进行计算，会损失一些精度。这个库仍然在开发中，因此API还不稳定。

## Demo
**Build:** 在运行Brouhaha-Demo之前需要首先编译BrouhahaMetal，然后将生成的文件BrouhahaMetal.metallib拷贝到Brouhaha-Demo的bundle中。
<br>
**LeNet:** 这个Demo是使用神经网络识别图片中的数字。具体的算法参考：http://yann.lecun.com/exdb/lenet/。模型文件来源于网路，抱歉忘记了出处。
![](Images/lente.gif)

**ArtTransform:** 这个Demo使用卷积神经网络进行图片风格的转换。算法参考：https://arxiv.org/abs/1603.08155，模型文件来源于：https://github.com/lengstrom/fast-style-transfer#video-stylization。
<br>
![](Images/arttransform.gif)

## Brouhaha的优势
Brouhaha使用GPU代替CPU来提升运算速度。同时使用float16代替float32顺然降低一些精度，但是会提升整体的运行速度。Brouhaha只依赖于iOS Metal，因此支持iOS8的设备，不像Apple Core ML和其他的第三方库需要iOS10+。

## 未来工作
1. 支持float32。
2. 支持RNN。