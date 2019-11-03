# coding:utf-8

import os
import numpy as np
import tensorflow as tf
import transform

# 打开tensorflow的可视化工具
# tensorboard --logdir "/Users/yanyuanchi/code/python/readtffile/see"

# 参数存储路径
params_path = "/Users/yanyuanchi/code/python/readtffile/params/"

def save_np_array(array, path):
    print "参数:", path
    print "数据shape:", np.shape(array)
    print "-------------------------------------------------------------------"

    f = file(path, "wb")

    array.flatten().astype("float32").tofile(f)

    f.close()

# tf 的卷积的权值存储为(h, w, input_channel, output_channel)
# 需要转换成(output_channel, h, w, input_channel)
def save_tf_conv_np_array(array, path):
    array = np.moveaxis(array, -1, 0)
    save_np_array(array, path)

# tf 的卷积的权值存储为(h, w, output_channel, input_channel)
# 需要转换成(output_channel, h, w, input_channel)
def save_tf_tranpose_conv_np_array(array, path):
    array = np.moveaxis(array, 2, 0)

    outChannel, h, w, inChannel = np.shape(array)

    for out in range(outChannel):
        temp = array[out].copy()

        for y in range(h):
            for x in range(w):
                in1 = temp[h - y - 1][w - x - 1]
                out1 = array[out][y][x]

                for l in range(inChannel):
                    out1[l] = in1[l]

    save_np_array(array, path)

image_height = 228
image_width  = 228

checkpoint_dir = "/Users/yanyuanchi/code/python/readtffile/model/la_muse.ckpt"

g = tf.Graph()
# allow_soft_placement=True ： 如果你指定的设备不存在，允许TF自动分配设备
soft_config = tf.ConfigProto(allow_soft_placement=True)
soft_config.gpu_options.allow_growth = True

with g.as_default(), g.device("/cpu:0"), tf.Session(config=soft_config) as sess:
    batch_shape = (1, image_height, image_width, 3)
    img_placeholder = tf.placeholder(tf.float32, shape=batch_shape, name='img_placeholder')
    preds = transform.net(img_placeholder)

    saver = tf.train.Saver()
    saver.restore(sess, checkpoint_dir)

    variables = tf.trainable_variables()
    # variables = tf.model_variables()

    # for var in variables:
    #     print var.name
    # print variables

    # for var in variables:
    #     name = var.name
    #     realValue = var._variable.eval()
    #
    #     print name
    #     print np.shape(realValue)
    #     print "-------------------------------------"


    # 这个项目对应的结构是

    # 3个卷积
    # 1:卷积(weight)，batchnormalization(alpha, beta) 3个变量
    # 2:卷积(weight)，batchnormalization(alpha, beta) 3个变量
    # 3:卷积(weight)，batchnormalization(alpha, beta) 3个变量

    # 5个res层 bn 的变量顺序为 beta alpha
    # 4:res层 卷积（weights），batchnormalization（alpha，beta）卷积（weights），batchnormalization（alpha，beta） 6个变量
    # 5:res层 卷积（weights），batchnormalization（alpha，beta）卷积（weights），batchnormalization（alpha，beta） 6个变量
    # 6:res层 卷积（weights），batchnormalization（alpha，beta）卷积（weights），batchnormalization（alpha，beta） 6个变量
    # 7:res层 卷积（weights），batchnormalization（alpha，beta）卷积（weights），batchnormalization（alpha，beta） 6个变量
    # 8:res层 卷积（weights），batchnormalization（alpha，beta）卷积（weights），batchnormalization（alpha，beta） 6个变量

    # 两个转置卷积
    # 9:转置卷积(weights),batchnormalization(alpha, beta) 3个变量
    # 10:转置卷积(weights),batchnormalization(alpha, beta) 3个变量

    # 卷积
    # 11:卷积(weights),batchnormalization(alpha, beta) 3个变量

    i = 0
    for var in variables:
        name      = var.name
        realValue = var._variable.eval()

        if i < 9:
            name = "conv" + str(i / 3 + 1)
            if 0 == i % 3:
                save_tf_conv_np_array(realValue, params_path + name + "_weight")
            elif 1 == i % 3:
                save_np_array(realValue, params_path + name + "_beta")
            else:
                save_np_array(realValue, params_path + name + "_alpha")
        elif i < 39:
            # res层
            j = i - 9
            name = "res" + str(j / 6 + 1)
            if j % 6 < 3:
                name += "_conv1"
            else:
                name += "_conv2"

            z = (j % 6)

            if 0 == z % 3:
                save_tf_conv_np_array(realValue, params_path + name + "_weight")
            elif 1 == z % 3:
                save_np_array(realValue, params_path + name + "_beta")
            else:
                save_np_array(realValue, params_path + name + "_alpha")
        elif i < 45:
            j = i - 39

            name = "transpose_conv" + str(j / 3 + 1)

            z = j % 6

            if 0 == z % 3:
                # (h, w, outputchannel, inputchannel)
                # to (outputchannel, h, w, inputchannel)
                save_tf_tranpose_conv_np_array(realValue, params_path + name + "_weight")
            elif 1 == z % 3:
                save_np_array(realValue, params_path + name + "_beta")
            else:
                save_np_array(realValue, params_path + name + "_alpha")
        elif i < 48:
            j = i - 45
            name = "conv4"

            if 0 == j % 3:
                save_tf_conv_np_array(realValue, params_path + name + "_weight")
            elif 1 == j % 3:
                save_np_array(realValue, params_path + name + "_beta")
            else:
                save_np_array(realValue, params_path + name + "_alpha")
        else:
            raise Exception("error")

        i += 1




