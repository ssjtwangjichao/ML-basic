# coding=utf-8
#算法解析： RNN算法

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials import mnist
from tensorflow.contrib import rnn
mnist = mnist.input_data.read_data_sets('./data',one_hot=True)

lr = 0.01  # 学习率
epochs = 30
batch_size = 64
input_x = 28 # 输入序列的长度，因为是28×28的大小，所以每一个序列我们设置长度为28，每一个输入都是28个像素点
time_steps = 28 #因为没有张图像为28×28,而每一个序列长度为1×28,所以总共28个1×28,
output_y = 10  #输入为10,因为共10类
hidden_n = 128 #隐层的大小，这个参数就是比如我们输入是1×28的矩阵大小，隐藏为128,就是将输入维度变为1×128,当然lstm输入也是1×128

