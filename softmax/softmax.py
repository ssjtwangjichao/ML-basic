# coding=utf-8
#算法解析： tensorflow实现线性回归


# 引入python模块
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 引用mnist数据集
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 构造模型
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]), name="w")
b = tf.Variable(tf.zeros([10]), name="b")
y = tf.nn.softmax(tf.matmul(x, W) + b)

print(y)

# 定义损失和优化函数
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

#初始化所有变量
init_op = tf.global_variables_initializer()
saver = tf.train.Saver(tf.global_variables())

# 训练并保存模型文件
with tf.Session() as sess:
    sess.run(init_op)

    for i in range(10000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    save_path = saver.save(sess, "tmp_mnist/model.ckpt")
    print("Model saved in file: ", save_path)

