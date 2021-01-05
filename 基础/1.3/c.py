import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('data/', one_hot=True)
# trainimg = mnist.train.images
# trainlabel = mnist.train.labels
# testimg = mnist.test.images
# testlabel = mnist.test.labels
# print("MNIST LOAD")
# print(trainimg.shape)
# print(trainlabel.shape)
# print(testimg.shape)
# print(testimg.shape)
#
# print(trainimg)
# print(trainlabel[0])

# 将其指定的形状[无，784]，其中784是一个单一的维数变平28由28像素MNIST图像，并且无指示第一维度，对应于批量大小，可以是任何大小
# 为训练数据集的输入 x 和标签 y 创建占位符
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])  # 每个样本输出十个值
# 创建学习变量、权重和偏置
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
# 创建逻辑回归模型
# 预测值结果
actv = tf.nn.softmax(tf.matmul(x, W) + b)
# 损失值
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(actv), reduction_indices=1))

learning_rate = 0.01
optm = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
# 0:列最大值索引  1：行最大值索引
# equal：对比索引
pred = tf.equal(tf.arg_max(actv, 1), tf.arg_max(y, 1))  # 预测 /真实
# 求准确率，记得类型转换
accr = tf.reduce_mean(tf.cast(pred, 'float'))
init = tf.global_variables_initializer()
sess = tf.Session()

training_epochs = 50
batch_size = 100
display_step = 5

sess.run(init)

for epoch in range(training_epochs):
    avg_cost = 0;
    num_batch = int(mnist.train.num_examples / batch_size)
    for i in range(num_batch):
        # 返回值batch_xs，batch_ys仍是二阶张量,说白了就是只有100行的mnist数组
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(optm, feed_dict={x: batch_xs, y: batch_ys})
        feeds = {x: batch_xs, y: batch_ys}
        avg_cost += sess.run(cost, feed_dict=feeds) / num_batch

    if (epoch % display_step == 0):
        feeds_train = {x: batch_xs, y: batch_ys}
        feeds_test = {x: mnist.test.images, y: mnist.test.labels}
        train_acc = sess.run(accr, feed_dict=feeds_train)
        test_acc = sess.run(accr, feed_dict=feeds_test)
        print("Epoch:%03d %03d -- cost:%.9f -- train_acc:%.3f -- test_acc: %.3f" % (
        epoch, training_epochs, avg_cost, train_acc, test_acc))
print("DONE")
