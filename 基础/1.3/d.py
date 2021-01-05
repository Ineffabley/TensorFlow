import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('data/', one_hot=True)


# 2. 为训练数据集的输入 x 和标签 y 创建占位符
X = tf.placeholder(tf.float32, [None, 784], name='X')
Y = tf.placeholder(tf.float32, [None, 10], name='Y')
# 3. 创建学习变量、权重和偏置：
w = tf.Variable(tf.zeros([784, 10]), name='w')
b = tf.Variable(tf.zeros([10]), name='b')

# 4. 创建逻辑回归模型。TensorFlow OP 给出了 name_scope（"wx_b"）：
# name_scope('wx_b'):
with tf.name_scope('wx_b') as scope:
    # 根据张量加法规则,(m,n)张量可以和(n,1)张量相加，效果就是(m,n)中每一行都和(n,1)相加
    # 例如，[[0,0,0],[0,0,0]]+[1,2,3]=[[1,2,3],[1,2,3]]
    y_hat = tf.nn.softmax(tf.matmul(X, w) + b)

# 5. 训练时添加 summary 操作来收集数据。
# 使用直方图以便看到权重和偏置随时间相对于彼此值的变化关系。可以通过 TensorBoard Histogtam 选项卡看到
w_h = tf.summary.histogram('weights', w)
b_h = tf.summary.histogram('biases', b)

# 6. 定义交叉熵（cross-entropy）和损失（loss）函数，并添加 name scope 和 summary 以实现更好的可视化。
# 使用 scalar summary 来获得随时间变化的损失函数。scalar summary 在 Events 选项卡下可见：
with tf.name_scope('cross-entropy') as scope:
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_hat, labels=Y))
    tf.summary.scalar('cross-entropy', loss)

# 7. 定义优化器。为了更好地可视化，定义一个 name_scope
with tf.name_scope('Train') as scope:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

# 8. 进行初始化：
init_op = tf.global_variables_initializer()

# 9. 组合所有的 summary 操作
merged_summary_op = tf.summary.merge_all()
# 10. 定义精度.逻辑回归的精度是这么定义的嘛？？
correct_prediction = tf.square(Y - y_hat)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float32'))

# 11.定义会话并将所有的 summary 存储在定义的文件夹中
batch_size = 100
with tf.Session() as sess:
    sess.run(init_op)
    writer = tf.summary.FileWriter('Z:\python_code\CBIANCHENG_relevant_code\Graphs', sess.graph)
    for epoch in range(100):
        avg_loss = 0
        num_of_batch = int(mnist.train.num_examples / batch_size) # =int(55000/100)=550
        for i in range(num_of_batch):
            # get next batch of data, 返回值batch_xs，batch_ys仍是二阶张量，说白了就是只有100行的mnist数组
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, l, summary_str = sess.run([optimizer, loss, merged_summary_op], feed_dict={X: batch_xs, Y: batch_ys})
            avg_loss += l
            writer.add_summary(summary_str, epoch * num_of_batch + i)  # add all summary per batch
        avg_loss = avg_loss / num_of_batch
        print('epoch{0}:loss{1}'.format(epoch, avg_loss))
    print('done')
    print(sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))
    writer.close()