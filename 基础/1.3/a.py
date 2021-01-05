import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#随机生成1000个点，在y=0.1x+0.3的直线周围
num_points=1000
vectors_set=[]
for i in range(num_points):
    x1=np.random.normal(0.0,0.55)  #均值，标准差
    y1=x1*0.1+0.3+np.random.normal(0.0,0.03)
    vectors_set.append([x1,y1])

#生成一些样本
x_data=[v[0] for v in vectors_set]
y_data=[v[1] for v in vectors_set]
#绘制散点图
#plt.scatter(x_data,y_data,c='r')
#plt.show()

W=tf.Variable(tf.random_uniform([1],-1.0,1.0),name='W')
b=tf.Variable(tf.zeros([1]),name='b')
# 经计算得出预估值y
y=W*x_data+b

#以预估值和实际值之间的均方误差作为损失
loss=tf.reduce_mean(tf.square(y-y_data),name='loss')
#递归下降预测，0.5是学习率
optimizer=tf.train.GradientDescentOptimizer(0.5)
#训练，最小化误差
train=optimizer.minimize(loss,name='train')

sess=tf.Session()
init=tf.global_variables_initializer()
sess.run(init)

#初始化的值
print("W=",sess.run(W),"b=",sess.run(b),'loss=',sess.run(loss))

#执行20次训练
for step in range(20):
    sess.run(train)
    #训练好的值
    print("W=", sess.run(W), "b=", sess.run(b), 'loss=', sess.run(loss))

plt.scatter(x_data,y_data,c='r')
plt.plot(x_data,sess.run(W)*x_data+sess.run(b))
plt.show()