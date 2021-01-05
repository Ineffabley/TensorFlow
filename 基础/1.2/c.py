#随机数
import  tensorflow as tf
#服从指定正态分布的序列”中随机取出指定个数的值
#tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
#       shape: 输出张量的形状，必选
#         mean: 正态分布的均值，默认为0
#         stddev: 正态分布的标准差，默认为1.0
#         dtype: 输出的类型，默认为tf.float32
#         seed: 随机数种子，是一个整数，当设置之后，每次生成的随机数都一样
#         name: 操作的名称

norm=tf.random_normal([2,3],mean=-1,stddev=4)
c=tf.constant([[1,2],[3,4],[5,6]])
#用来对一个元素序列进行重新排序(随机的)
shuff=tf.random_shuffle(c)

sess=tf.Session()
print(sess.run(norm))
print(sess.run(shuff))