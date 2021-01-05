import tensorflow as tf
#创建矩阵
sess = tf.Session()
tf.zeros([3, 4], tf.int32)  # 创建一个三行四列的矩阵，数据类型为int32
print(sess.run(tf.zeros([3, 4], tf.int32)))
tensor = [[1, 2, 3], [4, 5, 6]]
tf.zeros_like(tensor)
tf.ones([2, 3], tf.int32)
tensor1 = [[1, 2, 3], [4, 5, 6]]
tf.ones_like(tensor1)
tensor2 = tf.constant([1, 2, 3, 4, 5, 6, 7])
tensor3 = tf.constant(-1.0, shape=[3, 4])  # 3行4列 -1.0
print(sess.run(tensor2))
print(sess.run(tensor3))
#这个函数主要的参数就这三个，start代表起始的值，end表示结束的值，num表示在这个区间里生成数字的个数，生成的数组是等间隔生成的。
tensor4 = tf.linspace(10.0, 12.0, 3, name='linspace')
print(sess.run(tensor4))
print(sess.run(tf.range(3, 18, 3)))  # 指定步长为3
